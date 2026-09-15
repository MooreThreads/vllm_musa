import math

import torch
from vllm import envs as vllm_envs
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    fused_topk,
    fused_topk_bias,
)
from vllm.model_executor.utils import maybe_disable_graph_partition
from vllm.platforms import current_platform

try:
    from mate import moe_fused_gate as mate_moe_fused_gate

    is_fused_gate = True
except ImportError as e:
    raise ImportError(
        "MUSA platform requires MATE to be installed. Please install mate first."
    ) from e


def _can_use_musa_jit_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    correction_bias: torch.Tensor | None,
) -> bool:
    return (
        current_platform.is_musa()
        and hidden_states.device == gating_output.device
        and gating_output.device.type == "musa"
        and hidden_states.dim() == 2
        and gating_output.dim() == 2
        and hidden_states.shape[0] == gating_output.shape[0]
        and gating_output.is_contiguous()
        and gating_output.dtype in (torch.float32, torch.float16, torch.bfloat16)
        and 0 < topk <= gating_output.shape[1] <= 1024
        and (
            correction_bias is None
            or (
                correction_bias.device == gating_output.device
                and correction_bias.dim() == 1
                and correction_bias.shape[0] == gating_output.shape[1]
                and correction_bias.dtype == torch.float32
                and correction_bias.is_contiguous()
            )
        )
    )


def _maybe_import_musa_jit_topk():
    try:
        from vllm_musa.jit_kernel.csrc import topk as musa_jit_topk
    except (ImportError, ModuleNotFoundError):
        return None
    return musa_jit_topk


def _musa_jit_fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None,
    correction_bias: torch.Tensor | None = None,
    scoring_func: str = "softmax",
    shared_expert_gate_output: torch.Tensor | None = None,
    num_fused_shared_experts: int = 0,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not _can_use_musa_jit_topk(hidden_states, gating_output, topk, correction_bias):
        return None

    has_shared_experts = (
        shared_expert_gate_output is not None and num_fused_shared_experts > 0
    )
    has_combined_shared_experts = (
        shared_expert_gate_output is None and num_fused_shared_experts > 0
    )
    if has_shared_experts and not (
        scoring_func == "softmax"
        and correction_bias is None
        and gating_output.shape[1] == 256
        and topk == 8
        and num_fused_shared_experts == 1
        and shared_expert_gate_output.device == gating_output.device
        and shared_expert_gate_output.dim() == 2
        and shared_expert_gate_output.shape == (gating_output.shape[0], 1)
        and shared_expert_gate_output.dtype
        in (torch.float32, torch.float16, torch.bfloat16)
        and shared_expert_gate_output.is_contiguous()
    ):
        return None
    if has_combined_shared_experts and not (
        scoring_func == "softmax"
        and renormalize
        and correction_bias is None
        and gating_output.shape[1] == 257
        and topk == 8
        and num_fused_shared_experts == 1
    ):
        return None

    # The MUSA softmax kernel's correction-bias ABI returns probabilities of
    # ``logits + bias``. vLLM selects with ``softmax(logits) + bias`` while
    # returning the unbiased softmax weights, so preserve the upstream path.
    if correction_bias is not None and scoring_func == "softmax":
        return None

    musa_jit_topk = _maybe_import_musa_jit_topk()
    if musa_jit_topk is None:
        return None

    output_topk = topk + (
        num_fused_shared_experts
        if has_shared_experts or has_combined_shared_experts
        else 0
    )
    topk_weights = torch.empty(
        gating_output.shape[0],
        output_topk,
        dtype=torch.float32,
        device=gating_output.device,
    )
    topk_ids = torch.empty(
        gating_output.shape[0],
        output_topk,
        dtype=torch.int32,
        device=gating_output.device,
    )

    if scoring_func == "softmax":
        musa_jit_topk.topk_softmax(
            topk_weights,
            topk_ids,
            gating_output,
            renormalize,
            correction_bias=correction_bias,
            shared_expert_gate_output=shared_expert_gate_output,
            num_fused_shared_experts=num_fused_shared_experts,
        )
    elif scoring_func == "sigmoid":
        musa_jit_topk.topk_sigmoid(
            topk_weights,
            topk_ids,
            gating_output,
            renormalize,
            correction_bias=correction_bias,
            shared_expert_gate_output=shared_expert_gate_output,
            num_fused_shared_experts=num_fused_shared_experts,
        )
    else:
        return None

    if indices_type is not None and topk_ids.dtype != indices_type:
        topk_ids = topk_ids.to(indices_type)
    return topk_weights, topk_ids


def _compute_routing(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    indices_type: torch.dtype | None,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute routing using grouped top-k."""

    def valid_grouping() -> bool:
        # Check if num_experts is greater than num_expert_group
        # and is divisible by num_expert_group
        num_experts = router_logits.shape[-1]
        if num_experts <= self.num_expert_group:
            return False
        return num_experts % self.num_expert_group == 0

    if not valid_grouping():
        scoring_func = getattr(self, "scoring_func", "softmax")
        if self.e_score_correction_bias is not None:
            jit_result = _musa_jit_fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
                indices_type=indices_type,
                correction_bias=self.e_score_correction_bias.data,
                scoring_func=scoring_func,
            )
            if jit_result is not None:
                topk_weights, topk_ids = jit_result
                if self.routed_scaling_factor != 1.0:
                    topk_weights *= self.routed_scaling_factor
                return topk_weights, topk_ids

            topk_weights, topk_ids = fused_topk_bias(
                hidden_states=hidden_states,
                gating_output=router_logits,
                scoring_func=scoring_func,
                e_score_correction_bias=self.e_score_correction_bias.data,
                topk=self.top_k,
                renormalize=self.renormalize,
                indices_type=indices_type,
            )
            if self.routed_scaling_factor != 1.0:
                topk_weights *= self.routed_scaling_factor
        else:
            jit_result = _musa_jit_fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
                indices_type=indices_type,
                scoring_func=scoring_func,
            )
            if jit_result is not None:
                topk_weights, topk_ids = jit_result
                if self.routed_scaling_factor != 1.0:
                    topk_weights *= self.routed_scaling_factor
                return topk_weights, topk_ids

            topk_weights, topk_ids, token_expert_indices = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
                indices_type=indices_type,
            )
            if self.routed_scaling_factor != 1.0:
                topk_weights *= self.routed_scaling_factor
        return topk_weights, topk_ids
    # ==================== MUSA ADAPTATION ====================
    topk_weights, topk_ids = grouped_topk(
        hidden_states=hidden_states,
        gating_output=router_logits,
        topk=self.top_k,
        renormalize=self.renormalize,
        num_expert_group=self.num_expert_group,
        topk_group=self.topk_group,
        scoring_func=self.scoring_func,
        routed_scaling_factor=self.routed_scaling_factor,
        e_score_correction_bias=self.e_score_correction_bias,
        num_fused_shared_experts=self.num_fused_shared_experts,
    )
    # ========================== END ==========================

    return topk_weights, topk_ids


def is_power_of_two(n):
    return n > 0 and math.log2(n).is_integer()


@torch.compile(
    dynamic=True,
    backend=current_platform.simple_compile_backend,
    options=maybe_disable_graph_partition(current_platform.simple_compile_backend),
)
def _grouped_topk_general(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    num_fused_shared_experts: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # XXX (MUSA): Fixed the issue of incorrect accuracy when enabling graph on s5000 with the biased_grouped_topk
    # For detailed problem description, please refer to JIRA#686
    # ==================== MUSA ADAPTATION ====================
    if hidden_states.dtype != torch.float32:
        hidden_states = hidden_states.to(dtype=torch.float32)
    if gating_output.dtype != torch.float32:
        gating_output = gating_output.to(dtype=torch.float32)

    if (
        is_fused_gate
        and (
            gating_output.shape[1] // num_expert_group <= 32
            or (
                num_expert_group == 1 and gating_output.shape[1] in {160, 256, 384}
            )  # XXX (MUSA): will support more cases in the future
        )
        and (
            e_score_correction_bias is not None
            and is_power_of_two(e_score_correction_bias.shape[0])
        )
    ):
        apply_routed_scaling_factor_on_output = routed_scaling_factor != 1.0
        topk_weights, topk_ids = mate_moe_fused_gate(
            gating_output,
            e_score_correction_bias,
            num_expert_group,
            topk_group,
            topk,
            num_fused_shared_experts,
            routed_scaling_factor if routed_scaling_factor is not None else 1.0,
            renormalize,
            apply_routed_scaling_factor_on_output,
        )

        return topk_weights, topk_ids
    # ========================== END ==========================

    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.size(0)
    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
    else:
        group_scores = (
            scores.view(num_token, num_expert_group, -1).max(dim=-1).values
        )  # [n, n_group]

    # For batch invariance, use sorted=True to ensure deterministic expert selection
    use_sorted = vllm_envs.VLLM_BATCH_INVARIANT
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=use_sorted)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.size(-1) // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(
            tmp_scores, k=topk, dim=-1, sorted=use_sorted
        )

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def _can_use_musa_tilelang_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str,
    e_score_correction_bias: torch.Tensor | None,
    num_fused_shared_experts: int,
) -> bool:
    return (
        current_platform.is_musa()
        and scoring_func == "softmax"
        and e_score_correction_bias is None
        and num_fused_shared_experts == 0
        and gating_output.device.type == "musa"
        and hidden_states.device == gating_output.device
        and hidden_states.dim() == 2
        and gating_output.dim() == 2
        and hidden_states.shape[0] == gating_output.shape[0]
        and gating_output.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and 0 < num_expert_group
        and 0 < topk_group <= num_expert_group
        and gating_output.shape[1] % num_expert_group == 0
        and 0 < topk <= topk_group * (gating_output.shape[1] // num_expert_group)
    )


def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    num_fused_shared_experts: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch MUSA grouped top-k before entering the compiled Torch fallback."""
    # Nemotron 3.5 uses one expert group with one selected group, so grouped
    # routing reduces to the biased sigmoid top-k operation. Keep this case on
    # the native MUSA kernel; multi-group routing uses the implementation below.
    if (
        current_platform.is_musa()
        and scoring_func == "sigmoid"
        and num_expert_group == 1
        and topk_group == 1
        and gating_output.shape[1] == 128
        and num_fused_shared_experts == 0
        and e_score_correction_bias is not None
    ):
        jit_result = _musa_jit_fused_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            indices_type=torch.int32,
            correction_bias=e_score_correction_bias,
            scoring_func=scoring_func,
        )
        if jit_result is not None:
            topk_weights, topk_ids = jit_result
            if routed_scaling_factor != 1.0:
                topk_weights = topk_weights * routed_scaling_factor
            return topk_weights, topk_ids

    if _can_use_musa_tilelang_grouped_topk(
        hidden_states,
        gating_output,
        topk,
        num_expert_group,
        topk_group,
        scoring_func,
        e_score_correction_bias,
        num_fused_shared_experts,
    ):
        from vllm_musa.jit_kernel.tilelang.grouped_topk import (
            grouped_topk_softmax_tilelang,
        )

        try:
            result = grouped_topk_softmax_tilelang(
                gating_output=gating_output,
                topk=topk,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                renormalize=renormalize,
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=routed_scaling_factor != 1.0,
            )
            return result
        # TileLang selects its serial kernel when this shape does not fit the
        # parallel kernel's resource limits. Keep the compiled implementation
        # as a fallback only for unavailable TileLang capabilities.
        except NotImplementedError:
            pass
    return _grouped_topk_general(
        hidden_states,
        gating_output,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        scoring_func,
        routed_scaling_factor,
        e_score_correction_bias,
        num_fused_shared_experts,
    )


import vllm.model_executor.layers.fused_moe.router.grouped_topk_router

vllm.model_executor.layers.fused_moe.router.grouped_topk_router.GroupedTopKRouter._compute_routing = (
    _compute_routing
)
