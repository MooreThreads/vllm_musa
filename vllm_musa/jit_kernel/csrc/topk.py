from __future__ import annotations

from typing import Optional

import torch
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_musa.jit_kernel.csrc.jit import load_musa_jit
from vllm_musa.jit_kernel.utils import cache_once


@cache_once
def _topk_module():
    return load_musa_jit(
        "vllm_musa_topk_gating",
        ("topk/topk_gating.mu",),
    )


def _call_topk_kernel(name: str, *args: object) -> None:
    """Launch a TVM-FFI top-k kernel on PyTorch's current MUSA stream."""
    kernel = getattr(_topk_module(), name)
    # TVM FFI keeps an independent stream context.  Synchronize it with the
    # active torch.musa stream so graph replay and MTP do not race producer and
    # consumer kernels when the current stream is non-default.
    if getattr(torch.version, "musa", None) is not None:
        import tvm_ffi

        with tvm_ffi.use_torch_musa_stream():
            kernel(*args)
    else:
        kernel(*args)


def _topk_softmax_impl(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    moe_softcapping: float = 0.0,
    correction_bias: Optional[torch.Tensor] = None,
    shared_expert_gate_output: Optional[torch.Tensor] = None,
    num_fused_shared_experts: int = 0,
) -> None:
    has_correction_bias = correction_bias is not None
    bias_arg = correction_bias if has_correction_bias else topk_weights.reshape(-1)
    has_shared_experts = (
        shared_expert_gate_output is not None and num_fused_shared_experts > 0
    )
    shared_arg = (
        shared_expert_gate_output
        if has_shared_experts
        else topk_weights.reshape(-1)
    )
    _call_topk_kernel(
        "sgl_musa_topk_softmax",
        topk_weights,
        topk_ids,
        gating_output,
        bool(renormalize),
        float(moe_softcapping),
        bias_arg,
        bool(has_correction_bias),
        shared_arg,
        int(num_fused_shared_experts),
        bool(has_shared_experts),
    )


def _topk_sigmoid_impl(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    correction_bias: Optional[torch.Tensor] = None,
    shared_expert_gate_output: Optional[torch.Tensor] = None,
    num_fused_shared_experts: int = 0,
) -> None:
    has_correction_bias = correction_bias is not None
    bias_arg = correction_bias if has_correction_bias else topk_weights.reshape(-1)
    has_shared_experts = (
        shared_expert_gate_output is not None and num_fused_shared_experts > 0
    )
    shared_arg = (
        shared_expert_gate_output
        if has_shared_experts
        else topk_weights.reshape(-1)
    )
    _call_topk_kernel(
        "sgl_musa_topk_sigmoid",
        topk_weights,
        topk_ids,
        gating_output,
        bool(renormalize),
        bias_arg,
        bool(has_correction_bias),
        shared_arg,
        int(num_fused_shared_experts),
        bool(has_shared_experts),
    )


def _topk_softmax_custom(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    shared_expert_gate_output: torch.Tensor,
    renormalize: bool = False,
    moe_softcapping: float = 0.0,
    has_correction_bias: bool = False,
    num_fused_shared_experts: int = 0,
    has_shared_experts: bool = False,
) -> None:
    bias_arg = correction_bias if has_correction_bias else topk_weights.reshape(-1)
    shared_arg = (
        shared_expert_gate_output
        if has_shared_experts
        else topk_weights.reshape(-1)
    )
    _call_topk_kernel(
        "sgl_musa_topk_softmax",
        topk_weights,
        topk_ids,
        gating_output,
        bool(renormalize),
        float(moe_softcapping),
        bias_arg,
        bool(has_correction_bias),
        shared_arg,
        int(num_fused_shared_experts),
        bool(has_shared_experts),
    )


def _topk_softmax_custom_fake(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    shared_expert_gate_output: torch.Tensor,
    renormalize: bool = False,
    moe_softcapping: float = 0.0,
    has_correction_bias: bool = False,
    num_fused_shared_experts: int = 0,
    has_shared_experts: bool = False,
) -> None:
    return


def _topk_sigmoid_custom(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    shared_expert_gate_output: torch.Tensor,
    renormalize: bool = False,
    has_correction_bias: bool = False,
    num_fused_shared_experts: int = 0,
    has_shared_experts: bool = False,
) -> None:
    bias_arg = correction_bias if has_correction_bias else topk_weights.reshape(-1)
    shared_arg = (
        shared_expert_gate_output
        if has_shared_experts
        else topk_weights.reshape(-1)
    )
    _call_topk_kernel(
        "sgl_musa_topk_sigmoid",
        topk_weights,
        topk_ids,
        gating_output,
        bool(renormalize),
        bias_arg,
        bool(has_correction_bias),
        shared_arg,
        int(num_fused_shared_experts),
        bool(has_shared_experts),
    )


def _topk_sigmoid_custom_fake(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    shared_expert_gate_output: torch.Tensor,
    renormalize: bool = False,
    has_correction_bias: bool = False,
    num_fused_shared_experts: int = 0,
    has_shared_experts: bool = False,
) -> None:
    return


direct_register_custom_op(
    op_name="musa_topk_softmax",
    op_func=_topk_softmax_custom,
    mutates_args=["topk_weights", "topk_ids"],
    fake_impl=_topk_softmax_custom_fake,
)

direct_register_custom_op(
    op_name="musa_topk_sigmoid",
    op_func=_topk_sigmoid_custom,
    mutates_args=["topk_weights", "topk_ids"],
    fake_impl=_topk_sigmoid_custom_fake,
)


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    moe_softcapping: float = 0.0,
    correction_bias: Optional[torch.Tensor] = None,
    shared_expert_gate_output: Optional[torch.Tensor] = None,
    num_fused_shared_experts: int = 0,
) -> None:
    """sgl_kernel-compatible top-k softmax entry point."""
    has_correction_bias = correction_bias is not None
    bias_arg = correction_bias if has_correction_bias else topk_weights.reshape(-1)
    has_shared_experts = (
        shared_expert_gate_output is not None and num_fused_shared_experts > 0
    )
    shared_arg = (
        shared_expert_gate_output
        if has_shared_experts
        else topk_weights.reshape(-1)
    )
    torch.ops.vllm.musa_topk_softmax(
        topk_weights,
        topk_ids,
        gating_output,
        bias_arg,
        shared_arg,
        renormalize,
        moe_softcapping,
        has_correction_bias,
        num_fused_shared_experts,
        has_shared_experts,
    )


def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    correction_bias: Optional[torch.Tensor] = None,
    shared_expert_gate_output: Optional[torch.Tensor] = None,
    num_fused_shared_experts: int = 0,
) -> None:
    has_correction_bias = correction_bias is not None
    bias_arg = correction_bias if has_correction_bias else topk_weights.reshape(-1)
    has_shared_experts = (
        shared_expert_gate_output is not None and num_fused_shared_experts > 0
    )
    shared_arg = (
        shared_expert_gate_output
        if has_shared_experts
        else topk_weights.reshape(-1)
    )
    torch.ops.vllm.musa_topk_sigmoid(
        topk_weights,
        topk_ids,
        gating_output,
        bias_arg,
        shared_arg,
        renormalize,
        has_correction_bias,
        num_fused_shared_experts,
        has_shared_experts,
    )
