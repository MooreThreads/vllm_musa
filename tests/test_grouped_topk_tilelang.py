"""MUSA correctness coverage for the TileLang grouped softmax top-k kernel."""

from __future__ import annotations

import os

import pytest
import torchada  # noqa: F401
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_musa(),
    reason="MUSA grouped top-k test",
)


MODEL_SHAPES = (
    (64, 1, 1, 6),
    (160, 8, 3, 6),
    (256, 8, 4, 8),
    (256, 1, 1, 8),
)
GENERIC_SHAPES = (
    (48, 3, 2, 5),
    (96, 6, 2, 7),
    (128, 4, 3, 9),
    (128, 32, 3, 9),
    (128, 64, 4, 7),
)
PARALLEL_BOUNDARY_SHAPES = (
    (512, 16, 4, 32),
    (128, 128, 7, 7),
)
SERIAL_SHAPES = (
    (513, 3, 2, 7),
    (258, 129, 4, 7),
    (96, 3, 2, 40),
    (96, 3, 3, 40),
)
TOKEN_COUNTS = (
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    15,
    16,
    17,
    31,
    32,
    33,
    63,
    64,
    65,
    127,
    128,
    129,
    255,
    256,
    257,
    511,
    512,
    513,
    1023,
    1024,
    1025,
    2047,
    2048,
    2049,
    3071,
    3072,
)


def _make_unique_logits(
    num_tokens: int,
    num_experts: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Each 128-value block occupies a distinct BF16 exponent. The values are
    # exactly representable in both FP16 and BF16, remain unique through the
    # largest exercised expert count, and stay in a softmax-safe range.
    value_ids = torch.arange(num_experts, device="musa", dtype=torch.int64)
    mantissa = (value_ids % 128).to(torch.float32) / 128.0
    exponent = -(value_ids // 128).to(torch.float32)
    values = -(1.0 + mantissa) * torch.pow(2.0, exponent)
    expert_ids = torch.arange(num_experts, device="musa", dtype=torch.int64)
    token_ids = torch.arange(num_tokens, device="musa", dtype=torch.int64)
    permutation = (
        expert_ids.unsqueeze(0) * 131 + token_ids.unsqueeze(1) * 17
    ) % num_experts
    return values[permutation].to(dtype)


def _dense(
    weights: torch.Tensor,
    ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    output = torch.zeros(
        (weights.shape[0], num_experts),
        device=weights.device,
        dtype=weights.dtype,
    )
    return output.scatter_add(1, ids.long(), weights)


def _torch_reference(
    gating_output: torch.Tensor,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm_musa.model_executor.layers.fused_moe.router.grouped_topk_router import (
        _grouped_topk_general,
    )

    hidden_states = torch.empty(
        (gating_output.shape[0], 1),
        device=gating_output.device,
        dtype=gating_output.dtype,
    )
    return _grouped_topk_general(
        hidden_states,
        gating_output,
        topk,
        renormalize,
        num_expert_group,
        topk_group,
        "softmax",
        routed_scaling_factor,
        None,
        0,
    )


def _cpu_fp64_reference(
    gating_output: torch.Tensor,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute grouped softmax top-k independently on CPU in float64."""
    logits = gating_output.detach().to(device="cpu", dtype=torch.float64)
    num_tokens, num_experts = logits.shape
    experts_per_group = num_experts // num_expert_group
    weights = torch.empty((num_tokens, topk), dtype=torch.float64)
    ids = torch.empty((num_tokens, topk), dtype=torch.int64)

    for token_id in range(num_tokens):
        scores = torch.softmax(logits[token_id], dim=-1)
        group_scores = scores.reshape(num_expert_group, experts_per_group).amax(dim=-1)
        selected_groups = torch.argsort(
            group_scores, descending=True, stable=True
        )[:topk_group]
        group_mask = torch.zeros(num_expert_group, dtype=torch.bool)
        group_mask[selected_groups] = True
        expert_mask = group_mask.repeat_interleave(experts_per_group)
        candidate_ids = torch.arange(num_experts)[expert_mask]
        candidate_order = torch.argsort(
            scores[candidate_ids], descending=True, stable=True
        )[:topk]
        selected_ids = candidate_ids[candidate_order]
        selected_weights = scores[selected_ids]
        if renormalize:
            selected_weights = selected_weights / selected_weights.sum()
        if routed_scaling_factor != 1.0:
            selected_weights = selected_weights * routed_scaling_factor
        ids[token_id] = selected_ids
        weights[token_id] = selected_weights

    return weights, ids


@pytest.mark.parametrize("non_default_stream", [False, True])
def test_nemotron_sigmoid_topk_native_stream(non_default_stream: bool) -> None:
    """The Nemotron single-group route stays correct on the active MUSA stream."""
    from vllm_musa.model_executor.layers.fused_moe.router.grouped_topk_router import (
        grouped_topk,
    )

    torch.manual_seed(123)
    num_tokens, num_experts, topk = 8, 128, 6
    hidden_states = torch.empty(
        (num_tokens, 2688), device="musa", dtype=torch.bfloat16
    )
    gating_output = torch.randn(
        (num_tokens, num_experts), device="musa", dtype=torch.bfloat16
    )
    correction_bias = torch.randn(
        (num_experts,), device="musa", dtype=torch.float32
    )

    stream = torch.musa.Stream() if non_default_stream else None
    context = torch.musa.stream(stream) if stream is not None else None
    if context is not None:
        context.__enter__()
    try:
        weights, ids = grouped_topk(
            hidden_states,
            gating_output,
            topk,
            True,
            1,
            1,
            "sigmoid",
            1.0,
            correction_bias,
            0,
        )
        if stream is not None:
            stream.synchronize()
    finally:
        if context is not None:
            context.__exit__(None, None, None)

    scores = gating_output.float().sigmoid()
    ref_ids = torch.topk(scores + correction_bias, topk, dim=-1, sorted=False)[1]
    ref_weights = scores.gather(1, ref_ids)
    ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True)
    dense = torch.zeros_like(scores)
    dense.scatter_add_(1, ids.long(), weights)
    ref_dense = torch.zeros_like(scores)
    ref_dense.scatter_add_(1, ref_ids, ref_weights)
    assert torch.equal(torch.sort(ids, dim=-1).values, torch.sort(ref_ids, dim=-1).values)
    assert torch.allclose(dense, ref_dense, atol=2e-5, rtol=2e-5)


def _cpu_torch_topk_reference(
    gating_output: torch.Tensor,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fallback expression with CPU tensors and CPU ``torch.topk``.

    This intentionally calls ``torch.topk`` directly; it is not an independent
    reimplementation of top-k selection.  FP32 is used to mirror the MUSA
    fallback's explicit input casts before softmax/top-k.
    """
    logits = gating_output.detach().to(device="cpu", dtype=torch.float32)
    scores = torch.softmax(logits, dim=-1)
    num_tokens, num_experts = scores.shape
    experts_per_group = num_experts // num_expert_group
    group_scores = scores.reshape(
        num_tokens, num_expert_group, experts_per_group
    ).amax(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=True).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, group_idx, True)
    score_mask = group_mask.unsqueeze(-1).expand(
        num_tokens, num_expert_group, experts_per_group
    ).reshape(num_tokens, num_experts)
    tmp_scores = scores.masked_fill(~score_mask, float("-inf"))
    topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=True)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids


def _tilelang_result(
    gating_output: torch.Tensor,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    renormalize: bool,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm_musa.jit_kernel.tilelang.grouped_topk import (
        grouped_topk_softmax_tilelang,
    )

    return grouped_topk_softmax_tilelang(
        gating_output,
        topk,
        num_expert_group,
        topk_group,
        renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=routed_scaling_factor != 1.0,
    )


def _assert_matches_reference(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
    num_experts: int,
) -> None:
    actual_weights, actual_ids = actual
    expected_weights, expected_ids = expected
    assert actual_weights.dtype == expected_weights.dtype == torch.float32
    assert actual_ids.dtype == expected_ids.dtype == torch.int32
    torch.testing.assert_close(
        _dense(actual_weights, actual_ids, num_experts),
        _dense(expected_weights, expected_ids, num_experts),
        rtol=2e-4,
        atol=2e-6,
    )


def _assert_matches_cpu_reference(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> None:
    actual_weights, actual_ids = actual
    expected_weights, expected_ids = expected
    assert torch.equal(actual_ids.cpu().to(torch.int64), expected_ids)
    torch.testing.assert_close(
        actual_weights.cpu().to(torch.float64), expected_weights.to(torch.float64),
        rtol=3e-5,
        atol=3e-6,
    )


def _assert_tie_compatible_with_cpu_reference(
    actual: tuple[torch.Tensor, torch.Tensor],
    cpu_reference: tuple[torch.Tensor, torch.Tensor],
    cpu_logits: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
) -> None:
    """Validate an alternate, but score-equivalent, top-k tie resolution.

    The CPU reference fixes ties by ascending id. The legacy MUSA fallback
    uses ``torch.topk``, whose tied-index order is not an API guarantee. This
    helper permits only a cutoff-tie substitution: selected score multisets,
    normalized weights, and selected-group eligibility must still agree with
    the independent CPU implementation.
    """
    actual_weights, actual_ids = actual
    expected_weights, expected_ids = cpu_reference
    num_experts = cpu_logits.shape[1]
    experts_per_group = num_experts // num_expert_group
    actual_ids_cpu = actual_ids.cpu().to(torch.int64)
    expected_ids_cpu = expected_ids.to(torch.int64)
    logits = cpu_logits.to(torch.float64)

    actual_scores = logits.gather(1, actual_ids_cpu)
    expected_scores = logits.gather(1, expected_ids_cpu)
    torch.testing.assert_close(
        actual_scores.sort(dim=-1, descending=True).values,
        expected_scores.sort(dim=-1, descending=True).values,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_weights.cpu().to(torch.float64).sort(dim=-1, descending=True).values,
        expected_weights.sort(dim=-1, descending=True).values,
        rtol=3e-5,
        atol=3e-6,
    )

    group_scores = logits.reshape(-1, num_expert_group, experts_per_group).amax(
        dim=-1
    )
    group_cutoff = group_scores.topk(topk_group, dim=-1).values.amin(
        dim=-1, keepdim=True
    )
    actual_groups = actual_ids_cpu // experts_per_group
    assert (group_scores.gather(1, actual_groups) >= group_cutoff).all()
    for ids, groups in zip(actual_ids_cpu, actual_groups, strict=True):
        assert torch.unique(ids).numel() == ids.numel()
        assert torch.unique(groups).numel() <= topk_group


@pytest.mark.parametrize("num_experts,num_groups,topk_group,topk", MODEL_SHAPES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("renormalize", (False, True))
@pytest.mark.parametrize("routed_scaling_factor", (1.0, 2.5))
def test_grouped_topk_tilelang_matches_independent_cpu_fp64(
    num_experts: int,
    num_groups: int,
    topk_group: int,
    topk: int,
    dtype: torch.dtype,
    renormalize: bool,
    routed_scaling_factor: float,
) -> None:
    logits = _make_unique_logits(7, num_experts, dtype)
    cpu_logits = logits.cpu()
    actual = _tilelang_result(
        logits,
        topk,
        num_groups,
        topk_group,
        renormalize,
        routed_scaling_factor,
    )
    expected = _cpu_fp64_reference(
        cpu_logits,
        topk,
        num_groups,
        topk_group,
        renormalize,
        routed_scaling_factor,
    )
    _assert_matches_cpu_reference(actual, expected)


@pytest.mark.parametrize("num_experts,num_groups,topk_group,topk", MODEL_SHAPES)
def test_grouped_topk_tilelang_matches_cpu_torch_topk_on_unique_logits(
    num_experts: int,
    num_groups: int,
    topk_group: int,
    topk: int,
) -> None:
    """Compare TileLang with CPU ``torch.topk(sorted=True)`` on unique logits."""
    logits = _make_unique_logits(7, num_experts, torch.float32)
    cpu_logits = logits.cpu()
    actual = _tilelang_result(
        logits, topk, num_groups, topk_group, True, routed_scaling_factor=1.0
    )
    expected = _cpu_torch_topk_reference(
        cpu_logits, topk, num_groups, topk_group, True, 1.0
    )
    _assert_matches_cpu_reference(actual, expected)


@pytest.mark.parametrize("num_experts,num_groups,topk_group,topk", MODEL_SHAPES)
@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("renormalize", (False, True))
def test_grouped_topk_tilelang_model_shapes_match_torch(
    num_experts: int,
    num_groups: int,
    topk_group: int,
    topk: int,
    num_tokens: int,
    dtype: torch.dtype,
    renormalize: bool,
) -> None:
    logits = _make_unique_logits(num_tokens, num_experts, dtype)
    actual = _tilelang_result(
        logits,
        topk,
        num_groups,
        topk_group,
        renormalize,
    )
    expected = _torch_reference(
        logits,
        topk,
        num_groups,
        topk_group,
        renormalize,
        1.0,
    )
    _assert_matches_reference(actual, expected, num_experts)


@pytest.mark.parametrize("num_experts,num_groups,topk_group,topk", GENERIC_SHAPES)
@pytest.mark.parametrize("num_tokens", (1, 33, 257))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("renormalize", (False, True))
def test_grouped_topk_tilelang_generic_shapes_match_torch(
    num_experts: int,
    num_groups: int,
    topk_group: int,
    topk: int,
    num_tokens: int,
    dtype: torch.dtype,
    renormalize: bool,
) -> None:
    logits = _make_unique_logits(num_tokens, num_experts, dtype)
    actual = _tilelang_result(logits, topk, num_groups, topk_group, renormalize)
    expected = _torch_reference(
        logits,
        topk,
        num_groups,
        topk_group,
        renormalize,
        1.0,
    )
    _assert_matches_reference(actual, expected, num_experts)


@pytest.mark.parametrize(
    "num_experts,num_groups,topk_group,topk",
    PARALLEL_BOUNDARY_SHAPES + SERIAL_SHAPES,
)
@pytest.mark.parametrize("num_tokens", (1, 3))
@pytest.mark.parametrize("renormalize", (False, True))
def test_grouped_topk_tilelang_parallel_and_serial_limits_match_torch(
    num_experts: int,
    num_groups: int,
    topk_group: int,
    topk: int,
    num_tokens: int,
    renormalize: bool,
) -> None:
    logits = _make_unique_logits(num_tokens, num_experts, torch.bfloat16)
    actual = _tilelang_result(logits, topk, num_groups, topk_group, renormalize)
    expected = _torch_reference(
        logits,
        topk,
        num_groups,
        topk_group,
        renormalize,
        1.0,
    )
    _assert_matches_reference(actual, expected, num_experts)


@pytest.mark.skipif(
    os.environ.get("VLLM_MUSA_RUN_EXHAUSTIVE_GROUPED_TOPK") != "1",
    reason="set VLLM_MUSA_RUN_EXHAUSTIVE_GROUPED_TOPK=1 for token counts 1--3072",
)
def test_grouped_topk_tilelang_every_token_count() -> None:
    for num_tokens in range(1, 3073):
        logits = _make_unique_logits(num_tokens, 160, torch.bfloat16)
        actual = _tilelang_result(logits, 6, 8, 3, True, 16.0)
        expected = _torch_reference(logits, 6, 8, 3, True, 16.0)
        _assert_matches_reference(actual, expected, 160)


def test_grouped_topk_tilelang_handles_noncontiguous_parallel_input() -> None:
    storage = _make_unique_logits(66, 320, torch.bfloat16)
    logits = storage[::2, ::2]
    assert logits.shape == (33, 160)
    assert not logits.is_contiguous()
    actual = _tilelang_result(logits, 6, 8, 3, True, 16.0)
    expected = _torch_reference(logits, 6, 8, 3, True, 16.0)
    _assert_matches_reference(actual, expected, 160)


def test_grouped_topk_tilelang_handles_noncontiguous_serial_input() -> None:
    storage = _make_unique_logits(6, 1026, torch.bfloat16)
    logits = storage[::2, ::2]
    assert logits.shape == (3, 513)
    assert not logits.is_contiguous()
    actual = _tilelang_result(logits, 7, 3, 2, True)
    expected = _torch_reference(logits, 7, 3, 2, True, 1.0)
    _assert_matches_reference(actual, expected, 513)


@pytest.mark.parametrize(
    "num_experts,num_groups,topk_group,topk", MODEL_SHAPES + GENERIC_SHAPES
)
def test_grouped_topk_tilelang_unique_logits_are_deterministic_and_group_limited(
    num_experts: int,
    num_groups: int,
    topk_group: int,
    topk: int,
) -> None:
    logits = _make_unique_logits(17, num_experts, torch.bfloat16)
    weights, ids = _tilelang_result(logits, topk, num_groups, topk_group, True)
    repeated_weights, repeated_ids = _tilelang_result(
        logits,
        topk,
        num_groups,
        topk_group,
        True,
    )

    torch.testing.assert_close(repeated_weights, weights)
    torch.testing.assert_close(repeated_ids, ids)
    expected = _torch_reference(logits, topk, num_groups, topk_group, True, 1.0)
    _assert_matches_reference((weights, ids), expected, num_experts)

    scores = torch.softmax(logits.float(), dim=-1)
    group_scores = scores.reshape(17, num_groups, -1).amax(dim=-1)
    group_cutoff = group_scores.topk(topk_group).values.amin(dim=-1, keepdim=True)
    groups = ids.long() // (num_experts // num_groups)
    assert (group_scores.gather(1, groups) >= group_cutoff).all()
    for token_ids, token_groups in zip(ids, groups, strict=True):
        assert torch.unique(token_ids).numel() == topk
        assert torch.unique(token_groups).numel() <= topk_group


@pytest.mark.parametrize("renormalize", (False, True))
def test_grouped_topk_tilelang_applies_routed_scaling_factor(
    renormalize: bool,
) -> None:
    logits = _make_unique_logits(33, 160, torch.bfloat16)
    base_weights, base_ids = _tilelang_result(logits, 6, 8, 3, renormalize)
    scaled_weights, scaled_ids = _tilelang_result(
        logits,
        6,
        8,
        3,
        renormalize,
        2.5,
    )
    torch.testing.assert_close(scaled_ids, base_ids)
    torch.testing.assert_close(scaled_weights, base_weights * 2.5)
    if renormalize:
        torch.testing.assert_close(
            scaled_weights.sum(dim=-1),
            torch.full((33,), 2.5, device="musa", dtype=torch.float32),
        )


@pytest.mark.parametrize(
    "num_experts,num_groups,topk_group,routed_topk",
    MODEL_SHAPES + ((96, 6, 2, 7), (96, 3, 2, 40)),
)
@pytest.mark.parametrize("num_tokens", (1, 33, 257))
@pytest.mark.parametrize("renormalize", (False, True))
@pytest.mark.parametrize("apply_routed_scale", (False, True))
def test_grouped_topk_tilelang_shared_expert_and_routed_scale(
    num_experts: int,
    num_groups: int,
    topk_group: int,
    routed_topk: int,
    num_tokens: int,
    renormalize: bool,
    apply_routed_scale: bool,
) -> None:
    """Cover the kernel ABI slot that appends one fused shared expert."""
    from vllm_musa.jit_kernel.tilelang.grouped_topk import (
        grouped_topk_softmax_tilelang,
    )

    scaling_factor = 2.5
    logits = _make_unique_logits(num_tokens, num_experts, torch.bfloat16)
    routed_weights, routed_ids = grouped_topk_softmax_tilelang(
        logits,
        routed_topk,
        num_groups,
        topk_group,
        renormalize,
        routed_scaling_factor=scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scale,
    )
    shared_weights, shared_ids = grouped_topk_softmax_tilelang(
        logits,
        routed_topk + 1,
        num_groups,
        topk_group,
        renormalize,
        num_fused_shared_experts=1,
        routed_scaling_factor=scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scale,
    )

    torch.testing.assert_close(shared_ids[:, :-1], routed_ids)
    torch.testing.assert_close(shared_weights[:, :-1], routed_weights)
    torch.testing.assert_close(
        shared_ids[:, -1],
        torch.full((num_tokens,), num_experts, device="musa", dtype=torch.int32),
    )
    torch.testing.assert_close(
        shared_weights[:, -1], routed_weights.sum(dim=-1) / scaling_factor
    )


@pytest.mark.parametrize("num_experts,num_groups,topk_group,topk", MODEL_SHAPES)
def test_grouped_topk_tilelang_handles_very_negative_logits(
    num_experts: int,
    num_groups: int,
    topk_group: int,
    topk: int,
) -> None:
    logits = _make_unique_logits(17, num_experts, torch.float32) - 20000.0
    actual = _tilelang_result(logits, topk, num_groups, topk_group, True, 16.0)
    expected = _torch_reference(logits, topk, num_groups, topk_group, True, 16.0)
    _assert_matches_reference(actual, expected, num_experts)


def test_grouped_topk_tilelang_handles_empty_input() -> None:
    weights, ids = _tilelang_result(
        torch.empty((0, 48), device="musa", dtype=torch.bfloat16),
        5,
        3,
        2,
        True,
    )
    assert weights.shape == ids.shape == (0, 5)
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int32


@pytest.mark.parametrize(
    "num_experts,num_groups,topk_group,topk",
    (
        (0, 1, 1, 1),
        (48, 0, 1, 1),
        (48, 5, 1, 1),
        (48, 3, 0, 1),
        (48, 3, 4, 1),
        (48, 3, 2, 33),
    ),
)
def test_grouped_topk_tilelang_rejects_invalid_configurations(
    num_experts: int,
    num_groups: int,
    topk_group: int,
    topk: int,
) -> None:
    logits = torch.empty((1, num_experts), device="musa", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        _tilelang_result(logits, topk, num_groups, topk_group, True)


def test_grouped_topk_tilelang_rejects_unsupported_shared_expert_count() -> None:
    from vllm_musa.jit_kernel.tilelang.grouped_topk import (
        grouped_topk_softmax_tilelang,
    )

    logits = torch.empty((1, 48), device="musa", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        grouped_topk_softmax_tilelang(
            logits,
            7,
            3,
            2,
            True,
            num_fused_shared_experts=2,
            routed_scaling_factor=1.0,
        )


def _make_dispatch_logits(num_tokens: int, num_experts: int) -> torch.Tensor:
    return _make_unique_logits(num_tokens, num_experts, torch.bfloat16)


@pytest.mark.parametrize("num_experts,num_groups,topk_group,topk", MODEL_SHAPES)
@pytest.mark.parametrize("num_tokens", (1, 33, 3072))
def test_grouped_topk_router_dispatches_tilelang(
    monkeypatch: pytest.MonkeyPatch,
    num_experts: int,
    num_groups: int,
    topk_group: int,
    topk: int,
    num_tokens: int,
) -> None:
    import vllm_musa.jit_kernel.tilelang as tilelang_kernels
    from vllm_musa.model_executor.layers.fused_moe.router.grouped_topk_router import (
        grouped_topk,
    )

    called = False
    original = tilelang_kernels.grouped_topk_softmax_tilelang

    def traced_tilelang(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal called
        called = True
        return original(*args, **kwargs)

    monkeypatch.setattr(
        tilelang_kernels,
        "grouped_topk_softmax_tilelang",
        traced_tilelang,
    )
    logits = _make_dispatch_logits(num_tokens, num_experts)
    hidden_states = torch.empty(
        (num_tokens, 1), device="musa", dtype=torch.bfloat16
    )
    actual = grouped_topk(
        hidden_states,
        logits,
        topk,
        True,
        num_groups,
        topk_group,
        "softmax",
        16.0,
    )
    expected = original(
        logits,
        topk,
        num_groups,
        topk_group,
        True,
        routed_scaling_factor=16.0,
        apply_routed_scaling_factor_on_output=True,
    )
    assert called
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_grouped_topk_router_matches_cpu_fp64_and_forced_torch_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_musa.model_executor.layers.fused_moe.router.grouped_topk_router as router

    logits = _make_unique_logits(17, 160, torch.bfloat16)
    cpu_logits = logits.cpu()
    hidden_states = torch.empty((17, 32), device="musa", dtype=torch.bfloat16)
    cpu_reference = _cpu_fp64_reference(
        cpu_logits,
        topk=6,
        num_expert_group=8,
        topk_group=3,
        renormalize=True,
        routed_scaling_factor=2.5,
    )
    tilelang_result = router.grouped_topk(
        hidden_states,
        logits,
        6,
        True,
        8,
        3,
        "softmax",
        2.5,
    )
    _assert_matches_cpu_reference(tilelang_result, cpu_reference)

    monkeypatch.setattr(
        router,
        "_can_use_musa_tilelang_grouped_topk",
        lambda *args: False,
    )
    fallback_result = router.grouped_topk(
        hidden_states,
        logits,
        6,
        True,
        8,
        3,
        "softmax",
        2.5,
    )
    _assert_tie_compatible_with_cpu_reference(
        fallback_result,
        cpu_reference,
        cpu_logits,
        num_expert_group=8,
        topk_group=3,
    )


@pytest.mark.parametrize(
    "scoring_func,has_bias,num_fused_shared_experts",
    (
        ("sigmoid", False, 0),
        ("softmax", True, 0),
        ("softmax", False, 1),
    ),
)
def test_grouped_topk_router_does_not_dispatch_unsupported_semantics(
    monkeypatch: pytest.MonkeyPatch,
    scoring_func: str,
    has_bias: bool,
    num_fused_shared_experts: int,
) -> None:
    import vllm_musa.model_executor.layers.fused_moe.router.grouped_topk_router as router

    fallback_weights = torch.empty((1, 1), device="musa", dtype=torch.float32)
    fallback_ids = torch.empty((1, 1), device="musa", dtype=torch.int32)

    def fail_tilelang(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("unsupported routing semantics reached TileLang")

    def fake_fallback(
        *args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return fallback_weights, fallback_ids

    monkeypatch.setattr(
        "vllm_musa.jit_kernel.tilelang.grouped_topk_softmax_tilelang",
        fail_tilelang,
    )
    monkeypatch.setattr(router, "_grouped_topk_general", fake_fallback)
    logits = _make_dispatch_logits(1, 160)
    hidden_states = torch.empty((1, 1), device="musa", dtype=torch.bfloat16)
    bias = torch.zeros(160, device="musa") if has_bias else None

    actual = router.grouped_topk(
        hidden_states,
        logits,
        6,
        True,
        8,
        3,
        scoring_func,
        1.0,
        bias,
        num_fused_shared_experts,
    )
    assert actual[0] is fallback_weights
    assert actual[1] is fallback_ids


def test_grouped_topk_router_falls_back_when_tilelang_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm_musa.model_executor.layers.fused_moe.router.grouped_topk_router as router

    fallback_weights = torch.empty((1, 1), device="musa", dtype=torch.float32)
    fallback_ids = torch.empty((1, 1), device="musa", dtype=torch.int32)

    def unavailable_tilelang(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def fake_fallback(
        *args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return fallback_weights, fallback_ids

    monkeypatch.setattr(
        "vllm_musa.jit_kernel.tilelang.grouped_topk_softmax_tilelang",
        unavailable_tilelang,
    )
    monkeypatch.setattr(router, "_grouped_topk_general", fake_fallback)
    logits = _make_dispatch_logits(1, 160)
    hidden_states = torch.empty((1, 1), device="musa", dtype=torch.bfloat16)

    actual = router.grouped_topk(hidden_states, logits, 6, True, 8, 3)
    assert actual[0] is fallback_weights
    assert actual[1] is fallback_ids
