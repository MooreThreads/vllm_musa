# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep MUSA fused norm dispatch usable when Dynamo has no vLLM pass context."""

import pytest

pytest.importorskip("torchada")
import torch

from vllm import ir
from vllm_musa.kernels.musa_ops import _fused_add_rms_norm_supports_args

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="Requires a MUSA device",
)


@pytest.mark.parametrize("hidden_size", [2688, 5120])
def test_compiled_fused_norm_dispatch(hidden_size):
    torch.manual_seed(42)
    x = torch.randn(8, hidden_size, device="musa", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, device="musa", dtype=x.dtype)

    def capability(x, residual, weight):
        if _fused_add_rms_norm_supports_args(x, residual, weight, 1e-6):
            return x + 1
        return x + 2

    # Removing all guards raises in get_pass_context(); the broad guard would
    # compile but incorrectly reject the available C++ provider.
    compiled_capability = torch.compile(
        capability, backend="eager", fullgraph=True, dynamic=True
    )
    torch.testing.assert_close(compiled_capability(x, residual, weight), x + 1)

    def forward(x, residual, weight):
        return ir.ops.fused_add_rms_norm(x, residual, weight, 1e-6)

    with torch.no_grad(), ir.ops.fused_add_rms_norm.set_priority(["musa", "native"]):
        eager = forward(x, residual, weight)
        compiled = torch.compile(forward, fullgraph=True, dynamic=True)
        for _ in range(2):
            actual = compiled(x, residual, weight)
            for result, expected in zip(actual, eager):
                torch.testing.assert_close(result, expected, atol=0.02, rtol=0.02)
