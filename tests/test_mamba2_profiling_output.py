# SPDX-License-Identifier: Apache-2.0
"""Profiling must initialize the custom-op output consumed by later layers."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from vllm.model_executor.layers.mamba import mamba_mixer2  # noqa: E402


@pytest.mark.parametrize("tokens", [1, 32, 4112])
def test_profile_initializes_supplied_output(monkeypatch, tokens):
    monkeypatch.setattr(
        mamba_mixer2, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None)
    )
    monkeypatch.setattr(
        mamba_mixer2, "current_platform", SimpleNamespace(is_musa=lambda: True)
    )
    warmed = []
    mixer = SimpleNamespace(
        tped_intermediate_size=4,
        tped_conv_size=8,
        tped_dt_size=2,
        cache_config=SimpleNamespace(mamba_cache_mode="none", mamba_block_size=128),
        _warmup_ssd_kernels=lambda states: warmed.append(states.shape),
        split_hidden_states_B_C_fn=lambda values: (
            values[..., :4],
            values[..., 4:6],
            values[..., 6:],
        ),
    )
    projected = torch.zeros(tokens, 14)
    output = torch.full((tokens, 4), float("nan"))
    mamba_mixer2.MambaMixer2.conv_ssm_forward(mixer, projected, output)
    assert warmed == [projected.shape]
    assert torch.equal(output, torch.zeros_like(output))
