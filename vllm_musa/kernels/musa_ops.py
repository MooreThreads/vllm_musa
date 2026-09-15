# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA IR-op providers for vllm.ir.ops.*

Currently registers:
- rms_norm: delegates to torch.ops._C.rms_norm only when that op has a MUSA
  dispatch kernel. In vLLM v0.22 the upstream layernorm implementation moved
  to _C_stable_libtorch; vllm-musa builds that extension as a schema-only
  compatibility shim, so this provider self-disables until a real MUSA kernel
  is present.
- fused_add_rms_norm: exposes one ``musa`` provider whose internal capability
  dispatch uses the exact FP32-sum JIT kernel for measured H5120 BF16 work and
  the pre-existing C-extension kernel for broader standard RMSNorm shapes. It
  is registered as in-place so vLLM's IR functionalization pass, rather than a
  model-name check, owns activation donation.

Engine log from baseline confirms:
    ir_op_priority=IrOpPriorityConfig(rms_norm=['native'])

i.e. before this provider lands, every decoder layer per token uses
the pure-PyTorch native rms_norm. With the "musa" provider plus a
priority override in `vllm_musa.platform`, the dispatcher / Inductor
lowering picks the MUSA kernel when `_dispatch_has_kernel_for_dispatch_key`
confirms one is registered.
"""

import torch
from torch import Tensor
from vllm import ir
from vllm.platforms import current_platform

from vllm_musa.tuning import FUSED_ADD_RMSNORM_MIN_ROWS

# vllm._C must be loaded before torch.ops._C.rms_norm is resolvable.
current_platform.import_kernels()


def _has_C_rms_norm() -> bool:
    try:
        if not hasattr(torch.ops._C, "rms_norm"):
            return False
        return torch._C._dispatch_has_kernel_for_dispatch_key("_C::rms_norm", "MUSA")
    except Exception:
        return False


MUSA_RMS_NORM_SUPPORTED = current_platform.is_musa() and _has_C_rms_norm()


def _rms_norm_supports_args(
    x: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    return (
        variance_size is None
        and weight is not None
        and x.dim() >= 2
        and x.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == x.dtype
        and weight.is_contiguous()
    )


@ir.ops.rms_norm.register_impl(
    "musa",
    supports_args=_rms_norm_supports_args,
    supported=MUSA_RMS_NORM_SUPPORTED,
)
def rms_norm(
    x: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> Tensor:
    """MUSA provider for vllm.ir.ops.rms_norm.

    Delegates to torch.ops._C.rms_norm only when a MUSA dispatch kernel is
    registered for the upstream op namespace.
    """
    assert variance_size is None
    assert weight is not None
    output = torch.empty_like(x)
    torch.ops._C.rms_norm(output, x, weight, epsilon)
    return output


def _jit_fused_add_rms_norm_supports_args(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    del epsilon

    if weight is None or x.dim() != 2 or x_residual.dim() != 2 or weight.dim() != 1:
        return False

    # IR lowering selects one provider for an entire vLLM compile range. Do not
    # inspect a symbolic tensor's example-value hint here: vLLM deliberately
    # drops shape guards, so that choice would also be reused below the measured
    # profitability boundary. The platform inserts an endpoint immediately
    # below FUSED_ADD_RMSNORM_MIN_ROWS for eligible H5120/BF16 models, and the
    # lowering pass exposes that range through its pass context. Eager dispatch
    # has concrete shapes.
    # Dynamo cannot trace get_pass_context(); keep the C++ provider eligible.
    if torch.compiler.is_compiling():
        return False

    try:
        from vllm.compilation.passes.inductor_pass import get_pass_context

        profitable_rows = (
            get_pass_context().compile_range.start >= FUSED_ADD_RMSNORM_MIN_ROWS
        )
    except AssertionError:
        # Eager dispatch has concrete shapes.
        profitable_rows = x.shape[0] >= FUSED_ADD_RMSNORM_MIN_ROWS

    return (
        variance_size is None
        and x.device.type == "musa"
        and x_residual.device.type == "musa"
        and weight.device.type == "musa"
        and x.shape == x_residual.shape
        and profitable_rows
        and x.shape[1] == 5120
        and weight.numel() == 5120
        and x.dtype == torch.bfloat16
        and x_residual.dtype == x.dtype
        and weight.dtype in (x.dtype, torch.float32)
        and x.is_contiguous()
        and x_residual.is_contiguous()
        and weight.is_contiguous()
    )


def _c_ext_fused_add_rms_norm_supports_args(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    del epsilon
    if weight is None or x.dim() != 2 or x_residual.dim() != 2 or weight.dim() != 1:
        return False

    hidden_size = x.shape[1]
    return (
        variance_size is None
        and x.device.type == "musa"
        and x_residual.device.type == "musa"
        and weight.device.type == "musa"
        and x.shape == x_residual.shape
        and hidden_size == weight.numel()
        and hidden_size % 8 == 0
        and hidden_size <= 16384
        and x.dtype in (torch.float16, torch.bfloat16)
        and x_residual.dtype == x.dtype
        and weight.dtype == x.dtype
        and x.is_contiguous()
        and x_residual.is_contiguous()
        and weight.is_contiguous()
    )


def _select_musa_fused_add_rms_norm_impl(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> str | None:
    """Select the concrete kernel hidden behind the public ``musa`` provider.

    The JIT capability check is compile-range aware. Calling the same selector
    from both IR dispatch and the registered implementation keeps lowering on
    the kernel chosen for that entire range; it must not branch on a symbolic
    example tensor's row-count hint.
    """
    if _jit_fused_add_rms_norm_supports_args(
        x, x_residual, weight, epsilon, variance_size
    ):
        return "jit"
    if _c_ext_fused_add_rms_norm_supports_args(
        x, x_residual, weight, epsilon, variance_size
    ):
        return "c_ext"
    return None


def _fused_add_rms_norm_supports_args(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    return (
        _select_musa_fused_add_rms_norm_impl(
            x, x_residual, weight, epsilon, variance_size
        )
        is not None
    )


@ir.ops.fused_add_rms_norm.register_impl(
    "musa",
    inplace=True,
    supports_args=_fused_add_rms_norm_supports_args,
    supported=current_platform.is_musa(),
)
def fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """MUSA in-place provider for ``vllm.ir.ops.fused_add_rms_norm``."""
    assert variance_size is None
    assert weight is not None
    selected = _select_musa_fused_add_rms_norm_impl(
        x, x_residual, weight, epsilon, variance_size
    )
    if selected == "jit":
        from vllm_musa.jit_kernel.csrc.norm import fused_add_rmsnorm

        # IR supplies the effective scale. Gemma has already materialized
        # ``parameter.float() + 1``, so applying Gemma's offset here would
        # double it.
        return fused_add_rmsnorm(x, x_residual, weight, epsilon, gemma=False)
    if selected == "c_ext":
        from vllm_musa import _custom_ops as musa_ops

        musa_ops.musa_fused_add_rms_norm(x, x_residual, weight, epsilon)
        return x, x_residual
    raise AssertionError("musa provider invoked with unsupported arguments")
