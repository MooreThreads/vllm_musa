# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Code inside this file can safely assume musa platform, e.g. importing
pymtml. However, it should not initialize musa context.
"""

import os
from collections.abc import Callable
from functools import cache, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from packaging.version import Version

# isort: off
import torchada  # noqa: F401
import torch

# isort: on
from typing_extensions import ParamSpec
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
    from vllm.v1.attention.selector import AttentionSelectorConfig
else:
    VllmConfig = None
    CacheDType = None

import pymtml as pynvml

from vllm_musa.optimization_contract import (
    ModelFamily,
    OptimizationFeature,
)
from vllm_musa.optimization_contract import policy as contract_policy
from vllm_musa.optimization_contract import (
    resolve_optimization_contract,
)
from vllm_musa.tuning import FUSED_ADD_RMSNORM_MIN_ROWS

logger = init_logger(__name__)

_QWEN3_VL_ARCHITECTURES = {"Qwen3VLForConditionalGeneration"}


def _is_torch_211_or_newer() -> bool:
    return Version(torch.__version__.split("+", 1)[0]) >= Version("2.11")


_P = ParamSpec("_P")
_R = TypeVar("_R")


@cache
def _get_backend_priorities(
    use_mla: bool,
    device_capability: DeviceCapability,
    num_heads: int | None = None,
) -> list[AttentionBackendEnum]:
    """Get backend priorities with lazy import to avoid circular dependency."""
    if use_mla:
        return [
            AttentionBackendEnum.FLASHMLA,
            AttentionBackendEnum.FLASHMLA_SPARSE,
            AttentionBackendEnum.TRITON_MLA,
        ]
    else:
        return [
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.TRITON_ATTN,
            AttentionBackendEnum.TURBOQUANT,
        ]


def with_mtml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        pynvml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            # Note: We intentionally do NOT call nvmlShutdown() here because
            # pymtml has a bug where nvmlInit() fails after nvmlShutdown()
            # has been called. The library handles cleanup on process exit.
            pass

    return wrapper


def register_attention_backends() -> None:
    # Pre-register all attention backends
    register_backend(
        AttentionBackendEnum.FLASHMLA,
        class_path="vllm_musa.v1.attention.backends.mla.flashmla.MUSAFlashMLABackend",
    )
    register_backend(
        AttentionBackendEnum.FLASHMLA_SPARSE,
        class_path=(
            "vllm_musa.v1.attention.backends.mla.flashmla_sparse."
            "MUSAFlashMLASparseBackend"
        ),
    )
    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        class_path="vllm_musa.v1.attention.backends.flash_attn.MUSAFlashAttentionBackend",
    )
    register_backend(
        AttentionBackendEnum.TURBOQUANT,
        class_path=(
            "vllm_musa.v1.attention.backends.turboquant."
            "MUSATurboQuantAttentionBackend"
        ),
    )
    tree_attn_backend = getattr(AttentionBackendEnum, "TREE_ATTN", None)
    if tree_attn_backend is not None:
        # tree drafting via a MUSA-routed TreeAttention backend
        # (Triton unified_attention is already MUSA-patched; reshape_and_cache_flash
        # is wired through fa_utils.reshape_and_cache_flash). v0.22 removed this
        # enum member, so only register it on older upstream snapshots.
        register_backend(
            tree_attn_backend,
            class_path=(
                "vllm_musa.v1.attention.backends.tree_attn.MUSATreeAttentionBackend"
            ),
        )


def _will_capture_piecewise_cudagraph(vllm_config: Any) -> bool:
    """Whether this config will do PIECEWISE CUDAGraph capture under torch.compile.

    Called from ``check_and_update_config`` (``VllmConfig.__post_init__``), which runs
    BEFORE vLLM finalizes ``compilation_config.backend``/``mode`` and the
    ``cudagraph_mode`` default -- so we cannot gate on ``backend == "inductor"`` (unset
    here). Exclude only the cases that never capture piecewise: ``enforce_eager``,
    ``mode == NONE``, or an explicitly non-piecewise ``cudagraph_mode``. A ``None``
    ``cudagraph_mode`` means "not yet defaulted"; the v1 default is FULL_AND_PIECEWISE
    (piecewise), so None counts as piecewise.
    """
    from vllm.config.compilation import CompilationMode

    if getattr(vllm_config.model_config, "enforce_eager", False):
        return False
    comp = vllm_config.compilation_config
    if getattr(comp, "mode", None) == CompilationMode.NONE:
        return False
    cg_mode = getattr(comp, "cudagraph_mode", None)
    if cg_mode is not None and not cg_mode.has_piecewise_cudagraphs():
        return False
    return True


def _should_route_quantized_piecewise_ops_native(vllm_config: Any) -> bool:
    """Whether the quantized PIECEWISE correctness route applies to this model."""
    return getattr(
        vllm_config, "quant_config", None
    ) is not None and _will_capture_piecewise_cudagraph(vllm_config)


def _configure_fused_add_rmsnorm_compile_range(
    vllm_config: Any, *, native_custom_ops: bool
) -> bool:
    """Split compilation at the measured fused-add RMSNorm profit boundary."""
    ir_priority = getattr(
        getattr(vllm_config, "kernel_config", None),
        "ir_op_priority",
        None,
    )
    fused_priority = getattr(ir_priority, "fused_add_rms_norm", None)
    native_custom_ops = native_custom_ops or bool(
        fused_priority and fused_priority[0] == "native"
    )
    model_config = getattr(vllm_config, "model_config", None)
    get_hidden_size = getattr(model_config, "get_hidden_size", None)
    max_tokens = getattr(
        getattr(vllm_config, "scheduler_config", None),
        "max_num_batched_tokens",
        0,
    )
    if (
        native_custom_ops
        or model_config is None
        or not callable(get_hidden_size)
        or get_hidden_size() != 5120
        or getattr(model_config, "dtype", None) != torch.bfloat16
        or getattr(model_config, "enforce_eager", False)
        or max_tokens is None
        or max_tokens < FUSED_ADD_RMSNORM_MIN_ROWS
    ):
        return False

    comp = vllm_config.compilation_config
    endpoints = list(comp.compile_ranges_endpoints or [])
    fallback_endpoint = FUSED_ADD_RMSNORM_MIN_ROWS - 1
    if fallback_endpoint in endpoints:
        return False
    comp.compile_ranges_endpoints = sorted([*endpoints, fallback_endpoint])
    return True


class MUSAPlatformBase(Platform):
    _enum = PlatformEnum.OOT  # Out-of-tree platform
    device_name: str = "musa"
    device_type: str = "musa"
    dispatch_key: str = "MUSA"
    ray_device_key: str = "GPU"
    dist_backend: str = "mccl"  # MUSA's NCCL equivalent
    device_control_env_var: str = "MUSA_VISIBLE_DEVICES"
    ray_noset_device_env_vars: list[str] = [
        "RAY_EXPERIMENTAL_NOSET_MUSA_VISIBLE_DEVICES",
    ]

    @classmethod
    def get_pass_manager_cls(cls) -> str:
        return "vllm_musa.compilation.passes.MusaPostGradPassManager"

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        # MUSA GPUs support BF16 and FP16
        return [torch.bfloat16, torch.float16, torch.float32]

    def is_cuda_alike(self) -> bool:
        """MUSA is CUDA-alike for compatibility purposes."""
        return True

    def is_musa(self) -> bool:
        """This is the MUSA platform."""
        return True

    @classmethod
    def is_float64_supported(cls) -> bool:
        return False

    def is_sleep_mode_available(self) -> bool:
        """MUSA supports sleep mode."""
        return True

    @classmethod
    def import_kernels(cls) -> None:
        """Import upstream vLLM kernels, including v0.22 stable ABI ops."""
        super().import_kernels()
        try:
            import vllm._C_stable_libtorch  # noqa: F401
        except ImportError as e:
            logger.warning("Failed to import from vllm._C_stable_libtorch: %r", e)

    @classmethod
    def check_runner_kv_caches_multi_layer(cls) -> None:
        """Allow multiple attention caches sharing one decoder layer index.

        The MUSA V2 runner uses the same ordered ``runner_kv_caches`` binding
        contract as CUDA, ROCm, and XPU. DeepSeek-V4 exercises this with its
        sparse indexer and MLA caches in the same decoder block.
        """
        pass

    @classmethod
    def import_ir_kernels(cls) -> None:
        """Import upstream and MUSA-OOT IR-op providers.

        Order matters: upstream first (registers `native` / `vllm_c` /
        `oink` / etc.), then OOT MUSA providers so they appear in the
        registry alongside upstream impls. Used by
        `vllm.config.kernel.KernelConfig.set_priority()` to ensure
        every provider mentioned in `ir_op_priority` is registered
        before the dispatcher needs it.
        """
        super().import_ir_kernels()
        try:
            import vllm_musa.kernels  # noqa: F401
        except ImportError as exc:
            from vllm.logger import init_logger

            init_logger(__name__).info(
                "vllm_musa.kernels unavailable (%s); MUSA IR providers "
                "will not be registered. Upstream providers remain "
                "available.",
                exc,
            )

    @classmethod
    def get_default_ir_op_priority(cls, vllm_config):
        """Platform-default priority list for vllm.ir.ops on MUSA.

        When compiling with Inductor, prefer the `native` (pure-PyTorch)
        rms_norm implementation. Prefer the measured in-place `musa`
        fused_add_rms_norm implementation in both compiled and eager paths.
        That single provider dispatches internally between the measured JIT
        kernel and the broad pre-existing C-extension kernel, then falls back
        to native through normal IR priority; IR functionalization owns
        activation donation.
        This mirrors the upstream `cuda.py` pattern
        (`default = ["native"] if using_inductor else ["vllm_c", "native"]`):
        under Inductor the native rms_norm is a handful of
        elementwise/reduction ops the compiler can fuse with its
        neighbours, whereas a `torch.ops._C.*` custom op is an opaque
        fusion barrier; in eager mode there is no fusion to lose so the
        kernel provider is taken directly.

        The non-fused rms_norm choice remains upstream-consistent; the fused
        provider is independently gated by its measured supports_args scope.
        """
        from vllm.config.compilation import CompilationMode
        from vllm.config.kernel import IrOpPriorityConfig

        cc = vllm_config.compilation_config
        using_inductor = cc.backend == "inductor" and cc.mode != CompilationMode.NONE
        if using_inductor:
            # Let Inductor fuse the native reference impl. Keep the
            # `musa` custom-op provider out of the priority here — it is
            # a fusion barrier (no measurable batch1 effect, but native
            # is the upstream-aligned default).
            default = ["native"]
            rms_norm = ["native"]
        else:
            # Eager path: no Inductor fusion to lose, so take the rms_norm
            # kernel. Keep other IR ops native by default.
            default = ["native"]
            rms_norm = ["musa", "native"]
        # VLLM_MUSA_CUSTOM_OP_USE_NATIVE is a correctness route for quantized
        # models under piecewise CUDAGraph capture. It must remain authoritative
        # for the IR provider too; unlike supports_args, this is global provider
        # enablement and therefore belongs in the priority policy.
        from vllm_musa.utils.environ import envs as musa_envs

        native_custom_ops = musa_envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get()
        if not musa_envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.is_set():
            # VllmConfig installs platform IR priorities before invoking
            # check_and_update_config(), where the process-wide safety route is
            # normally published for custom-op forward paths. Derive the same
            # policy from this config here so quantized PIECEWISE compilation
            # cannot retain an unsafe fused-add provider in its frozen priority.
            native_custom_ops = _should_route_quantized_piecewise_ops_native(
                vllm_config
            )
        fused_add_rms_norm = ["native"] if native_custom_ops else ["musa", "native"]
        # The direct Triton HOP is profitable for every validated dense shape
        # family; supports_args remains the semantic/shape guard. Keep routed
        # MoE on native by default because a 100-prompt Qwen3.6 TP8 sweep
        # regressed 2.81% as eager-faithful BF16 rounding changed expert routes.
        # Users can still explicitly select the generic provider for A/B work.
        contract = resolve_optimization_contract(vllm_config)
        gated_qkv_rms_norm_rope = (
            ["musa_inductor", "native"]
            if using_inductor and contract.model.has_routed_experts is not True
            else ["native"]
        )
        return IrOpPriorityConfig.with_default(
            default,
            rms_norm=rms_norm,
            fused_add_rms_norm=fused_add_rms_norm,
            gated_qkv_rms_norm_rope=gated_qkv_rms_norm_rope,
        )

    @classmethod
    def support_deep_gemm(cls) -> bool:
        """
        Returns if DeepGEMM is supported by the current platform.
        """
        return True

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        torch.cuda.set_device(device)
        # With this trick we can force the device to be set eagerly
        # see https://github.com/pytorch/pytorch/issues/155668
        # for why and when it is needed
        _ = torch.zeros(1, device=device)

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        torch.musa.manual_seed_all(seed)

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        raise NotImplementedError

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        raise NotImplementedError

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        raise NotImplementedError

    @classmethod
    def is_fully_connected(cls, device_ids: list[int]) -> bool:
        raise NotImplementedError

    @classmethod
    def log_warnings(cls):
        pass

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: "VllmConfig") -> None:
        # Ensure custom ops are enabled for MUSA platform so that
        # OOT forward implementations (forward_oot) are dispatched.
        # This must be set here (before VllmConfig.__post_init__ defaults)
        # to prevent custom_ops from being defaulted to ['none'] when
        # the inductor backend is active.
        compilation_config = vllm_config.compilation_config
        if all(s not in compilation_config.custom_ops for s in ("all", "none")):
            compilation_config.custom_ops.append("all")

        # torch 2.11's Inductor tiling heuristic turns Qwen3-VL's decode-time
        # clone/index-select/split kernel into a slower 2D grid on MUSA. The
        # torch 2.9 1D grid is restored by limiting this model's pointwise
        # kernels to one tile dimension. Preserve an explicit user override.
        architectures = set(
            getattr(
                getattr(vllm_config.model_config, "hf_config", None),
                "architectures",
                None,
            )
            or ()
        )
        inductor_config = compilation_config.inductor_compile_config
        if (
            _is_torch_211_or_newer()
            and architectures & _QWEN3_VL_ARCHITECTURES
            and "triton.max_tiles" not in inductor_config
        ):
            inductor_config["triton.max_tiles"] = 1
            logger.info(
                "Using triton.max_tiles=1 for Qwen3-VL on torch >= 2.11 "
                "to preserve the MUSA decode pointwise layout"
            )

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        # when dflash spec-decode is active, coerce the draft-loop
        # CUDAGraph capture to block-aligned FULL sizes (see below). The dflash
        # source patch (DFlashProposer.dummy_run signature) is applied at BUILD
        # time to the cloned vLLM (setup.py -> series/0030 dflash.patch), so the
        # installed dflash.py the spawn workers import is already patched — no
        # runtime source-patching is needed here.
        spec_config = getattr(vllm_config, "speculative_config", None)
        # Detect dflash via the STABLE `use_dflash()` API (the same predicate
        # gpu_model_runner branches on) rather than the `method` string, whose
        # representation can change across vLLM versions; fall back to the string
        # only if use_dflash() is unavailable.
        _use_dflash = getattr(spec_config, "use_dflash", None)
        if spec_config is not None and (
            _use_dflash()
            if callable(_use_dflash)
            else getattr(spec_config, "method", None) == "dflash"
        ):
            # the dflash draft-loop FULL CUDAGraph capture is
            # default-on (opt out with VLLM_MUSA_DFLASH_FULL_WRAP=0). It makes
            # the draft forward process a (1 + num_speculative_tokens)-token
            # verify block, so every captured cudagraph size must be a whole
            # multiple of that block. vLLM's default capture sizes
            # ([1, 2, 4, 8, 16, ...]) are not block-aligned, and capturing the
            # draft at a sub-block size raises "MUSA error: an illegal memory
            # access". Coerce to pure FULL + the block-aligned subset of the
            # configured sizes (default: the single BS=1 block). Multi-block
            # (BS>1) draft capture is tracked separately.
            import os as _df_os

            if _df_os.environ.get("VLLM_MUSA_DFLASH_FULL_WRAP", "1") != "0":
                _comp = getattr(vllm_config, "compilation_config", None)
                _nspec = getattr(spec_config, "num_speculative_tokens", None)
                if _comp is not None and _nspec:
                    from vllm.config import CUDAGraphMode as _CGM

                    _block = 1 + int(_nspec)
                    # Capture exactly the single BS=1 verify block — the proven,
                    # fast size (no padding). vLLM's default capture sizes are
                    # either not block-aligned (1, 2, 4, 8, 16, ...) or far too
                    # large (the only multiples of the block in the default set
                    # are 72, 144, ... = BS>=8, which leave a BS=1 9-token decode
                    # padded to 72 and running ~35 tok/s vs 48 at the exact
                    # block). extends this to BS=1,2,4,8 = block-aligned
                    # [9,18,36,72]; a BS=N decode uses the size-N*block graph exactly.
                    _maxseq = (
                        getattr(
                            getattr(vllm_config, "scheduler_config", None),
                            "max_num_seqs",
                            8,
                        )
                        or 8
                    )
                    _comp.cudagraph_capture_sizes = [
                        _block * k for k in (1, 2, 4, 8) if k <= max(1, _maxseq)
                    ]
                    _comp.max_cudagraph_capture_size = _comp.cudagraph_capture_sizes[-1]
                    _comp.cudagraph_mode = _CGM.FULL
                    logger.info(
                        "dflash draft capture default-on; coerced "
                        "cudagraph_mode=FULL, capture_sizes=%s (block=%d)",
                        _comp.cudagraph_capture_sizes,
                        _block,
                    )

        # FP8 correctness under torch.compile + PIECEWISE CUDAGraph capture: the
        # MUSA RMSNorm / SiluAndMul / per-token-group FP8-quant custom-op kernels
        # (forward_oot) are opaque to Inductor's quant fusion and leave the quant's
        # (q, scale) in buffers it cannot lifetime-track, so the captured piecewise
        # graph reads a stale scale and the FP8 forward emits degenerate output.
        # Route them to the native (decomposable, Inductor-fused, CUDAGraph-safe)
        # path while piecewise capture is active -- no perf loss (Inductor fuses the
        # quant). Eager / FULL_DECODE_ONLY and bf16 are unaffected, as is the
        # deep_gemm linear path (it fuses quant+GEMM, so its buffers never escape).
        from vllm_musa.utils.environ import envs as musa_envs

        # Only quantized models can enter the FP8 quant path described above.
        # Keeping this process-wide route enabled for unquantized BF16 models
        # unnecessarily disables their safe normalization kernels.
        route_native = _should_route_quantized_piecewise_ops_native(vllm_config)
        logger.debug(
            "MUSA custom-op native-routing: quantized_piecewise=%s", route_native
        )
        # Set the flag process-wide (not on the config): the op forward_oot paths read
        # this env live and spawn workers inherit os.environ. First-writer-wins (the
        # is_set() guard) is safe -- the native path is correct in every cudagraph mode,
        # so an inherited value can never yield wrong output, only the native path.
        if route_native and not musa_envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.is_set():
            musa_envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.set(True)
            logger.info(
                "Routing MUSA RMSNorm/SiluAndMul/QuantFP8 to the native "
                "(Inductor-fused) path: the custom-op kernels corrupt FP8 output "
                "under piecewise CUDAGraph capture. Set "
                "VLLM_MUSA_CUSTOM_OP_USE_NATIVE=0 to force the kernels."
            )

        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config

        # The fused-add provider is profitable from
        # FUSED_ADD_RMSNORM_MIN_ROWS onward. vLLM picks IR implementations once
        # per dynamic compile range and drops shape guards, so split immediately
        # below that threshold rather than freezing a decision from a symbolic
        # example-value hint. This is shape/dtype based and applies to any model
        # exposing the IR op.
        if _configure_fused_add_rmsnorm_compile_range(
            vllm_config,
            native_custom_ops=musa_envs.VLLM_MUSA_CUSTOM_OP_USE_NATIVE.get(),
        ):
            logger.info(
                "Splitting MUSA fused-add RMSNorm compile ranges at %d rows",
                FUSED_ADD_RMSNORM_MIN_ROWS - 1,
            )

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_musa.worker.MTGPUWorker"

        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        # TODO(lucas): handle this more gracefully
        # Note: model_config may be None during testing
        # Note: block_size is initialized in
        # HybridAttentionMambaModelConfig.verify_and_update_config
        # for models with both attention and mamba,
        # and doesn't need to be reinitialized here
        if (
            model_config is not None
            and model_config.use_mla
            and cache_config.block_size is not None
        ):
            use_sparse = hasattr(vllm_config.model_config.hf_config, "index_topk")
            # If `--attention-config.backend` is not set and we are using MLA,
            # then we default to FlashMLA backend.
            use_flashmla = False
            use_flashmla_sparse = False

            from vllm_musa.v1.attention.ops.flashmla import is_flashmla_dense_supported

            if vllm_config.attention_config.backend is None:
                # Default case: use FlashMLA if supported
                if is_flashmla_dense_supported()[0]:
                    use_flashmla = True
            else:
                # Forced case
                backend = vllm_config.attention_config.backend
                use_flashmla = backend == AttentionBackendEnum.FLASHMLA
                use_flashmla_sparse = backend == AttentionBackendEnum.FLASHMLA_SPARSE

            if (
                use_flashmla
                and is_flashmla_dense_supported()[0]
                and cache_config.block_size % 64 != 0
            ):
                cache_config.block_size = 64
                logger.info("Forcing kv cache block size to 64 for FlashMLA backend.")

            if use_sparse:
                if not use_flashmla_sparse:
                    use_flashmla_sparse = True

                sparse_block_size = (
                    contract_policy.deepseek_v4_flashmla_sparse_page_size(vllm_config)
                )
                if use_flashmla_sparse and cache_config.block_size != sparse_block_size:
                    cache_config.block_size = sparse_block_size
                    logger.info(
                        "Forcing kv cache block size to %d for FlashMLASparse backend.",
                        sparse_block_size,
                    )

        scheduler_config = vllm_config.scheduler_config
        # Note: model_config may be None during testing
        if (
            model_config is not None
            and model_config.is_mm_prefix_lm
            and scheduler_config.is_multimodal_model
            and not scheduler_config.disable_chunked_mm_input
        ):
            logger.warning(
                "Forcing --disable_chunked_mm_input for models "
                "with multimodal-bidirectional attention."
            )
            scheduler_config.disable_chunked_mm_input = True

    @classmethod
    def update_block_size_for_backend(cls, vllm_config: "VllmConfig") -> None:
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        # The separate Mamba pools have independent physical pages and block-id
        # spaces. In the no-prefix-cache mode used by Qwen3.5/Qwen3.6 serving,
        # do not let upstream inflate the attention block to the full recurrent
        # state page (1056 tokens for Qwen3.6-35B-A3B). That inflation wastes
        # attention KV capacity and forces avoidable request preemption.
        separate_mamba_pages = (
            model_config is not None
            and cache_config is not None
            and model_config.is_hybrid
            and cache_config.mamba_cache_mode == "none"
            and resolve_optimization_contract(vllm_config).prefers(
                OptimizationFeature.HYBRID_SEPARATE_MAMBA_POOL
            )
        )
        if separate_mamba_pages:
            backend_cls = cls._find_non_ssm_backend(vllm_config)
            if backend_cls is not None:
                from vllm.config.vllm import set_current_vllm_config

                target_block_size = (
                    cache_config.block_size
                    if cache_config.user_specified_block_size
                    else 64
                )
                with set_current_vllm_config(vllm_config):
                    supported = backend_cls.supports_block_size(target_block_size)
                if supported:
                    cache_config.block_size = target_block_size
                    cache_config.mamba_page_size_padded = None
                    logger.info(
                        "[MUSA]Keeping separate-pool hybrid attention block "
                        "size at %d for %s backend.",
                        target_block_size,
                        backend_cls.get_name(),
                    )
                    return

        # MUSA: 64 is the only optimal KV page here (paged FMHA/MLA decode takes
        # the TME bulk-gather path). Let upstream pick and mamba-align the page
        # first, then pin every non-hybrid backend that supports a 64 page to 64,
        # whatever super() picked. A user --block-size and fixed-page kernels that
        # cannot take 64 (sparse MLA at 256) are left as super() resolved them.
        super().update_block_size_for_backend(vllm_config)
        if (
            model_config is None
            or cache_config is None
            or cache_config.user_specified_block_size
            or model_config.is_hybrid
        ):
            return
        if (
            resolve_optimization_contract(vllm_config).model.family
            is ModelFamily.DEEPSEEK_V4
        ):
            return
        backend_cls = cls._find_non_ssm_backend(vllm_config)
        if backend_cls is None:
            return
        from vllm.config.vllm import set_current_vllm_config

        with set_current_vllm_config(vllm_config):
            supports_64 = backend_cls.supports_block_size(64)
        if supports_64 and cache_config.block_size != 64:
            logger.info(
                "[MUSA]Setting attention block size to 64 for %s backend.",
                backend_cls.get_name(),
            )
            cache_config.block_size = 64

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.max_memory_allocated(device)

    @classmethod
    def get_valid_backends(
        cls,
        device_capability: DeviceCapability,
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> tuple[
        list[tuple["AttentionBackendEnum", int]],
        dict["AttentionBackendEnum", list[str]],
    ]:
        valid_backends_priorities = []
        invalid_reasons = {}

        backend_priorities = _get_backend_priorities(
            attn_selector_config.use_mla,
            device_capability,
            num_heads,
        )
        for priority, backend in enumerate(backend_priorities):
            try:
                backend_class = backend.get_class()
                invalid_reasons_i = backend_class.validate_configuration(
                    device_capability=device_capability,
                    **attn_selector_config._asdict(),
                )
            except ImportError:
                invalid_reasons_i = ["ImportError"]
            if invalid_reasons_i:
                invalid_reasons[backend] = invalid_reasons_i
            else:
                valid_backends_priorities.append((backend, priority))

        return valid_backends_priorities, invalid_reasons

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        register_attention_backends()
        device_capability = cls.get_device_capability()
        assert device_capability is not None

        attn_selector_config = attn_selector_config._replace(block_size=None)
        # First try checking just the selected backend, if there is one.
        if selected_backend is not None:
            try:
                backend_class = selected_backend.get_class()
                invalid_reasons = backend_class.validate_configuration(
                    device_capability=device_capability,
                    **attn_selector_config._asdict(),
                )
            except ImportError:
                invalid_reasons = ["ImportError"]
            if invalid_reasons:
                raise ValueError(
                    f"Selected backend {selected_backend} is not valid for "
                    f"this configuration. Reason: {invalid_reasons}"
                )
            else:
                logger.info("Using %s backend.", selected_backend)
                return selected_backend.get_path()

        # No selected backend or the selected backend is invalid,
        # so we try finding a valid backend.
        valid_backends_priorities, invalid_reasons = cls.get_valid_backends(
            device_capability=device_capability,
            attn_selector_config=attn_selector_config,
            num_heads=num_heads,
        )
        reasons_str = (
            "{"
            + ", ".join(
                f"{backend.name}: [{', '.join(reasons)}]"
                for backend, reasons in invalid_reasons.items()
            )
            + "}"
        )
        config_str = attn_selector_config.__repr__()
        logger.debug_once(
            f"Some attention backends are not valid for {cls.device_name} with "
            f"{config_str}. Reasons: {reasons_str}."
        )
        if len(valid_backends_priorities) == 0:
            raise ValueError(
                f"No valid attention backend found for {cls.device_name} "
                f"with {config_str}. Reasons: {reasons_str}."
            )

        # We have found some valid backends. Select the one with the
        # highest priority.
        sorted_indices = sorted(
            range(len(valid_backends_priorities)),
            key=lambda i: valid_backends_priorities[i][1],
        )
        selected_index = sorted_indices[0]
        selected_backend = valid_backends_priorities[selected_index][0]
        logger.info_once(
            "Using %s attention backend out of potential backends: %s.",
            selected_backend.name,
            "[" + ", ".join(f"'{b[0].name}'" for b in valid_backends_priorities) + "]",
            scope="local",
        )

        return selected_backend.get_path()

    @classmethod
    def get_supported_vit_attn_backends(cls) -> list["AttentionBackendEnum"]:
        return [
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.TRITON_ATTN,
            AttentionBackendEnum.TORCH_SDPA,
        ]

    @classmethod
    def get_vit_attn_backend(
        cls,
        head_size: int,
        dtype: torch.dtype,
        backend: "AttentionBackendEnum | None" = None,
    ) -> "AttentionBackendEnum":
        if backend is not None:
            assert backend in cls.get_supported_vit_attn_backends(), (
                f"Backend {backend} is not supported for vit attention. "
                f"Supported backends are: {cls.get_supported_vit_attn_backends()}"
            )
            logger.info_once(f"Using backend {backend} for vit attention")
            return backend

        cc = cls.get_device_capability()
        for vit_attn_backend in cls.get_supported_vit_attn_backends():
            if vit_attn_backend == AttentionBackendEnum.TORCH_SDPA:
                continue
            try:
                backend_class = vit_attn_backend.get_class()
                is_backend_supported = backend_class.supports_head_size(
                    head_size
                ) and backend_class.supports_dtype(dtype)
                if cc is not None:
                    is_backend_supported = (
                        is_backend_supported
                        and backend_class.supports_compute_capability(cc)
                    )
                if is_backend_supported:
                    logger.info_once(
                        f"Using backend {vit_attn_backend} for vit attention"
                    )
                    return vit_attn_backend
            except ImportError:
                pass

        return AttentionBackendEnum.TORCH_SDPA

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"  # noqa

    @classmethod
    def supports_fp8(cls) -> bool:
        return cls.has_device_capability((3, 1))

    @classmethod
    def use_custom_allreduce(cls) -> bool:
        return True

    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm.compilation.cuda_graph.CUDAGraphWrapper"

    @classmethod
    def device_count(cls) -> int:
        return torch.cuda.device_count()

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype):
        # MUSA devices support bfloat16 natively, no capability check needed.
        pass

    @classmethod
    def insert_blocks_to_device(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from src_cache to dst_cache on GPU."""
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.to(dst_cache.device)

    @classmethod
    def swap_out_blocks_to_host(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from GPU to host (CPU)."""
        _src_cache = src_cache[:, src_block_indices]
        dst_cache[:, dst_block_indices] = _src_cache.cpu()

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True

    @classmethod
    def num_compute_units(cls, device_id=0):
        return torch.cuda.get_device_properties(device_id).multi_processor_count


# MTML utils
# Note that MTML is not affected by `MUSA_VISIBLE_DEVICES`,
# all the related functions work on real physical device ids.
# the major benefit of using MTML is that it will not initialize MUSA
class MtmlMUSAPlatform(MUSAPlatformBase):
    @classmethod
    @cache
    @with_mtml_context
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        try:
            # XXX (MUSA): CUDA uses physical device ids (cls.device_id_to_physical_device_id(device_id)),
            # but torch.musa.get_device_capability uses logical device ids when MUSA_VISIBLE_DEVICES is set.
            # Since pymtml doesn't do remapping, so we can only use device_id here.
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            return DeviceCapability(major=major, minor=minor)
        except RuntimeError:
            return None

    @classmethod
    @with_mtml_context
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        try:
            return super().has_device_capability(capability, device_id)
        except RuntimeError:
            return False

    @classmethod
    @with_mtml_context
    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        return cls._get_physical_device_name(physical_device_id)

    @classmethod
    @with_mtml_context
    def get_device_uuid(cls, device_id: int = 0) -> str:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return pynvml.nvmlDeviceGetUUID(handle)

    @classmethod
    @with_mtml_context
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)

    @classmethod
    @with_mtml_context
    def get_device_numa_node(cls, device_id: int = 0) -> int | None:
        """Get the NUMA node ID for a GPU device."""
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)

        try:
            numa_node = pynvml.nvmlDeviceGetNumaNodeId(handle)
            if cls._numa_node_has_cpus(numa_node):
                return numa_node
            logger.debug(
                "NUMA node %d for GPU %d has no CPUs, falling back to "
                "CPU-affinity-based detection",
                numa_node,
                device_id,
            )
        except Exception:
            pass

        try:
            cpu_ids = cls._get_device_cpu_affinity(handle)
            if cpu_ids:
                numa_node = cls._get_numa_node_for_cpu(cpu_ids[0])
                if numa_node is not None:
                    logger.debug(
                        "Determined NUMA node %d for GPU %d via CPU affinity",
                        numa_node,
                        device_id,
                    )
                    return numa_node
        except Exception as e:
            logger.warning("Failed to get NUMA node for GPU %d: %s", device_id, e)

        return None

    @classmethod
    def _numa_node_has_cpus(cls, node_id: int) -> bool:
        """Check whether a NUMA node has any CPUs assigned to it."""
        cpulist_file = Path(f"/sys/devices/system/node/node{node_id}/cpulist")
        try:
            return cpulist_file.read_text().strip() != ""
        except (OSError, ValueError):
            return False

    @classmethod
    def _get_device_cpu_affinity(cls, handle) -> list[int]:
        """Get the list of CPU IDs associated with a GPU via MTML."""
        cpu_count = os.cpu_count()
        if cpu_count is None:
            return []

        cpu_set_size = (cpu_count + 63) // 64
        cpu_affinity_mask = pynvml.nvmlDeviceGetCpuAffinity(handle, cpu_set_size)

        cpu_ids = []
        for i, mask in enumerate(cpu_affinity_mask):
            for bit in range(64):
                cpu_id = i * 64 + bit
                if cpu_id >= cpu_count:
                    break
                if mask & (1 << bit):
                    cpu_ids.append(cpu_id)
        return cpu_ids

    @classmethod
    def _get_numa_node_for_cpu(cls, cpu_id: int) -> int | None:
        """Determine which NUMA node a CPU belongs to."""
        node_path = Path("/sys/devices/system/node")
        if not node_path.exists():
            return None

        for node_dir in node_path.iterdir():
            if not node_dir.name.startswith("node"):
                continue
            try:
                node_id = int(node_dir.name[4:])
                cpulist_file = node_dir / "cpulist"
                if cpulist_file.exists():
                    cpulist = cpulist_file.read_text().strip()
                    if cls._cpu_in_cpulist(cpu_id, cpulist):
                        return node_id
            except (ValueError, OSError):
                continue
        return None

    @classmethod
    def _cpu_in_cpulist(cls, cpu_id: int, cpulist: str) -> bool:
        """Check if a CPU ID is in a cpulist string such as '0-3,8-11'."""
        for part in cpulist.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-", 1)
                if int(start) <= cpu_id <= int(end):
                    return True
            elif part.isdigit() and int(part) == cpu_id:
                return True
        return False

    @classmethod
    @with_mtml_context
    def get_all_device_numa_nodes(cls) -> list[int] | None:
        """Get NUMA nodes for all visible GPU devices."""
        try:
            numa_nodes = []
            for device_id in range(cls.device_count()):
                numa_node = cls.get_device_numa_node(device_id)
                if numa_node is None:
                    logger.warning(
                        "Could not detect NUMA node for GPU %d, "
                        "disabling automatic NUMA binding",
                        device_id,
                    )
                    return None
                numa_nodes.append(numa_node)
            return numa_nodes
        except Exception as e:
            logger.warning("Failed to get NUMA nodes for GPUs: %s", e)
            return None

    @classmethod
    @with_mtml_context
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        """
        query if the set of gpus are fully connected by mtlink (1 hop)
        """
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in physical_device_ids]
        for i, handle in enumerate(handles):
            for j, peer_handle in enumerate(handles):
                if i < j:
                    try:
                        p2p_status = pynvml.nvmlDeviceGetP2PStatus(
                            handle,
                            peer_handle,
                            pynvml.NVML_P2P_CAPS_INDEX_NVLINK,
                        )
                        if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                            return False
                    except pynvml.NVMLError:
                        logger.exception(
                            "MtLink detection failed. This is normal if"
                            " your machine has no MtLink equipped."
                        )
                        return False
        return True

    @classmethod
    def _get_physical_device_name(cls, device_id: int = 0) -> str:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        return pynvml.nvmlDeviceGetName(handle)

    @classmethod
    @with_mtml_context
    def log_warnings(cls):
        device_ids: int = pynvml.nvmlDeviceGetCount()
        if device_ids > 1:
            device_names = [cls._get_physical_device_name(i) for i in range(device_ids)]
            if (
                len(set(device_names)) > 1
                and os.environ.get("MUSA_DEVICE_ORDER") != "PCI_BUS_ID"
            ):
                logger.warning(
                    "Detected different devices in the system: %s. Please"
                    " make sure to set `MUSA_DEVICE_ORDER=PCI_BUS_ID` to "
                    "avoid unexpected behavior.",
                    ", ".join(device_names),
                )


class NonMtmlMUSAPlatform(MUSAPlatformBase):
    @classmethod
    @cache
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.cuda.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        logger.exception(
            "MtLink detection not possible, as context support was"
            " not found. Assuming no MtLink available."
        )
        return False


# Autodetect either MTML-enabled or non-MTML platform
# based on whether MTML is available.
mtml_available = False
try:
    try:
        pynvml.nvmlInit()
        mtml_available = True
    except Exception:
        # MTML may not be supported on all systems.
        mtml_available = False
finally:
    # Note: We intentionally do NOT call nvmlShutdown() here because
    # pymtml has a bug where nvmlInit() fails after nvmlShutdown()
    # has been called. The library handles cleanup on process exit.
    pass

MUSAPlatform = MtmlMUSAPlatform if mtml_available else NonMtmlMUSAPlatform

MUSAPlatform.log_warnings()

__all__ = [
    "MUSAPlatform",
    "MUSAPlatformBase",
    "MtmlMUSAPlatform",
    "NonMtmlMUSAPlatform",
    "mtml_available",
    "with_mtml_context",
]
