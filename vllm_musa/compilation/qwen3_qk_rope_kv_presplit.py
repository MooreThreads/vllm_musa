# SPDX-License-Identifier: Apache-2.0
"""Fail-closed pre-split fusion for dense Qwen3 QK norm/RoPE/KV update."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any

import torch
from torch import fx

KV_UPDATE_SPLITTING_OP = "vllm::unified_kv_cache_update"
FUSED_SPLITTING_OP = "vllm::musa_qwen3_qk_rope_and_unified_kv_cache_update"
_MISSING = object()

_SUPPORTED_SPLITS = {
    (2048, 1024, 1024): (16, 8),
    (4096, 1024, 1024): (32, 8),
}


def qwen3_qk_rope_kv_backend_supported(
    vllm_config: Any,
    expected_sites: int,
) -> bool:
    try:
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention.attention import Attention

        layers = get_layers_from_vllm_config(vllm_config, Attention)
    except Exception:
        return False
    if len(layers) != expected_sites:
        return False
    try:
        return all(
            getattr(
                layer.impl,
                "qwen3_qk_rope_kvcache_supported",
                lambda: False,
            )()
            for layer in layers.values()
        )
    except Exception:
        return False


@dataclass(frozen=True)
class Qwen3QKRoPEKVCandidate:
    rope: fx.Node
    kv_update: fx.Node
    attention: fx.Node
    q_head: fx.Node
    k_head: fx.Node
    value: fx.Node
    q_weight: fx.Node
    k_weight: fx.Node
    positions: fx.Node
    cos_sin_cache: fx.Node
    layer_name: str | fx.Node


def _is_call(node: fx.Node, qualified_name: str) -> bool:
    if node.op != "call_function":
        return False
    target = str(node.target)
    expected = qualified_name.replace("::", ".")
    if target == expected or target == f"{expected}.default":
        return True
    module = getattr(node.target, "__module__", None)
    name = getattr(node.target, "__name__", None)
    return (
        isinstance(module, str)
        and isinstance(name, str)
        and (f"{module}.{name}" == expected)
    )


def _arg(
    node: fx.Node,
    index: int,
    name: str,
    default: Any = _MISSING,
) -> Any:
    if name in node.kwargs:
        return node.kwargs[name]
    if len(node.args) > index:
        return node.args[index]
    if default is not _MISSING:
        return default
    return _MISSING


def _tensor_value(node: fx.Node) -> Any:
    for key in ("val", "example_value"):
        value = node.meta.get(key)
        if value is not None:
            return value
    return node.meta.get("tensor_meta")


def _shape(node: fx.Node) -> tuple[Any, ...] | None:
    shape = getattr(_tensor_value(node), "shape", None)
    return tuple(shape) if shape is not None else None


def _shape_tail(node: fx.Node) -> tuple[Any, Any] | None:
    shape = _shape(node)
    return None if shape is None or len(shape) < 2 else (shape[-2], shape[-1])


def _dtype(node: fx.Node) -> torch.dtype | None:
    return getattr(_tensor_value(node), "dtype", None)


def _view_base(node: fx.Node) -> fx.Node | None:
    is_view = node.op == "call_method" and node.target in {"view", "reshape"}
    is_view = is_view or any(
        _is_call(node, target)
        for target in ("aten::view", "aten::reshape", "aten::_unsafe_view")
    )
    if not is_view or not node.args or not isinstance(node.args[0], fx.Node):
        return None
    return node.args[0]


def _split_getitem(node: fx.Node, index: int) -> fx.Node | None:
    if (
        node.op != "call_function"
        or node.target is not operator.getitem
        or len(node.args) != 2
        or node.args[1] != index
        or not isinstance(node.args[0], fx.Node)
    ):
        return None
    return node.args[0]


def _split_sizes(node: fx.Node) -> tuple[int, ...] | None:
    if node.op == "call_method" and node.target == "split":
        sizes = _arg(node, 1, "split_size")
        dim = _arg(node, 2, "dim", 0)
    elif _is_call(node, "aten::split_with_sizes"):
        sizes = _arg(node, 1, "split_sizes")
        dim = _arg(node, 2, "dim", 0)
    else:
        return None
    if not isinstance(sizes, (list, tuple)) or dim not in (-1, 1):
        return None
    return tuple(int(size) for size in sizes)


def _rms_norm_parts(node: fx.Node) -> tuple[fx.Node, fx.Node] | None:
    if not (_is_call(node, "aten::rms_norm") or _is_call(node, "torch::rms_norm")):
        return None
    tensor = _arg(node, 0, "input")
    normalized_shape = _arg(node, 1, "normalized_shape")
    weight = _arg(node, 2, "weight")
    eps = _arg(node, 3, "eps", None)
    if (
        not isinstance(tensor, fx.Node)
        or not isinstance(weight, fx.Node)
        or not isinstance(normalized_shape, (list, tuple))
        or tuple(normalized_shape) != (128,)
        or eps is None
        or float(eps) != 1e-6
    ):
        return None
    return tensor, weight


def _only_users(node: fx.Node, expected: set[fx.Node]) -> bool:
    return set(node.users) == expected


def plan_qwen3_qk_rope_kv_presplit(
    graph_module: fx.GraphModule,
    expected_sites: int,
) -> tuple[Qwen3QKRoPEKVCandidate, ...] | None:
    nodes = tuple(graph_module.graph.nodes)
    all_rope = [node for node in nodes if _is_call(node, "vllm::musa_rotary_embedding")]
    all_kv = [node for node in nodes if _is_call(node, KV_UPDATE_SPLITTING_OP)]
    if len(all_rope) != expected_sites or len(all_kv) != expected_sites:
        return None

    candidates: list[Qwen3QKRoPEKVCandidate] = []
    seen_rope: set[fx.Node] = set()
    seen_layer_names: set[str | fx.Node] = set()
    for kv_update in all_kv:
        if len(kv_update.users) != 1:
            return None
        attention = next(iter(kv_update.users))
        if not _is_call(attention, "vllm::unified_attention_with_output"):
            return None
        query = _arg(attention, 0, "query")
        key = _arg(attention, 1, "key")
        value = _arg(attention, 2, "value")
        layer_name = _arg(attention, 4, "layer_name")
        kv_layer_name = _arg(kv_update, 2, "layer_name")
        dummy = _arg(attention, 7, "kv_cache_dummy_dep", None)
        # torch >= 2.11 hoists vLLM's opaque LayerName into an FX placeholder.
        # Preserve that node as the fused custom op already accepts LayerNameType.
        same_layer_name = (
            kv_layer_name is layer_name
            if isinstance(layer_name, fx.Node)
            else kv_layer_name == layer_name
        )
        if (
            not all(isinstance(node, fx.Node) for node in (query, key, value))
            or dummy is not kv_update
            or not isinstance(layer_name, (str, fx.Node))
            or _arg(kv_update, 0, "key") is not key
            or _arg(kv_update, 1, "value") is not value
            or not same_layer_name
            or layer_name in seen_layer_names
        ):
            return None

        q_flat = _view_base(query)
        k_flat = _view_base(key)
        if q_flat is None or k_flat is None:
            return None
        q_norm = _view_base(q_flat)
        k_norm = _view_base(k_flat)
        if q_norm is None or k_norm is None:
            return None
        q_parts = _rms_norm_parts(q_norm)
        k_parts = _rms_norm_parts(k_norm)
        if q_parts is None or k_parts is None:
            return None
        q_head, q_weight = q_parts
        k_head, k_weight = k_parts
        q_base = _view_base(q_head)
        k_base = _view_base(k_head)
        value_base = _view_base(value)
        if q_base is None or k_base is None or value_base is None:
            return None
        q_split = _split_getitem(q_base, 0)
        k_split = _split_getitem(k_base, 1)
        v_split = _split_getitem(value_base, 2)
        if q_split is None or q_split is not k_split or q_split is not v_split:
            return None
        split_sizes = _split_sizes(q_split)
        expected_heads = _SUPPORTED_SPLITS.get(split_sizes or ())
        if expected_heads is None:
            return None
        q_heads, kv_heads = expected_heads
        if (
            _shape_tail(q_head) != (q_heads, 128)
            or _shape_tail(k_head) != (kv_heads, 128)
            or _shape_tail(value) != (kv_heads, 128)
            or _shape(q_weight) != (128,)
            or _shape(k_weight) != (128,)
            or any(
                _dtype(node) != torch.bfloat16
                for node in (q_head, k_head, value, q_weight, k_weight)
            )
        ):
            return None

        rope_candidates = {
            user
            for user in q_flat.users
            if _is_call(user, "vllm::musa_rotary_embedding")
        } & {
            user
            for user in k_flat.users
            if _is_call(user, "vllm::musa_rotary_embedding")
        }
        if len(rope_candidates) != 1:
            return None
        rope = rope_candidates.pop()
        positions = _arg(rope, 0, "positions")
        cos_sin_cache = _arg(rope, 4, "cos_sin_cache")
        if (
            rope in seen_rope
            or rope.users
            or not isinstance(positions, fx.Node)
            or not isinstance(cos_sin_cache, fx.Node)
            or _arg(rope, 1, "query") is not q_flat
            or _arg(rope, 2, "key") is not k_flat
            or _arg(rope, 3, "head_size") != 128
            or _arg(rope, 5, "is_neox") is not True
            or _dtype(cos_sin_cache) != torch.bfloat16
            or not _only_users(q_flat, {query, rope})
            or not _only_users(k_flat, {key, rope})
            or not _only_users(q_norm, {q_flat})
            or not _only_users(k_norm, {k_flat})
            or not _only_users(q_head, {q_norm})
            or not _only_users(k_head, {k_norm})
            or not _only_users(query, {attention})
            or not _only_users(key, {kv_update, attention})
            or not _only_users(value, {kv_update, attention})
        ):
            return None

        candidates.append(
            Qwen3QKRoPEKVCandidate(
                rope=rope,
                kv_update=kv_update,
                attention=attention,
                q_head=q_head,
                k_head=k_head,
                value=value,
                q_weight=q_weight,
                k_weight=k_weight,
                positions=positions,
                cos_sin_cache=cos_sin_cache,
                layer_name=layer_name,
            )
        )
        seen_rope.add(rope)
        seen_layer_names.add(layer_name)

    if len(candidates) != expected_sites or seen_rope != set(all_rope):
        return None
    return tuple(candidates)


def _replace_arg(node: fx.Node, index: int, name: str, value: Any) -> None:
    if name in node.kwargs:
        kwargs = dict(node.kwargs)
        kwargs[name] = value
        node.kwargs = kwargs
        return
    args = list(node.args)
    args[index] = value
    node.args = tuple(args)


def apply_qwen3_qk_rope_kv_presplit(
    graph_module: fx.GraphModule,
    candidates: tuple[Qwen3QKRoPEKVCandidate, ...],
) -> int:
    from vllm_musa.kernels import qwen3_qk_rope_kv  # noqa: F401

    fused_op = torch.ops.vllm.musa_qwen3_qk_rope_and_unified_kv_cache_update.default
    empty_like = torch.ops.aten.empty_like.default
    graph = graph_module.graph
    for candidate in candidates:
        old_query = _arg(candidate.attention, 0, "query")
        old_key = _arg(candidate.attention, 1, "key")
        old_q_flat = _view_base(old_query) if isinstance(old_query, fx.Node) else None
        old_k_flat = _view_base(old_key) if isinstance(old_key, fx.Node) else None
        old_q_norm = _view_base(old_q_flat) if isinstance(old_q_flat, fx.Node) else None
        old_k_norm = _view_base(old_k_flat) if isinstance(old_k_flat, fx.Node) else None
        with graph.inserting_before(candidate.kv_update):
            q_out = graph.call_function(empty_like, args=(candidate.q_head,))
            k_out = graph.call_function(empty_like, args=(candidate.k_head,))
            fused = graph.call_function(
                fused_op,
                kwargs={
                    "q": candidate.q_head,
                    "k": candidate.k_head,
                    "v": candidate.value,
                    "q_weight": candidate.q_weight,
                    "k_weight": candidate.k_weight,
                    "positions": candidate.positions,
                    "cos_sin_cache": candidate.cos_sin_cache,
                    "q_out": q_out,
                    "k_out": k_out,
                    # The pre-split graph is the dense Qwen3 path.  These
                    # values are part of the planner's match contract (see
                    # the RoPE and RMSNorm checks above), but they are not
                    # represented by the old graph nodes.  Pass them
                    # explicitly because the fused op also serves Qwen3.5
                    # and therefore has no schema defaults.
                    "is_neox": True,
                    "mrope_section_t": 64,
                    "mrope_section_h": 0,
                    "mrope_section_w": 0,
                    "is_interleaved": False,
                    "eps": 1e-6,
                    "gemma": False,
                    "layer_name": candidate.layer_name,
                },
            )
        q_out.meta = dict(candidate.q_head.meta)
        k_out.meta = dict(candidate.k_head.meta)
        fused.meta = dict(candidate.kv_update.meta)
        _replace_arg(candidate.attention, 0, "query", q_out)
        _replace_arg(candidate.attention, 1, "key", k_out)
        candidate.kv_update.replace_all_uses_with(fused)
        graph.erase_node(candidate.kv_update)
        if candidate.rope.users:
            raise RuntimeError("MUSA Qwen3 RoPE node gained an unexpected user")
        graph.erase_node(candidate.rope)

        seen: set[fx.Node] = set()
        for dead_node in (
            old_query,
            old_key,
            old_q_flat,
            old_k_flat,
            old_q_norm,
            old_k_norm,
        ):
            if (
                isinstance(dead_node, fx.Node)
                and dead_node not in seen
                and not dead_node.users
            ):
                graph.erase_node(dead_node)
                seen.add(dead_node)

    graph.lint()
    graph_module.recompile()
    return len(candidates)


__all__ = [
    "FUSED_SPLITTING_OP",
    "KV_UPDATE_SPLITTING_OP",
    "Qwen3QKRoPEKVCandidate",
    "apply_qwen3_qk_rope_kv_presplit",
    "plan_qwen3_qk_rope_kv_presplit",
    "qwen3_qk_rope_kv_backend_supported",
]
