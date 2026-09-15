# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gc
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torchada")
torch = pytest.importorskip("torch")
from torch import fx  # noqa: E402

from vllm_musa.compilation.qwen3_qk_rope_kv_presplit import (  # noqa: E402
    apply_qwen3_qk_rope_kv_presplit,
    plan_qwen3_qk_rope_kv_presplit,
)

_FUSED_OP_NAME = "musa_qwen3_qk_rope_and_unified_kv_cache_update"
_FUSED_OP_SCHEMA = (
    "musa_qwen3_qk_rope_and_unified_kv_cache_update(Tensor q, Tensor k, "
    "Tensor v, Tensor q_weight, Tensor k_weight, Tensor positions, "
    "Tensor cos_sin_cache, Tensor(a!) q_out, Tensor(b!) k_out, "
    "bool is_neox, int mrope_section_t, int mrope_section_h, "
    "int mrope_section_w, bool is_interleaved, float eps, bool gemma, "
    "str layer_name) -> Tensor"
)
_GRAPH_OP_SCHEMAS = (
    (
        "musa_rotary_embedding",
        "musa_rotary_embedding(Tensor positions, Tensor(a!) query, Tensor(b!) key, "
        "int head_size, Tensor cos_sin_cache, bool is_neox) -> ()",
    ),
    (
        "unified_kv_cache_update",
        "unified_kv_cache_update(Tensor key, Tensor value, str layer_name) -> Tensor",
    ),
    (
        "unified_attention_with_output",
        "unified_attention_with_output(Tensor query, Tensor key, Tensor value, "
        "Tensor(a!) output, str layer_name, float scale, float output_scale, "
        "Tensor? kv_cache_dummy_dep) -> Tensor",
    ),
)


def _dispatch_has_schema(name: str) -> bool:
    try:
        torch._C._dispatch_find_schema_or_throw(f"vllm::{name}", "")
    except RuntimeError:
        return False
    return True


@contextmanager
def _temporary_schema(name: str, schema: str):
    library = None
    if not _dispatch_has_schema(name):
        library = torch.library.Library("vllm", "FRAGMENT")
        library.define(schema)
    try:
        yield
    finally:
        if library is not None:
            library._destroy()


@pytest.fixture(autouse=True)
def _graph_op_schemas():
    initial = {name: _dispatch_has_schema(name) for name, _ in _GRAPH_OP_SCHEMAS}
    libraries = []
    for name, schema in _GRAPH_OP_SCHEMAS:
        if not initial[name]:
            library = torch.library.Library("vllm", "FRAGMENT")
            library.define(schema)
            libraries.append(library)
    yield
    for library in reversed(libraries):
        library._destroy()
    gc.collect()
    assert {
        name: _dispatch_has_schema(name) for name, _ in _GRAPH_OP_SCHEMAS
    } == initial


def _meta(node: fx.Node, shape: tuple[int, ...], dtype=torch.bfloat16) -> fx.Node:
    node.meta["example_value"] = SimpleNamespace(shape=shape, dtype=dtype)
    return node


def _make_graph(
    *,
    q_heads: int = 16,
    extra_q_user: bool = False,
    hoisted_layer_name: bool = False,
):
    kv_heads = 8
    q_width = q_heads * 128
    kv_width = kv_heads * 128
    graph = fx.Graph()
    qkv = _meta(graph.placeholder("qkv"), (1, q_width + 2 * kv_width))
    q_weight = _meta(graph.placeholder("q_weight"), (128,))
    k_weight = _meta(graph.placeholder("k_weight"), (128,))
    positions = _meta(graph.placeholder("positions"), (1,), torch.int64)
    cos = _meta(graph.placeholder("cos"), (32, 128))
    output = _meta(graph.placeholder("output"), (1, q_heads, 128))
    layer_name: str | fx.Node = "model.layers.0.self_attn"
    if hoisted_layer_name:
        layer_name = graph.placeholder("layer_name")
    split = graph.call_method(
        "split",
        args=(qkv, [q_width, kv_width, kv_width]),
        kwargs={"dim": -1},
    )
    q_item = graph.call_function(__import__("operator").getitem, args=(split, 0))
    k_item = graph.call_function(__import__("operator").getitem, args=(split, 1))
    v_item = graph.call_function(__import__("operator").getitem, args=(split, 2))
    q_head = _meta(
        graph.call_method("view", args=(q_item, 1, q_heads, 128)),
        (1, q_heads, 128),
    )
    k_head = _meta(
        graph.call_method("view", args=(k_item, 1, kv_heads, 128)),
        (1, kv_heads, 128),
    )
    value = _meta(
        graph.call_method("view", args=(v_item, 1, kv_heads, 128)),
        (1, kv_heads, 128),
    )
    q_norm = _meta(
        graph.call_function(torch.rms_norm, args=(q_head, [128], q_weight, 1e-6)),
        (1, q_heads, 128),
    )
    k_norm = _meta(
        graph.call_function(torch.rms_norm, args=(k_head, [128], k_weight, 1e-6)),
        (1, kv_heads, 128),
    )
    q_flat = _meta(
        graph.call_method("view", args=(q_norm, 1, q_width)),
        (1, q_width),
    )
    k_flat = _meta(
        graph.call_method("view", args=(k_norm, 1, kv_width)),
        (1, kv_width),
    )
    graph.call_function(
        torch.ops.vllm.musa_rotary_embedding,
        args=(positions, q_flat, k_flat, 128, cos, True),
    )
    query = _meta(
        graph.call_method("view", args=(q_flat, 1, q_heads, 128)),
        (1, q_heads, 128),
    )
    key = _meta(
        graph.call_method("view", args=(k_flat, 1, kv_heads, 128)),
        (1, kv_heads, 128),
    )
    if extra_q_user:
        graph.call_function(torch.ops.aten.clone.default, args=(q_flat,))
    kv_update = graph.call_function(
        torch.ops.vllm.unified_kv_cache_update,
        kwargs={"key": key, "value": value, "layer_name": layer_name},
    )
    attention = graph.call_function(
        torch.ops.vllm.unified_attention_with_output,
        kwargs={
            "query": query,
            "key": key,
            "value": value,
            "output": output,
            "layer_name": layer_name,
            "scale": 0.08838834764831845,
            "output_scale": 1.0,
            "kv_cache_dummy_dep": kv_update,
        },
    )
    graph.output(attention)
    return fx.GraphModule({}, graph)


def test_plan_matches_qwen3_06b_and_8b_shapes():
    for q_heads in (16, 32):
        graph_module = _make_graph(q_heads=q_heads)
        candidates = plan_qwen3_qk_rope_kv_presplit(graph_module, 1)
        assert candidates is not None
        assert len(candidates) == 1


def test_plan_matches_torch211_hoisted_layer_name():
    graph_module = _make_graph(hoisted_layer_name=True)
    candidates = plan_qwen3_qk_rope_kv_presplit(graph_module, 1)
    assert candidates is not None
    assert len(candidates) == 1
    assert isinstance(candidates[0].layer_name, fx.Node)


def test_plan_is_atomic_on_extra_q_user():
    graph_module = _make_graph(extra_q_user=True)
    before = str(graph_module.graph)
    assert plan_qwen3_qk_rope_kv_presplit(graph_module, 1) is None
    assert str(graph_module.graph) == before


def test_temporary_schema_is_order_isolated():
    unique_name = "musa_qwen3_qk_rope_presplit_test_isolation"
    cases = (
        (unique_name, f"{unique_name}() -> ()"),
        (_FUSED_OP_NAME, _FUSED_OP_SCHEMA),
    )
    for name, schema in cases:
        registered_before = _dispatch_has_schema(name)
        with _temporary_schema(name, schema):
            assert _dispatch_has_schema(name)
        assert _dispatch_has_schema(name) is registered_before


@pytest.mark.parametrize("hoisted_layer_name", [False, True])
def test_apply_replaces_norm_rope_and_cache(monkeypatch, hoisted_layer_name):
    module_name = "vllm_musa.kernels.qwen3_qk_rope_kv"
    if module_name not in sys.modules:
        kernels_package = sys.modules.get("vllm_musa.kernels")
        if kernels_package is None:
            kernels_package = types.ModuleType("vllm_musa.kernels")
            kernels_package.__path__ = [
                str(Path(__file__).parents[1] / "vllm_musa" / "kernels")
            ]
            monkeypatch.setitem(sys.modules, "vllm_musa.kernels", kernels_package)
        fake_module = types.ModuleType(module_name)
        monkeypatch.setitem(sys.modules, module_name, fake_module)
        monkeypatch.setattr(
            kernels_package,
            "qwen3_qk_rope_kv",
            fake_module,
            raising=False,
        )

    with _temporary_schema(_FUSED_OP_NAME, _FUSED_OP_SCHEMA):
        graph_module = _make_graph(hoisted_layer_name=hoisted_layer_name)
        layer_name = (
            next(node for node in graph_module.graph.nodes if node.name == "layer_name")
            if hoisted_layer_name
            else None
        )
        candidates = plan_qwen3_qk_rope_kv_presplit(graph_module, 1)
        assert candidates is not None
        assert apply_qwen3_qk_rope_kv_presplit(graph_module, candidates) == 1
        targets = [str(node.target) for node in graph_module.graph.nodes]
        assert (
            sum(
                "musa_qwen3_qk_rope_and_unified_kv_cache_update" in target
                for target in targets
            )
            == 1
        )
        assert not any("musa_rotary_embedding" in target for target in targets)
        assert "vllm.unified_kv_cache_update.default" not in targets
        assert not any("rms_norm" in target for target in targets)
        attention_nodes = [
            node
            for node in graph_module.graph.nodes
            if str(node.target) == "vllm.unified_attention_with_output"
        ]
        fused_nodes = [
            node
            for node in graph_module.graph.nodes
            if "musa_qwen3_qk_rope_and_unified_kv_cache_update" in str(node.target)
        ]
        assert len(attention_nodes) == 1
        assert len(fused_nodes) == 1
        assert fused_nodes[0] in attention_nodes[0].all_input_nodes
        assert fused_nodes[0].kwargs["is_neox"] is True
        assert fused_nodes[0].kwargs["mrope_section_t"] == 64
        assert fused_nodes[0].kwargs["mrope_section_h"] == 0
        assert fused_nodes[0].kwargs["mrope_section_w"] == 0
        assert fused_nodes[0].kwargs["is_interleaved"] is False
        assert fused_nodes[0].kwargs["eps"] == 1e-6
        assert fused_nodes[0].kwargs["gemma"] is False
        if layer_name is not None:
            assert fused_nodes[0].kwargs["layer_name"] is layer_name
