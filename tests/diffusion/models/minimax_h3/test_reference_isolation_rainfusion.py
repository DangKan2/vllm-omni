# SPDX-License-Identifier: Apache-2.0
"""Hard reference isolation plus the existing rectangular RainFusion ABI."""

from __future__ import annotations

import math
import os
import sys
import types
from bisect import bisect_left
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from tests.diffusion.models.minimax_h3.test_reference_isolation import (
    MODEL,
    ROOT,
    Metadata,
    dense,
    layer_fixture,
    load_function,
    make_state,
)


@dataclass(frozen=True)
class Span:
    start: int
    latent_grid: tuple[int, int, int]
    role: str


@dataclass(frozen=True)
class Layout:
    used_len: int
    video_spans: tuple = ()
    prefix_len: int | None = None
    latent_grid: tuple | None = None


def prepared_state():
    state = make_state((1, 2, 5, 6), used=12, total=16)
    layout = Layout(
        12, (Span(1, (1, 1, 2), "reference"), Span(5, (1, 1, 2), "reference"), Span(8, (1, 2, 2), "target"))
    )
    namespace = {"math": math, "bisect_left": bisect_left, "VideoTokenSpan": Span, "VideoTokenLayout": Layout}
    compact = load_function(MODEL / "minimax_h3_transformer.py", "_compact_reference_video_layout", namespace)
    kv = load_function(MODEL / "minimax_h3_transformer.py", "_reference_kv_video_layout", namespace)
    q_layout = compact(layout, state.reference_positions, 8)
    k_layout = kv(layout, state.reference_positions, q_layout)
    state.set_video_geometry(q_layout, k_layout)
    return state


@pytest.mark.parametrize("compact", [False, True])
def test_sparse_dispatch_contains_only_dynamic_q_and_correctly_permuted_kv(compact):
    state = prepared_state()
    torch.manual_seed(61)
    q, k, v = [torch.randn(1, 16, 2, 4, dtype=torch.float64) for _ in range(3)]
    seen = []

    def sparse_dynamic(query, key, value, metadata):
        assert query.shape[1] == 8 and key.shape[1] == 12
        torch.testing.assert_close(query, q[:, state.dynamic_positions], rtol=0, atol=0)
        torch.testing.assert_close(key, k[:, state.key_order], rtol=0, atol=0)
        assert metadata.extra["minimax_h3_compact_reference_rows"] == 4
        assert metadata.video_layout.video_spans == (Span(4, (1, 2, 2), "target"),)
        assert metadata.extra["minimax_h3_reference_kv_video_layout"].video_spans == (
            Span(0, (1, 1, 2), "reference"),
            Span(2, (1, 1, 2), "reference"),
            Span(8, (1, 2, 2), "target"),
        )
        seen.append(True)
        return dense(query, key, value, metadata)

    oracle = state.attend(q, k, v, Metadata(), compact=False, dense_forward=dense)
    if compact:
        actual = state.attend(
            q[:, state.dynamic_positions],
            k[:, state.key_order],
            v[:, state.key_order],
            Metadata(extra={"minimax_h3_compact_reference_rows": 4}),
            compact=True,
            dense_forward=dense,
            dynamic_forward=sparse_dynamic,
        )
        expected = oracle[:, state.dynamic_positions]
    else:
        actual = state.attend(q, k, v, Metadata(), compact=False, dense_forward=dense, dynamic_forward=sparse_dynamic)
        expected = oracle
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert seen == [True] and state.dynamic_backend_calls == 1


def test_actual_shared_layer_sends_reference_to_dense_and_dynamic_to_rainfusion():
    state = prepared_state()
    q = torch.randn(1, 16, 2, 4)
    layer, strategy, calls, forward = layer_fixture("RAINFUSION_ATTN")

    def rf(query, key, value, metadata):
        calls.append("rainfusion_dynamic")
        assert query.shape[1] == 8 and key.shape[1] == 12
        return dense(query, key, value, metadata)

    layer.attention.forward = rf
    forward(layer, q, q, q, Metadata(extra={"minimax_h3_reference_isolation": (state, False)}))
    assert calls == ["pre", "dense", "rainfusion_dynamic", "post"]


def test_existing_rainfusion_dispatch_keeps_independent_reference_video_geometry():
    state = prepared_state()
    q = torch.randn(1, 8, 2, 4)
    kv = torch.randn(1, 12, 2, 4)
    metadata = state._dense_metadata(Metadata(), q, 12)
    metadata.video_layout = state.dynamic_video_layout
    metadata.extra.update(
        minimax_h3_compact_reference_rows=4, minimax_h3_reference_kv_video_layout=state.kv_video_layout
    )
    seen = []

    def resolve(meta):
        assert meta.video_layout is state.kv_video_layout
        assert meta.extra["max_seqlen_q"] == 12
        return object()

    backend = SimpleNamespace(
        qkv_layout="BSND",
        _resolve_plan=resolve,
        _can_use_hybrid=lambda *a: True,
        _forward_hybrid_npu=lambda q, k, v, meta, plan, refs, **kw: seen.append((refs, kw["q_spans"])) or q,
    )
    path = ROOT / "vllm_omni/diffusion/attention/backends/rainfusion_attn.py"
    dispatch = load_function(
        path,
        "forward_npu",
        {
            "_reset_mask_diagnostics": lambda: None,
            "REFERENCE_KV_ENABLED": "minimax_h3_reference_kv_enabled",
            "REFERENCE_KV_VIDEO_LAYOUT": "minimax_h3_reference_kv_video_layout",
            "_INPUT_LAYOUT": "BSND",
            "replace": __import__("dataclasses").replace,
        },
    )
    out = dispatch(backend, q, kv, kv, metadata)
    assert out is q and seen == [(4, ((4, (1, 2, 2)),))]


@pytest.mark.parametrize("sparse_rows,raises,expected", [(8, False, 1), (0, False, 0), (8, True, 0)])
def test_sparse_counter_records_successful_sparse_kernel_path_only(monkeypatch, sparse_rows, raises, expected):
    state = prepared_state()
    plan = SimpleNamespace(
        sparse_query_rows=sparse_rows,
        dense_query_rows=8 - sparse_rows,
        key_rows=12,
        protected_keys=torch.tensor([0]),
        key_blocks=1,
    )

    def kernel(q, k, v, *a, **kw):
        if raises:
            raise RuntimeError("injected sparse operator failure")
        return q

    fake = types.ModuleType("vllm_omni.diffusion.attention.ops.rainfusion_hybrid")
    fake.get_hybrid_plan = lambda *a, **kw: plan
    fake.hybrid_attention = kernel
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setenv("VLLM_OMNI_MINIMAX_H3_REF_KV_DIAGNOSTICS", "0")
    backend = SimpleNamespace(
        rainfusion=SimpleNamespace(sparsity=0.8),
        softmax_scale=0.5,
        hybrid_kernel_dtype="fp16",
        hybrid_kernel="bsa",
        layer_idx=0,
    )
    metadata = Metadata(extra={"max_seqlen_q": 8, "max_seqlen_k": 12, "minimax_h3_isolation_metrics": state})
    sparse_plan = SimpleNamespace(
        video_spans=[{"start": 0, "latent_shape": (1, 1, 4)}, {"start": 8, "latent_shape": (1, 2, 2)}]
    )
    fn = load_function(
        ROOT / "vllm_omni/diffusion/attention/backends/rainfusion_attn.py",
        "_forward_hybrid_npu",
        {
            "math": math,
            "os": os,
            "_DIAGNOSTICS_ENV": "VLLM_OMNI_MINIMAX_H3_REF_KV_DIAGNOSTICS",
            "_INNER_PRECISE": 0,
        },
    )
    q, kv = torch.randn(1, 8, 2, 4), torch.randn(1, 12, 2, 4)
    if raises:
        with pytest.raises(RuntimeError, match="injected"):
            fn(backend, q, kv, kv, metadata, sparse_plan, 4, q_spans=((4, (1, 2, 2)),))
    else:
        fn(backend, q, kv, kv, metadata, sparse_plan, 4, q_spans=((4, (1, 2, 2)),))
    assert state.sparse_layer_calls == expected


def test_missing_geometry_stays_dense_and_geometry_cannot_change_mid_request():
    state = make_state()
    q = torch.randn(1, 9, 2, 4)
    state.attend(
        q,
        q,
        q,
        Metadata(),
        compact=False,
        dense_forward=dense,
        dynamic_forward=lambda *a: pytest.fail("invalid geometry must not enter RF"),
    )
    state.set_video_geometry(None, None)
    with pytest.raises(RuntimeError, match="immutable"):
        state.set_video_geometry(None, None)
    assert state.sparse_layer_calls == 0 and state.dynamic_backend_calls == 0
