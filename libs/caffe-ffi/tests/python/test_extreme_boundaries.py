"""P2-B: Extreme boundary condition tests.

Tests cover:
- TestExtremeBoundaries: large input dimensions, NaN/Inf/zero inputs, extreme weights,
  deep networks, repeated reshape, lifecycle stress, high-concurrency-like serial loops
"""
from __future__ import annotations

import gc
import numpy as np
import pytest

import caffe_ffi
from caffe_ffi import (
    Net, Blob,
    net_param_from_string, net_from_param,
    total_allocated_bytes, live_blob_count,
)
from .conftest import require_cpp_extension, perf_trace


# ─── Prototxt builders ─────────────────────────────────────────────

def _make_mlp_prototxt(batch: int, input_dim: int, hidden_dim: int, n_hidden: int) -> str:
    """Deep MLP: Input -> (IP->ReLU) * n_hidden -> IP -> Softmax."""
    lines = [
        'name: "extreme_mlp"',
        'layer {',
        '  name: "data"', '  type: "Input"', '  top: "data"',
        f'  input_param {{ shape {{ dim: {batch} dim: {input_dim} }} }}',
        '}',
    ]
    prev = "data"
    for i in range(n_hidden):
        lines.extend([
            'layer {',
            f'  name: "ip{i}"', '  type: "InnerProduct"',
            f'  bottom: "{prev}"', f'  top: "ip{i}"',
            f'  inner_product_param {{ num_output: {hidden_dim} bias_term: true }}',
            '}',
            'layer {',
            f'  name: "relu{i}"', '  type: "ReLU"',
            f'  bottom: "ip{i}"', f'  top: "ip{i}"',
            '}',
        ])
        prev = f"ip{i}"
    lines.extend([
        'layer {',
        '  name: "ip_out"', '  type: "InnerProduct"',
        f'  bottom: "{prev}"', '  top: "ip_out"',
        '  inner_product_param { num_output: 10 bias_term: true }',
        '}',
        'layer {',
        '  name: "prob"', '  type: "Softmax"',
        '  bottom: "ip_out"', '  top: "prob"',
        '}',
    ])
    return "\n".join(lines)


def _make_split_mlp_prototxt(batch: int, input_dim: int, n_branches: int) -> str:
    """Split-based MLP: Input -> Split -> N branches (each IP) -> Concat -> IP -> Softmax."""
    lines = [
        'name: "split_extreme"',
        'layer {',
        '  name: "data"', '  type: "Input"', '  top: "data"',
        f'  input_param {{ shape {{ dim: {batch} dim: {input_dim} }} }}',
        '}',
        'layer {',
        '  name: "split"', '  type: "Split"', '  bottom: "data"',
    ]
    for i in range(n_branches):
        lines.append(f'  top: "b{i}"')
    lines.append('}')
    for i in range(n_branches):
        lines.extend([
            'layer {',
            f'  name: "ip_b{i}"', '  type: "InnerProduct"',
            f'  bottom: "b{i}"', f'  top: "b{i}_out"',
            f'  inner_product_param {{ num_output: {input_dim} bias_term: false }}',
            '}',
        ])
    lines.append('layer {')
    lines.append('  name: "concat"')
    lines.append('  type: "Concat"')
    for i in range(n_branches):
        lines.append(f'  bottom: "b{i}_out"')
    lines.append('  top: "concat"')
    lines.append('  concat_param { axis: 1 }')
    lines.append('}')
    lines.extend([
        'layer {',
        '  name: "ip_out"', '  type: "InnerProduct"',
        '  bottom: "concat"', '  top: "ip_out"',
        '  inner_product_param { num_output: 10 bias_term: true }',
        '}',
        'layer {',
        '  name: "prob"', '  type: "Softmax"',
        '  bottom: "ip_out"', '  top: "prob"',
        '}',
    ])
    return "\n".join(lines)


def _set_random_weights_all(net: Net, seed: int = 42, scale: float = 0.01) -> None:
    """Set random weights for all InnerProduct layers in net."""
    rng = np.random.RandomState(seed)
    for layer in net.layers_array():
        if layer.type == "InnerProduct" and len(layer.blobs) >= 1:
            W = layer.blobs[0]
            w_data = rng.randn(*W.shape).astype(np.float32) * scale
            W.from_numpy(w_data)
            if len(layer.blobs) >= 2:
                b = layer.blobs[1]
                b.from_numpy(np.zeros(b.shape, dtype=np.float32))


def _set_constant_weights(net: Net, value: float) -> None:
    """Set all InnerProduct weights to a constant value, bias to zero."""
    for layer in net.layers_array():
        if layer.type == "InnerProduct" and len(layer.blobs) >= 1:
            w = np.full(layer.blobs[0].shape, value, dtype=np.float32)
            layer.blobs[0].from_numpy(w)
            if len(layer.blobs) >= 2:
                b = np.zeros(layer.blobs[1].shape, dtype=np.float32)
                layer.blobs[1].from_numpy(b)


# ─── Test Class ────────────────────────────────────────────────────

@require_cpp_extension
class TestExtremeBoundaries:
    """Extreme boundary condition tests for memory stability and correctness."""

    # ── Large input dimensions ─────────────────────────────────────

    @pytest.mark.leak_check(False)
    def test_large_input_2048(self, ptrace):
        """Large batch+feature: batch=64, feat=2048 forward succeeds, no crash."""
        batch, feat = 64, 2048
        proto = _make_mlp_prototxt(batch=batch, input_dim=feat, hidden_dim=512, n_hidden=2)
        with ptrace(f"Net(large_mlp_b{batch}_d{feat})"):
            param = net_param_from_string(proto)
            net = net_from_param(param)
        _set_random_weights_all(net, seed=1, scale=0.001)

        rng = np.random.RandomState(0)
        inp = rng.randn(batch, feat).astype(np.float32) * 0.01
        with ptrace(f"Forward(large_mlp_b{batch}_d{feat})") as t:
            net.Forward({"data": inp})
            t['batch'] = batch
            t['feat_dim'] = feat
            t['nbytes_input'] = batch * feat * 4

        prob = net.blob_by_name("prob").to_numpy()
        assert prob.shape == (batch, 10)
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.leak_check(False)
    def test_split_large_input_1024(self, ptrace):
        """Split with large feat_dim=1024 batch=32, 2 branches."""
        batch, feat, n_branches = 32, 1024, 2
        proto = _make_split_mlp_prototxt(batch=batch, input_dim=feat, n_branches=n_branches)
        with ptrace(f"Net(split_large_b{batch}_d{feat}_n{n_branches})"):
            param = net_param_from_string(proto)
            net = net_from_param(param)
        _set_random_weights_all(net, seed=2, scale=0.001)

        rng = np.random.RandomState(42)
        inp = rng.randn(batch, feat).astype(np.float32) * 0.01
        with ptrace(f"Forward(split_large_b{batch}_d{feat})") as t:
            net.Forward({"data": inp})
            t['total_copied_bytes'] = batch * feat * 4 * n_branches

        prob = net.blob_by_name("prob").to_numpy()
        assert prob.shape == (batch, 10)
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-5)

    # ── NaN / Inf inputs ───────────────────────────────────────────

    def test_nan_input_no_crash(self, ptrace):
        """NaN inputs should not cause segfault."""
        batch, feat = 8, 16
        proto = _make_mlp_prototxt(batch=batch, input_dim=feat, hidden_dim=16, n_hidden=1)
        param = net_param_from_string(proto)
        net = net_from_param(param)
        _set_random_weights_all(net, seed=3)

        inp = np.full((batch, feat), np.nan, dtype=np.float32)
        caffe_ffi.set_log_level(caffe_ffi.LOG_LEVEL_ERROR)
        try:
            with ptrace("Forward(nan_input)"):
                net.Forward({"data": inp})
            prob = net.blob_by_name("prob").to_numpy()
            assert np.any(np.isnan(prob)) or np.all(np.isfinite(prob))
        finally:
            caffe_ffi.set_log_level(caffe_ffi.LOG_LEVEL_WARN)

    def test_inf_input_no_crash(self, ptrace):
        """Inf inputs should not cause segfault."""
        batch, feat = 8, 16
        proto = _make_mlp_prototxt(batch=batch, input_dim=feat, hidden_dim=16, n_hidden=1)
        param = net_param_from_string(proto)
        net = net_from_param(param)
        _set_random_weights_all(net, seed=4)

        inp = np.full((batch, feat), np.inf, dtype=np.float32)
        caffe_ffi.set_log_level(caffe_ffi.LOG_LEVEL_ERROR)
        try:
            with ptrace("Forward(inf_input)"):
                net.Forward({"data": inp})
        finally:
            caffe_ffi.set_log_level(caffe_ffi.LOG_LEVEL_WARN)

    # ── Zero inputs ────────────────────────────────────────────────

    def test_zero_input_deterministic(self, ptrace):
        """All-zero input should produce deterministic output."""
        batch, feat = 4, 8
        proto = _make_mlp_prototxt(batch=batch, input_dim=feat, hidden_dim=8, n_hidden=1)
        param = net_param_from_string(proto)
        net = net_from_param(param)
        _set_random_weights_all(net, seed=5)

        zero = np.zeros((batch, feat), dtype=np.float32)
        with ptrace("Forward(zero_input_1)"):
            net.Forward({"data": zero})
        p1 = net.blob_by_name("prob").to_numpy().copy()

        with ptrace("Forward(zero_input_2)"):
            net.Forward({"data": zero})
        p2 = net.blob_by_name("prob").to_numpy().copy()

        np.testing.assert_array_equal(p1, p2)
        assert np.all(np.isfinite(p1))

    # ── Extreme weight values ──────────────────────────────────────

    @pytest.mark.leak_check(False)
    def test_extreme_weights_large(self, ptrace):
        """Very large weights (1e6) shouldn't crash; may produce Inf but no segfault."""
        batch, feat = 2, 4
        proto = _make_mlp_prototxt(batch=batch, input_dim=feat, hidden_dim=4, n_hidden=1)
        param = net_param_from_string(proto)
        net = net_from_param(param)
        _set_constant_weights(net, 1e6)

        inp = np.ones((batch, feat), dtype=np.float32)
        caffe_ffi.set_log_level(caffe_ffi.LOG_LEVEL_ERROR)
        try:
            with ptrace("Forward(large_weights)"):
                net.Forward({"data": inp})
        finally:
            caffe_ffi.set_log_level(caffe_ffi.LOG_LEVEL_WARN)

    def test_extreme_weights_tiny(self, ptrace):
        """Very small weights (1e-6) produce finite output."""
        batch, feat = 4, 8
        proto = _make_mlp_prototxt(batch=batch, input_dim=feat, hidden_dim=8, n_hidden=1)
        param = net_param_from_string(proto)
        net = net_from_param(param)
        _set_constant_weights(net, 1e-6)

        inp = np.ones((batch, feat), dtype=np.float32)
        with ptrace("Forward(tiny_weights)"):
            net.Forward({"data": inp})
        prob = net.blob_by_name("prob").to_numpy()
        assert np.all(np.isfinite(prob)), "Tiny weights should produce finite output"

    # ── Deep network ───────────────────────────────────────────────

    def test_deep_network_20_layers(self, ptrace):
        """20-layer MLP forward succeeds without memory issues."""
        batch, feat, hidden, n_hidden = 4, 16, 16, 18
        proto = _make_mlp_prototxt(batch=batch, input_dim=feat, hidden_dim=hidden, n_hidden=n_hidden)
        with ptrace(f"Net(deep{n_hidden+2})"):
            param = net_param_from_string(proto)
            net = net_from_param(param)
        _set_random_weights_all(net, seed=6, scale=0.1)

        inp = np.random.RandomState(7).randn(batch, feat).astype(np.float32) * 0.1
        with ptrace(f"Forward(deep{n_hidden+2})") as t:
            net.Forward({"data": inp})
            t['n_layers'] = len(net.layers_array())

        prob = net.blob_by_name("prob").to_numpy()
        assert prob.shape == (batch, 10)
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-5)

    # ── Lifecycle stress: repeated create/forward/destroy ──────────

    def test_lifecycle_stress_50_creates(self, ptrace):
        """Repeatedly create, forward, and destroy networks; no memory leak."""
        batch, feat = 4, 8
        proto = _make_mlp_prototxt(batch=batch, input_dim=feat, hidden_dim=8, n_hidden=1)
        param = net_param_from_string(proto)
        inp = np.random.RandomState(8).randn(batch, feat).astype(np.float32) * 0.1

        mem_before = total_allocated_bytes()
        for i in range(50):
            with ptrace(f"Lifecycle(iter_{i})", verbose=(i < 3)):
                net = net_from_param(param)
                _set_random_weights_all(net, seed=i, scale=0.1)
                net.Forward({"data": inp})
                del net
                gc.collect()

        gc.collect()
        mem_after = total_allocated_bytes()
        leak = mem_after - mem_before
        assert leak < 1024 * 1024, f"Lifecycle stress leaked {leak} bytes"

    # ── Repeated forward memory stability ──────────────────────────

    def test_repeated_forward_100_times(self, ptrace):
        """100 forwards on same net; blobs should be reused, no growth."""
        batch, feat = 8, 32
        proto = _make_mlp_prototxt(batch=batch, input_dim=feat, hidden_dim=32, n_hidden=2)
        param = net_param_from_string(proto)
        net = net_from_param(param)
        _set_random_weights_all(net, seed=9, scale=0.05)

        inp = np.random.RandomState(10).randn(batch, feat).astype(np.float32) * 0.1

        blobs_before = live_blob_count()
        mem_before = total_allocated_bytes()
        for i in range(100):
            with ptrace(f"RepeatedFwd({i})", verbose=(i % 20 == 0)):
                net.Forward({"data": inp})

        gc.collect()
        blobs_after = live_blob_count()
        mem_after = total_allocated_bytes()
        assert blobs_after == blobs_before, \
            f"Blob count grew: {blobs_before} -> {blobs_after}"
        assert mem_after <= mem_before + 4096, \
            f"Memory grew by {mem_after - mem_before} bytes over 100 forwards"

    # ── Minimal input ──────────────────────────────────────────────

    def test_minimal_1x1(self, ptrace):
        """Minimal 1x1 scalar-like network."""
        proto = _make_mlp_prototxt(batch=1, input_dim=1, hidden_dim=1, n_hidden=1)
        param = net_param_from_string(proto)
        net = net_from_param(param)
        _set_random_weights_all(net, seed=11, scale=0.5)

        inp = np.array([[0.5]], dtype=np.float32)
        with ptrace("Forward(1x1)"):
            net.Forward({"data": inp})
        prob = net.blob_by_name("prob").to_numpy()
        assert prob.shape == (1, 10)
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-6)
