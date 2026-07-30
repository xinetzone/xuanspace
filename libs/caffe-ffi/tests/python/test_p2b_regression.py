"""P2-B Complete Regression Test Suite.

Unified regression script combining:
- TestSplitTopologies: Split layer correctness (basic copy, concat roundtrip,
  residual connections, in-place branch isolation, N=1 passthrough, determinism)
- TestSplitPerformanceScaling: memcpy performance benchmark across input sizes
- TestExtremeBoundaries: large inputs, NaN/Inf/zero inputs, extreme weights,
  deep networks, lifecycle stress, repeated forward, minimal inputs
- TestSplitMemoryStability: Split-specific memory stress (high fanout,
  repeated create/destroy with Split, multi-level Split chains)

Run with:
    pytest tests/python/test_p2b_regression.py -v
    pytest tests/python/test_p2b_regression.py -v -s  # verbose with [SPLIT-PERF] logs
    pytest tests/python/test_p2b_regression.py::TestSplitMemoryStability -v
"""
from __future__ import annotations

import gc
import threading
import numpy as np
import pytest

import caffe_ffi
from caffe_ffi import (
    Net,
    net_param_from_string, net_from_param,
    total_allocated_bytes, live_blob_count,
)
from .conftest import require_cpp_extension, perf_trace


# ═══════════════════════════════════════════════════════════════════════
# Prototxt builders
# ═══════════════════════════════════════════════════════════════════════

def _make_basic_split_prototxt(num_top: int = 2, feat_dim: int = 4) -> str:
    """Basic 1->N split: data -> Split{top1,...,topN}."""
    lines = [
        'name: "basic_split"',
        'layer {',
        '  name: "data"', '  type: "Input"', '  top: "data"',
        f'  input_param {{ shape {{ dim: 2 dim: {feat_dim} }} }}',
        '}',
        'layer {',
        '  name: "split"', '  type: "Split"', '  bottom: "data"',
    ]
    for i in range(num_top):
        lines.append(f'  top: "split_{i}"')
    lines.append('}')
    return "\n".join(lines)


def _make_split_concat_prototxt(num_branches: int = 3, feat_dim: int = 4) -> str:
    """Split into N branches then Concat back."""
    lines = [
        'name: "split_concat"',
        'layer {',
        '  name: "data"', '  type: "Input"', '  top: "data"',
        f'  input_param {{ shape {{ dim: 2 dim: {feat_dim} }} }}',
        '}',
        'layer {',
        '  name: "split"', '  type: "Split"', '  bottom: "data"',
    ]
    for i in range(num_branches):
        lines.append(f'  top: "branch_{i}"')
    lines.append('}')
    lines.append('layer {')
    lines.append('  name: "concat"')
    lines.append('  type: "Concat"')
    for i in range(num_branches):
        lines.append(f'  bottom: "branch_{i}"')
    lines.append('  top: "concat_out"')
    lines.append('  concat_param { axis: 1 }')
    lines.append('}')
    return "\n".join(lines)


def _make_residual_split_prototxt(feat_dim: int = 8) -> str:
    """Residual: data -> Split -> (identity, FC+ReLU) -> Eltwise SUM -> Softmax."""
    return f"""name: "residual_split"
layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape {{ dim: 2 dim: {feat_dim} }} }}
}}
layer {{
  name: "split"
  type: "Split"
  bottom: "data"
  top: "identity"
  top: "fc_path"
}}
layer {{
  name: "fc1"
  type: "InnerProduct"
  bottom: "fc_path"
  top: "fc1_out"
  inner_product_param {{ num_output: {feat_dim} bias_term: true }}
}}
layer {{
  name: "relu1"
  type: "ReLU"
  bottom: "fc1_out"
  top: "fc1_out"
}}
layer {{
  name: "add"
  type: "Eltwise"
  bottom: "identity"
  bottom: "fc1_out"
  top: "residual_out"
  eltwise_param {{ operation: SUM }}
}}
layer {{
  name: "prob"
  type: "Softmax"
  bottom: "residual_out"
  top: "prob"
}}
"""


def _make_split_inplace_branch_prototxt(feat_dim: int = 8) -> str:
    """Split + in-place ReLU on one branch; verifies sibling isolation."""
    return f"""name: "split_inplace"
layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape {{ dim: 2 dim: {feat_dim} }} }}
}}
layer {{
  name: "split"
  type: "Split"
  bottom: "data"
  top: "raw_branch"
  top: "relu_branch"
}}
layer {{
  name: "relu_on_branch"
  type: "ReLU"
  bottom: "relu_branch"
  top: "relu_branch"
}}
layer {{
  name: "raw_ip"
  type: "InnerProduct"
  bottom: "raw_branch"
  top: "raw_out"
  inner_product_param {{ num_output: {feat_dim} bias_term: false }}
}}
layer {{
  name: "relu_ip"
  type: "InnerProduct"
  bottom: "relu_branch"
  top: "relu_out"
  inner_product_param {{ num_output: {feat_dim} bias_term: false }}
}}
"""


def _make_n1_split_passthrough_prototxt(feat_dim: int = 4) -> str:
    """N=1 Split: identity passthrough."""
    return f"""name: "n1_split"
layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape {{ dim: 2 dim: {feat_dim} }} }}
}}
layer {{
  name: "split"
  type: "Split"
  bottom: "data"
  top: "passthrough"
}}
layer {{
  name: "ip"
  type: "InnerProduct"
  bottom: "passthrough"
  top: "out"
  inner_product_param {{ num_output: 2 bias_term: true }}
}}
layer {{
  name: "prob"
  type: "Softmax"
  bottom: "out"
  top: "prob"
}}
"""


def _make_split_mlp_perf_prototxt(batch: int, feat_dim: int, n_branches: int,
                                   branch_ip_dim: int | None = None) -> str:
    """Split + per-branch IP + Concat + IP + Softmax."""
    if branch_ip_dim is None:
        branch_ip_dim = feat_dim
    lines = [
        'name: "split_perf"',
        'layer {', '  name: "data"', '  type: "Input"', '  top: "data"',
        f'  input_param {{ shape {{ dim: {batch} dim: {feat_dim} }} }}', '}',
        'layer {', '  name: "split"', '  type: "Split"', '  bottom: "data"',
    ]
    for i in range(n_branches):
        lines.append(f'  top: "b{i}"')
    lines.append('}')
    for i in range(n_branches):
        lines.extend([
            'layer {',
            f'  name: "ip_b{i}"', '  type: "InnerProduct"',
            f'  bottom: "b{i}"', f'  top: "b{i}_out"',
            f'  inner_product_param {{ num_output: {branch_ip_dim} bias_term: false }}',
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
    concat_dim = branch_ip_dim * n_branches
    lines.extend([
        'layer {',
        '  name: "ip_out"', '  type: "InnerProduct"',
        '  bottom: "concat"', '  top: "ip_out"',
        f'  inner_product_param {{ num_output: 10 bias_term: true }}',
        '}',
        'layer {',
        '  name: "prob"', '  type: "Softmax"',
        '  bottom: "ip_out"', '  top: "prob"',
        '}',
    ])
    return "\n".join(lines)


def _make_mlp_prototxt(batch: int, input_dim: int, hidden_dim: int,
                        n_hidden: int) -> str:
    """Deep MLP: Input -> (IP->ReLU)*n -> IP -> Softmax."""
    lines = [
        'name: "extreme_mlp"',
        'layer {', '  name: "data"', '  type: "Input"', '  top: "data"',
        f'  input_param {{ shape {{ dim: {batch} dim: {input_dim} }} }}', '}',
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


def _make_multi_level_split_prototxt(batch: int, feat_dim: int, depth: int = 3) -> str:
    """Chain of Split layers: data -> Split -> (Split -> (Split -> ...))."""
    lines = [
        'name: "multi_split"',
        'layer {', '  name: "data"', '  type: "Input"', '  top: "data"',
        f'  input_param {{ shape {{ dim: {batch} dim: {feat_dim} }} }}', '}',
    ]
    prev_name = "data"
    for d in range(depth):
        split_name = f"split{d}"
        lines.extend([
            'layer {',
            f'  name: "{split_name}"', '  type: "Split"',
            f'  bottom: "{prev_name}"',
            f'  top: "{split_name}_a"',
            f'  top: "{split_name}_b"',
            '}',
        ])
        prev_name = f"{split_name}_a"
    # Concat all leaf branches
    lines.append('layer {')
    lines.append('  name: "concat"')
    lines.append('  type: "Concat"')
    for d in range(depth):
        lines.append(f'  bottom: "split{d}_b"')
    lines.append(f'  bottom: "{prev_name}"')
    lines.append('  top: "concat_out"')
    lines.append('  concat_param { axis: 1 }')
    lines.append('}')
    lines.extend([
        'layer {',
        '  name: "prob"', '  type: "Softmax"',
        '  bottom: "concat_out"', '  top: "prob"',
        '}',
    ])
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _set_random_weights(net: Net, rng: np.random.RandomState,
                        scale: float = 0.1) -> None:
    """Set small random weights into all InnerProduct layers."""
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


# ═══════════════════════════════════════════════════════════════════════
# Test Class 1: Split Topology Correctness
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestSplitTopologies:
    """Split layer correctness tests for multi-branch topologies."""

    def test_split_1to2_copies_data(self, ptrace):
        """1->2 Split: both tops must exactly match bottom."""
        prototxt = _make_basic_split_prototxt(num_top=2, feat_dim=4)
        with ptrace("Net(basic_split_1to2)") as t:
            net = net_from_param(net_param_from_string(prototxt))
            t['layers'] = len(net.layers_array())
        inp = np.random.RandomState(123).randn(2, 4).astype(np.float32)
        with ptrace("Forward(basic_split_1to2)"):
            net.Forward({"data": inp})
        np.testing.assert_array_equal(net.blob_by_name("split_0").to_numpy(), inp)
        np.testing.assert_array_equal(net.blob_by_name("split_1").to_numpy(), inp)

    def test_split_1to3_concat_roundtrip(self, ptrace):
        """1->3 Split + Concat: output is 3 copies concatenated."""
        feat_dim, n = 4, 3
        net = net_from_param(net_param_from_string(
            _make_split_concat_prototxt(n, feat_dim)))
        inp = np.random.RandomState(42).randn(2, feat_dim).astype(np.float32)
        with ptrace("Forward(split_concat_3)"):
            net.Forward({"data": inp})
        concat = net.blob_by_name("concat_out").to_numpy()
        assert concat.shape == (2, feat_dim * n)
        expected = np.concatenate([inp] * n, axis=1)
        np.testing.assert_array_equal(concat, expected)

    def test_residual_with_split(self, ptrace):
        """Residual connection: Split->(identity, FC+ReLU)->Eltwise SUM."""
        feat_dim = 8
        net = net_from_param(net_param_from_string(
            _make_residual_split_prototxt(feat_dim)))
        _set_random_weights(net, np.random.RandomState(7), scale=0.1)
        inp = (np.random.RandomState(99).randn(2, feat_dim).astype(np.float32)
               + 1.0)
        with ptrace("Forward(residual_split)"):
            net.Forward({"data": inp})
        np.testing.assert_array_equal(
            net.blob_by_name("identity").to_numpy(), inp)
        prob = net.blob_by_name("prob").to_numpy()
        assert not np.any(np.isnan(prob))
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-6)

    def test_split_inplace_branch_isolation(self, ptrace):
        """In-place ReLU on one branch must NOT corrupt sibling data."""
        feat_dim = 8
        net = net_from_param(net_param_from_string(
            _make_split_inplace_branch_prototxt(feat_dim)))
        _set_random_weights(net, np.random.RandomState(55), scale=0.1)
        inp = np.random.RandomState(55).randn(2, feat_dim).astype(np.float32)
        with ptrace("Forward(split_inplace)"):
            net.Forward({"data": inp})
        np.testing.assert_array_equal(
            net.blob_by_name("raw_branch").to_numpy(), inp,
            err_msg="In-place ReLU corrupted sibling branch!")
        np.testing.assert_array_equal(
            net.blob_by_name("relu_branch").to_numpy(), np.maximum(inp, 0))

    def test_n1_split_passthrough(self, ptrace):
        """N=1 Split: acts as identity passthrough (Phase 1 zero-copy: shares data pointer)."""
        feat_dim = 4
        net = net_from_param(net_param_from_string(
            _make_n1_split_passthrough_prototxt(feat_dim)))
        _set_random_weights(net, np.random.RandomState(77), scale=0.1)
        inp = np.random.RandomState(77).randn(2, feat_dim).astype(np.float32)
        with ptrace("Forward(n1_split)"):
            net.Forward({"data": inp})
        # Data correctness: passthrough blob has same values as input
        np.testing.assert_array_equal(
            net.blob_by_name("passthrough").to_numpy(), inp)
        prob = net.blob_by_name("prob").to_numpy()
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-6)
        # Phase 1 zero-copy verification: N=1 Split top shares the same data pointer as bottom
        data_ptr = net.blob_by_name("data").data_tensor.ctypes.data
        pass_ptr = net.blob_by_name("passthrough").data_tensor.ctypes.data
        assert data_ptr == pass_ptr, (
            f"N=1 Split zero-copy broken: data ptr=0x{data_ptr:x}, passthrough ptr=0x{pass_ptr:x}"
        )

    def test_split_deterministic_repeated_forward(self, ptrace):
        """Same input produces identical results across repeated forwards."""
        net = net_from_param(net_param_from_string(
            _make_basic_split_prototxt(num_top=2, feat_dim=4)))
        inp = np.random.RandomState(2024).randn(2, 4).astype(np.float32)
        results = []
        for i in range(5):
            with ptrace(f"Forward(deterministic_{i})"):
                net.Forward({"data": inp})
            results.append(net.blob_by_name("split_0").to_numpy().copy())
        for r in results[1:]:
            np.testing.assert_array_equal(r, results[0])


# ═══════════════════════════════════════════════════════════════════════
# Test Class 2: Split Performance Scaling (memcpy bottleneck measurement)
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestSplitPerformanceScaling:
    """Measure Split memcpy performance across input sizes to detect bottlenecks.

    Performance data is written to CSV; [SPLIT-PERF] logs appear in stderr at
    WARN level for direct inspection.
    """

    @pytest.mark.leak_check(False)
    @pytest.mark.parametrize("batch,feat_dim,n_branches", [
        (1, 128, 2),
        (1, 128, 4),
        (32, 256, 2),
        (32, 256, 4),
        (32, 512, 2),
        (32, 1024, 2),
        (16, 2048, 2),
        (16, 2048, 4),
    ])
    def test_split_perf_scaling(self, ptrace, batch, feat_dim, n_branches):
        """Per-config memcpy scaling test. Analyze CSV to find bottleneck."""
        proto = _make_split_mlp_perf_prototxt(
            batch=batch, feat_dim=feat_dim, n_branches=n_branches)
        net = net_from_param(net_param_from_string(proto))
        _set_random_weights(net, np.random.RandomState(0), scale=0.01)
        inp = np.random.RandomState(0).randn(batch, feat_dim).astype(np.float32)

        net.Forward({"data": inp})  # warm-up
        label = f"SplitPerf(b={batch},d={feat_dim},n={n_branches})"
        with ptrace(label) as t:
            net.Forward({"data": inp})
            t['batch'] = batch
            t['feat_dim'] = feat_dim
            t['n_branches'] = n_branches
            t['total_bytes_split'] = batch * feat_dim * 4 * n_branches

        prob = net.blob_by_name("prob").to_numpy()
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# Test Class 3: Extreme Boundary Conditions
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestExtremeBoundaries:
    """Extreme boundary tests: large inputs, NaN/Inf, extreme weights, deep nets."""

    @pytest.mark.leak_check(False)
    def test_large_input_2048(self, ptrace):
        """batch=64, feat=2048: forward succeeds, no crash, valid softmax."""
        batch, feat = 64, 2048
        net = net_from_param(net_param_from_string(
            _make_mlp_prototxt(batch, feat, hidden_dim=512, n_hidden=2)))
        _set_random_weights(net, np.random.RandomState(1), scale=0.001)
        inp = np.random.RandomState(0).randn(batch, feat).astype(np.float32) * 0.01
        with ptrace(f"Forward(large_b{batch}_d{feat})") as t:
            net.Forward({"data": inp})
            t['nbytes_input'] = batch * feat * 4
        prob = net.blob_by_name("prob").to_numpy()
        assert prob.shape == (batch, 10)
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.leak_check(False)
    def test_split_large_input_1024(self, ptrace):
        """Split: batch=32, feat=1024, N=2 branches."""
        batch, feat, n = 32, 1024, 2
        net = net_from_param(net_param_from_string(
            _make_split_mlp_perf_prototxt(batch, feat, n)))
        _set_random_weights(net, np.random.RandomState(2), scale=0.001)
        inp = np.random.RandomState(42).randn(batch, feat).astype(np.float32) * 0.01
        with ptrace(f"Forward(split_large_b{batch}_d{feat})") as t:
            net.Forward({"data": inp})
            t['split_copy_bytes'] = batch * feat * 4 * n
        prob = net.blob_by_name("prob").to_numpy()
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-5)

    def test_nan_input_no_crash(self, ptrace):
        """NaN inputs: no segfault (NaN propagation acceptable)."""
        batch, feat = 8, 16
        net = net_from_param(net_param_from_string(
            _make_mlp_prototxt(batch, feat, 16, 1)))
        _set_random_weights(net, np.random.RandomState(3))
        inp = np.full((batch, feat), np.nan, dtype=np.float32)
        caffe_ffi.set_log_level(caffe_ffi.LOG_LEVEL_ERROR)
        try:
            with ptrace("Forward(nan_input)"):
                net.Forward({"data": inp})
        finally:
            caffe_ffi.set_log_level(caffe_ffi.LOG_LEVEL_WARN)

    def test_inf_input_no_crash(self, ptrace):
        """Inf inputs: no segfault."""
        batch, feat = 8, 16
        net = net_from_param(net_param_from_string(
            _make_mlp_prototxt(batch, feat, 16, 1)))
        _set_random_weights(net, np.random.RandomState(4))
        inp = np.full((batch, feat), np.inf, dtype=np.float32)
        caffe_ffi.set_log_level(caffe_ffi.LOG_LEVEL_ERROR)
        try:
            with ptrace("Forward(inf_input)"):
                net.Forward({"data": inp})
        finally:
            caffe_ffi.set_log_level(caffe_ffi.LOG_LEVEL_WARN)

    def test_zero_input_deterministic(self, ptrace):
        """All-zero input: deterministic output across runs."""
        batch, feat = 4, 8
        net = net_from_param(net_param_from_string(
            _make_mlp_prototxt(batch, feat, 8, 1)))
        _set_random_weights(net, np.random.RandomState(5))
        zero = np.zeros((batch, feat), dtype=np.float32)
        with ptrace("Forward(zero_1)"):
            net.Forward({"data": zero})
        p1 = net.blob_by_name("prob").to_numpy().copy()
        with ptrace("Forward(zero_2)"):
            net.Forward({"data": zero})
        p2 = net.blob_by_name("prob").to_numpy().copy()
        np.testing.assert_array_equal(p1, p2)
        assert np.all(np.isfinite(p1))

    @pytest.mark.leak_check(False)
    def test_extreme_weights_large(self, ptrace):
        """Weights=1e6: no crash (Inf output acceptable)."""
        batch, feat = 2, 4
        net = net_from_param(net_param_from_string(
            _make_mlp_prototxt(batch, feat, 4, 1)))
        _set_constant_weights(net, 1e6)
        inp = np.ones((batch, feat), dtype=np.float32)
        caffe_ffi.set_log_level(caffe_ffi.LOG_LEVEL_ERROR)
        try:
            with ptrace("Forward(large_w)"):
                net.Forward({"data": inp})
        finally:
            caffe_ffi.set_log_level(caffe_ffi.LOG_LEVEL_WARN)

    def test_extreme_weights_tiny(self, ptrace):
        """Weights=1e-6: finite output."""
        batch, feat = 4, 8
        net = net_from_param(net_param_from_string(
            _make_mlp_prototxt(batch, feat, 8, 1)))
        _set_constant_weights(net, 1e-6)
        inp = np.ones((batch, feat), dtype=np.float32)
        with ptrace("Forward(tiny_w)"):
            net.Forward({"data": inp})
        assert np.all(np.isfinite(net.blob_by_name("prob").to_numpy()))

    def test_deep_network_20_layers(self, ptrace):
        """20-layer MLP: forward succeeds."""
        batch, feat, hidden, n_hidden = 4, 16, 16, 18
        net = net_from_param(net_param_from_string(
            _make_mlp_prototxt(batch, feat, hidden, n_hidden)))
        _set_random_weights(net, np.random.RandomState(6), scale=0.1)
        inp = np.random.RandomState(7).randn(batch, feat).astype(np.float32) * 0.1
        with ptrace(f"Forward(deep{n_hidden+2})") as t:
            net.Forward({"data": inp})
            t['n_layers'] = len(net.layers_array())
        prob = net.blob_by_name("prob").to_numpy()
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-5)

    def test_minimal_1x1(self, ptrace):
        """1x1 scalar network: forward succeeds."""
        net = net_from_param(net_param_from_string(
            _make_mlp_prototxt(1, 1, 1, 1)))
        _set_random_weights(net, np.random.RandomState(11), scale=0.5)
        inp = np.array([[0.5]], dtype=np.float32)
        with ptrace("Forward(1x1)"):
            net.Forward({"data": inp})
        prob = net.blob_by_name("prob").to_numpy()
        assert prob.shape == (1, 10)
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# Test Class 4: Memory Stability (Split-specific + lifecycle stress)
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestSplitMemoryStability:
    """Memory stability: lifecycle stress, repeated forward, high fanout,
    multi-level chains, concurrent-like access patterns."""

    def test_lifecycle_stress_50_creates(self, ptrace):
        """50x create/forward/destroy: <1MB leak tolerance."""
        batch, feat = 4, 8
        param = net_param_from_string(
            _make_mlp_prototxt(batch, feat, 8, 1))
        inp = np.random.RandomState(8).randn(batch, feat).astype(np.float32) * 0.1
        mem_before = total_allocated_bytes()
        for i in range(50):
            with ptrace(f"Lifecycle({i})", verbose=(i < 3)):
                net = net_from_param(param)
                _set_random_weights(net, np.random.RandomState(i), scale=0.1)
                net.Forward({"data": inp})
                del net
                gc.collect()
        gc.collect()
        leak = total_allocated_bytes() - mem_before
        assert leak < 1024 * 1024, f"Lifecycle leaked {leak} bytes"

    def test_repeated_forward_100_times(self, ptrace):
        """100 forwards: blob count stable, memory growth <4KB."""
        batch, feat = 8, 32
        net = net_from_param(net_param_from_string(
            _make_mlp_prototxt(batch, feat, 32, 2)))
        _set_random_weights(net, np.random.RandomState(9), scale=0.05)
        inp = np.random.RandomState(10).randn(batch, feat).astype(np.float32) * 0.1
        blobs_before = live_blob_count()
        mem_before = total_allocated_bytes()
        for i in range(100):
            with ptrace(f"RepeatedFwd({i})", verbose=(i % 20 == 0)):
                net.Forward({"data": inp})
        gc.collect()
        assert live_blob_count() == blobs_before, "Blob count grew!"
        assert total_allocated_bytes() <= mem_before + 4096, \
            "Memory grew excessively over 100 forwards"

    def test_split_high_fanout_8(self, ptrace):
        """Split N=8 (high fanout): all branches produce identical data."""
        batch, feat, n = 4, 64, 8
        net = net_from_param(net_param_from_string(
            _make_split_mlp_perf_prototxt(batch, feat, n, branch_ip_dim=16)))
        _set_random_weights(net, np.random.RandomState(20), scale=0.01)
        inp = np.random.RandomState(20).randn(batch, feat).astype(np.float32)
        with ptrace(f"Forward(split_fanout{n})") as t:
            net.Forward({"data": inp})
            t['fanout'] = n
            t['copy_bytes'] = batch * feat * 4 * n
        for i in range(n):
            np.testing.assert_array_equal(
                net.blob_by_name(f"b{i}").to_numpy(), inp)

    def test_split_lifecycle_stress(self, ptrace):
        """Split nets: 30x create/forward/destroy cycle with N=4 fanout."""
        batch, feat, n = 4, 32, 4
        param = net_param_from_string(
            _make_split_mlp_perf_prototxt(batch, feat, n, branch_ip_dim=16))
        inp = np.random.RandomState(30).randn(batch, feat).astype(np.float32) * 0.1
        mem_before = total_allocated_bytes()
        for i in range(30):
            with ptrace(f"SplitLifecycle({i})", verbose=(i < 3)):
                net = net_from_param(param)
                _set_random_weights(net, np.random.RandomState(i), scale=0.05)
                net.Forward({"data": inp})
                del net
                gc.collect()
        gc.collect()
        leak = total_allocated_bytes() - mem_before
        assert leak < 1024 * 1024, f"Split lifecycle leaked {leak} bytes"

    def test_multi_level_split_chain(self, ptrace):
        """3-level Split chain: forward succeeds, no corruption."""
        batch, feat, depth = 2, 16, 3
        net = net_from_param(net_param_from_string(
            _make_multi_level_split_prototxt(batch, feat, depth)))
        _set_random_weights(net, np.random.RandomState(40), scale=0.1)
        inp = np.random.RandomState(40).randn(batch, feat).astype(np.float32) * 0.1
        with ptrace(f"Forward(multi_split_d{depth})") as t:
            net.Forward({"data": inp})
            t['depth'] = depth
        prob = net.blob_by_name("prob").to_numpy()
        assert not np.any(np.isnan(prob))
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-5)

    def test_concurrent_net_creation_stress(self, ptrace):
        """8 threads simultaneously create/forward/destroy nets (serial stress)."""
        batch, feat = 2, 8
        param = net_param_from_string(
            _make_split_mlp_perf_prototxt(batch, feat, 2, branch_ip_dim=4))
        errors: list[Exception] = []

        def worker(seed: int):
            try:
                for _ in range(5):
                    net = net_from_param(param)
                    _set_random_weights(net, np.random.RandomState(seed), scale=0.1)
                    inp = np.random.RandomState(seed).randn(
                        batch, feat).astype(np.float32)
                    net.Forward({"data": inp})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(8)]
        mem_before = total_allocated_bytes()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        gc.collect()
        assert len(errors) == 0, f"Concurrent errors: {errors[:3]}"
        leak = total_allocated_bytes() - mem_before
        assert leak < 2 * 1024 * 1024, \
            f"Concurrent stress leaked {leak} bytes"


# ═══════════════════════════════════════════════════════════════════════
# CSV Performance Report Helper (run with: pytest ... -s)
# ═══════════════════════════════════════════════════════════════════════

def print_performance_summary():
    """Post-session summary: parse CSV and report memcpy bottleneck analysis.

    This function is called manually (or from a script) after running tests.
    It reads the latest perf_log_*.csv from .temp/ and prints analysis.
    """
    import csv as _csv
    from pathlib import Path as _Path
    temp_dir = _Path(__file__).parent / ".temp"
    csv_files = sorted(temp_dir.glob("perf_log_*.csv"))
    if not csv_files:
        print("No CSV performance logs found. Run tests first.")
        return
    latest = csv_files[-1]
    print(f"\n{'='*70}")
    print(f"Performance Summary from: {latest.name}")
    print(f"{'='*70}")
    split_rows = []
    with open(latest, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            if "SplitPerf" in row["operation"]:
                split_rows.append(row)
    if split_rows:
        print(f"\nSplit memcpy performance ({len(split_rows)} configs):")
        print(f"{'Config':<35} {'elapsed_ms':>10} {'total_bytes':>12}")
        print("-" * 60)
        for r in split_rows:
            print(f"{r['operation']:<35} {float(r['elapsed_ms']):>10.3f}"
                  f" {r['extra_fields']:>40}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    print_performance_summary()
