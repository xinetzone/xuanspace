"""P2-A: Complex network topology, dynamic shape, and large-scale forward tests.

Tests cover:
- TestNetTopologies: multi-branch, multi-input, deep networks, in-place ops, residual connections
- TestNetReshapeDynamics: varying batch sizes, channel counts, dynamic reshaping
- TestLargeScaleForward: large batches, repeated forwards, memory stability
"""
from __future__ import annotations

import numpy as np
import pytest

import caffe_ffi
from caffe_ffi import (
    Net, Blob, Layer,
    net_param_from_string, net_from_param,
)
from .conftest import require_cpp_extension


# ─── Helper prototxt builders ──────────────────────────────────────

def _make_two_branch_prototxt(fc_dim: int = 4) -> str:
    """Two-branch network with two separate Inputs: data1/data2 -> FC -> Concat -> prob.
    Note: current caffe-ffi does not auto-insert Split layers, so each blob
    can be used as bottom by exactly one layer.
    """
    return f"""name: "two_branch"
layer {{
  name: "data1"
  type: "Input"
  top: "data1"
  input_param {{ shape {{ dim: 2 dim: 3 }} }}
}}
layer {{
  name: "data2"
  type: "Input"
  top: "data2"
  input_param {{ shape {{ dim: 2 dim: 3 }} }}
}}
layer {{
  name: "branch1_fc"
  type: "InnerProduct"
  bottom: "data1"
  top: "branch1"
  inner_product_param {{ num_output: {fc_dim} bias_term: true }}
}}
layer {{
  name: "branch1_relu"
  type: "ReLU"
  bottom: "branch1"
  top: "branch1"
}}
layer {{
  name: "branch2_fc"
  type: "InnerProduct"
  bottom: "data2"
  top: "branch2"
  inner_product_param {{ num_output: {fc_dim} bias_term: true }}
}}
layer {{
  name: "branch2_relu"
  type: "ReLU"
  bottom: "branch2"
  top: "branch2"
}}
layer {{
  name: "concat"
  type: "Concat"
  bottom: "branch1"
  bottom: "branch2"
  top: "concat"
  concat_param {{ axis: 1 }}
}}
layer {{
  name: "prob"
  type: "Softmax"
  bottom: "concat"
  top: "prob"
}}
"""


def _make_multi_input_prototxt() -> str:
    """Two-input network: data1 + data2 -> concat -> ip -> prob."""
    return """name: "multi_input"
layer {
  name: "data1"
  type: "Input"
  top: "data1"
  input_param { shape { dim: 2 dim: 3 } }
}
layer {
  name: "data2"
  type: "Input"
  top: "data2"
  input_param { shape { dim: 2 dim: 3 } }
}
layer {
  name: "concat"
  type: "Concat"
  bottom: "data1"
  bottom: "data2"
  top: "concat"
  concat_param { axis: 1 }
}
layer {
  name: "ip"
  type: "InnerProduct"
  bottom: "concat"
  top: "ip"
  inner_product_param { num_output: 2 bias_term: true }
}
layer {
  name: "prob"
  type: "Softmax"
  bottom: "ip"
  top: "prob"
}
"""


def _make_deep_mlp_prototxt(n_hidden: int = 3, hidden_dim: int = 8) -> str:
    """Deep MLP: Input -> (IP -> ReLU) * n_hidden -> IP -> Softmax."""
    lines = ['name: "deep_mlp"', 'layer {', '  name: "data"', '  type: "Input"',
             '  top: "data"', '  input_param { shape { dim: 2 dim: 4 } }', '}']
    prev_top = "data"
    for i in range(n_hidden):
        lines.extend([
            'layer {',
            f'  name: "ip{i+1}"',
            '  type: "InnerProduct"',
            f'  bottom: "{prev_top}"',
            f'  top: "ip{i+1}"',
            f'  inner_product_param {{ num_output: {hidden_dim} bias_term: true }}',
            '}',
            'layer {',
            f'  name: "relu{i+1}"',
            '  type: "ReLU"',
            f'  bottom: "ip{i+1}"',
            f'  top: "ip{i+1}"',
            '}',
        ])
        prev_top = f"ip{i+1}"
    lines.extend([
        'layer {',
        f'  name: "ip_out"',
        '  type: "InnerProduct"',
        f'  bottom: "{prev_top}"',
        '  top: "ip_out"',
        '  inner_product_param { num_output: 2 bias_term: true }',
        '}',
        'layer {',
        '  name: "prob"',
        '  type: "Softmax"',
        '  bottom: "ip_out"',
        '  top: "prob"',
        '}',
    ])
    return "\n".join(lines)


def _make_inplace_chain_prototxt() -> str:
    """Chain of in-place ReLU/Dropout operations."""
    return """name: "inplace_chain"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 2 dim: 4 } }
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "data"
  top: "x"
  inner_product_param { num_output: 4 bias_term: true }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "x"
  top: "x"
}
layer {
  name: "ip2"
  type: "InnerProduct"
  bottom: "x"
  top: "x"
  inner_product_param { num_output: 4 bias_term: true }
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "x"
  top: "x"
}
layer {
  name: "ip3"
  type: "InnerProduct"
  bottom: "x"
  top: "x3"
  inner_product_param { num_output: 2 bias_term: true }
}
layer {
  name: "prob"
  type: "Softmax"
  bottom: "x3"
  top: "prob"
}
"""


def _make_residual_prototxt() -> str:
    """Eltwise SUM network using two separate paths (no blob reuse needed).
    Note: current caffe-ffi requires each blob to be consumed by exactly one layer,
    so true residual connections (reusing a blob as both identity and branch input)
    require explicit Split which is not yet auto-inserted. This tests Eltwise SUM
    with two independent processing paths.
    """
    return """name: "dual_path_sum"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 2 dim: 4 } }
}
layer {
  name: "path1_fc"
  type: "InnerProduct"
  bottom: "data"
  top: "path1"
  inner_product_param { num_output: 4 bias_term: false }
}
layer {
  name: "path1_relu"
  type: "ReLU"
  bottom: "path1"
  top: "path1"
}
layer {
  name: "data2"
  type: "Input"
  top: "data2"
  input_param { shape { dim: 2 dim: 4 } }
}
layer {
  name: "path2_fc"
  type: "InnerProduct"
  bottom: "data2"
  top: "path2"
  inner_product_param { num_output: 4 bias_term: true }
}
layer {
  name: "path2_relu"
  type: "ReLU"
  bottom: "path2"
  top: "path2"
}
layer {
  name: "add"
  type: "Eltwise"
  bottom: "path1"
  bottom: "path2"
  top: "added"
  eltwise_param { operation: SUM }
}
layer {
  name: "ip_out"
  type: "InnerProduct"
  bottom: "added"
  top: "ip_out"
  inner_product_param { num_output: 2 bias_term: true }
}
layer {
  name: "prob"
  type: "Softmax"
  bottom: "ip_out"
  top: "prob"
}
"""


def _make_lenet_like_prototxt() -> str:
    """LeNet-like conv net (using InnerProduct to simulate conv outputs for simplicity)."""
    return """name: "lenet_like"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 2 dim: 1 } }
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "data"
  top: "ip1"
  inner_product_param { num_output: 8 bias_term: true }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "ip1"
  top: "ip1"
}
layer {
  name: "ip2"
  type: "InnerProduct"
  bottom: "ip1"
  top: "ip2"
  inner_product_param { num_output: 4 bias_term: true }
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "ip2"
  top: "ip2"
}
layer {
  name: "ip3"
  type: "InnerProduct"
  bottom: "ip2"
  top: "ip3"
  inner_product_param { num_output: 2 bias_term: true }
}
layer {
  name: "prob"
  type: "Softmax"
  bottom: "ip3"
  top: "prob"
}
"""


def _load_identity_weights(net: Net, rng: np.random.RandomState | None = None) -> None:
    """Load small random weights into all InnerProduct layers by name."""
    if rng is None:
        rng = np.random.RandomState(42)
    for layer in net.layers_array():
        if layer.type == "InnerProduct" and len(layer.blobs) >= 1:
            W = layer.blobs[0]
            w_shape = W.shape
            # Small random weights scaled by 0.1
            w_data = (rng.randn(*w_shape).astype(np.float32)) * 0.1
            W.from_numpy(w_data)
            if len(layer.blobs) >= 2:
                b = layer.blobs[1]
                b_data = np.zeros(b.shape, dtype=np.float32)
                b.from_numpy(b_data)


# ─── P2-A1: Network topology tests ────────────────────────────────

@require_cpp_extension
class TestNetTopologies:
    """Complex network topology tests: branches, multi-input, deep, in-place, residual."""

    def test_two_branch_concat_shape(self, ptrace):
        """Two-branch Concat (two inputs) produces correct output dimension (2*fc_dim)."""
        fc_dim = 4
        with ptrace("build two-branch net (dual-input)") as t:
            proto = _make_two_branch_prototxt(fc_dim=fc_dim)
            net = net_from_param(net_param_from_string(proto))
            _load_identity_weights(net)
            t['layers'] = len(net.layers_array())
        inp1 = np.random.randn(2, 3).astype(np.float32)
        inp2 = np.random.randn(2, 3).astype(np.float32)
        with ptrace("forward two-branch net") as t:
            out = net.forward({"data1": inp1, "data2": inp2})
            t['out_keys'] = list(out.keys())
        assert "prob" in out
        assert out["prob"].shape == (2, fc_dim * 2), \
            f"Expected (2, {fc_dim*2}), got {out['prob'].shape}"
        np.testing.assert_allclose(
            out["prob"].sum(axis=1), np.ones(2), rtol=1e-5,
        )

    def test_two_branch_concat_values_deterministic(self, ptrace):
        """Two-branch forward is deterministic across calls."""
        proto = _make_two_branch_prototxt(fc_dim=4)
        with ptrace("build two-branch net"):
            net = net_from_param(net_param_from_string(proto))
            _load_identity_weights(net)
        rng = np.random.RandomState(123)
        inp1 = rng.randn(2, 3).astype(np.float32)
        inp2 = rng.randn(2, 3).astype(np.float32)
        with ptrace("forward x2") as t:
            out1 = net.forward({"data1": inp1, "data2": inp2})
            out2 = net.forward({"data1": inp1, "data2": inp2})
            t['max_diff'] = float(np.max(np.abs(out1["prob"] - out2["prob"])))
        np.testing.assert_array_equal(out1["prob"], out2["prob"])

    def test_multi_input_concat(self, ptrace):
        """Multi-input network (two Input layers) with Concat works correctly."""
        with ptrace("build multi-input net") as t:
            proto = _make_multi_input_prototxt()
            net = net_from_param(net_param_from_string(proto))
            _load_identity_weights(net)
            t['layers'] = len(net.layers_array())
            t['blobs'] = len(net.blobs_array())
        inp1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        inp2 = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)
        with ptrace("forward multi-input net") as t:
            out = net.forward({"data1": inp1, "data2": inp2})
            t['out_shape'] = str(out["prob"].shape)
        assert "prob" in out
        assert out["prob"].shape == (2, 2)
        np.testing.assert_allclose(
            out["prob"].sum(axis=1), np.ones(2), rtol=1e-5,
        )

    def test_multi_input_different_batch_sizes_error(self, ptrace):
        """Multi-input with mismatched batch sizes should not crash (may produce error)."""
        proto = _make_multi_input_prototxt()
        with ptrace("build multi-input net"):
            net = net_from_param(net_param_from_string(proto))
            _load_identity_weights(net)
        inp1 = np.random.randn(2, 3).astype(np.float32)
        inp2 = np.random.randn(4, 3).astype(np.float32)  # Mismatched batch!
        with ptrace("forward mismatched batch (expect error/undefined)") as t:
            try:
                out = net.forward({"data1": inp1, "data2": inp2})
                t['result'] = 'no_crash'
            except (ValueError, RuntimeError) as e:
                t['result'] = f'raised: {type(e).__name__}'

    def test_deep_mlp_forward(self, ptrace):
        """Deep MLP with n_hidden=5 hidden layers forwards correctly."""
        n_hidden = 5
        hidden_dim = 8
        with ptrace(f"build deep MLP ({n_hidden} hidden, dim={hidden_dim})") as t:
            proto = _make_deep_mlp_prototxt(n_hidden=n_hidden, hidden_dim=hidden_dim)
            net = net_from_param(net_param_from_string(proto))
            _load_identity_weights(net)
            t['layers'] = len(net.layers_array())
        inp = np.random.randn(2, 4).astype(np.float32)
        with ptrace("forward deep MLP") as t:
            out = net.forward({"data": inp})
            t['out_shape'] = str(out["prob"].shape)
        assert out["prob"].shape == (2, 2)
        np.testing.assert_allclose(
            out["prob"].sum(axis=1), np.ones(2), rtol=1e-5,
        )

    def test_inplace_chain_forward(self, ptrace):
        """Chain of in-place ReLU operations doesn't corrupt data."""
        with ptrace("build in-place chain net") as t:
            proto = _make_inplace_chain_prototxt()
            net = net_from_param(net_param_from_string(proto))
            _load_identity_weights(net)
            t['layers'] = len(net.layers_array())
        inp = np.random.randn(2, 4).astype(np.float32)
        with ptrace("forward in-place chain x3") as t:
            for i in range(3):
                out = net.forward({"data": inp})
            t['iterations'] = 3
        assert out["prob"].shape == (2, 2)
        np.testing.assert_allclose(
            out["prob"].sum(axis=1), np.ones(2), rtol=1e-5,
        )

    def test_inplace_inner_product_shape_change_rejected(self, ptrace):
        """In-place InnerProduct with output size != input size is rejected (ASan guard).

        Regression guard for Task 17b: an in-place InnerProduct whose num_output
        differs from the input feature dim would resize the shared bottom/top buffer,
        causing Forward_cpu to read beyond the truncated buffer (heap-buffer-overflow
        caught by ASan). The layer must reject this at Reshape time instead.
        """
        proto = """name: "inplace_ip_shape_change"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 2 dim: 4 } }
}
layer {
  name: "ip"
  type: "InnerProduct"
  bottom: "data"
  top: "data"
  inner_product_param { num_output: 2 bias_term: true }
}
"""
        with ptrace("in-place InnerProduct shape-change → rejected"):
            with pytest.raises((ValueError, RuntimeError)):
                net_from_param(net_param_from_string(proto))

    def test_residual_eltwise_sum(self, ptrace):
        """Eltwise SUM of two independent processing paths (dual-path add) produces valid output."""
        with ptrace("build dual-path sum net") as t:
            proto = _make_residual_prototxt()
            net = net_from_param(net_param_from_string(proto))
            _load_identity_weights(net)
            t['layers'] = len(net.layers_array())
        inp1 = np.random.randn(2, 4).astype(np.float32)
        inp2 = np.random.randn(2, 4).astype(np.float32)
        with ptrace("forward dual-path sum net") as t:
            out = net.forward({"data": inp1, "data2": inp2})
            t['out_shape'] = str(out["prob"].shape)
        assert "prob" in out
        assert out["prob"].shape == (2, 2)
        np.testing.assert_allclose(
            out["prob"].sum(axis=1), np.ones(2), rtol=1e-5,
        )

    def test_lenet_like_classification(self, ptrace):
        """LeNet-like multi-layer network produces valid probabilities."""
        with ptrace("build lenet-like net") as t:
            proto = _make_lenet_like_prototxt()
            net = net_from_param(net_param_from_string(proto))
            _load_identity_weights(net)
            t['layers'] = len(net.layers_array())
        inp = np.random.randn(2, 1).astype(np.float32)
        with ptrace("forward lenet-like net") as t:
            out = net.forward({"data": inp})
            t['prob_range'] = f"[{out['prob'].min():.4f}, {out['prob'].max():.4f}]"
        assert out["prob"].shape == (2, 2)
        assert np.all(out["prob"] >= 0) and np.all(out["prob"] <= 1)
        np.testing.assert_allclose(
            out["prob"].sum(axis=1), np.ones(2), rtol=1e-5,
        )

    def test_eltwise_coeff_sum(self, ptrace):
        """Eltwise SUM with coefficients: 2*a + 0.5*b produces correct weighted sum."""
        proto = """name: "eltwise_coeff"
layer { name: "a" type: "Input" top: "a" input_param { shape { dim: 1 dim: 3 } } }
layer { name: "b" type: "Input" top: "b" input_param { shape { dim: 1 dim: 3 } } }
layer {
  name: "sum"
  type: "Eltwise"
  bottom: "a"
  bottom: "b"
  top: "sum"
  eltwise_param { operation: SUM coeff: 2.0 coeff: 0.5 }
}"""
        with ptrace("build eltwise coeff net"):
            net = net_from_param(net_param_from_string(proto))
        a = np.array([[1, 2, 3]], dtype=np.float32)
        b = np.array([[4, 6, 8]], dtype=np.float32)
        with ptrace("forward eltwise coeff") as t:
            out = net.forward({"a": a, "b": b})
            expected = 2.0 * a + 0.5 * b
            t['max_diff'] = float(np.max(np.abs(out["sum"] - expected)))
        np.testing.assert_allclose(out["sum"], expected, rtol=1e-5)

    def test_eltwise_prod(self, ptrace):
        """Eltwise PROD produces element-wise product."""
        proto = """name: "eltwise_prod"
layer { name: "a" type: "Input" top: "a" input_param { shape { dim: 1 dim: 3 } } }
layer { name: "b" type: "Input" top: "b" input_param { shape { dim: 1 dim: 3 } } }
layer {
  name: "prod"
  type: "Eltwise"
  bottom: "a"
  bottom: "b"
  top: "prod"
  eltwise_param { operation: PROD }
}"""
        with ptrace("build eltwise prod net"):
            net = net_from_param(net_param_from_string(proto))
        a = np.array([[1, 2, 3]], dtype=np.float32)
        b = np.array([[4, 5, 6]], dtype=np.float32)
        with ptrace("forward eltwise prod") as t:
            out = net.forward({"a": a, "b": b})
            expected = a * b
            t['max_diff'] = float(np.max(np.abs(out["prod"] - expected)))
        np.testing.assert_allclose(out["prod"], expected, rtol=1e-5)


# ─── P2-A2: Dynamic shape tests ──────────────────────────────────

@require_cpp_extension
class TestNetReshapeDynamics:
    """Dynamic shape tests: varying batch sizes, channel counts, reshaping."""

    @pytest.fixture
    def simple_mlp(self, ptrace):
        """Simple MLP: data(2x3) -> ip(4) -> relu -> ip(2) -> prob."""
        proto = """name: "dyn_mlp"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 } } }
layer { name: "ip1" type: "InnerProduct" bottom: "data" top: "ip1" inner_product_param { num_output: 4 bias_term: true } }
layer { name: "relu1" type: "ReLU" bottom: "ip1" top: "ip1" }
layer { name: "ip2" type: "InnerProduct" bottom: "ip1" top: "ip2" inner_product_param { num_output: 2 bias_term: true } }
layer { name: "prob" type: "Softmax" bottom: "ip2" top: "prob" }"""
        with ptrace("build dynamic MLP") as t:
            net = net_from_param(net_param_from_string(proto))
            _load_identity_weights(net, rng=np.random.RandomState(99))
            t['layers'] = len(net.layers_array())
        return net

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_varying_batch_sizes(self, simple_mlp, batch_size, ptrace):
        """Forward works with batch sizes 1, 2, 4, 8."""
        inp = np.random.randn(batch_size, 3).astype(np.float32)
        with ptrace(f"forward batch={batch_size}") as t:
            out = simple_mlp.forward({"data": inp})
            t['out_shape'] = str(out["prob"].shape)
        assert out["prob"].shape == (batch_size, 2), \
            f"batch={batch_size}: expected ({batch_size}, 2), got {out['prob'].shape}"
        np.testing.assert_allclose(
            out["prob"].sum(axis=1), np.ones(batch_size), rtol=1e-5,
        )

    def test_batch_size_1(self, simple_mlp, ptrace):
        """Batch size 1 (single sample inference) works correctly."""
        inp = np.array([[0.5, -0.3, 1.2]], dtype=np.float32)
        with ptrace("forward batch=1") as t:
            out = simple_mlp.forward({"data": inp})
            t['prob'] = str(out["prob"].flatten())
        assert out["prob"].shape == (1, 2)
        assert np.allclose(out["prob"].sum(), 1.0, rtol=1e-5)

    def test_large_batch_32(self, simple_mlp, ptrace):
        """Larger batch (32) forwards without error."""
        batch = 32
        inp = np.random.randn(batch, 3).astype(np.float32)
        with ptrace(f"forward batch={batch}") as t:
            out = simple_mlp.forward({"data": inp})
            t['out_shape'] = str(out["prob"].shape)
        assert out["prob"].shape == (batch, 2)

    def test_large_batch_128(self, simple_mlp, ptrace):
        """Batch size 128 forwards without memory issues."""
        batch = 128
        inp = np.random.randn(batch, 3).astype(np.float32)
        with ptrace(f"forward batch={batch}") as t:
            out = simple_mlp.forward({"data": inp})
            t['out_shape'] = str(out["prob"].shape)
        assert out["prob"].shape == (batch, 2)

    def test_forwards_with_increasing_batch_sizes(self, simple_mlp, ptrace):
        """Sequentially increasing batch sizes (1→2→4→8→16) all work."""
        import time as _time
        batch_sizes = [1, 2, 4, 8, 16]
        results = []
        fwd_times = []
        with ptrace(f"forward increasing batches {batch_sizes}") as t:
            for bs in batch_sizes:
                inp = np.random.randn(bs, 3).astype(np.float32)
                _t0 = _time.perf_counter()
                out = simple_mlp.forward({"data": inp})
                _dt = (_time.perf_counter() - _t0) * 1000.0
                fwd_times.append((bs, round(_dt, 3)))
                assert out["prob"].shape == (bs, 2)
                results.append(out["prob"].sum())
            t['n_batches'] = len(batch_sizes)
            t['fwd_times_ms'] = str(fwd_times)
        # Each batch should produce valid probability distributions
        for i, bs in enumerate(batch_sizes):
            assert np.isclose(results[i], float(bs), rtol=1e-4), \
                f"batch={bs}: sum(prob)={results[i]}, expected {bs}"

    def test_forwards_with_decreasing_batch_sizes(self, simple_mlp, ptrace):
        """Sequentially decreasing batch sizes (16→8→4→2→1) all work."""
        import time as _time
        batch_sizes = [16, 8, 4, 2, 1]
        fwd_times = []
        with ptrace(f"forward decreasing batches {batch_sizes}") as t:
            for bs in batch_sizes:
                inp = np.random.randn(bs, 3).astype(np.float32)
                _t0 = _time.perf_counter()
                out = simple_mlp.forward({"data": inp})
                _dt = (_time.perf_counter() - _t0) * 1000.0
                fwd_times.append((bs, round(_dt, 3)))
                assert out["prob"].shape == (bs, 2)
            t['n_batches'] = len(batch_sizes)
            t['fwd_times_ms'] = str(fwd_times)

    def test_same_input_different_batch_layout(self, ptrace):
        """Same logical samples produce consistent probabilities regardless of batch layout."""
        proto = """name: "consistent"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 2 } } }
layer { name: "ip" type: "InnerProduct" bottom: "data" top: "ip" inner_product_param { num_output: 2 bias_term: true } }
layer { name: "prob" type: "Softmax" bottom: "ip" top: "prob" }"""
        with ptrace("build consistency net"):
            net = net_from_param(net_param_from_string(proto))
            _load_identity_weights(net, rng=np.random.RandomState(7))

        sample = np.array([[0.1, 0.2]], dtype=np.float32)
        with ptrace("forward single sample"):
            out1 = net.forward({"data": sample})

        batch_two = np.vstack([sample, sample])
        with ptrace("forward batch=2 (two copies)"):
            out2 = net.forward({"data": batch_two})

        # First sample of batch should match single-sample result
        np.testing.assert_allclose(out1["prob"][0], out2["prob"][0], rtol=1e-5)
        np.testing.assert_allclose(out2["prob"][0], out2["prob"][1], rtol=1e-5)

    def test_input_dimension_mismatch_no_crash(self, simple_mlp, ptrace):
        """Wrong input feature dimension (2 instead of 3) should raise, not segfault."""
        inp = np.random.randn(2, 2).astype(np.float32)  # Wrong: 2 features, net expects 3
        with ptrace("forward wrong dim (expect error)") as t:
            try:
                out = simple_mlp.forward({"data": inp})
                t['result'] = 'no_crash (shape adapted?)'
                t['out_shape'] = str(out.get("prob", out).shape)
            except (ValueError, RuntimeError, IndexError) as e:
                t['result'] = f'raised: {type(e).__name__}'

    def test_blob_reshape_between_forwards(self, ptrace):
        """Reshaping input blob between forwards handles new shape."""
        proto = """name: "reshape_test"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 } } }
layer { name: "ip" type: "InnerProduct" bottom: "data" top: "ip" inner_product_param { num_output: 2 bias_term: false } }
layer { name: "prob" type: "Softmax" bottom: "ip" top: "prob" }"""
        with ptrace("build reshape test net"):
            net = net_from_param(net_param_from_string(proto))
            _load_identity_weights(net)

        # First forward with batch=2
        inp1 = np.random.randn(2, 3).astype(np.float32)
        with ptrace("forward batch=2"):
            out1 = net.forward({"data": inp1})
        assert out1["prob"].shape == (2, 2)

        # Second forward with batch=4 (automatic reshape?)
        inp2 = np.random.randn(4, 3).astype(np.float32)
        with ptrace("forward batch=4 (after reshape)") as t:
            out2 = net.forward({"data": inp2})
            t['out_shape'] = str(out2["prob"].shape)
        assert out2["prob"].shape[0] == 4


# ─── P2-A3: Large-scale forward stability ────────────────────────

@require_cpp_extension
class TestLargeScaleForward:
    """Large-scale forward tests: repeated iterations, large batches, memory stability."""

    @pytest.fixture
    def small_mlp(self, ptrace):
        """Small MLP for fast repeated testing."""
        proto = """name: "stable_mlp"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 4 dim: 3 } } }
layer { name: "ip1" type: "InnerProduct" bottom: "data" top: "ip1" inner_product_param { num_output: 4 bias_term: true } }
layer { name: "relu1" type: "ReLU" bottom: "ip1" top: "ip1" }
layer { name: "ip2" type: "InnerProduct" bottom: "ip1" top: "ip2" inner_product_param { num_output: 2 bias_term: true } }
layer { name: "prob" type: "Softmax" bottom: "ip2" top: "prob" }"""
        with ptrace("build small_mlp (4x3→4→2)") as t:
            net = net_from_param(net_param_from_string(proto))
            rng = np.random.RandomState(42)
            n_ip = 0
            for layer in net.layers_array():
                if layer.type == "InnerProduct" and len(layer.blobs) >= 1:
                    W = layer.blobs[0]
                    W.from_numpy((rng.randn(*W.shape).astype(np.float32)) * 0.1)
                    if len(layer.blobs) >= 2:
                        layer.blobs[1].from_numpy(np.zeros(layer.blobs[1].shape, dtype=np.float32))
                    n_ip += 1
            t['layers'] = len(net.layers_array())
            t['ip_layers'] = n_ip
        return net

    def test_100_forwards_no_memory_growth(self, small_mlp, ptrace):
        """100 forward passes with no memory leak (Δblobs=0)."""
        from caffe_ffi import live_blob_count, total_allocated_bytes
        rng = np.random.RandomState(0)
        n_iters = 100
        mem_before = total_allocated_bytes()
        blobs_before = live_blob_count()
        # Sample timing for first, middle, last iterations
        sample_indices = {0, n_iters // 2, n_iters - 1}
        forward_times_ms = []
        import time as _time
        with ptrace(f"forward x{n_iters}") as t:
            for i in range(n_iters):
                inp = rng.randn(4, 3).astype(np.float32)
                _t0 = _time.perf_counter()
                out = small_mlp.forward({"data": inp})
                _dt = (_time.perf_counter() - _t0) * 1000.0
                if i in sample_indices:
                    forward_times_ms.append((i, round(_dt, 3)))
                assert out["prob"].shape == (4, 2)
            t['iterations'] = n_iters
            t['sample_times_ms'] = str(forward_times_ms)
        mem_after = total_allocated_bytes()
        blobs_after = live_blob_count()
        delta_blobs = blobs_after - blobs_before
        delta_mem = mem_after - mem_before
        t['delta_blobs'] = delta_blobs
        t['delta_mem'] = delta_mem
        assert delta_blobs == 0, f"Memory leak: +{delta_blobs} Blobs after {n_iters} forwards"
        assert delta_mem <= 4096, f"Memory leak: +{delta_mem} bytes after {n_iters} forwards"

    def test_500_forwards_deterministic(self, small_mlp, ptrace):
        """500 forwards with same input always produce same output."""
        import time as _time
        inp = np.random.RandomState(123).randn(4, 3).astype(np.float32)
        n_iters = 500
        sample_indices = {0, n_iters // 4, n_iters // 2, 3 * n_iters // 4, n_iters - 1}
        sample_times = []
        with ptrace(f"forward x{n_iters} determinism check") as t:
            out_first = small_mlp.forward({"data": inp})["prob"].copy()
            max_diff = 0.0
            for i in range(n_iters - 1):
                _t0 = _time.perf_counter()
                out = small_mlp.forward({"data": inp})
                _dt = (_time.perf_counter() - _t0) * 1000.0
                diff = float(np.max(np.abs(out["prob"] - out_first)))
                if diff > max_diff:
                    max_diff = diff
                if i in sample_indices:
                    sample_times.append((i, round(_dt, 3)))
            t['iterations'] = n_iters
            t['max_diff'] = max_diff
            t['sample_times_ms'] = str(sample_times)
        assert max_diff == 0.0, f"Non-deterministic after {n_iters} iters: max_diff={max_diff}"

    def test_large_batch_256_stable(self, ptrace):
        """Batch size 256 forward produces correct output."""
        proto = """name: "large_batch"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 256 dim: 16 } } }
layer { name: "ip1" type: "InnerProduct" bottom: "data" top: "ip1" inner_product_param { num_output: 32 bias_term: true } }
layer { name: "relu1" type: "ReLU" bottom: "ip1" top: "ip1" }
layer { name: "ip2" type: "InnerProduct" bottom: "ip1" top: "ip2" inner_product_param { num_output: 10 bias_term: true } }
layer { name: "prob" type: "Softmax" bottom: "ip2" top: "prob" }"""
        with ptrace("build large-batch net (256x16→32→10)") as t:
            net = net_from_param(net_param_from_string(proto))
            _load_identity_weights(net, rng=np.random.RandomState(55))
            t['layers'] = len(net.layers_array())
        batch = 256
        inp = np.random.randn(batch, 16).astype(np.float32)
        with ptrace(f"forward batch={batch}") as t:
            out = net.forward({"data": inp})
            t['out_shape'] = str(out["prob"].shape)
        assert out["prob"].shape == (batch, 10)
        np.testing.assert_allclose(
            out["prob"].sum(axis=1), np.ones(batch), rtol=1e-5,
        )

    def test_alternating_batch_sizes_stable(self, small_mlp, ptrace):
        """Alternating batch sizes (1↔32) 50 times each is stable."""
        import time as _time
        rng = np.random.RandomState(77)
        n_pairs = 50
        batch1_times = []
        batch32_times = []
        with ptrace(f"alternating batches 1↔32 x{n_pairs}") as t:
            for i in range(n_pairs):
                # Batch 1
                inp1 = rng.randn(1, 3).astype(np.float32)
                _t0 = _time.perf_counter()
                out1 = small_mlp.forward({"data": inp1})
                _dt = (_time.perf_counter() - _t0) * 1000.0
                assert out1["prob"].shape == (1, 2)
                # Batch 32
                inp32 = rng.randn(32, 3).astype(np.float32)
                _t0 = _time.perf_counter()
                out32 = small_mlp.forward({"data": inp32})
                _dt32 = (_time.perf_counter() - _t0) * 1000.0
                assert out32["prob"].shape == (32, 2)
                if i == 0 or i == n_pairs - 1:
                    batch1_times.append(round(_dt, 3))
                    batch32_times.append(round(_dt32, 3))
            t['switches'] = n_pairs
            t['batch1_sample_ms'] = str(batch1_times)
            t['batch32_sample_ms'] = str(batch32_times)
        # No assertion beyond no crash and correct shapes

    def test_net_destruction_and_recreation(self, ptrace):
        """Creating and destroying many nets in sequence doesn't leak."""
        import time as _time
        from caffe_ffi import live_blob_count, total_allocated_bytes
        n_nets = 20
        blobs_before = live_blob_count()
        mem_before = total_allocated_bytes()
        create_times = []
        forward_times = []
        with ptrace(f"create+destroy {n_nets} nets") as t:
            for i in range(n_nets):
                proto = f"""name: "tmp_net_{i}"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: 2 dim: 3 }} }} }}
layer {{ name: "ip" type: "InnerProduct" bottom: "data" top: "ip" inner_product_param {{ num_output: 2 bias_term: true }} }}
layer {{ name: "prob" type: "Softmax" bottom: "ip" top: "prob" }}"""
                _t0 = _time.perf_counter()
                net = net_from_param(net_param_from_string(proto))
                _load_identity_weights(net, rng=np.random.RandomState(i))
                _dt_create = (_time.perf_counter() - _t0) * 1000.0
                inp = np.random.randn(2, 3).astype(np.float32)
                _t0 = _time.perf_counter()
                out = net.forward({"data": inp})
                _dt_fwd = (_time.perf_counter() - _t0) * 1000.0
                assert out["prob"].shape == (2, 2)
                if i in (0, n_nets - 1):
                    create_times.append(round(_dt_create, 3))
                    forward_times.append(round(_dt_fwd, 3))
                del net
            t['nets'] = n_nets
            t['create_sample_ms'] = str(create_times)
            t['fwd_sample_ms'] = str(forward_times)
        import gc
        for _ in range(5):
            gc.collect()
        blobs_after = live_blob_count()
        mem_after = total_allocated_bytes()
        t['delta_blobs'] = blobs_after - blobs_before
        t['delta_mem'] = mem_after - mem_before
        # After GC, blob count should be stable (small tolerance for Python ref cycles)
        assert blobs_after - blobs_before <= 2, \
            f"Net leak: +{blobs_after - blobs_before} Blobs after {n_nets} create/destroy cycles"

    def test_multi_net_parallel_usage(self, ptrace):
        """Multiple independent Net objects coexist and forward correctly."""
        import time as _time
        n_nets = 5
        nets = []
        with ptrace(f"create {n_nets} independent nets") as t:
            for i in range(n_nets):
                proto = f"""name: "net_{i}"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: 2 dim: 2 }} }} }}
layer {{ name: "ip" type: "InnerProduct" bottom: "data" top: "ip" inner_product_param {{ num_output: 2 bias_term: true }} }}
layer {{ name: "prob" type: "Softmax" bottom: "ip" top: "prob" }}"""
                net = net_from_param(net_param_from_string(proto))
                _load_identity_weights(net, rng=np.random.RandomState(i * 100))
                nets.append(net)
            t['nets'] = len(nets)

        inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        results = []
        fwd_times = []
        with ptrace(f"forward {n_nets} nets independently") as t:
            for i, net in enumerate(nets):
                _t0 = _time.perf_counter()
                out = net.forward({"data": inp})
                _dt = (_time.perf_counter() - _t0) * 1000.0
                results.append(out["prob"])
                fwd_times.append(round(_dt, 3))
                assert out["prob"].shape == (2, 2)
            t['results'] = len(results)
            t['fwd_times_ms'] = str(fwd_times)

        # All outputs should be valid probability distributions
        for i, prob in enumerate(results):
            np.testing.assert_allclose(
                prob.sum(axis=1), np.ones(2), rtol=1e-5,
                err_msg=f"net[{i}] probabilities don't sum to 1",
            )
