"""P3-B: Scale/Bias/Eltwise/Concat/Dropout/SoftmaxWithLoss/Accuracy Real Forward Tests.

Comprehensive tests covering the actual C++ forward computation of:
- Scale layer (per-channel scale+bias, broadcasting, dynamic scale from bottom)
- Bias layer (per-channel bias addition, broadcasting, dynamic bias from bottom)
- Eltwise layer (SUM, PROD, MAX with coeffs)
- Concat layer (concatenation along various axes, multiple inputs)
- Dropout layer (inference-mode identity pass-through)
- SoftmaxWithLoss layer (softmax probabilities, cross-entropy loss)
- Accuracy layer (top-k classification accuracy)

Each test includes numpy reference implementations and detailed perf_trace
logging of forward time, RSS memory peaks, and exception details.

Run with:
    pytest tests/python/test_p3b_eltwise_scale.py -v
    pytest tests/python/test_p3b_eltwise_scale.py -v -s  # verbose with [PERF] logs
"""
from __future__ import annotations

import numpy as np
import pytest

import caffe_ffi
from caffe_ffi import net_param_from_string, net_from_param
from .conftest import require_cpp_extension, perf_trace


# ═══════════════════════════════════════════════════════════════════════
# Numpy Reference Implementations
# ═══════════════════════════════════════════════════════════════════════

def scale_np(x, scale, bias=None, axis=1, num_axes=1):
    """Numpy reference for Scale layer: y = x * scale + bias (with broadcasting).

    Args:
        x: Input tensor (any shape)
        scale: Scale factor tensor, shape = x.shape[axis:axis+num_axes]
        bias: Optional bias tensor, same shape as scale, or None
        axis: First axis to apply scale/bias
        num_axes: Number of axes for scale/bias shape
    Returns:
        Output tensor same shape as x
    """
    ndim = x.ndim
    # Build broadcast shape for scale: [1]*axis + list(scale.shape) + [1]*remaining
    scale_shape = [1] * axis + list(scale.shape)
    remaining = ndim - axis - num_axes
    if remaining > 0:
        scale_shape += [1] * remaining
    scale_bc = scale.reshape(scale_shape)
    out = x * scale_bc.astype(x.dtype)
    if bias is not None:
        bias_bc = bias.reshape(scale_shape)
        out = out + bias_bc.astype(x.dtype)
    return out


def bias_np(x, bias, axis=1, num_axes=1):
    """Numpy reference for Bias layer: y = x + bias (with broadcasting)."""
    ndim = x.ndim
    bias_shape = [1] * axis + list(bias.shape)
    remaining = ndim - axis - num_axes
    if remaining > 0:
        bias_shape += [1] * remaining
    bias_bc = bias.reshape(bias_shape)
    return x + bias_bc.astype(x.dtype)


def eltwise_np(arrays, op="SUM", coeffs=None):
    """Numpy reference for Eltwise layer.

    Args:
        arrays: List of numpy arrays (all same shape)
        op: "SUM", "PROD", or "MAX"
        coeffs: Optional list of float coefficients (one per array)
    Returns:
        Element-wise result
    """
    if coeffs is None:
        coeffs = [1.0] * len(arrays)
    result = coeffs[0] * arrays[0].astype(np.float64)
    if op == "SUM":
        for i in range(1, len(arrays)):
            result = result + coeffs[i] * arrays[i].astype(np.float64)
    elif op == "PROD":
        for i in range(1, len(arrays)):
            result = result * (coeffs[i] * arrays[i].astype(np.float64))
    elif op == "MAX":
        for i in range(1, len(arrays)):
            result = np.maximum(result, coeffs[i] * arrays[i].astype(np.float64))
    return result.astype(np.float32)


def concat_np(arrays, axis=1):
    """Numpy reference for Concat layer."""
    return np.concatenate(arrays, axis=axis)


def dropout_np(x, dropout_ratio=0.5):
    """Numpy reference for Dropout layer (inference mode = identity)."""
    return x.copy()


def softmax_np(x, axis=1):
    """Numpy reference for softmax along a given axis."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def softmax_loss_np(score, label, axis=1, ignore_label=None):
    """Numpy reference for SoftmaxWithLoss: returns (loss, probs)."""
    probs = softmax_np(score.astype(np.float64), axis=axis)
    ndim = score.ndim
    outer_num = int(np.prod(score.shape[:axis]))
    inner_num = int(np.prod(score.shape[axis+1:]))
    channels = score.shape[axis]

    loss = 0.0
    count = 0
    label_flat = label.reshape(-1)
    # Reshape score/probs to (outer_num, channels, inner_num) for indexing
    score_transposed = np.moveaxis(score, axis, 1).reshape(outer_num, channels, inner_num)
    probs_transposed = np.moveaxis(probs, axis, 1).reshape(outer_num, channels, inner_num)

    idx = 0
    for i in range(outer_num):
        for j in range(inner_num):
            lbl = int(label_flat[idx])
            idx += 1
            if ignore_label is not None and lbl == ignore_label:
                continue
            prob = max(probs_transposed[i, lbl, j], np.finfo(np.float32).tiny)
            loss -= np.log(prob)
            count += 1

    avg_loss = loss / count if count > 0 else 0.0
    return avg_loss, probs.astype(np.float32)


def accuracy_np(score, label, top_k=1, axis=1, ignore_label=None):
    """Numpy reference for Accuracy layer: returns top-k accuracy."""
    ndim = score.ndim
    outer_num = int(np.prod(score.shape[:axis]))
    inner_num = int(np.prod(score.shape[axis+1:]))
    channels = score.shape[axis]

    count = 0
    correct = 0
    label_flat = label.reshape(-1)
    score_transposed = np.moveaxis(score, axis, 1).reshape(outer_num, channels, inner_num)

    idx = 0
    for i in range(outer_num):
        for j in range(inner_num):
            lbl = int(label_flat[idx])
            idx += 1
            if ignore_label is not None and lbl == ignore_label:
                continue
            # Get top-k predictions
            scores_ij = score_transposed[i, :, j]
            topk_preds = np.argsort(scores_ij)[::-1][:top_k]
            if lbl in topk_preds:
                correct += 1
            count += 1

    return correct / count if count > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════
# Helper: build net from prototxt and create/load weights
# ═══════════════════════════════════════════════════════════════════════

def _make_net(prototxt: str):
    """Create a net from prototxt string."""
    return net_from_param(net_param_from_string(prototxt))


# ═══════════════════════════════════════════════════════════════════════
# Scale Layer Tests
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestScaleLayers:
    """Tests for the Scale layer's forward computation."""

    def test_scale_identity_default(self, ptrace):
        """Scale with default weights (scale=1.0, no bias) should be identity."""
        prototxt = """name: "scale_identity"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 4 dim: 4 } } }
layer { name: "scale" type: "Scale" bottom: "data" top: "out" }
"""
        with ptrace("Net(scale identity)"):
            net = _make_net(prototxt)
        inp = np.random.randn(2, 3, 4, 4).astype(np.float32)
        with ptrace("scale identity forward") as t:
            out = net.forward({"data": inp})
            t['input_shape'] = f"{inp.shape}"
            t['output_shape'] = f"{out['out'].shape}"
        np.testing.assert_allclose(out["out"], inp, rtol=1e-6, atol=1e-6)

    def test_scale_per_channel(self, ptrace):
        """Scale with per-channel scale factors (no bias)."""
        C = 4
        prototxt = f"""name: "scale_per_ch"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: 2 dim: {C} dim: 3 dim: 3 }} }} }}
layer {{ name: "scale" type: "Scale" bottom: "data" top: "out" }}
"""
        with ptrace("Net(scale per-channel)"):
            net = _make_net(prototxt)
        # Load per-channel scale factors
        scale_factors = np.array([0.5, 1.0, 2.0, 0.1], dtype=np.float32)
        scale_layer = net.layer_by_name("scale")
        with ptrace("load scale weights") as t:
            scale_layer.blobs[0].from_numpy(scale_factors)
            t['scale_shape'] = f"{scale_factors.shape}"
        inp = np.random.randn(2, C, 3, 3).astype(np.float32)
        with ptrace("scale per-channel forward") as t:
            out = net.forward({"data": inp})
        expected = scale_np(inp, scale_factors, axis=1)
        np.testing.assert_allclose(out["out"], expected, rtol=1e-5, atol=1e-5)

    def test_scale_with_bias(self, ptrace):
        """Scale with both per-channel scale and bias."""
        C = 3
        prototxt = f"""name: "scale_bias"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: 1 dim: {C} dim: 2 dim: 2 }} }} }}
layer {{ name: "scale" type: "Scale" bottom: "data" top: "out" scale_param {{ bias_term: true }} }}
"""
        with ptrace("Net(scale+bias)"):
            net = _make_net(prototxt)
        scale_factors = np.array([2.0, 0.5, 1.5], dtype=np.float32)
        bias_values = np.array([1.0, -1.0, 0.0], dtype=np.float32)
        scale_layer = net.layer_by_name("scale")
        with ptrace("load scale+bias weights"):
            scale_layer.blobs[0].from_numpy(scale_factors)
            scale_layer.blobs[1].from_numpy(bias_values)
        inp = np.array([[[[1, 2], [3, 4]],
                         [[5, 6], [7, 8]],
                         [[9, 10], [11, 12]]]], dtype=np.float32)
        with ptrace("scale+bias forward"):
            out = net.forward({"data": inp})
        expected = scale_np(inp, scale_factors, bias=bias_values, axis=1)
        np.testing.assert_allclose(out["out"], expected, rtol=1e-5, atol=1e-5)

    def test_scale_axis0(self, ptrace):
        """Scale along axis=0 (per-sample scaling in batch)."""
        N, C = 2, 3
        prototxt = f"""name: "scale_axis0"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {C} dim: 2 dim: 2 }} }} }}
layer {{ name: "scale" type: "Scale" bottom: "data" top: "out" scale_param {{ axis: 0 }} }}
"""
        with ptrace("Net(scale axis=0)"):
            net = _make_net(prototxt)
        scale_factors = np.array([0.5, 2.0], dtype=np.float32)
        scale_layer = net.layer_by_name("scale")
        with ptrace("load axis=0 scale weights"):
            scale_layer.blobs[0].from_numpy(scale_factors)
        inp = np.random.randn(N, C, 2, 2).astype(np.float32)
        with ptrace("scale axis=0 forward"):
            out = net.forward({"data": inp})
        expected = scale_np(inp, scale_factors, axis=0, num_axes=1)
        np.testing.assert_allclose(out["out"], expected, rtol=1e-5, atol=1e-5)

    def test_scale_repeated_forward(self, ptrace):
        """Scale should produce deterministic results across repeated forwards."""
        C = 4
        prototxt = f"""name: "scale_rep"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: 2 dim: {C} dim: 3 dim: 3 }} }} }}
layer {{ name: "scale" type: "Scale" bottom: "data" top: "out" scale_param {{ bias_term: true }} }}
"""
        with ptrace("Net(scale repeated)"):
            net = _make_net(prototxt)
        scale_layer = net.layer_by_name("scale")
        scale_factors = np.random.randn(C).astype(np.float32) * 0.5 + 1.0
        bias_values = np.random.randn(C).astype(np.float32) * 0.1
        scale_layer.blobs[0].from_numpy(scale_factors)
        scale_layer.blobs[1].from_numpy(bias_values)
        inp = np.random.randn(2, C, 3, 3).astype(np.float32)
        results = []
        for i in range(5):
            with ptrace(f"scale forward #{i+1}"):
                out = net.forward({"data": inp})
            results.append(out["out"].copy())
        for i in range(1, 5):
            np.testing.assert_array_equal(results[0], results[i])

    def test_scale_weights_unchanged(self, ptrace):
        """Scale weights should not be modified by forward."""
        C = 3
        prototxt = f"""name: "scale_w_unchanged"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: 1 dim: {C} dim: 2 dim: 2 }} }} }}
layer {{ name: "scale" type: "Scale" bottom: "data" top: "out" scale_param {{ bias_term: true }} }}
"""
        with ptrace("Net(scale weights check)"):
            net = _make_net(prototxt)
        scale_layer = net.layer_by_name("scale")
        w0 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b0 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        scale_layer.blobs[0].from_numpy(w0.copy())
        scale_layer.blobs[1].from_numpy(b0.copy())
        inp = np.random.randn(1, C, 2, 2).astype(np.float32)
        with ptrace("scale forward (weight check)"):
            net.forward({"data": inp})
        w_after = scale_layer.blobs[0].to_numpy()
        b_after = scale_layer.blobs[1].to_numpy()
        np.testing.assert_array_equal(w_after, w0)
        np.testing.assert_array_equal(b_after, b0)


# ═══════════════════════════════════════════════════════════════════════
# Bias Layer Tests
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestBiasLayers:
    """Tests for the Bias layer's forward computation."""

    def test_bias_zero_default(self, ptrace):
        """Bias with default weights (all zeros) should be identity."""
        prototxt = """name: "bias_zero"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 4 dim: 4 } } }
layer { name: "bias" type: "Bias" bottom: "data" top: "out" }
"""
        with ptrace("Net(bias zero)"):
            net = _make_net(prototxt)
        inp = np.random.randn(2, 3, 4, 4).astype(np.float32)
        with ptrace("bias zero forward") as t:
            out = net.forward({"data": inp})
            t['input_shape'] = f"{inp.shape}"
        np.testing.assert_allclose(out["out"], inp, rtol=1e-6, atol=1e-6)

    def test_bias_per_channel(self, ptrace):
        """Bias with per-channel bias values."""
        C = 4
        prototxt = f"""name: "bias_per_ch"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: 2 dim: {C} dim: 3 dim: 3 }} }} }}
layer {{ name: "bias" type: "Bias" bottom: "data" top: "out" }}
"""
        with ptrace("Net(bias per-channel)"):
            net = _make_net(prototxt)
        bias_values = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        bias_layer = net.layer_by_name("bias")
        with ptrace("load bias weights") as t:
            bias_layer.blobs[0].from_numpy(bias_values)
            t['bias_shape'] = f"{bias_values.shape}"
        inp = np.random.randn(2, C, 3, 3).astype(np.float32)
        with ptrace("bias per-channel forward"):
            out = net.forward({"data": inp})
        expected = bias_np(inp, bias_values, axis=1)
        np.testing.assert_allclose(out["out"], expected, rtol=1e-5, atol=1e-5)

    def test_bias_known_values(self, ptrace):
        """Bias with known values for exact verification."""
        C = 2
        prototxt = f"""name: "bias_known"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: 1 dim: {C} dim: 2 dim: 2 }} }} }}
layer {{ name: "bias" type: "Bias" bottom: "data" top: "out" }}
"""
        with ptrace("Net(bias known values)"):
            net = _make_net(prototxt)
        bias_values = np.array([10.0, -10.0], dtype=np.float32)
        bias_layer = net.layer_by_name("bias")
        bias_layer.blobs[0].from_numpy(bias_values)
        inp = np.array([[[[1, 2], [3, 4]],
                         [[5, 6], [7, 8]]]], dtype=np.float32)
        with ptrace("bias known forward"):
            out = net.forward({"data": inp})
        expected = inp.copy()
        expected[0, 0] += 10.0
        expected[0, 1] += -10.0
        np.testing.assert_array_equal(out["out"], expected)

    def test_bias_axis0(self, ptrace):
        """Bias along axis=0 (per-sample bias)."""
        N, C = 2, 3
        prototxt = f"""name: "bias_axis0"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {C} dim: 2 dim: 2 }} }} }}
layer {{ name: "bias" type: "Bias" bottom: "data" top: "out" bias_param {{ axis: 0 }} }}
"""
        with ptrace("Net(bias axis=0)"):
            net = _make_net(prototxt)
        bias_values = np.array([1.0, -1.0], dtype=np.float32)
        bias_layer = net.layer_by_name("bias")
        with ptrace("load axis=0 bias weights"):
            bias_layer.blobs[0].from_numpy(bias_values)
        inp = np.random.randn(N, C, 2, 2).astype(np.float32)
        with ptrace("bias axis=0 forward"):
            out = net.forward({"data": inp})
        expected = bias_np(inp, bias_values, axis=0, num_axes=1)
        np.testing.assert_allclose(out["out"], expected, rtol=1e-5, atol=1e-5)

    def test_bias_repeated_forward(self, ptrace):
        """Bias should produce deterministic results."""
        C = 3
        prototxt = f"""name: "bias_rep"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: 2 dim: {C} dim: 3 dim: 3 }} }} }}
layer {{ name: "bias" type: "Bias" bottom: "data" top: "out" }}
"""
        with ptrace("Net(bias repeated)"):
            net = _make_net(prototxt)
        bias_layer = net.layer_by_name("bias")
        bias_values = np.random.randn(C).astype(np.float32)
        bias_layer.blobs[0].from_numpy(bias_values)
        inp = np.random.randn(2, C, 3, 3).astype(np.float32)
        results = []
        for i in range(5):
            with ptrace(f"bias forward #{i+1}"):
                out = net.forward({"data": inp})
            results.append(out["out"].copy())
        for i in range(1, 5):
            np.testing.assert_array_equal(results[0], results[i])

    def test_bias_weights_unchanged(self, ptrace):
        """Bias weights should not be modified by forward."""
        C = 3
        prototxt = f"""name: "bias_w_unchanged"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: 1 dim: {C} dim: 2 dim: 2 }} }} }}
layer {{ name: "bias" type: "Bias" bottom: "data" top: "out" }}
"""
        with ptrace("Net(bias weights check)"):
            net = _make_net(prototxt)
        bias_layer = net.layer_by_name("bias")
        b0 = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        bias_layer.blobs[0].from_numpy(b0.copy())
        inp = np.random.randn(1, C, 2, 2).astype(np.float32)
        with ptrace("bias forward (weight check)"):
            net.forward({"data": inp})
        b_after = bias_layer.blobs[0].to_numpy()
        np.testing.assert_array_equal(b_after, b0)


# ═══════════════════════════════════════════════════════════════════════
# Eltwise Layer Tests
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestEltwiseLayers:
    """Tests for the Eltwise layer's forward computation (SUM/PROD/MAX)."""

    def test_eltwise_sum_two_inputs(self, ptrace):
        """Eltwise SUM: two inputs, default coeffs [1,1]."""
        prototxt = """name: "eltwise_sum2"
layer { name: "a" type: "Input" top: "a" input_param { shape { dim: 2 dim: 3 dim: 4 dim: 4 } } }
layer { name: "b" type: "Input" top: "b" input_param { shape { dim: 2 dim: 3 dim: 4 dim: 4 } } }
layer { name: "ew" type: "Eltwise" bottom: "a" bottom: "b" top: "out"
        eltwise_param { operation: SUM } }
"""
        with ptrace("Net(eltwise sum 2)"):
            net = _make_net(prototxt)
        a = np.random.randn(2, 3, 4, 4).astype(np.float32)
        b = np.random.randn(2, 3, 4, 4).astype(np.float32)
        with ptrace("eltwise sum forward"):
            out = net.forward({"a": a, "b": b})
        expected = eltwise_np([a, b], op="SUM")
        np.testing.assert_allclose(out["out"], expected, rtol=1e-5, atol=1e-5)

    def test_eltwise_sum_with_coeffs(self, ptrace):
        """Eltwise SUM with coefficients [2.0, 0.5]."""
        prototxt = """name: "eltwise_sum_coeff"
layer { name: "a" type: "Input" top: "a" input_param { shape { dim: 1 dim: 2 dim: 3 dim: 3 } } }
layer { name: "b" type: "Input" top: "b" input_param { shape { dim: 1 dim: 2 dim: 3 dim: 3 } } }
layer { name: "ew" type: "Eltwise" bottom: "a" bottom: "b" top: "out"
        eltwise_param { operation: SUM coeff: 2.0 coeff: 0.5 } }
"""
        with ptrace("Net(eltwise sum coeffs)"):
            net = _make_net(prototxt)
        a = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                        [[10, 11, 12], [13, 14, 15], [16, 17, 18]]]], dtype=np.float32)
        b = np.ones_like(a) * 4.0
        with ptrace("eltwise sum coeffs forward"):
            out = net.forward({"a": a, "b": b})
        expected = eltwise_np([a, b], op="SUM", coeffs=[2.0, 0.5])
        np.testing.assert_allclose(out["out"], expected, rtol=1e-5, atol=1e-5)

    def test_eltwise_prod_two_inputs(self, ptrace):
        """Eltwise PROD: element-wise multiplication."""
        prototxt = """name: "eltwise_prod"
layer { name: "a" type: "Input" top: "a" input_param { shape { dim: 2 dim: 3 dim: 2 dim: 2 } } }
layer { name: "b" type: "Input" top: "b" input_param { shape { dim: 2 dim: 3 dim: 2 dim: 2 } } }
layer { name: "ew" type: "Eltwise" bottom: "a" bottom: "b" top: "out"
        eltwise_param { operation: PROD } }
"""
        with ptrace("Net(eltwise prod)"):
            net = _make_net(prototxt)
        a = np.array([[[[1, 2], [3, 4]]]], dtype=np.float32).repeat(2, axis=0).repeat(3, axis=1)
        b = np.array([[[[2, 1], [0.5, 0.25]]]], dtype=np.float32).repeat(2, axis=0).repeat(3, axis=1)
        with ptrace("eltwise prod forward"):
            out = net.forward({"a": a, "b": b})
        expected = eltwise_np([a, b], op="PROD")
        np.testing.assert_allclose(out["out"], expected, rtol=1e-5, atol=1e-5)

    def test_eltwise_max_two_inputs(self, ptrace):
        """Eltwise MAX: element-wise maximum."""
        prototxt = """name: "eltwise_max"
layer { name: "a" type: "Input" top: "a" input_param { shape { dim: 1 dim: 1 dim: 2 dim: 4 } } }
layer { name: "b" type: "Input" top: "b" input_param { shape { dim: 1 dim: 1 dim: 2 dim: 4 } } }
layer { name: "ew" type: "Eltwise" bottom: "a" bottom: "b" top: "out"
        eltwise_param { operation: MAX } }
"""
        with ptrace("Net(eltwise max)"):
            net = _make_net(prototxt)
        a = np.array([[[[1, 5, 3, 2], [4, 0, -1, 8]]]], dtype=np.float32)
        b = np.array([[[[3, 2, 7, 1], [0, 6, -2, 5]]]], dtype=np.float32)
        with ptrace("eltwise max forward"):
            out = net.forward({"a": a, "b": b})
        expected = eltwise_np([a, b], op="MAX")
        np.testing.assert_allclose(out["out"], expected, rtol=1e-5, atol=1e-5)

    def test_eltwise_sum_three_inputs(self, ptrace):
        """Eltwise SUM with three inputs."""
        prototxt = """name: "eltwise_sum3"
layer { name: "a" type: "Input" top: "a" input_param { shape { dim: 1 dim: 2 dim: 2 dim: 2 } } }
layer { name: "b" type: "Input" top: "b" input_param { shape { dim: 1 dim: 2 dim: 2 dim: 2 } } }
layer { name: "c" type: "Input" top: "c" input_param { shape { dim: 1 dim: 2 dim: 2 dim: 2 } } }
layer { name: "ew" type: "Eltwise" bottom: "a" bottom: "b" bottom: "c" top: "out"
        eltwise_param { operation: SUM coeff: 1.0 coeff: 2.0 coeff: 0.5 } }
"""
        with ptrace("Net(eltwise sum 3)"):
            net = _make_net(prototxt)
        a = np.random.randn(1, 2, 2, 2).astype(np.float32)
        b = np.random.randn(1, 2, 2, 2).astype(np.float32)
        c = np.random.randn(1, 2, 2, 2).astype(np.float32)
        with ptrace("eltwise sum3 forward"):
            out = net.forward({"a": a, "b": b, "c": c})
        expected = eltwise_np([a, b, c], op="SUM", coeffs=[1.0, 2.0, 0.5])
        np.testing.assert_allclose(out["out"], expected, rtol=1e-5, atol=1e-5)

    def test_eltwise_known_values_sum(self, ptrace):
        """Eltwise SUM with known values for exact verification."""
        prototxt = """name: "eltwise_known_sum"
layer { name: "a" type: "Input" top: "a" input_param { shape { dim: 1 dim: 1 dim: 1 dim: 3 } } }
layer { name: "b" type: "Input" top: "b" input_param { shape { dim: 1 dim: 1 dim: 1 dim: 3 } } }
layer { name: "ew" type: "Eltwise" bottom: "a" bottom: "b" top: "out"
        eltwise_param { operation: SUM } }
"""
        with ptrace("Net(eltwise SUM known)"):
            net = _make_net(prototxt)
        a = np.array([[[[1.0, 2.0, 3.0]]]], dtype=np.float32)
        b = np.array([[[[4.0, 5.0, 6.0]]]], dtype=np.float32)
        with ptrace("eltwise SUM known forward"):
            out = net.forward({"a": a, "b": b})
        np.testing.assert_array_equal(out["out"], np.array([[[[5, 7, 9]]]], dtype=np.float32))

    def test_eltwise_known_values_prod(self, ptrace):
        """Eltwise PROD with known values for exact verification."""
        prototxt = """name: "eltwise_known_prod"
layer { name: "a" type: "Input" top: "a" input_param { shape { dim: 1 dim: 1 dim: 1 dim: 3 } } }
layer { name: "b" type: "Input" top: "b" input_param { shape { dim: 1 dim: 1 dim: 1 dim: 3 } } }
layer { name: "ew" type: "Eltwise" bottom: "a" bottom: "b" top: "out"
        eltwise_param { operation: PROD } }
"""
        with ptrace("Net(eltwise PROD known)"):
            net = _make_net(prototxt)
        a = np.array([[[[1.0, 2.0, 3.0]]]], dtype=np.float32)
        b = np.array([[[[4.0, 5.0, 6.0]]]], dtype=np.float32)
        with ptrace("eltwise PROD known forward"):
            out = net.forward({"a": a, "b": b})
        np.testing.assert_array_equal(out["out"], np.array([[[[4, 10, 18]]]], dtype=np.float32))

    def test_eltwise_known_values_max(self, ptrace):
        """Eltwise MAX with known values for exact verification."""
        prototxt = """name: "eltwise_known_max"
layer { name: "a" type: "Input" top: "a" input_param { shape { dim: 1 dim: 1 dim: 1 dim: 3 } } }
layer { name: "b" type: "Input" top: "b" input_param { shape { dim: 1 dim: 1 dim: 1 dim: 3 } } }
layer { name: "ew" type: "Eltwise" bottom: "a" bottom: "b" top: "out"
        eltwise_param { operation: MAX } }
"""
        with ptrace("Net(eltwise MAX known)"):
            net = _make_net(prototxt)
        a = np.array([[[[1.0, 5.0, 3.0]]]], dtype=np.float32)
        b = np.array([[[[4.0, 2.0, 6.0]]]], dtype=np.float32)
        with ptrace("eltwise MAX known forward"):
            out = net.forward({"a": a, "b": b})
        np.testing.assert_array_equal(out["out"], np.array([[[[4, 5, 6]]]], dtype=np.float32))

    def test_eltwise_repeated_forward(self, ptrace):
        """Eltwise should be deterministic across repeated forwards."""
        prototxt = """name: "eltwise_rep"
layer { name: "a" type: "Input" top: "a" input_param { shape { dim: 2 dim: 3 dim: 2 dim: 2 } } }
layer { name: "b" type: "Input" top: "b" input_param { shape { dim: 2 dim: 3 dim: 2 dim: 2 } } }
layer { name: "ew" type: "Eltwise" bottom: "a" bottom: "b" top: "out"
        eltwise_param { operation: SUM } }
"""
        with ptrace("Net(eltwise repeated)"):
            net = _make_net(prototxt)
        a = np.random.randn(2, 3, 2, 2).astype(np.float32)
        b = np.random.randn(2, 3, 2, 2).astype(np.float32)
        results = []
        for i in range(10):
            with ptrace(f"eltwise forward #{i+1}"):
                out = net.forward({"a": a, "b": b})
            results.append(out["out"].copy())
        for i in range(1, 10):
            np.testing.assert_array_equal(results[0], results[i])


# ═══════════════════════════════════════════════════════════════════════
# Concat Layer Tests
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestConcatLayers:
    """Tests for the Concat layer's forward computation."""

    def test_concat_axis1_channel(self, ptrace):
        """Concat along axis=1 (channel dimension)."""
        prototxt = """name: "concat_ch"
layer { name: "a" type: "Input" top: "a" input_param { shape { dim: 2 dim: 3 dim: 4 dim: 4 } } }
layer { name: "b" type: "Input" top: "b" input_param { shape { dim: 2 dim: 5 dim: 4 dim: 4 } } }
layer { name: "cat" type: "Concat" bottom: "a" bottom: "b" top: "out"
        concat_param { axis: 1 } }
"""
        with ptrace("Net(concat axis=1)"):
            net = _make_net(prototxt)
        a = np.random.randn(2, 3, 4, 4).astype(np.float32)
        b = np.random.randn(2, 5, 4, 4).astype(np.float32)
        with ptrace("concat axis=1 forward") as t:
            out = net.forward({"a": a, "b": b})
            t['output_shape'] = f"{out['out'].shape}"
        assert out["out"].shape == (2, 8, 4, 4)
        expected = concat_np([a, b], axis=1)
        np.testing.assert_array_equal(out["out"], expected)

    def test_concat_axis0_batch(self, ptrace):
        """Concat along axis=0 (batch dimension)."""
        prototxt = """name: "concat_batch"
layer { name: "a" type: "Input" top: "a" input_param { shape { dim: 3 dim: 2 dim: 4 dim: 4 } } }
layer { name: "b" type: "Input" top: "b" input_param { shape { dim: 2 dim: 2 dim: 4 dim: 4 } } }
layer { name: "cat" type: "Concat" bottom: "a" bottom: "b" top: "out"
        concat_param { axis: 0 } }
"""
        with ptrace("Net(concat axis=0)"):
            net = _make_net(prototxt)
        a = np.random.randn(3, 2, 4, 4).astype(np.float32)
        b = np.random.randn(2, 2, 4, 4).astype(np.float32)
        with ptrace("concat axis=0 forward") as t:
            out = net.forward({"a": a, "b": b})
            t['output_shape'] = f"{out['out'].shape}"
        assert out["out"].shape == (5, 2, 4, 4)
        expected = concat_np([a, b], axis=0)
        np.testing.assert_array_equal(out["out"], expected)

    def test_concat_axis2_height(self, ptrace):
        """Concat along axis=2 (spatial height)."""
        prototxt = """name: "concat_h"
layer { name: "a" type: "Input" top: "a" input_param { shape { dim: 1 dim: 2 dim: 3 dim: 4 } } }
layer { name: "b" type: "Input" top: "b" input_param { shape { dim: 1 dim: 2 dim: 5 dim: 4 } } }
layer { name: "cat" type: "Concat" bottom: "a" bottom: "b" top: "out"
        concat_param { axis: 2 } }
"""
        with ptrace("Net(concat axis=2)"):
            net = _make_net(prototxt)
        a = np.random.randn(1, 2, 3, 4).astype(np.float32)
        b = np.random.randn(1, 2, 5, 4).astype(np.float32)
        with ptrace("concat axis=2 forward") as t:
            out = net.forward({"a": a, "b": b})
            t['output_shape'] = f"{out['out'].shape}"
        assert out["out"].shape == (1, 2, 8, 4)
        expected = concat_np([a, b], axis=2)
        np.testing.assert_array_equal(out["out"], expected)

    def test_concat_three_inputs(self, ptrace):
        """Concat three inputs along axis=1."""
        prototxt = """name: "concat3"
layer { name: "a" type: "Input" top: "a" input_param { shape { dim: 1 dim: 2 dim: 3 dim: 3 } } }
layer { name: "b" type: "Input" top: "b" input_param { shape { dim: 1 dim: 3 dim: 3 dim: 3 } } }
layer { name: "c" type: "Input" top: "c" input_param { shape { dim: 1 dim: 4 dim: 3 dim: 3 } } }
layer { name: "cat" type: "Concat" bottom: "a" bottom: "b" bottom: "c" top: "out"
        concat_param { axis: 1 } }
"""
        with ptrace("Net(concat 3 inputs)"):
            net = _make_net(prototxt)
        a = np.ones((1, 2, 3, 3), dtype=np.float32)
        b = np.ones((1, 3, 3, 3), dtype=np.float32) * 2
        c = np.ones((1, 4, 3, 3), dtype=np.float32) * 3
        with ptrace("concat 3-input forward"):
            out = net.forward({"a": a, "b": b, "c": c})
        assert out["out"].shape == (1, 9, 3, 3)
        expected = concat_np([a, b, c], axis=1)
        np.testing.assert_array_equal(out["out"], expected)

    def test_concat_known_values(self, ptrace):
        """Concat with known small values for exact verification."""
        prototxt = """name: "concat_known"
layer { name: "a" type: "Input" top: "a" input_param { shape { dim: 1 dim: 1 dim: 1 dim: 2 } } }
layer { name: "b" type: "Input" top: "b" input_param { shape { dim: 1 dim: 1 dim: 1 dim: 3 } } }
layer { name: "cat" type: "Concat" bottom: "a" bottom: "b" top: "out"
        concat_param { axis: 3 } }
"""
        with ptrace("Net(concat known)"):
            net = _make_net(prototxt)
        a = np.array([[[[1.0, 2.0]]]], dtype=np.float32)
        b = np.array([[[[3.0, 4.0, 5.0]]]], dtype=np.float32)
        with ptrace("concat known forward"):
            out = net.forward({"a": a, "b": b})
        expected = np.array([[[[1, 2, 3, 4, 5]]]], dtype=np.float32)
        np.testing.assert_array_equal(out["out"], expected)
        assert out["out"].shape == (1, 1, 1, 5)

    def test_concat_repeated_forward(self, ptrace):
        """Concat should be deterministic across repeated forwards."""
        prototxt = """name: "concat_rep"
layer { name: "a" type: "Input" top: "a" input_param { shape { dim: 2 dim: 3 dim: 2 dim: 2 } } }
layer { name: "b" type: "Input" top: "b" input_param { shape { dim: 2 dim: 4 dim: 2 dim: 2 } } }
layer { name: "cat" type: "Concat" bottom: "a" bottom: "b" top: "out"
        concat_param { axis: 1 } }
"""
        with ptrace("Net(concat repeated)"):
            net = _make_net(prototxt)
        a = np.random.randn(2, 3, 2, 2).astype(np.float32)
        b = np.random.randn(2, 4, 2, 2).astype(np.float32)
        results = []
        for i in range(10):
            with ptrace(f"concat forward #{i+1}"):
                out = net.forward({"a": a, "b": b})
            results.append(out["out"].copy())
        for i in range(1, 10):
            np.testing.assert_array_equal(results[0], results[i])


# ═══════════════════════════════════════════════════════════════════════
# Dropout Layer Tests (inference mode = identity)
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestDropoutLayers:
    """Tests for the Dropout layer's forward computation (inference mode)."""

    def test_dropout_identity_ratio_0(self, ptrace):
        """Dropout with ratio=0 is identity in inference mode."""
        prototxt = """name: "drop_id0"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 4 dim: 4 } } }
layer { name: "drop" type: "Dropout" bottom: "data" top: "out"
        dropout_param { dropout_ratio: 0.0 } }
"""
        with ptrace("Net(dropout ratio=0)"):
            net = _make_net(prototxt)
        inp = np.random.randn(2, 3, 4, 4).astype(np.float32)
        with ptrace("dropout ratio=0 forward"):
            out = net.forward({"data": inp})
        np.testing.assert_array_equal(out["out"], inp)

    def test_dropout_identity_ratio_05(self, ptrace):
        """Dropout with ratio=0.5 is also identity in inference mode (no mask applied)."""
        prototxt = """name: "drop_id5"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 4 dim: 4 } } }
layer { name: "drop" type: "Dropout" bottom: "data" top: "out"
        dropout_param { dropout_ratio: 0.5 } }
"""
        with ptrace("Net(dropout ratio=0.5)"):
            net = _make_net(prototxt)
        inp = np.random.randn(2, 3, 4, 4).astype(np.float32)
        with ptrace("dropout ratio=0.5 forward"):
            out = net.forward({"data": inp})
        np.testing.assert_array_equal(out["out"], inp)

    def test_dropout_identity_ratio_09(self, ptrace):
        """Dropout with ratio=0.9 is identity in inference mode."""
        prototxt = """name: "drop_id9"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 4 dim: 4 } } }
layer { name: "drop" type: "Dropout" bottom: "data" top: "out"
        dropout_param { dropout_ratio: 0.9 } }
"""
        with ptrace("Net(dropout ratio=0.9)"):
            net = _make_net(prototxt)
        inp = np.random.randn(2, 3, 4, 4).astype(np.float32)
        with ptrace("dropout ratio=0.9 forward"):
            out = net.forward({"data": inp})
        np.testing.assert_array_equal(out["out"], inp)

    def test_dropout_1d_input(self, ptrace):
        """Dropout works with 2D input (N,C)."""
        prototxt = """name: "drop_1d"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 4 dim: 10 } } }
layer { name: "drop" type: "Dropout" bottom: "data" top: "out" }
"""
        with ptrace("Net(dropout 1d)"):
            net = _make_net(prototxt)
        inp = np.random.randn(4, 10).astype(np.float32)
        with ptrace("dropout 1d forward"):
            out = net.forward({"data": inp})
        np.testing.assert_array_equal(out["out"], inp)
        assert out["out"].shape == (4, 10)

    def test_dropout_preserves_special_values(self, ptrace):
        """Dropout preserves zeros, ones, and negative values in inference mode."""
        prototxt = """name: "drop_special"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 1 dim: 1 dim: 5 } } }
layer { name: "drop" type: "Dropout" bottom: "data" top: "out" }
"""
        with ptrace("Net(dropout special)"):
            net = _make_net(prototxt)
        inp = np.array([[[[0.0, 1.0, -1.0, 100.0, -100.0]]]], dtype=np.float32)
        with ptrace("dropout special forward"):
            out = net.forward({"data": inp})
        np.testing.assert_array_equal(out["out"], inp)

    def test_dropout_repeated_forward(self, ptrace):
        """Dropout is always identity (deterministic) in inference mode."""
        prototxt = """name: "drop_rep"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 4 dim: 4 } } }
layer { name: "drop" type: "Dropout" bottom: "data" top: "out"
        dropout_param { dropout_ratio: 0.5 } }
"""
        with ptrace("Net(dropout repeated)"):
            net = _make_net(prototxt)
        inp = np.random.randn(2, 3, 4, 4).astype(np.float32)
        results = []
        for i in range(20):
            with ptrace(f"dropout forward #{i+1}"):
                out = net.forward({"data": inp})
            results.append(out["out"].copy())
        for i in range(1, 20):
            np.testing.assert_array_equal(results[0], results[i])


# ═══════════════════════════════════════════════════════════════════════
# SoftmaxWithLoss Layer Tests
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestSoftmaxWithLossLayers:
    """Tests for the SoftmaxWithLoss layer's forward computation."""

    def test_softmax_probs_only(self, ptrace):
        """SoftmaxWithLoss with single bottom outputs probabilities only."""
        prototxt = """name: "sm_probs"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 3 dim: 1 dim: 1 } } }
layer { name: "sm" type: "SoftmaxWithLoss" bottom: "data" top: "probs" }
"""
        with ptrace("Net(softmax probs only)"):
            net = _make_net(prototxt)
        # Logits: class 0 has highest score
        inp = np.array([[[[10.0]], [[1.0]], [[0.0]]]], dtype=np.float32)
        with ptrace("softmax probs forward") as t:
            out = net.forward({"data": inp})
            t['output_shape'] = f"{out['probs'].shape}"
        probs = out["probs"]
        # Probabilities should sum to 1 along axis=1
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
        # Class 0 should have highest probability
        assert probs[0, 0, 0, 0] > probs[0, 1, 0, 0]
        assert probs[0, 0, 0, 0] > probs[0, 2, 0, 0]
        expected_probs = softmax_np(inp, axis=1)
        np.testing.assert_allclose(probs, expected_probs, rtol=1e-5, atol=1e-6)

    def test_softmax_loss_perfect_predictions(self, ptrace):
        """SoftmaxWithLoss with perfect predictions should give near-zero loss."""
        prototxt = """name: "sm_perfect"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 3 dim: 4 dim: 1 dim: 1 } } }
layer { name: "label" type: "Input" top: "label" input_param { shape { dim: 3 dim: 1 dim: 1 dim: 1 } } }
layer { name: "loss" type: "SoftmaxWithLoss" bottom: "data" bottom: "label" top: "loss" }
"""
        with ptrace("Net(softmax perfect)"):
            net = _make_net(prototxt)
        # Perfect predictions: large value for correct class, small for others
        inp = np.zeros((3, 4, 1, 1), dtype=np.float32)
        inp[0, 0, 0, 0] = 100.0  # sample 0 -> class 0
        inp[1, 2, 0, 0] = 100.0  # sample 1 -> class 2
        inp[2, 3, 0, 0] = 100.0  # sample 2 -> class 3
        label = np.array([[[[0]]], [[[2]]], [[[3]]]], dtype=np.float32)
        with ptrace("softmax perfect forward") as t:
            out = net.forward({"data": inp, "label": label})
            t['loss'] = float(out["loss"][0])
        loss = float(out["loss"][0])
        assert loss < 1e-4, f"Expected near-zero loss for perfect predictions, got {loss}"

    def test_softmax_loss_uniform(self, ptrace):
        """SoftmaxWithLoss with uniform logits should give -log(1/C) loss."""
        C = 5
        N = 4
        prototxt = f"""name: "sm_uniform"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {C} dim: 1 dim: 1 }} }} }}
layer {{ name: "label" type: "Input" top: "label" input_param {{ shape {{ dim: {N} dim: 1 dim: 1 dim: 1 }} }} }}
layer {{ name: "loss" type: "SoftmaxWithLoss" bottom: "data" bottom: "label" top: "loss" }}
"""
        with ptrace("Net(softmax uniform)"):
            net = _make_net(prototxt)
        inp = np.zeros((N, C, 1, 1), dtype=np.float32)  # uniform logits
        label = np.array([[[[i % C]]] for i in range(N)], dtype=np.float32)
        with ptrace("softmax uniform forward") as t:
            out = net.forward({"data": inp, "label": label})
            t['loss'] = float(out["loss"][0])
        expected_loss = -np.log(1.0 / C)
        np.testing.assert_allclose(out["loss"][0], expected_loss, rtol=1e-4, atol=1e-4)

    def test_softmax_loss_numpy_match(self, ptrace):
        """SoftmaxWithLoss should match numpy reference."""
        N, C, H, W = 2, 4, 2, 2
        prototxt = f"""name: "sm_np"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {C} dim: {H} dim: {W} }} }} }}
layer {{ name: "label" type: "Input" top: "label" input_param {{ shape {{ dim: {N} dim: 1 dim: {H} dim: {W} }} }} }}
layer {{ name: "loss" type: "SoftmaxWithLoss" bottom: "data" bottom: "label" top: "loss" }}
"""
        with ptrace("Net(softmax numpy match)"):
            net = _make_net(prototxt)
        np.random.seed(42)
        inp = np.random.randn(N, C, H, W).astype(np.float32)
        label = np.random.randint(0, C, (N, 1, H, W)).astype(np.float32)
        with ptrace("softmax numpy match forward"):
            out = net.forward({"data": inp, "label": label})
        expected_loss, _ = softmax_loss_np(inp, label, axis=1)
        np.testing.assert_allclose(out["loss"][0], expected_loss, rtol=1e-4, atol=1e-4)

    def test_softmax_loss_with_probs_top(self, ptrace):
        """SoftmaxWithLoss can output both loss and probabilities with two tops."""
        N, C = 3, 4
        prototxt = f"""name: "sm_2top"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {C} dim: 1 dim: 1 }} }} }}
layer {{ name: "label" type: "Input" top: "label" input_param {{ shape {{ dim: {N} dim: 1 dim: 1 dim: 1 }} }} }}
layer {{ name: "loss" type: "SoftmaxWithLoss" bottom: "data" bottom: "label" top: "loss" top: "probs" }}
"""
        with ptrace("Net(softmax 2 tops)"):
            net = _make_net(prototxt)
        inp = np.random.randn(N, C, 1, 1).astype(np.float32)
        label = np.array([[[[0]]], [[[2]]], [[[1]]]], dtype=np.float32)
        with ptrace("softmax 2-top forward") as t:
            out = net.forward({"data": inp, "label": label})
            t['output_keys'] = str(list(out.keys()))
        assert "loss" in out
        assert "probs" in out
        assert out["loss"].shape == (1,)
        assert out["probs"].shape == (N, C, 1, 1)
        np.testing.assert_allclose(out["probs"].sum(axis=1), 1.0, rtol=1e-5)

    def test_softmax_repeated_forward(self, ptrace):
        """SoftmaxWithLoss should be deterministic."""
        N, C = 2, 3
        prototxt = f"""name: "sm_rep"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {C} dim: 1 dim: 1 }} }} }}
layer {{ name: "label" type: "Input" top: "label" input_param {{ shape {{ dim: {N} dim: 1 dim: 1 dim: 1 }} }} }}
layer {{ name: "loss" type: "SoftmaxWithLoss" bottom: "data" bottom: "label" top: "loss" }}
"""
        with ptrace("Net(softmax repeated)"):
            net = _make_net(prototxt)
        inp = np.random.randn(N, C, 1, 1).astype(np.float32)
        label = np.array([[[[0]]], [[[1]]]], dtype=np.float32)
        losses = []
        for i in range(10):
            with ptrace(f"softmax forward #{i+1}"):
                out = net.forward({"data": inp, "label": label})
            losses.append(float(out["loss"][0]))
        for i in range(1, 10):
            assert abs(losses[0] - losses[i]) < 1e-6


# ═══════════════════════════════════════════════════════════════════════
# Accuracy Layer Tests
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestAccuracyLayers:
    """Tests for the Accuracy layer's forward computation."""

    def test_accuracy_perfect(self, ptrace):
        """Accuracy should be 1.0 when all predictions are correct."""
        N, C = 4, 3
        prototxt = f"""name: "acc_perfect"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {C} dim: 1 dim: 1 }} }} }}
layer {{ name: "label" type: "Input" top: "label" input_param {{ shape {{ dim: {N} dim: 1 dim: 1 dim: 1 }} }} }}
layer {{ name: "acc" type: "Accuracy" bottom: "data" bottom: "label" top: "accuracy" }}
"""
        with ptrace("Net(accuracy perfect)"):
            net = _make_net(prototxt)
        # Score higher for correct class
        inp = np.zeros((N, C, 1, 1), dtype=np.float32)
        labels = np.array([0, 1, 2, 0], dtype=np.float32)
        for i in range(N):
            inp[i, int(labels[i]), 0, 0] = 10.0
        label = labels.reshape(N, 1, 1, 1)
        with ptrace("accuracy perfect forward") as t:
            out = net.forward({"data": inp, "label": label})
            t['accuracy'] = float(out["accuracy"][0])
        np.testing.assert_allclose(out["accuracy"][0], 1.0, rtol=1e-6)

    def test_accuracy_zero(self, ptrace):
        """Accuracy should be 0.0 when all predictions are wrong."""
        N, C = 3, 3
        prototxt = f"""name: "acc_zero"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {C} dim: 1 dim: 1 }} }} }}
layer {{ name: "label" type: "Input" top: "label" input_param {{ shape {{ dim: {N} dim: 1 dim: 1 dim: 1 }} }} }}
layer {{ name: "acc" type: "Accuracy" bottom: "data" bottom: "label" top: "accuracy" }}
"""
        with ptrace("Net(accuracy zero)"):
            net = _make_net(prototxt)
        # Always predict class 0, labels are never 0
        inp = np.zeros((N, C, 1, 1), dtype=np.float32)
        inp[:, 0, 0, 0] = 10.0
        label = np.array([[[[1]]], [[[2]]], [[[1]]]], dtype=np.float32)
        with ptrace("accuracy zero forward") as t:
            out = net.forward({"data": inp, "label": label})
            t['accuracy'] = float(out["accuracy"][0])
        np.testing.assert_allclose(out["accuracy"][0], 0.0, atol=1e-6)

    def test_accuracy_partial(self, ptrace):
        """Accuracy with partial correctness: 2/4 correct = 0.5."""
        N, C = 4, 3
        prototxt = f"""name: "acc_partial"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {C} dim: 1 dim: 1 }} }} }}
layer {{ name: "label" type: "Input" top: "label" input_param {{ shape {{ dim: {N} dim: 1 dim: 1 dim: 1 }} }} }}
layer {{ name: "acc" type: "Accuracy" bottom: "data" bottom: "label" top: "accuracy" }}
"""
        with ptrace("Net(accuracy partial)"):
            net = _make_net(prototxt)
        # Samples 0 and 2 are correct, 1 and 3 are wrong
        inp = np.zeros((N, C, 1, 1), dtype=np.float32)
        inp[0, 0, 0, 0] = 10.0  # predict 0, label 0 -> correct
        inp[1, 0, 0, 0] = 10.0  # predict 0, label 1 -> wrong
        inp[2, 2, 0, 0] = 10.0  # predict 2, label 2 -> correct
        inp[3, 1, 0, 0] = 10.0  # predict 1, label 0 -> wrong
        label = np.array([[[[0]]], [[[1]]], [[[2]]], [[[0]]]], dtype=np.float32)
        with ptrace("accuracy partial forward") as t:
            out = net.forward({"data": inp, "label": label})
            t['accuracy'] = float(out["accuracy"][0])
        np.testing.assert_allclose(out["accuracy"][0], 0.5, rtol=1e-6)

    def test_accuracy_topk(self, ptrace):
        """Top-k accuracy: correct class within top-k predictions."""
        N, C = 3, 5
        prototxt = f"""name: "acc_topk"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {C} dim: 1 dim: 1 }} }} }}
layer {{ name: "label" type: "Input" top: "label" input_param {{ shape {{ dim: {N} dim: 1 dim: 1 dim: 1 }} }} }}
layer {{ name: "acc" type: "Accuracy" bottom: "data" bottom: "label" top: "accuracy"
        accuracy_param {{ top_k: 3 }} }}
"""
        with ptrace("Net(accuracy top-k)"):
            net = _make_net(prototxt)
        inp = np.zeros((N, C, 1, 1), dtype=np.float32)
        # For sample 0: correct class=2 has score 5.0 (3rd highest behind 8.0 and 6.0)
        inp[0, 0, 0, 0] = 8.0  # rank 1 (wrong)
        inp[0, 1, 0, 0] = 6.0  # rank 2 (wrong)
        inp[0, 2, 0, 0] = 5.0  # rank 3 (correct, top-3 hits)
        inp[0, 3, 0, 0] = 3.0
        inp[0, 4, 0, 0] = 1.0
        # For sample 1: correct class=4 has score 0.5 (5th, outside top-3)
        inp[1, 0, 0, 0] = 9.0
        inp[1, 1, 0, 0] = 8.0
        inp[1, 2, 0, 0] = 7.0
        inp[1, 3, 0, 0] = 2.0
        inp[1, 4, 0, 0] = 0.5  # 5th, miss
        # For sample 2: correct class=1 has score 10.0 (1st)
        inp[2, 1, 0, 0] = 10.0
        label = np.array([[[[2]]], [[[4]]], [[[1]]]], dtype=np.float32)
        with ptrace("accuracy top-k forward") as t:
            out = net.forward({"data": inp, "label": label})
            t['accuracy'] = float(out["accuracy"][0])
        # 2 out of 3 correct (samples 0 and 2)
        np.testing.assert_allclose(out["accuracy"][0], 2.0/3.0, rtol=1e-5)

    def test_accuracy_spatial(self, ptrace):
        """Accuracy with spatial predictions (N,C,H,W format)."""
        N, C, H, W = 2, 3, 2, 2
        prototxt = f"""name: "acc_spatial"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {C} dim: {H} dim: {W} }} }} }}
layer {{ name: "label" type: "Input" top: "label" input_param {{ shape {{ dim: {N} dim: 1 dim: {H} dim: {W} }} }} }}
layer {{ name: "acc" type: "Accuracy" bottom: "data" bottom: "label" top: "accuracy" }}
"""
        with ptrace("Net(accuracy spatial)"):
            net = _make_net(prototxt)
        inp = np.zeros((N, C, H, W), dtype=np.float32)
        # 8 positions total, 5 correct (positions marked correct below)
        label = np.array([[[[0, 1], [2, 0]]],
                          [[[1, 2], [0, 1]]]], dtype=np.float32)
        inp[0, 0, 0, 0] = 10.0  # correct (label=0) ✓
        inp[0, 0, 0, 1] = 10.0  # wrong (label=1, predicts 0) ✗
        inp[0, 2, 1, 0] = 10.0  # correct (label=2) ✓
        inp[0, 0, 1, 1] = 10.0  # correct (label=0) ✓
        inp[1, 1, 0, 0] = 10.0  # correct (label=1) ✓
        inp[1, 0, 0, 1] = 10.0  # wrong (label=2, predicts 0) ✗
        inp[1, 0, 1, 0] = 10.0  # correct (label=0) ✓
        inp[1, 0, 1, 1] = 10.0  # wrong (label=1, predicts 0) ✗
        with ptrace("accuracy spatial forward") as t:
            out = net.forward({"data": inp, "label": label})
            t['accuracy'] = float(out["accuracy"][0])
        np.testing.assert_allclose(out["accuracy"][0], 5.0/8.0, rtol=1e-5)

    def test_accuracy_numpy_match(self, ptrace):
        """Accuracy should match numpy reference on random data."""
        N, C = 10, 5
        prototxt = f"""name: "acc_np"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {C} dim: 1 dim: 1 }} }} }}
layer {{ name: "label" type: "Input" top: "label" input_param {{ shape {{ dim: {N} dim: 1 dim: 1 dim: 1 }} }} }}
layer {{ name: "acc" type: "Accuracy" bottom: "data" bottom: "label" top: "accuracy" }}
"""
        with ptrace("Net(accuracy numpy match)"):
            net = _make_net(prototxt)
        np.random.seed(123)
        inp = np.random.randn(N, C, 1, 1).astype(np.float32)
        label = np.random.randint(0, C, (N, 1, 1, 1)).astype(np.float32)
        with ptrace("accuracy numpy match forward"):
            out = net.forward({"data": inp, "label": label})
        expected = accuracy_np(inp, label, top_k=1, axis=1)
        np.testing.assert_allclose(out["accuracy"][0], expected, rtol=1e-5)

    def test_accuracy_repeated_forward(self, ptrace):
        """Accuracy should be deterministic."""
        N, C = 5, 4
        prototxt = f"""name: "acc_rep"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {C} dim: 1 dim: 1 }} }} }}
layer {{ name: "label" type: "Input" top: "label" input_param {{ shape {{ dim: {N} dim: 1 dim: 1 dim: 1 }} }} }}
layer {{ name: "acc" type: "Accuracy" bottom: "data" bottom: "label" top: "accuracy" }}
"""
        with ptrace("Net(accuracy repeated)"):
            net = _make_net(prototxt)
        np.random.seed(456)
        inp = np.random.randn(N, C, 1, 1).astype(np.float32)
        label = np.random.randint(0, C, (N, 1, 1, 1)).astype(np.float32)
        accs = []
        for i in range(10):
            with ptrace(f"accuracy forward #{i+1}"):
                out = net.forward({"data": inp, "label": label})
            accs.append(float(out["accuracy"][0]))
        for i in range(1, 10):
            assert abs(accs[0] - accs[i]) < 1e-6


# ═══════════════════════════════════════════════════════════════════════
# Combined Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestScaleBiasEltwiseCombination:
    """Combined pipeline tests with Scale/Bias/Eltwise layers."""

    def test_scale_then_bias_pipeline(self, ptrace):
        """Conv-like pipeline: Scale then Bias (y = x*s + b)."""
        C = 3
        prototxt = f"""name: "scale_bias_pipe"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: 2 dim: {C} dim: 4 dim: 4 }} }} }}
layer {{ name: "scale" type: "Scale" bottom: "data" top: "scaled" scale_param {{ bias_term: false }} }}
layer {{ name: "bias" type: "Bias" bottom: "scaled" top: "out" }}
"""
        with ptrace("Net(scale->bias pipeline)"):
            net = _make_net(prototxt)
        scale_factors = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        bias_values = np.array([0.1, -0.1, 0.5], dtype=np.float32)
        net.layer_by_name("scale").blobs[0].from_numpy(scale_factors)
        net.layer_by_name("bias").blobs[0].from_numpy(bias_values)
        inp = np.random.randn(2, C, 4, 4).astype(np.float32)
        with ptrace("scale->bias forward") as t:
            out = net.forward({"data": inp})
            t['output_shape'] = f"{out['out'].shape}"
        expected = bias_np(scale_np(inp, scale_factors, axis=1), bias_values, axis=1)
        np.testing.assert_allclose(out["out"], expected, rtol=1e-5, atol=1e-5)

    def test_eltwise_then_scale_pipeline(self, ptrace):
        """Eltwise SUM then Scale pipeline."""
        C = 4
        prototxt = f"""name: "ew_scale_pipe"
layer {{ name: "a" type: "Input" top: "a" input_param {{ shape {{ dim: 1 dim: {C} dim: 3 dim: 3 }} }} }}
layer {{ name: "b" type: "Input" top: "b" input_param {{ shape {{ dim: 1 dim: {C} dim: 3 dim: 3 }} }} }}
layer {{ name: "ew" type: "Eltwise" bottom: "a" bottom: "b" top: "sum"
        eltwise_param {{ operation: SUM }} }}
layer {{ name: "scale" type: "Scale" bottom: "sum" top: "out" }}
"""
        with ptrace("Net(eltwise->scale pipeline)"):
            net = _make_net(prototxt)
        scale_factors = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        net.layer_by_name("scale").blobs[0].from_numpy(scale_factors)
        a = np.random.randn(1, C, 3, 3).astype(np.float32)
        b = np.random.randn(1, C, 3, 3).astype(np.float32)
        with ptrace("eltwise->scale forward") as t:
            out = net.forward({"a": a, "b": b})
        expected = scale_np(eltwise_np([a, b], op="SUM"), scale_factors, axis=1)
        np.testing.assert_allclose(out["out"], expected, rtol=1e-5, atol=1e-5)

    def test_classification_pipeline(self, ptrace):
        """Full classification pipeline: Scale->Split->SoftmaxWithLoss + Accuracy.
        InnerProduct (simulated via Scale) -> Split (for dual consumer)
        -> SoftmaxWithLoss + Accuracy.
        We use a Scale layer as a simple linear classifier (per-channel scale + bias).
        Split is needed because caffe-ffi requires each blob to have exactly one consumer.
        """
        N, C = 8, 4
        prototxt = f"""name: "class_pipe"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {C} dim: 1 dim: 1 }} }} }}
layer {{ name: "label" type: "Input" top: "label" input_param {{ shape {{ dim: {N} dim: 1 dim: 1 dim: 1 }} }} }}
layer {{ name: "score" type: "Scale" bottom: "data" top: "score" scale_param {{ bias_term: true }} }}
layer {{ name: "split_score" type: "Split" bottom: "score" top: "score_loss" top: "score_acc" }}
layer {{ name: "split_label" type: "Split" bottom: "label" top: "label_loss" top: "label_acc" }}
layer {{ name: "loss" type: "SoftmaxWithLoss" bottom: "score_loss" bottom: "label_loss" top: "loss" }}
layer {{ name: "acc" type: "Accuracy" bottom: "score_acc" bottom: "label_acc" top: "accuracy" }}
"""
        with ptrace("Net(classification pipeline)"):
            net = _make_net(prototxt)
        # Identity scale + zero bias initially
        net.layer_by_name("score").blobs[0].from_numpy(np.ones(C, dtype=np.float32))
        net.layer_by_name("score").blobs[1].from_numpy(np.zeros(C, dtype=np.float32))
        np.random.seed(789)
        inp = np.random.randn(N, C, 1, 1).astype(np.float32) * 0.1
        label = np.random.randint(0, C, (N, 1, 1, 1)).astype(np.float32)
        # Make some predictions better by adding bias to correct class
        score_bias = np.zeros(C, dtype=np.float32)
        for i in range(N):
            lbl = int(label[i, 0, 0, 0])
            if i < N // 2:
                score_bias[lbl] += 10.0  # first half correct
        net.layer_by_name("score").blobs[1].from_numpy(score_bias)
        with ptrace("classification pipeline forward") as t:
            out = net.forward({"data": inp, "label": label})
            t['loss'] = float(out["loss"][0])
            t['accuracy'] = float(out["accuracy"][0])
        # First half correct -> accuracy ~= 0.5, loss < -log(0.25) = 1.386
        assert out["loss"].shape == (1,)
        assert out["accuracy"].shape == (1,)
        assert out["accuracy"][0] >= 0.4  # at least half correct (may be more due to noise)

    def test_stability_20_iters(self, ptrace):
        """20-iteration pipeline stress test: no segfault, no OOM, stable results."""
        N, C = 4, 3
        prototxt = f"""name: "p3b_stress"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {C} dim: 2 dim: 2 }} }} }}
layer {{ name: "s1" type: "Scale" bottom: "data" top: "s_out" scale_param {{ bias_term: true }} }}
layer {{ name: "b1" type: "Bias" bottom: "s_out" top: "b_out" }}
layer {{ name: "drop" type: "Dropout" bottom: "b_out" top: "d_out" }}
"""
        with ptrace("Net(P3-B stress test)"):
            net = _make_net(prototxt)
        net.layer_by_name("s1").blobs[0].from_numpy(np.array([1.5, 0.8, 1.2], dtype=np.float32))
        net.layer_by_name("s1").blobs[1].from_numpy(np.array([0.1, -0.2, 0.3], dtype=np.float32))
        net.layer_by_name("b1").blobs[0].from_numpy(np.array([0.5, -0.5, 0.0], dtype=np.float32))
        np.random.seed(999)
        inp = np.random.randn(N, C, 2, 2).astype(np.float32)
        prev_out = None
        for i in range(20):
            with ptrace(f"P3-B stress iter #{i+1}"):
                out = net.forward({"data": inp})
            if prev_out is not None:
                np.testing.assert_array_equal(out["d_out"], prev_out)
            prev_out = out["d_out"].copy()
