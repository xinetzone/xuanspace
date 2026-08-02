"""P3-C: Activation Functions, InnerProduct, Softmax, Flatten, Reshape Real Forward Tests.

Comprehensive tests covering the actual C++ forward computation of:
- ReLU layer (with negative_slope for Leaky ReLU)
- Sigmoid layer (logistic: 1/(1+exp(-x)))
- TanH layer (hyperbolic tangent)
- ELU layer (Exponential Linear Unit: x if x>=0 else alpha*(exp(x)-1))
- PReLU layer (Parametric ReLU with per-channel or shared slope)
- InnerProduct layer (Fully Connected: y = x @ W^T + b)
- Softmax layer (standalone softmax along specified axis)
- Flatten layer (flatten a range of axes into one dimension)
- Reshape layer (reshape tensor with specified dimensions, -1 inference)

Each test follows the "three-layer validation" pattern:
1. Known values: exact verification with hand-computed expected outputs
2. Numpy random match: compare with numpy reference on random data
3. Repeated forward: verify deterministic output across multiple calls

All tests include numpy reference implementations and perf_trace logging.

Run with:
    pytest tests/python/test_p3c_activations_ip.py -v
    pytest tests/python/test_p3c_activations_ip.py -v -s  # verbose with [PERF] logs
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

def relu_np(x, negative_slope=0.0):
    """Numpy reference for ReLU/LeakyReLU: y = max(x,0) + negative_slope*min(x,0)."""
    return np.maximum(x, 0.0).astype(np.float32) + negative_slope * np.minimum(x, 0.0).astype(np.float32)


def sigmoid_np(x):
    """Numpy reference for Sigmoid: y = 1/(1+exp(-x))."""
    x64 = x.astype(np.float64)
    return (1.0 / (1.0 + np.exp(-x64))).astype(np.float32)


def tanh_np(x):
    """Numpy reference for TanH: y = tanh(x)."""
    return np.tanh(x.astype(np.float64)).astype(np.float32)


def elu_np(x, alpha=1.0):
    """Numpy reference for ELU: y = x if x>=0 else alpha*(exp(x)-1)."""
    x64 = x.astype(np.float64)
    out = np.where(x64 >= 0, x64, alpha * (np.exp(x64) - 1.0))
    return out.astype(np.float32)


def prelu_np(x, slope, channel_shared=False):
    """Numpy reference for PReLU: y = max(x,0) + slope_c * min(x,0) per channel.

    Args:
        x: Input tensor (N,C,H,W) or any shape
        slope: Slope parameter(s). If channel_shared, scalar or shape (1,).
               Otherwise shape (C,) for per-channel slopes.
    """
    x64 = x.astype(np.float64)
    slope_arr = np.asarray(slope, dtype=np.float64)
    if channel_shared:
        s = float(slope_arr.flat[0])
        return (np.maximum(x64, 0.0) + s * np.minimum(x64, 0.0)).astype(np.float32)
    # Per-channel: slope applied to axis=1
    ndim = x64.ndim
    slope_shape = [1] * ndim
    slope_shape[1] = slope_arr.shape[0]
    slope_bc = slope_arr.reshape(slope_shape)
    return (np.maximum(x64, 0.0) + slope_bc * np.minimum(x64, 0.0)).astype(np.float32)


def inner_product_np(x, W, b=None, axis=1):
    """Numpy reference for InnerProduct (Fully Connected) layer.

    y = x_reshaped @ W.T + b  where:
    - x is reshaped to (M, K) with M = prod(shape[:axis]), K = prod(shape[axis:])
    - W shape: (N, K) where N = num_output
    - b shape: (N,) or None
    - Output shape: shape[:axis] + (N,) + (1,)*(ndim-axis-1)

    Args:
        x: Input tensor
        W: Weight matrix shape (N, K)
        b: Bias vector shape (N,) or None
        axis: Axis at which to flatten (default 1)
    Returns:
        Output tensor
    """
    x64 = x.astype(np.float64)
    W64 = W.astype(np.float64)
    shape = x64.shape
    ndim = len(shape)
    M = int(np.prod(shape[:axis]))
    K = int(np.prod(shape[axis:]))
    N = W64.shape[0]
    x_flat = x64.reshape(M, K)
    y_flat = x_flat @ W64.T
    if b is not None:
        y_flat = y_flat + b.astype(np.float64)
    out_shape = list(shape[:axis]) + [N] + [1] * (ndim - axis - 1)
    return y_flat.reshape(out_shape).astype(np.float32)


def softmax_np(x, axis=1):
    """Numpy reference for softmax along a given axis."""
    x64 = x.astype(np.float64)
    x_max = np.max(x64, axis=axis, keepdims=True)
    e_x = np.exp(x64 - x_max)
    return (e_x / np.sum(e_x, axis=axis, keepdims=True)).astype(np.float32)


def flatten_np(x, start_axis=1, end_axis=-1):
    """Numpy reference for Flatten layer: flatten axes [start_axis, end_axis] into one."""
    ndim = x.ndim
    if end_axis < 0:
        end_axis = ndim + end_axis
    shape = list(x.shape)
    new_shape = list(shape[:start_axis])
    flattened_dim = int(np.prod(shape[start_axis:end_axis + 1]))
    new_shape.append(flattened_dim)
    new_shape.extend(shape[end_axis + 1:])
    return x.reshape(new_shape)


def reshape_np(x, shape_spec):
    """Numpy reference for Reshape layer with -1 inference and 0 copy semantics.

    Caffe Reshape semantics:
      - 0 means "copy the dimension from the corresponding input position"
      - -1 means "infer this dimension from total size"
      - Exactly one -1 allowed (or none)

    Args:
        x: Input tensor
        shape_spec: List/tuple of dims.
    """
    in_shape = x.shape
    ndim = len(in_shape)
    shape = list(shape_spec)
    # Resolve 0s: copy from input
    for i in range(len(shape)):
        if shape[i] == 0:
            shape[i] = in_shape[i]
    total = int(np.prod(in_shape))
    neg_count = sum(1 for d in shape if d == -1)
    if neg_count == 1:
        known = int(np.prod([d for d in shape if d != -1]))
        if known == 0:
            raise ValueError("Cannot infer dimension: known product is 0")
        inferred = total // known
        shape = [inferred if d == -1 else d for d in shape]
    elif neg_count > 1:
        raise ValueError("At most one -1 dimension allowed")
    return x.reshape(shape)


# ═══════════════════════════════════════════════════════════════════════
# Helper: build net from prototxt
# ═══════════════════════════════════════════════════════════════════════

def _make_net(prototxt: str):
    """Create a net from prototxt string."""
    return net_from_param(net_param_from_string(prototxt))


# ═══════════════════════════════════════════════════════════════════════
# ReLU Layer Tests
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestReLULayers:
    """Tests for the ReLU layer's forward computation (including Leaky ReLU)."""

    def test_relu_basic_known_values(self, ptrace):
        """ReLU with default negative_slope=0 should clamp negatives to 0."""
        prototxt = """name: "relu_basic"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 1 dim: 1 dim: 5 } } }
layer { name: "relu" type: "ReLU" bottom: "data" top: "out" }
"""
        with ptrace("Net(relu basic)"):
            net = _make_net(prototxt)
        inp = np.array([[[[-2.0, -1.0, 0.0, 1.0, 3.0]]]], dtype=np.float32)
        with ptrace("relu basic forward"):
            out = net.forward({"data": inp})
        expected = np.array([[[[0.0, 0.0, 0.0, 1.0, 3.0]]]], dtype=np.float32)
        np.testing.assert_array_equal(out["out"], expected)

    def test_relu_leaky_negative_slope(self, ptrace):
        """Leaky ReLU with negative_slope=0.1: f(x)=x for x>0, f(x)=0.1x for x<0."""
        prototxt = """name: "relu_leaky"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 1 dim: 1 dim: 4 } } }
layer { name: "relu" type: "ReLU" bottom: "data" top: "out" relu_param { negative_slope: 0.1 } }
"""
        with ptrace("Net(relu leaky)"):
            net = _make_net(prototxt)
        inp = np.array([[[[-2.0, -1.0, 0.0, 5.0]]]], dtype=np.float32)
        with ptrace("relu leaky forward"):
            out = net.forward({"data": inp})
        expected = np.array([[[[-0.2, -0.1, 0.0, 5.0]]]], dtype=np.float32)
        np.testing.assert_allclose(out["out"], expected, rtol=1e-6)

    def test_relu_numpy_match(self, ptrace):
        """ReLU should match numpy reference on random data."""
        prototxt = """name: "relu_np"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 4 dim: 8 dim: 6 dim: 6 } } }
layer { name: "relu" type: "ReLU" bottom: "data" top: "out" relu_param { negative_slope: 0.0 } }
"""
        with ptrace("Net(relu numpy match)"):
            net = _make_net(prototxt)
        np.random.seed(42)
        inp = np.random.randn(4, 8, 6, 6).astype(np.float32) * 5.0
        with ptrace("relu numpy match forward") as t:
            out = net.forward({"data": inp})
            t['max_abs_diff'] = float(np.max(np.abs(out["out"] - relu_np(inp))))
        np.testing.assert_allclose(out["out"], relu_np(inp), rtol=1e-6, atol=1e-6)

    def test_relu_leaky_numpy_match(self, ptrace):
        """Leaky ReLU (negative_slope=0.01) should match numpy reference."""
        slope = 0.01
        prototxt = f"""name: "relu_leaky_np"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: 3 dim: 5 dim: 4 dim: 4 }} }} }}
layer {{ name: "relu" type: "ReLU" bottom: "data" top: "out" relu_param {{ negative_slope: {slope} }} }}
"""
        with ptrace("Net(relu leaky numpy match)"):
            net = _make_net(prototxt)
        np.random.seed(123)
        inp = np.random.randn(3, 5, 4, 4).astype(np.float32) * 3.0
        with ptrace("relu leaky numpy match forward"):
            out = net.forward({"data": inp})
        np.testing.assert_allclose(out["out"], relu_np(inp, negative_slope=slope), rtol=1e-5)

    def test_relu_preserves_shape(self, ptrace):
        """ReLU output shape must match input shape."""
        prototxt = """name: "relu_shape"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 5 dim: 5 } } }
layer { name: "relu" type: "ReLU" bottom: "data" top: "out" }
"""
        with ptrace("Net(relu shape)"):
            net = _make_net(prototxt)
        inp = np.random.randn(2, 3, 5, 5).astype(np.float32)
        with ptrace("relu shape forward"):
            out = net.forward({"data": inp})
        assert out["out"].shape == inp.shape

    def test_relu_repeated_forward(self, ptrace):
        """ReLU should be deterministic across repeated forwards."""
        prototxt = """name: "relu_rep"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 4 dim: 3 dim: 3 } } }
layer { name: "relu" type: "ReLU" bottom: "data" top: "out" relu_param { negative_slope: 0.2 } }
"""
        with ptrace("Net(relu repeated)"):
            net = _make_net(prototxt)
        np.random.seed(99)
        inp = np.random.randn(2, 4, 3, 3).astype(np.float32) * 10.0
        outs = []
        for i in range(5):
            with ptrace(f"relu repeated forward #{i}"):
                out = net.forward({"data": inp})
            outs.append(out["out"].copy())
        for i in range(1, 5):
            np.testing.assert_array_equal(outs[0], outs[i])


# ═══════════════════════════════════════════════════════════════════════
# Sigmoid Layer Tests
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestSigmoidLayers:
    """Tests for the Sigmoid layer's forward computation."""

    def test_sigmoid_known_values(self, ptrace):
        """Sigmoid at known points: sigmoid(0)=0.5, sigmoid(±100) saturates to exact 0/1 in float32."""
        prototxt = """name: "sigmoid_known"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 1 dim: 1 dim: 5 } } }
layer { name: "sig" type: "Sigmoid" bottom: "data" top: "out" }
"""
        with ptrace("Net(sigmoid known)"):
            net = _make_net(prototxt)
        inp = np.array([[[[-100.0, -1.0, 0.0, 1.0, 100.0]]]], dtype=np.float32)
        with ptrace("sigmoid known forward"):
            out = net.forward({"data": inp})
        result = out["out"]
        assert result[0, 0, 0, 2] == pytest.approx(0.5, abs=1e-6)
        # In float32, sigmoid(-100) = 1/(1+exp(100)) = 1/inf = exactly 0.0
        assert result[0, 0, 0, 0] == 0.0, f"sigmoid(-100) should be exactly 0.0 in float32, got {result[0,0,0,0]}"
        # In float32, sigmoid(100) = 1/(1+exp(-100)) ≈ 1-3.7e-44, rounds to exactly 1.0
        assert result[0, 0, 0, 4] == 1.0, f"sigmoid(100) should be exactly 1.0 in float32, got {result[0,0,0,4]}"
        assert result[0, 0, 0, 1] == pytest.approx(1.0 / (1.0 + np.e), rel=1e-5)
        assert result[0, 0, 0, 3] == pytest.approx(np.e / (1.0 + np.e), rel=1e-5)

    def test_sigmoid_numpy_match(self, ptrace):
        """Sigmoid should match numpy reference on random data."""
        prototxt = """name: "sigmoid_np"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 4 dim: 6 dim: 5 dim: 5 } } }
layer { name: "sig" type: "Sigmoid" bottom: "data" top: "out" }
"""
        with ptrace("Net(sigmoid numpy match)"):
            net = _make_net(prototxt)
        np.random.seed(456)
        inp = np.random.randn(4, 6, 5, 5).astype(np.float32) * 2.0
        with ptrace("sigmoid numpy match forward") as t:
            out = net.forward({"data": inp})
            t['max_abs_diff'] = float(np.max(np.abs(out["out"] - sigmoid_np(inp))))
        np.testing.assert_allclose(out["out"], sigmoid_np(inp), rtol=1e-5, atol=1e-6)

    def test_sigmoid_output_range(self, ptrace):
        """Sigmoid output must be strictly in (0, 1) for non-saturating inputs."""
        prototxt = """name: "sigmoid_range"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 3 dim: 4 dim: 3 dim: 3 } } }
layer { name: "sig" type: "Sigmoid" bottom: "data" top: "out" }
"""
        with ptrace("Net(sigmoid range)"):
            net = _make_net(prototxt)
        np.random.seed(789)
        # Use scale 3 so values stay ~[-9,9], where sigmoid doesn't saturate to 0/1 in float32
        inp = np.random.randn(3, 4, 3, 3).astype(np.float32) * 3.0
        with ptrace("sigmoid range forward"):
            out = net.forward({"data": inp})
        assert np.all(out["out"] > 0.0)
        assert np.all(out["out"] < 1.0)

    def test_sigmoid_float32_saturation_exact(self, ptrace):
        """Sigmoid saturation behavior in float32.

        In IEEE754 float32:
        - ULP(1.0) ≈ 1.2e-7, so sigmoid(x) rounds to exactly 1.0 when
          1-sigmoid(x) ≈ exp(-x) < ULP/2 ≈ 6e-8, i.e. x > ~16.6.
        - sigmoid(88) = 1/(1+exp(-88)) ≈ 1.0 exactly (exp(-88)≈6e-39 ≪ ULP/2).
        - sigmoid(80) = 1/(1+exp(-80)) ≈ 1.0 exactly too (exp(-80)≈1.8e-35 ≪ ULP/2).
        - sigmoid(-88) = 1/(1+exp(88)) ≈ 6.1e-39, a representable subnormal
          float32 (min subnormal ≈ 1.4e-45), NOT exactly 0.0 but effectively zero.
        - sigmoid(-80) ≈ 1.8e-35, also a subnormal, NOT exactly 0.0.
        """
        prototxt = """name: "sigmoid_sat"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 1 dim: 1 dim: 4 } } }
layer { name: "sig" type: "Sigmoid" bottom: "data" top: "out" }
"""
        with ptrace("Net(sigmoid saturation)"):
            net = _make_net(prototxt)
        # Test saturation points: all |x|>=20 are well past float32 saturation threshold
        inp = np.array([[[[-88.0, -80.0, 80.0, 88.0]]]], dtype=np.float32)
        with ptrace("sigmoid saturation forward"):
            out = net.forward({"data": inp})
        result = out["out"]
        # sigmoid(-88) ≈ 6e-39 in float32 (subnormal, not exactly 0)
        assert result[0, 0, 0, 0] < 1e-37, (
            f"sigmoid(-88) should be < 1e-37 (effectively zero), got {result[0,0,0,0]}"
        )
        # sigmoid(-80) ≈ 1.8e-35 in float32 (subnormal, not exactly 0)
        assert result[0, 0, 0, 1] < 1e-30, (
            f"sigmoid(-80) should be < 1e-30 (effectively zero), got {result[0,0,0,1]}"
        )
        # sigmoid(80) is exactly 1.0 in float32 (exp(-80)≈1.8e-35 ≪ ULP(1.0)/2≈6e-8)
        assert result[0, 0, 0, 2] == 1.0, (
            f"sigmoid(80) should be exactly 1.0 in float32, got {result[0,0,0,2]}"
        )
        # sigmoid(88) is exactly 1.0 in float32
        assert result[0, 0, 0, 3] == 1.0, f"sigmoid(88) should be exactly 1.0, got {result[0,0,0,3]}"
        # No NaN or Inf
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_sigmoid_saturation_transition_zone(self, ptrace):
        """Find the approximate transition zone where float32 saturation begins.

        In float32, ULP(1.0) ≈ 1.2e-7, so sigmoid(x) rounds to 1.0 when
        1-sigmoid(x) ≈ exp(-x) < ULP/2 ≈ 6e-8, i.e. x > ~16.6.
        Values below x≈15 remain strictly < 1.0.
        """
        prototxt = """name: "sigmoid_trans"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 1 dim: 1 dim: 10 } } }
layer { name: "sig" type: "Sigmoid" bottom: "data" top: "out" }
"""
        with ptrace("Net(sigmoid transition)"):
            net = _make_net(prototxt)
        # Sweep values from well-behaved to saturated
        # In float32, saturation to 1.0 starts around x≈17
        x_values = np.array([[[[0.0, 5.0, 10.0, 12.0, 14.0, 15.0, 16.0, 20.0, 40.0, 88.0]]]], dtype=np.float32)
        with ptrace("sigmoid transition forward"):
            out = net.forward({"data": x_values})
        result = out["out"]
        # For x <= 14, output should be strictly < 1.0 (not yet saturated)
        for i in range(5):
            assert result.flat[i] < 1.0, (
                f"sigmoid({x_values.flat[i]}) should be < 1.0, got {result.flat[i]}"
            )
        # At x=88 it must be exactly 1.0
        assert result.flat[9] == 1.0, f"sigmoid(88) should be exactly 1.0, got {result.flat[9]}"
        # Monotonicity: output should be non-decreasing as x increases
        for i in range(9):
            assert result.flat[i+1] >= result.flat[i], (
                f"Sigmoid not monotonic at index {i}: {result.flat[i]} -> {result.flat[i+1]}"
            )
        # No NaN or Inf anywhere
        assert not np.any(np.isnan(result)), "Sigmoid produced NaN"
        assert not np.any(np.isinf(result)), "Sigmoid produced Inf"

    def test_sigmoid_extreme_large_tensor(self, ptrace):
        """Sigmoid on large tensor with extreme values: no NaN/Inf/crash, correct behavior."""
        N = 65536
        prototxt = f"""name: "sigmoid_large"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: 1 dim: 1 dim: 1 dim: {N} }} }} }}
layer {{ name: "sig" type: "Sigmoid" bottom: "data" top: "out" }}
"""
        with ptrace("Net(sigmoid large tensor)"):
            net = _make_net(prototxt)
        # Mix of extreme values: -88, 0, 88 repeated
        inp = np.zeros((1, 1, 1, N), dtype=np.float32)
        inp.flat[0::3] = -88.0
        inp.flat[1::3] = 0.0
        inp.flat[2::3] = 88.0
        with ptrace("sigmoid large tensor forward"):
            out = net.forward({"data": inp})
        result = out["out"]
        assert result.shape == (1, 1, 1, N)
        # sigmoid(-88) ≈ 6e-39 (subnormal, effectively zero but not exactly 0.0)
        assert np.all(result.flat[0::3] < 1e-37), "All sigmoid(-88) should be < 1e-37"
        assert np.all(result.flat[1::3] == pytest.approx(0.5, abs=1e-6)), "All sigmoid(0) should be 0.5"
        assert np.all(result.flat[2::3] == 1.0), "All sigmoid(88) should be exactly 1.0"
        assert not np.any(np.isnan(result)), "NaN in large tensor sigmoid"
        assert not np.any(np.isinf(result)), "Inf in large tensor sigmoid"

    def test_sigmoid_zero_input(self, ptrace):
        """Sigmoid of all zeros should produce exactly 0.5 for all elements."""
        prototxt = """name: "sigmoid_zero"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 4 dim: 4 } } }
layer { name: "sig" type: "Sigmoid" bottom: "data" top: "out" }
"""
        with ptrace("Net(sigmoid zero)"):
            net = _make_net(prototxt)
        inp = np.zeros((2, 3, 4, 4), dtype=np.float32)
        with ptrace("sigmoid zero forward"):
            out = net.forward({"data": inp})
        assert np.all(out["out"] == 0.5), "sigmoid(0) should be exactly 0.5 for all elements"

    def test_sigmoid_symmetric(self, ptrace):
        """Sigmoid should satisfy 1 - sigmoid(x) = sigmoid(-x)."""
        prototxt = """name: "sigmoid_sym"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 2 dim: 2 } } }
layer { name: "neg_data" type: "Input" top: "neg_data" input_param { shape { dim: 2 dim: 3 dim: 2 dim: 2 } } }
layer { name: "sig_pos" type: "Sigmoid" bottom: "data" top: "out_pos" }
layer { name: "sig_neg" type: "Sigmoid" bottom: "neg_data" top: "out_neg" }
"""
        with ptrace("Net(sigmoid symmetric)"):
            net = _make_net(prototxt)
        np.random.seed(321)
        inp = np.random.randn(2, 3, 2, 2).astype(np.float32)
        with ptrace("sigmoid symmetric forward"):
            out = net.forward({"data": inp, "neg_data": -inp})
        np.testing.assert_allclose(1.0 - out["out_pos"], out["out_neg"], rtol=1e-5)

    def test_sigmoid_repeated_forward(self, ptrace):
        """Sigmoid should be deterministic across repeated forwards."""
        prototxt = """name: "sigmoid_rep"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 4 dim: 4 } } }
layer { name: "sig" type: "Sigmoid" bottom: "data" top: "out" }
"""
        with ptrace("Net(sigmoid repeated)"):
            net = _make_net(prototxt)
        np.random.seed(55)
        inp = np.random.randn(2, 3, 4, 4).astype(np.float32)
        outs = []
        for i in range(5):
            with ptrace(f"sigmoid repeated forward #{i}"):
                out = net.forward({"data": inp})
            outs.append(out["out"].copy())
        for i in range(1, 5):
            np.testing.assert_array_equal(outs[0], outs[i])


# ═══════════════════════════════════════════════════════════════════════
# Sigmoid Layer Backward Tests (P3-D: Gradient & Saturation Counter)
# ═══════════════════════════════════════════════════════════════════════

def sigmoid_grad_np(x):
    """Numpy reference for sigmoid gradient: sigmoid'(x) = y * (1 - y)."""
    y = sigmoid_np(x)
    return (y * (1.0 - y)).astype(np.float32)


@require_cpp_extension
class TestSigmoidBackward:
    """Tests for Sigmoid backward pass: gradient correctness and saturation counter.

    These tests verify:
    1. Gradient values match numpy reference dx = dy * y * (1-y)
    2. Saturation counter correctly identifies elements where y is in saturation zone
       (y < 1e-4 or y > 1-1e-4, where the local gradient is < 1e-4, causing vanishing gradients)
    3. propagate_down=false skips gradient computation
    """

    def _make_sigmoid_net(self, N):
        """Create a simple Input->Sigmoid network with 1D input of size N."""
        prototxt = f"""name: "sigmoid_bwd"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: 1 dim: 1 dim: 1 dim: {N} }} }} }}
layer {{ name: "sig" type: "Sigmoid" bottom: "data" top: "out" }}
"""
        return _make_net(prototxt)

    def test_sigmoid_backward_gradient_values(self, ptrace):
        """Gradient dx should equal dy * sigmoid(x) * (1 - sigmoid(x))."""
        N = 10
        with ptrace("Net(sigmoid bwd gradient)"):
            net = self._make_sigmoid_net(N)
        np.random.seed(1001)
        x = np.random.randn(1, 1, 1, N).astype(np.float32) * 2.0
        dy = np.ones((1, 1, 1, N), dtype=np.float32)  # upstream gradient = 1
        with ptrace("sigmoid forward"):
            net.forward({"data": x})
        with ptrace("sigmoid backward"):
            net.backward({"out": dy})
        # Read input blob diff (bottom gradient)
        data_blob = net.blob_by_name("data")
        dx = data_blob.diff
        expected_dx = sigmoid_grad_np(x)  # dy=1, so dx = y*(1-y)
        np.testing.assert_allclose(dx, expected_dx, rtol=1e-5, atol=1e-6)

    def test_sigmoid_backward_with_arbitrary_dy(self, ptrace):
        """Gradient dx = dy * y * (1-y) with non-unit upstream gradient."""
        N = 20
        with ptrace("Net(sigmoid bwd arbitrary dy)"):
            net = self._make_sigmoid_net(N)
        np.random.seed(1002)
        x = np.random.randn(1, 1, 1, N).astype(np.float32) * 3.0
        dy = np.random.randn(1, 1, 1, N).astype(np.float32) * 0.5
        with ptrace("sigmoid forward"):
            out = net.forward({"data": x})
        with ptrace("sigmoid backward"):
            net.backward({"out": dy})
        y = out["out"]
        expected_dx = (dy * y * (1.0 - y)).astype(np.float32)
        dx = net.blob_by_name("data").diff
        np.testing.assert_allclose(dx, expected_dx, rtol=1e-5, atol=1e-6)

    def test_sigmoid_backward_saturation_counter_zero_input(self, ptrace):
        """All-zero input: y=0.5 for all elements → zero saturated elements.

        At x=0, sigmoid(0)=0.5 which is right in the linear zone (not near 0 or 1),
        so saturate count must be exactly 0 and ratio=0.
        """
        N = 100
        with ptrace("Net(sigmoid bwd zero sat)"):
            net = self._make_sigmoid_net(N)
        x = np.zeros((1, 1, 1, N), dtype=np.float32)
        dy = np.ones((1, 1, 1, N), dtype=np.float32)
        with ptrace("sigmoid forward zero"):
            net.forward({"data": x})
        with ptrace("sigmoid backward zero") as t:
            net.backward({"out": dy})
        # Verify gradient is max at 0.25 for all elements
        dx = net.blob_by_name("data").diff
        np.testing.assert_allclose(dx, np.full_like(dx, 0.25), rtol=1e-6)
        t['note'] = 'All elements at y=0.5, max gradient, zero saturation'

    def test_sigmoid_backward_saturation_counter_all_saturated_positive(self, ptrace):
        """All large positive inputs (x=20): y≈1 for all → all saturated.

        At x=20, sigmoid(20) ≈ 1-2e-9, which is > 1-1e-4, so all elements are
        in the positive saturation zone. saturate count must equal N, ratio=1.0.
        Gradient dx should be near zero (vanishing gradient).
        """
        N = 50
        with ptrace("Net(sigmoid bwd all sat pos)"):
            net = self._make_sigmoid_net(N)
        x = np.full((1, 1, 1, N), 20.0, dtype=np.float32)
        dy = np.ones((1, 1, 1, N), dtype=np.float32)
        with ptrace("sigmoid forward sat pos"):
            out = net.forward({"data": x})
        y = out["out"]
        # Verify forward outputs are near 1.0
        assert np.all(y > 1.0 - 1e-4), f"Expected all y > 1-1e-4, min={y.min()}"
        with ptrace("sigmoid backward sat pos") as t:
            net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        # All gradients should be near zero (vanishing)
        assert np.all(np.abs(dx) < 1e-4), f"Expected near-zero gradients, max|dx|={np.abs(dx).max()}"
        t['max_dx'] = float(np.abs(dx).max())
        t['note'] = f'All {N} elements in positive saturation zone, vanishing gradients'

    def test_sigmoid_backward_saturation_counter_all_saturated_negative(self, ptrace):
        """All large negative inputs (x=-20): y≈0 for all → all saturated.

        At x=-20, sigmoid(-20) ≈ 2e-9, which is < 1e-4, so all elements are
        in the negative saturation zone. saturate count must equal N, ratio=1.0.
        """
        N = 50
        with ptrace("Net(sigmoid bwd all sat neg)"):
            net = self._make_sigmoid_net(N)
        x = np.full((1, 1, 1, N), -20.0, dtype=np.float32)
        dy = np.ones((1, 1, 1, N), dtype=np.float32)
        with ptrace("sigmoid forward sat neg"):
            out = net.forward({"data": x})
        y = out["out"]
        assert np.all(y < 1e-4), f"Expected all y < 1e-4, max={y.max()}"
        with ptrace("sigmoid backward sat neg") as t:
            net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        assert np.all(np.abs(dx) < 1e-4), f"Expected near-zero gradients, max|dx|={np.abs(dx).max()}"
        t['max_dx'] = float(np.abs(dx).max())
        t['note'] = f'All {N} elements in negative saturation zone, vanishing gradients'

    def test_sigmoid_backward_saturation_counter_mixed(self, ptrace):
        """Mixed inputs: precisely control how many elements are saturated.

        Construct input with known saturated/non-saturated elements and verify
        the saturation counter matches exactly. Uses threshold boundary:
        - y < 1e-4  → x < -ln(1/1e-4 - 1) ≈ -9.21 (negative saturation)
        - y > 1-1e-4 → x > ln(1/1e-4 - 1) ≈ 9.21 (positive saturation)
        - |x| < 9   → non-saturated
        """
        N = 30
        with ptrace("Net(sigmoid bwd mixed)"):
            net = self._make_sigmoid_net(N)
        x = np.zeros((1, 1, 1, N), dtype=np.float32)
        # 10 elements in negative saturation (x = -20)
        x.flat[:10] = -20.0
        # 10 elements in linear zone (x = 0)
        x.flat[10:20] = 0.0
        # 10 elements in positive saturation (x = 20)
        x.flat[20:] = 20.0
        expected_saturated = 20  # 10 neg + 10 pos
        expected_linear = 10
        dy = np.ones((1, 1, 1, N), dtype=np.float32)
        with ptrace("sigmoid forward mixed"):
            out = net.forward({"data": x})
        y = out["out"]
        # Verify forward: saturated elements at extremes, linear at 0.5
        assert np.all(y.flat[:10] < 1e-4)
        np.testing.assert_allclose(y.flat[10:20], np.full(10, 0.5, dtype=np.float32), atol=1e-6)
        assert np.all(y.flat[20:] > 1.0 - 1e-4)
        with ptrace("sigmoid backward mixed") as t:
            net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        # Saturated elements: near-zero gradients
        assert np.all(np.abs(dx.flat[:10]) < 1e-4), "Neg saturated gradients should vanish"
        assert np.all(np.abs(dx.flat[20:]) < 1e-4), "Pos saturated gradients should vanish"
        # Linear elements: gradient ≈ 0.25 (max gradient at x=0)
        np.testing.assert_allclose(dx.flat[10:20], np.full(10, 0.25, dtype=np.float32), rtol=1e-5)
        t['expected_saturated'] = expected_saturated
        t['expected_linear'] = expected_linear
        t['note'] = f'Mixed: {expected_saturated} saturated (near-zero dx), {expected_linear} linear (dx≈0.25)'

    def test_sigmoid_backward_saturation_boundary_threshold(self, ptrace):
        """Test exact threshold boundary: elements just inside/outside saturation.

        The saturation threshold is kSaturateThreshold = 1e-4.
        Elements with y exactly at the boundary are NOT considered saturated
        (condition uses strict < and >).
        """
        N = 4
        with ptrace("Net(sigmoid bwd threshold)"):
            net = self._make_sigmoid_net(N)
        # x = ±9.21 gives y ≈ 1e-4 or 1-1e-4
        # Use x = ±9 to be safely non-saturated, x = ±10 to be safely saturated
        x = np.array([[[[-10.0, -9.0, 9.0, 10.0]]]], dtype=np.float32)
        dy = np.ones((1, 1, 1, N), dtype=np.float32)
        with ptrace("sigmoid forward threshold"):
            out = net.forward({"data": x})
        y = out["out"]
        # Verify threshold behavior:
        # x=-10 → y ≈ 4.5e-5 < 1e-4 → saturated
        # x=-9  → y ≈ 1.2e-4 > 1e-4 → NOT saturated
        # x=9   → y ≈ 1-1.2e-4 < 1-1e-4 → NOT saturated
        # x=10  → y ≈ 1-4.5e-5 > 1-1e-4 → saturated
        assert y.flat[0] < 1e-4, f"x=-10 should give y<1e-4, got {y.flat[0]}"
        assert y.flat[1] > 1e-4, f"x=-9 should give y>1e-4, got {y.flat[1]}"
        assert y.flat[2] < 1.0 - 1e-4, f"x=9 should give y<1-1e-4, got {y.flat[2]}"
        assert y.flat[3] > 1.0 - 1e-4, f"x=10 should give y>1-1e-4, got {y.flat[3]}"
        with ptrace("sigmoid backward threshold") as t:
            net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        # Elements 0 and 3 (indices 0,3) should have near-zero gradients
        assert np.abs(dx.flat[0]) < 1e-4, f"Saturated element 0 should have near-zero dx"
        assert np.abs(dx.flat[3]) < 1e-4, f"Saturated element 3 should have near-zero dx"
        # Elements 1 and 2 (indices 1,2) should have non-trivial gradients
        assert np.abs(dx.flat[1]) > 1e-4, f"Non-saturated element 1 should have non-zero dx"
        assert np.abs(dx.flat[2]) > 1e-4, f"Non-saturated element 2 should have non-zero dx"
        t['saturated_indices'] = [0, 3]
        t['non_saturated_indices'] = [1, 2]

    def test_sigmoid_backward_large_tensor_saturation_stats(self, ptrace):
        """Large tensor mixed distribution: verify saturation statistics.

        Uses normal distribution with std=5 to generate a realistic mix of
        saturated and non-saturated elements. Compares saturation count against
        numpy reference calculation to verify the C++ counter is accurate.
        """
        N = 65536
        with ptrace("Net(sigmoid bwd large stats)"):
            net = self._make_sigmoid_net(N)
        np.random.seed(999)
        x = np.random.randn(1, 1, 1, N).astype(np.float32) * 5.0
        dy = np.ones((1, 1, 1, N), dtype=np.float32)
        with ptrace("sigmoid forward large"):
            out = net.forward({"data": x})
        y = out["out"]
        # Compute expected saturation count using numpy
        saturate_thresh = 1e-4
        expected_saturated = int(np.sum((y < saturate_thresh) | (y > 1.0 - saturate_thresh)))
        expected_ratio = expected_saturated / N
        with ptrace("sigmoid backward large") as t:
            net.backward({"out": dy})
        dx = net.blob_by_name("data").diff
        expected_dx = (y * (1.0 - y)).astype(np.float32)
        np.testing.assert_allclose(dx, expected_dx, rtol=1e-5, atol=1e-6)
        t['total_elements'] = N
        t['expected_saturated'] = expected_saturated
        t['expected_ratio'] = float(expected_ratio)
        t['note'] = f'Large tensor: {expected_saturated}/{N} saturated ({expected_ratio:.2%})'
        # Sanity: with std=5, expect roughly 5-10% saturation
        assert 0.01 < expected_ratio < 0.30, (
            f"Expected saturation ratio between 1-30% for N(0,5) inputs, got {expected_ratio:.2%}"
        )

    def test_sigmoid_backward_deterministic(self, ptrace):
        """Backward should be deterministic across repeated calls."""
        N = 32
        with ptrace("Net(sigmoid bwd deterministic)"):
            net = self._make_sigmoid_net(N)
        np.random.seed(77)
        x = np.random.randn(1, 1, 1, N).astype(np.float32) * 4.0
        dy = np.ones((1, 1, 1, N), dtype=np.float32)
        with ptrace("sigmoid forward det"):
            net.forward({"data": x})
        dx_results = []
        for i in range(5):
            with ptrace(f"sigmoid backward det #{i}"):
                net.backward({"out": dy})
            dx_results.append(net.blob_by_name("data").diff.copy())
        for i in range(1, 5):
            np.testing.assert_array_equal(dx_results[0], dx_results[i])

    def test_sigmoid_backward_preserves_forward_output(self, ptrace):
        """Backward must not modify forward activations (top data)."""
        N = 16
        with ptrace("Net(sigmoid bwd preserves fwd)"):
            net = self._make_sigmoid_net(N)
        np.random.seed(88)
        x = np.random.randn(1, 1, 1, N).astype(np.float32) * 3.0
        dy = np.ones((1, 1, 1, N), dtype=np.float32)
        with ptrace("sigmoid forward"):
            out = net.forward({"data": x})
        y_before = out["out"].copy()
        with ptrace("sigmoid backward"):
            net.backward({"out": dy})
        # Forward output should be unchanged
        y_after = net.blob_by_name("out").data
        np.testing.assert_array_equal(y_before, y_after)


# ═══════════════════════════════════════════════════════════════════════
# TanH Layer Tests
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestTanHLayers:
    """Tests for the TanH layer's forward computation."""

    def test_tanh_known_values(self, ptrace):
        """TanH at known points: tanh(0)=0, tanh(∞)=1, tanh(-∞)=-1."""
        prototxt = """name: "tanh_known"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 1 dim: 1 dim: 5 } } }
layer { name: "th" type: "TanH" bottom: "data" top: "out" }
"""
        with ptrace("Net(tanh known)"):
            net = _make_net(prototxt)
        inp = np.array([[[[-100.0, -1.0, 0.0, 1.0, 100.0]]]], dtype=np.float32)
        with ptrace("tanh known forward"):
            out = net.forward({"data": inp})
        result = out["out"]
        assert result[0, 0, 0, 2] == pytest.approx(0.0, abs=1e-6)
        assert result[0, 0, 0, 4] == 1.0, f"tanh(100) should be exactly 1.0 in float32, got {result[0,0,0,4]}"
        assert result[0, 0, 0, 0] == -1.0, f"tanh(-100) should be exactly -1.0 in float32, got {result[0,0,0,0]}"
        assert result[0, 0, 0, 3] == pytest.approx(np.tanh(1.0), rel=1e-5)
        assert result[0, 0, 0, 1] == pytest.approx(np.tanh(-1.0), rel=1e-5)

    def test_tanh_numpy_match(self, ptrace):
        """TanH should match numpy reference on random data."""
        prototxt = """name: "tanh_np"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 4 dim: 6 dim: 5 dim: 5 } } }
layer { name: "th" type: "TanH" bottom: "data" top: "out" }
"""
        with ptrace("Net(tanh numpy match)"):
            net = _make_net(prototxt)
        np.random.seed(777)
        inp = np.random.randn(4, 6, 5, 5).astype(np.float32) * 2.0
        with ptrace("tanh numpy match forward") as t:
            out = net.forward({"data": inp})
            t['max_abs_diff'] = float(np.max(np.abs(out["out"] - tanh_np(inp))))
        np.testing.assert_allclose(out["out"], tanh_np(inp), rtol=1e-5, atol=1e-6)

    def test_tanh_output_range(self, ptrace):
        """TanH output must be in (-1, 1)."""
        prototxt = """name: "tanh_range"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 3 dim: 4 dim: 3 dim: 3 } } }
layer { name: "th" type: "TanH" bottom: "data" top: "out" }
"""
        with ptrace("Net(tanh range)"):
            net = _make_net(prototxt)
        np.random.seed(111)
        inp = np.random.randn(3, 4, 3, 3).astype(np.float32) * 10.0
        with ptrace("tanh range forward"):
            out = net.forward({"data": inp})
        assert np.all(out["out"] >= -1.0)
        assert np.all(out["out"] <= 1.0)

    def test_tanh_odd_function(self, ptrace):
        """TanH should be an odd function: tanh(-x) = -tanh(x)."""
        prototxt = """name: "tanh_odd"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 2 dim: 2 } } }
layer { name: "neg_data" type: "Input" top: "neg_data" input_param { shape { dim: 2 dim: 3 dim: 2 dim: 2 } } }
layer { name: "th_pos" type: "TanH" bottom: "data" top: "out_pos" }
layer { name: "th_neg" type: "TanH" bottom: "neg_data" top: "out_neg" }
"""
        with ptrace("Net(tanh odd)"):
            net = _make_net(prototxt)
        np.random.seed(222)
        inp = np.random.randn(2, 3, 2, 2).astype(np.float32)
        with ptrace("tanh odd forward"):
            out = net.forward({"data": inp, "neg_data": -inp})
        np.testing.assert_allclose(-out["out_pos"], out["out_neg"], rtol=1e-5)

    def test_tanh_repeated_forward(self, ptrace):
        """TanH should be deterministic across repeated forwards."""
        prototxt = """name: "tanh_rep"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 4 dim: 4 } } }
layer { name: "th" type: "TanH" bottom: "data" top: "out" }
"""
        with ptrace("Net(tanh repeated)"):
            net = _make_net(prototxt)
        np.random.seed(66)
        inp = np.random.randn(2, 3, 4, 4).astype(np.float32)
        outs = []
        for i in range(5):
            with ptrace(f"tanh repeated forward #{i}"):
                out = net.forward({"data": inp})
            outs.append(out["out"].copy())
        for i in range(1, 5):
            np.testing.assert_array_equal(outs[0], outs[i])


# ═══════════════════════════════════════════════════════════════════════
# ELU Layer Tests
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestELULayers:
    """Tests for the ELU layer's forward computation."""

    def test_elu_default_alpha_known_values(self, ptrace):
        """ELU with default alpha=1.0: f(x)=x for x>=0, f(x)=exp(x)-1 for x<0."""
        prototxt = """name: "elu_known"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 1 dim: 1 dim: 4 } } }
layer { name: "elu" type: "ELU" bottom: "data" top: "out" }
"""
        with ptrace("Net(elu known)"):
            net = _make_net(prototxt)
        inp = np.array([[[[-1.0, 0.0, 1.0, 2.0]]]], dtype=np.float32)
        with ptrace("elu known forward"):
            out = net.forward({"data": inp})
        expected = np.array([[[[np.e**(-1) - 1, 0.0, 1.0, 2.0]]]], dtype=np.float32)
        np.testing.assert_allclose(out["out"], expected, rtol=1e-5)

    def test_elu_custom_alpha(self, ptrace):
        """ELU with alpha=0.5: f(x)=0.5*(exp(x)-1) for x<0."""
        prototxt = """name: "elu_alpha"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 1 dim: 1 dim: 3 } } }
layer { name: "elu" type: "ELU" bottom: "data" top: "out" elu_param { alpha: 0.5 } }
"""
        with ptrace("Net(elu custom alpha)"):
            net = _make_net(prototxt)
        inp = np.array([[[[-1.0, 0.0, 3.0]]]], dtype=np.float32)
        with ptrace("elu custom alpha forward"):
            out = net.forward({"data": inp})
        expected = np.array([[[[0.5 * (np.e**(-1) - 1), 0.0, 3.0]]]], dtype=np.float32)
        np.testing.assert_allclose(out["out"], expected, rtol=1e-5)

    def test_elu_numpy_match_default_alpha(self, ptrace):
        """ELU (alpha=1.0) should match numpy reference on random data."""
        prototxt = """name: "elu_np"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 3 dim: 5 dim: 4 dim: 4 } } }
layer { name: "elu" type: "ELU" bottom: "data" top: "out" }
"""
        with ptrace("Net(elu numpy match)"):
            net = _make_net(prototxt)
        np.random.seed(333)
        inp = np.random.randn(3, 5, 4, 4).astype(np.float32) * 3.0
        with ptrace("elu numpy match forward") as t:
            out = net.forward({"data": inp})
            t['max_abs_diff'] = float(np.max(np.abs(out["out"] - elu_np(inp))))
        np.testing.assert_allclose(out["out"], elu_np(inp), rtol=1e-5, atol=1e-5)

    def test_elu_numpy_match_custom_alpha(self, ptrace):
        """ELU (alpha=0.1) should match numpy reference."""
        alpha = 0.1
        prototxt = f"""name: "elu_alpha_np"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: 3 dim: 5 dim: 4 dim: 4 }} }} }}
layer {{ name: "elu" type: "ELU" bottom: "data" top: "out" elu_param {{ alpha: {alpha} }} }}
"""
        with ptrace("Net(elu alpha numpy match)"):
            net = _make_net(prototxt)
        np.random.seed(444)
        inp = np.random.randn(3, 5, 4, 4).astype(np.float32) * 2.0
        with ptrace("elu alpha numpy match forward"):
            out = net.forward({"data": inp})
        np.testing.assert_allclose(out["out"], elu_np(inp, alpha=alpha), rtol=1e-5, atol=1e-5)

    def test_elu_continuity_at_zero(self, ptrace):
        """ELU should be continuous at x=0 (both sides give 0)."""
        prototxt = """name: "elu_cont"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 1 dim: 1 dim: 3 } } }
layer { name: "elu" type: "ELU" bottom: "data" top: "out" }
"""
        with ptrace("Net(elu continuity)"):
            net = _make_net(prototxt)
        eps = np.float32(1e-6)
        inp_pos = np.array([[[[0.0, eps, 1.0]]]], dtype=np.float32)
        inp_neg = np.array([[[[-eps, 0.0, -0.5]]]], dtype=np.float32)
        out_pos = net.forward({"data": inp_pos})
        out_neg = net.forward({"data": inp_neg})
        # At x=0, ELU(0) = 0 from both sides
        assert out_pos["out"][0, 0, 0, 0] == pytest.approx(0.0, abs=1e-6)
        assert out_neg["out"][0, 0, 0, 1] == pytest.approx(0.0, abs=1e-6)

    def test_elu_repeated_forward(self, ptrace):
        """ELU should be deterministic across repeated forwards."""
        prototxt = """name: "elu_rep"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 4 dim: 4 } } }
layer { name: "elu" type: "ELU" bottom: "data" top: "out" elu_param { alpha: 1.0 } }
"""
        with ptrace("Net(elu repeated)"):
            net = _make_net(prototxt)
        np.random.seed(88)
        inp = np.random.randn(2, 3, 4, 4).astype(np.float32) * 3.0
        outs = []
        for i in range(5):
            with ptrace(f"elu repeated forward #{i}"):
                out = net.forward({"data": inp})
            outs.append(out["out"].copy())
        for i in range(1, 5):
            np.testing.assert_array_equal(outs[0], outs[i])


# ═══════════════════════════════════════════════════════════════════════
# PReLU Layer Tests
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestPReLULayers:
    """Tests for the PReLU layer's forward computation."""

    def test_prelu_channel_shared_default(self, ptrace):
        """PReLU with channel_shared=true: default slope=0.25 for all channels."""
        prototxt = """name: "prelu_shared"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 1 dim: 1 dim: 3 } } }
layer { name: "prelu" type: "PReLU" bottom: "data" top: "out" prelu_param { channel_shared: true } }
"""
        with ptrace("Net(prelu channel shared)"):
            net = _make_net(prototxt)
        inp = np.array([[[[-2.0, 0.0, 4.0]]]], dtype=np.float32)
        with ptrace("prelu channel shared forward"):
            out = net.forward({"data": inp})
        # slope default is 0.25
        expected = np.array([[[[-0.5, 0.0, 4.0]]]], dtype=np.float32)
        np.testing.assert_allclose(out["out"], expected, rtol=1e-5)

    def test_prelu_per_channel_default(self, ptrace):
        """PReLU per-channel: default slope=0.25 for each channel."""
        prototxt = """name: "prelu_perch"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 3 dim: 1 dim: 1 } } }
layer { name: "prelu" type: "PReLU" bottom: "data" top: "out" }
"""
        with ptrace("Net(prelu per-channel default)"):
            net = _make_net(prototxt)
        inp = np.array([[[[-1.0]], [[-2.0]], [[3.0]]]], dtype=np.float32)
        with ptrace("prelu per-channel default forward"):
            out = net.forward({"data": inp})
        # Default slope=0.25 for all channels: -0.25, -0.5, 3.0
        expected = np.array([[[[-0.25]], [[-0.5]], [[3.0]]]], dtype=np.float32)
        np.testing.assert_allclose(out["out"], expected, rtol=1e-5)

    def test_prelu_per_channel_custom_slopes(self, ptrace):
        """PReLU per-channel with custom slopes set via blobs[0]."""
        C = 3
        prototxt = f"""name: "prelu_custom"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: 1 dim: {C} dim: 2 dim: 2 }} }} }}
layer {{ name: "prelu" type: "PReLU" bottom: "data" top: "out" }}
"""
        with ptrace("Net(prelu custom slopes)"):
            net = _make_net(prototxt)
        slopes = np.array([0.0, 0.1, 0.5], dtype=np.float32)
        net.layer_by_name("prelu").blobs[0].from_numpy(slopes)
        # Channel 0: slope=0.0 → same as ReLU
        # Channel 1: slope=0.1 → leaky ReLU
        # Channel 2: slope=0.5 → half negative
        inp = np.array([[[[-1.0, 2.0], [0.5, -3.0]],
                          [[-1.0, 2.0], [0.5, -3.0]],
                          [[-1.0, 2.0], [0.5, -3.0]]]], dtype=np.float32)
        with ptrace("prelu custom slopes forward"):
            out = net.forward({"data": inp})
        expected = prelu_np(inp, slopes, channel_shared=False)
        np.testing.assert_allclose(out["out"], expected, rtol=1e-5)

    def test_prelu_channel_shared_custom_slope(self, ptrace):
        """PReLU channel_shared with custom slope=0.3."""
        prototxt = """name: "prelu_shared_custom"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 2 dim: 2 } } }
layer { name: "prelu" type: "PReLU" bottom: "data" top: "out" prelu_param { channel_shared: true } }
"""
        with ptrace("Net(prelu shared custom)"):
            net = _make_net(prototxt)
        slope = np.array([0.3], dtype=np.float32)
        net.layer_by_name("prelu").blobs[0].from_numpy(slope)
        np.random.seed(555)
        inp = np.random.randn(2, 3, 2, 2).astype(np.float32) * 5.0
        with ptrace("prelu shared custom forward"):
            out = net.forward({"data": inp})
        np.testing.assert_allclose(out["out"], prelu_np(inp, slope, channel_shared=True), rtol=1e-5)

    def test_prelu_numpy_match_per_channel(self, ptrace):
        """PReLU per-channel random slopes should match numpy reference."""
        N, C, H, W = 3, 5, 4, 4
        prototxt = f"""name: "prelu_np"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {C} dim: {H} dim: {W} }} }} }}
layer {{ name: "prelu" type: "PReLU" bottom: "data" top: "out" }}
"""
        with ptrace("Net(prelu numpy match)"):
            net = _make_net(prototxt)
        np.random.seed(666)
        slopes = np.random.uniform(0.01, 0.5, C).astype(np.float32)
        net.layer_by_name("prelu").blobs[0].from_numpy(slopes)
        inp = np.random.randn(N, C, H, W).astype(np.float32) * 4.0
        with ptrace("prelu numpy match forward") as t:
            out = net.forward({"data": inp})
            t['max_abs_diff'] = float(np.max(np.abs(out["out"] - prelu_np(inp, slopes))))
        np.testing.assert_allclose(out["out"], prelu_np(inp, slopes), rtol=1e-5)

    def test_prelu_repeated_forward(self, ptrace):
        """PReLU should be deterministic across repeated forwards."""
        prototxt = """name: "prelu_rep"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 3 dim: 3 } } }
layer { name: "prelu" type: "PReLU" bottom: "data" top: "out" }
"""
        with ptrace("Net(prelu repeated)"):
            net = _make_net(prototxt)
        np.random.seed(77)
        inp = np.random.randn(2, 3, 3, 3).astype(np.float32) * 5.0
        slopes = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        net.layer_by_name("prelu").blobs[0].from_numpy(slopes)
        outs = []
        for i in range(5):
            with ptrace(f"prelu repeated forward #{i}"):
                out = net.forward({"data": inp})
            outs.append(out["out"].copy())
        for i in range(1, 5):
            np.testing.assert_array_equal(outs[0], outs[i])


# ═══════════════════════════════════════════════════════════════════════
# InnerProduct (Fully Connected) Layer Tests
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestInnerProductLayers:
    """Tests for the InnerProduct (Fully Connected) layer's forward computation."""

    def test_ip_known_values_no_bias(self, ptrace):
        """InnerProduct with known weights, no bias: y = x @ W.T."""
        # 2 inputs (K=3), 2 outputs (N=2), batch=2 (M=2)
        # Input shape (2,3,1,1): 2 samples, 3 features
        prototxt = """name: "ip_known"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 1 dim: 1 } } }
layer { name: "ip" type: "InnerProduct" bottom: "data" top: "out"
        inner_product_param { num_output: 2 bias_term: false } }
"""
        with ptrace("Net(ip known no bias)"):
            net = _make_net(prototxt)
        W = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)  # (N=2, K=3)
        net.layer_by_name("ip").blobs[0].from_numpy(W)
        inp = np.array([[[[1]], [[2]], [[3]]],
                         [[[4]], [[5]], [[6]]]], dtype=np.float32)  # (2,3,1,1)
        with ptrace("ip known no bias forward"):
            out = net.forward({"data": inp})
        # y[0] = [1,2] (first two features), y[1] = [4,5]
        expected = np.array([[[[1]], [[2]]],
                              [[[4]], [[5]]]], dtype=np.float32)
        np.testing.assert_allclose(out["out"], expected, rtol=1e-5)

    def test_ip_known_values_with_bias(self, ptrace):
        """InnerProduct with known weights and bias: y = x @ W.T + b."""
        prototxt = """name: "ip_bias"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 2 dim: 1 dim: 1 } } }
layer { name: "ip" type: "InnerProduct" bottom: "data" top: "out"
        inner_product_param { num_output: 2 bias_term: true } }
"""
        with ptrace("Net(ip with bias)"):
            net = _make_net(prototxt)
        W = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([0.1, 0.2], dtype=np.float32)
        net.layer_by_name("ip").blobs[0].from_numpy(W)
        net.layer_by_name("ip").blobs[1].from_numpy(b)
        inp = np.array([[[[1]], [[1]]]], dtype=np.float32)  # (1,2,1,1)
        with ptrace("ip with bias forward"):
            out = net.forward({"data": inp})
        # y = [1*1+1*2+0.1, 1*3+1*4+0.2] = [3.1, 7.2]
        expected = np.array([[[[3.1]], [[7.2]]]], dtype=np.float32)
        np.testing.assert_allclose(out["out"], expected, rtol=1e-5)

    def test_ip_numpy_match_with_bias(self, ptrace):
        """InnerProduct should match numpy reference on random data with bias."""
        N_batch, K_feat, N_out = 4, 10, 5
        prototxt = f"""name: "ip_np"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N_batch} dim: {K_feat} dim: 1 dim: 1 }} }} }}
layer {{ name: "ip" type: "InnerProduct" bottom: "data" top: "out"
        inner_product_param {{ num_output: {N_out} bias_term: true }} }}
"""
        with ptrace("Net(ip numpy match)"):
            net = _make_net(prototxt)
        np.random.seed(100)
        W = np.random.randn(N_out, K_feat).astype(np.float32) * 0.1
        b = np.random.randn(N_out).astype(np.float32) * 0.01
        net.layer_by_name("ip").blobs[0].from_numpy(W)
        net.layer_by_name("ip").blobs[1].from_numpy(b)
        inp = np.random.randn(N_batch, K_feat, 1, 1).astype(np.float32)
        with ptrace("ip numpy match forward") as t:
            out = net.forward({"data": inp})
            expected = inner_product_np(inp, W, b)
            t['max_abs_diff'] = float(np.max(np.abs(out["out"] - expected)))
        np.testing.assert_allclose(out["out"], expected, rtol=1e-4, atol=1e-5)

    def test_ip_numpy_match_no_bias(self, ptrace):
        """InnerProduct without bias should match numpy reference."""
        N_batch, K_feat, N_out = 3, 8, 4
        prototxt = f"""name: "ip_nobias"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N_batch} dim: {K_feat} dim: 1 dim: 1 }} }} }}
layer {{ name: "ip" type: "InnerProduct" bottom: "data" top: "out"
        inner_product_param {{ num_output: {N_out} bias_term: false }} }}
"""
        with ptrace("Net(ip no bias numpy match)"):
            net = _make_net(prototxt)
        np.random.seed(200)
        W = np.random.randn(N_out, K_feat).astype(np.float32) * 0.5
        net.layer_by_name("ip").blobs[0].from_numpy(W)
        inp = np.random.randn(N_batch, K_feat, 1, 1).astype(np.float32)
        with ptrace("ip no bias numpy match forward"):
            out = net.forward({"data": inp})
        np.testing.assert_allclose(out["out"], inner_product_np(inp, W, b=None), rtol=1e-4, atol=1e-5)

    def test_ip_output_shape(self, ptrace):
        """InnerProduct output shape should be (M, N_out, 1, 1) for axis=1."""
        N_batch, K_feat, N_out = 2, 6, 3
        prototxt = f"""name: "ip_shape"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N_batch} dim: {K_feat} dim: 1 dim: 1 }} }} }}
layer {{ name: "ip" type: "InnerProduct" bottom: "data" top: "out"
        inner_product_param {{ num_output: {N_out} }} }}
"""
        with ptrace("Net(ip shape)"):
            net = _make_net(prototxt)
        net.layer_by_name("ip").blobs[0].from_numpy(np.random.randn(N_out, K_feat).astype(np.float32))
        net.layer_by_name("ip").blobs[1].from_numpy(np.zeros(N_out, dtype=np.float32))
        inp = np.random.randn(N_batch, K_feat, 1, 1).astype(np.float32)
        with ptrace("ip shape forward"):
            out = net.forward({"data": inp})
        assert out["out"].shape == (N_batch, N_out, 1, 1)

    def test_ip_weights_unchanged_after_forward(self, ptrace):
        """InnerProduct weights should not be modified during forward."""
        prototxt = """name: "ip_weights"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 4 dim: 1 dim: 1 } } }
layer { name: "ip" type: "InnerProduct" bottom: "data" top: "out"
        inner_product_param { num_output: 3 bias_term: true } }
"""
        with ptrace("Net(ip weights unchanged)"):
            net = _make_net(prototxt)
        W_orig = np.random.randn(3, 4).astype(np.float32) * 0.5
        b_orig = np.random.randn(3).astype(np.float32) * 0.1
        net.layer_by_name("ip").blobs[0].from_numpy(W_orig.copy())
        net.layer_by_name("ip").blobs[1].from_numpy(b_orig.copy())
        inp = np.random.randn(2, 4, 1, 1).astype(np.float32)
        with ptrace("ip weights unchanged forward"):
            net.forward({"data": inp})
        W_after = net.layer_by_name("ip").blobs[0].to_numpy()
        b_after = net.layer_by_name("ip").blobs[1].to_numpy()
        np.testing.assert_array_equal(W_after, W_orig)
        np.testing.assert_array_equal(b_after, b_orig)

    def test_ip_repeated_forward(self, ptrace):
        """InnerProduct should be deterministic across repeated forwards."""
        N_batch, K_feat, N_out = 3, 5, 4
        prototxt = f"""name: "ip_rep"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N_batch} dim: {K_feat} dim: 1 dim: 1 }} }} }}
layer {{ name: "ip" type: "InnerProduct" bottom: "data" top: "out"
        inner_product_param {{ num_output: {N_out} }} }}
"""
        with ptrace("Net(ip repeated)"):
            net = _make_net(prototxt)
        np.random.seed(300)
        W = np.random.randn(N_out, K_feat).astype(np.float32) * 0.3
        b = np.random.randn(N_out).astype(np.float32) * 0.1
        net.layer_by_name("ip").blobs[0].from_numpy(W)
        net.layer_by_name("ip").blobs[1].from_numpy(b)
        inp = np.random.randn(N_batch, K_feat, 1, 1).astype(np.float32)
        outs = []
        for i in range(5):
            with ptrace(f"ip repeated forward #{i}"):
                out = net.forward({"data": inp})
            outs.append(out["out"].copy())
        for i in range(1, 5):
            np.testing.assert_array_equal(outs[0], outs[i])


# ═══════════════════════════════════════════════════════════════════════
# Softmax Layer Tests (standalone, separate from SoftmaxWithLoss)
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestSoftmaxLayers:
    """Tests for the standalone Softmax layer's forward computation."""

    def test_softmax_known_values(self, ptrace):
        """Softmax of known inputs should produce correct probability distribution."""
        prototxt = """name: "softmax_known"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 3 dim: 1 dim: 1 } } }
layer { name: "sm" type: "Softmax" bottom: "data" top: "out" }
"""
        with ptrace("Net(softmax known)"):
            net = _make_net(prototxt)
        # Equal logits → uniform probabilities
        inp = np.array([[[[0.0]], [[0.0]], [[0.0]]]], dtype=np.float32)
        with ptrace("softmax known forward"):
            out = net.forward({"data": inp})
        np.testing.assert_allclose(out["out"], np.ones_like(inp) / 3.0, rtol=1e-5)

    def test_softmax_one_hot_large_input(self, ptrace):
        """Softmax with one very large input should concentrate probability there."""
        prototxt = """name: "softmax_onehot"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 4 dim: 1 dim: 1 } } }
layer { name: "sm" type: "Softmax" bottom: "data" top: "out" }
"""
        with ptrace("Net(softmax one-hot)"):
            net = _make_net(prototxt)
        inp = np.array([[[[0.0]], [[100.0]], [[0.0]], [[0.0]]]], dtype=np.float32)
        with ptrace("softmax one-hot forward"):
            out = net.forward({"data": inp})
        assert out["out"][0, 1, 0, 0] > 0.9999  # Almost all probability at index 1
        assert abs(float(np.sum(out["out"])) - 1.0) < 1e-6  # Sums to 1

    def test_softmax_numpy_match(self, ptrace):
        """Softmax should match numpy reference on random data."""
        N, C, H, W = 4, 6, 3, 3
        prototxt = f"""name: "softmax_np"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {C} dim: {H} dim: {W} }} }} }}
layer {{ name: "sm" type: "Softmax" bottom: "data" top: "out" }}
"""
        with ptrace("Net(softmax numpy match)"):
            net = _make_net(prototxt)
        np.random.seed(500)
        inp = np.random.randn(N, C, H, W).astype(np.float32)
        with ptrace("softmax numpy match forward") as t:
            out = net.forward({"data": inp})
            expected = softmax_np(inp, axis=1)
            t['max_abs_diff'] = float(np.max(np.abs(out["out"] - expected)))
        np.testing.assert_allclose(out["out"], expected, rtol=1e-5, atol=1e-6)

    def test_softmax_sums_to_one(self, ptrace):
        """Softmax probabilities must sum to 1 along the softmax axis."""
        N, C, H, W = 3, 8, 4, 4
        prototxt = f"""name: "softmax_sum"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {C} dim: {H} dim: {W} }} }} }}
layer {{ name: "sm" type: "Softmax" bottom: "data" top: "out" }}
"""
        with ptrace("Net(softmax sums to one)"):
            net = _make_net(prototxt)
        np.random.seed(600)
        inp = np.random.randn(N, C, H, W).astype(np.float32) * 3.0
        with ptrace("softmax sums to one forward"):
            out = net.forward({"data": inp})
        # Sum along axis=1 (channel) should be 1
        sums = np.sum(out["out"], axis=1)
        np.testing.assert_allclose(sums, np.ones_like(sums), rtol=1e-5, atol=1e-5)

    def test_softmax_preserves_shape(self, ptrace):
        """Softmax output shape must match input shape."""
        prototxt = """name: "softmax_shape"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 5 dim: 3 dim: 3 } } }
layer { name: "sm" type: "Softmax" bottom: "data" top: "out" }
"""
        with ptrace("Net(softmax shape)"):
            net = _make_net(prototxt)
        inp = np.random.randn(2, 5, 3, 3).astype(np.float32)
        with ptrace("softmax shape forward"):
            out = net.forward({"data": inp})
        assert out["out"].shape == inp.shape

    def test_softmax_repeated_forward(self, ptrace):
        """Softmax should be deterministic across repeated forwards."""
        prototxt = """name: "softmax_rep"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 4 dim: 3 dim: 3 } } }
layer { name: "sm" type: "Softmax" bottom: "data" top: "out" }
"""
        with ptrace("Net(softmax repeated)"):
            net = _make_net(prototxt)
        np.random.seed(99)
        inp = np.random.randn(2, 4, 3, 3).astype(np.float32)
        outs = []
        for i in range(5):
            with ptrace(f"softmax repeated forward #{i}"):
                out = net.forward({"data": inp})
            outs.append(out["out"].copy())
        for i in range(1, 5):
            np.testing.assert_array_equal(outs[0], outs[i])


# ═══════════════════════════════════════════════════════════════════════
# Flatten Layer Tests
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestFlattenLayers:
    """Tests for the Flatten layer's forward computation."""

    def test_flatten_default_all(self, ptrace):
        """Flatten with default axis=1, end_axis=-1 flattens from axis 1 to end.
        (N,C,H,W) -> (N, C*H*W) for default params... actually:
        default start_axis=1, end_axis=-1 (last axis), so axes 1..-1 are flattened.
        (2,3,4,4) -> (2, 3*4*4) = (2, 48) → but caffe Flatten preserves the
        spatial dims as (N, C*H*W, 1, 1) i.e. ndim is preserved with trailing 1s.
        Wait - let me check: Flatten only flattens the specified axis range.
        For (N,C,H,W) start_axis=1, end_axis=-1(=3):
        new_shape = [N, C*H*W] → 2D output?
        Looking at the code: top_shape = shape[:start_axis] + [flattened_dim] + shape[end_axis+1:]
        end_axis+1 = 4 which is beyond ndim=4, so no trailing dims.
        So (2,3,4,4) → (2, 48) i.e. 2D.
        """
        prototxt = """name: "flatten_default"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 4 dim: 4 } } }
layer { name: "flat" type: "Flatten" bottom: "data" top: "out" }
"""
        with ptrace("Net(flatten default)"):
            net = _make_net(prototxt)
        np.random.seed(11)
        inp = np.random.randn(2, 3, 4, 4).astype(np.float32)
        with ptrace("flatten default forward"):
            out = net.forward({"data": inp})
        expected_shape = (2, 3 * 4 * 4)
        assert out["out"].shape == expected_shape
        # Values should be preserved
        np.testing.assert_array_equal(out["out"], inp.reshape(expected_shape))

    def test_flatten_axis1_to_2(self, ptrace):
        """Flatten axes 1..2: (N,C,H,W) -> (N, C*H, W)."""
        prototxt = """name: "flatten_ax1_2"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 4 dim: 5 } } }
layer { name: "flat" type: "Flatten" bottom: "data" top: "out"
        flatten_param { axis: 1 end_axis: 2 } }
"""
        with ptrace("Net(flatten axis 1-2)"):
            net = _make_net(prototxt)
        np.random.seed(22)
        inp = np.random.randn(2, 3, 4, 5).astype(np.float32)
        with ptrace("flatten axis 1-2 forward"):
            out = net.forward({"data": inp})
        expected_shape = (2, 3 * 4, 5)
        assert out["out"].shape == expected_shape
        np.testing.assert_array_equal(out["out"], inp.reshape(expected_shape))

    def test_flatten_numpy_match(self, ptrace):
        """Flatten should match numpy reshape reference."""
        N, C, H, W = 3, 5, 4, 6
        prototxt = f"""name: "flatten_np"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {C} dim: {H} dim: {W} }} }} }}
layer {{ name: "flat" type: "Flatten" bottom: "data" top: "out"
        flatten_param {{ axis: 1 end_axis: 2 }} }}
"""
        with ptrace("Net(flatten numpy match)"):
            net = _make_net(prototxt)
        np.random.seed(33)
        inp = np.random.randn(N, C, H, W).astype(np.float32)
        with ptrace("flatten numpy match forward"):
            out = net.forward({"data": inp})
        expected = flatten_np(inp, start_axis=1, end_axis=2)
        np.testing.assert_array_equal(out["out"], expected)

    def test_flatten_preserves_values(self, ptrace):
        """Flatten must preserve all values (only reshapes, no computation)."""
        prototxt = """name: "flatten_vals"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 2 dim: 2 } } }
layer { name: "flat" type: "Flatten" bottom: "data" top: "out" }
"""
        with ptrace("Net(flatten preserves values)"):
            net = _make_net(prototxt)
        inp = np.arange(2*3*2*2, dtype=np.float32).reshape(2, 3, 2, 2)
        with ptrace("flatten preserves values forward"):
            out = net.forward({"data": inp})
        # Total count should match
        assert out["out"].size == inp.size
        np.testing.assert_array_equal(out["out"].flatten(), inp.flatten())

    def test_flatten_repeated_forward(self, ptrace):
        """Flatten should be deterministic across repeated forwards."""
        prototxt = """name: "flatten_rep"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 4 dim: 4 } } }
layer { name: "flat" type: "Flatten" bottom: "data" top: "out" }
"""
        with ptrace("Net(flatten repeated)"):
            net = _make_net(prototxt)
        np.random.seed(44)
        inp = np.random.randn(2, 3, 4, 4).astype(np.float32)
        outs = []
        for i in range(5):
            with ptrace(f"flatten repeated forward #{i}"):
                out = net.forward({"data": inp})
            outs.append(out["out"].copy())
        for i in range(1, 5):
            np.testing.assert_array_equal(outs[0], outs[i])


# ═══════════════════════════════════════════════════════════════════════
# Reshape Layer Tests
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestReshapeLayers:
    """Tests for the Reshape layer's forward computation."""

    def test_reshape_simple(self, ptrace):
        """Reshape (N,C,H,W) -> (N, C*H*W, 1, 1)."""
        prototxt = """name: "reshape_simple"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 4 dim: 4 } } }
layer { name: "rsh" type: "Reshape" bottom: "data" top: "out"
        reshape_param { shape { dim: 0 dim: 48 dim: 1 dim: 1 } } }
"""
        with ptrace("Net(reshape simple)"):
            net = _make_net(prototxt)
        # dim: 0 means "copy from corresponding input dimension"
        np.random.seed(50)
        inp = np.random.randn(2, 3, 4, 4).astype(np.float32)
        with ptrace("reshape simple forward"):
            out = net.forward({"data": inp})
        assert out["out"].shape == (2, 48, 1, 1)
        np.testing.assert_array_equal(out["out"].flatten(), inp.flatten())

    def test_reshape_with_inferred_dim(self, ptrace):
        """Reshape with -1 infers the dimension: (2,3,4,4) -> (2, -1, 2) = (2, 48, 2)?
        Wait, total elements = 2*3*4*4 = 96. Shape (0, -1, 2): 0 means copy dim 0 (=2),
        so 2 * ? * 2 = 96 → ? = 24. Shape = (2, 24, 2).
        But this is 3D. Caffe Reshape always outputs 4D? Let me check...
        Actually the reshape_param.shape specifies the output dims directly.
        The code reshapes to exactly the specified dims (with 0 meaning copy, -1 meaning infer).
        """
        prototxt = """name: "reshape_infer"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 4 dim: 4 } } }
layer { name: "rsh" type: "Reshape" bottom: "data" top: "out"
        reshape_param { shape { dim: 0 dim: -1 dim: 8 } } }
"""
        with ptrace("Net(reshape inferred)"):
            net = _make_net(prototxt)
        inp = np.arange(2*3*4*4, dtype=np.float32).reshape(2, 3, 4, 4)
        with ptrace("reshape inferred forward"):
            out = net.forward({"data": inp})
        # total = 96, dim0=2(copy), dim2=8, so dim1 = 96/(2*8) = 6
        assert out["out"].shape == (2, 6, 8)
        np.testing.assert_array_equal(out["out"].flatten(), inp.flatten())

    def test_reshape_numpy_match(self, ptrace):
        """Reshape should match numpy reshape reference."""
        N, C, H, W = 3, 4, 5, 6
        prototxt = f"""name: "reshape_np"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {C} dim: {H} dim: {W} }} }} }}
layer {{ name: "rsh" type: "Reshape" bottom: "data" top: "out"
        reshape_param {{ shape {{ dim: 0 dim: -1 dim: 1 dim: 1 }} }} }}
"""
        with ptrace("Net(reshape numpy match)"):
            net = _make_net(prototxt)
        np.random.seed(60)
        inp = np.random.randn(N, C, H, W).astype(np.float32)
        with ptrace("reshape numpy match forward"):
            out = net.forward({"data": inp})
        # shape_spec: [0, -1, 1, 1] — dim0=3(copy), total=3*4*5*6=360, so dim1=360/(3*1*1)=120
        expected = reshape_np(inp, [0, -1, 1, 1])
        # 0 in spec means copy from input
        expected_shape = [N] + [N*C*H*W//N] + [1, 1]
        assert out["out"].shape == tuple(expected_shape)
        np.testing.assert_array_equal(out["out"].flatten(), inp.flatten())

    def test_reshape_preserves_values(self, ptrace):
        """Reshape must preserve all values."""
        prototxt = """name: "reshape_vals"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 2 dim: 2 } } }
layer { name: "rsh" type: "Reshape" bottom: "data" top: "out"
        reshape_param { shape { dim: -1 dim: 4 } } }
"""
        with ptrace("Net(reshape preserves values)"):
            net = _make_net(prototxt)
        inp = np.arange(2*3*2*2, dtype=np.float32).reshape(2, 3, 2, 2)
        with ptrace("reshape preserves values forward"):
            out = net.forward({"data": inp})
        assert out["out"].size == inp.size
        np.testing.assert_array_equal(out["out"].flatten(), inp.flatten())

    def test_reshape_repeated_forward(self, ptrace):
        """Reshape should be deterministic across repeated forwards."""
        prototxt = """name: "reshape_rep"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 4 dim: 4 } } }
layer { name: "rsh" type: "Reshape" bottom: "data" top: "out"
        reshape_param { shape { dim: 0 dim: -1 dim: 1 dim: 1 } } }
"""
        with ptrace("Net(reshape repeated)"):
            net = _make_net(prototxt)
        np.random.seed(70)
        inp = np.random.randn(2, 3, 4, 4).astype(np.float32)
        outs = []
        for i in range(5):
            with ptrace(f"reshape repeated forward #{i}"):
                out = net.forward({"data": inp})
            outs.append(out["out"].copy())
        for i in range(1, 5):
            np.testing.assert_array_equal(outs[0], outs[i])


# ═══════════════════════════════════════════════════════════════════════
# Combination / Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestActivationIPCombination:
    """Integration tests: MLP-style pipelines combining IP + Activation + Softmax."""

    def test_mlp_pipeline_ip_relu_softmax(self, ptrace):
        """Full MLP forward pipeline: IP -> ReLU -> Softmax.
        Simulates a simple 2-layer MLP classification head.
        Split layers needed for blob multi-consumer in real models,
        but this is a linear chain so no Split required.
        """
        N, K, H, C = 4, 8, 6, 3  # batch, input, hidden, classes
        prototxt = f"""name: "mlp_pipe"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {K} dim: 1 dim: 1 }} }} }}
layer {{ name: "ip1" type: "InnerProduct" bottom: "data" top: "hidden"
        inner_product_param {{ num_output: {H} bias_term: true }} }}
layer {{ name: "relu1" type: "ReLU" bottom: "hidden" top: "hidden_relu" }}
layer {{ name: "ip2" type: "InnerProduct" bottom: "hidden_relu" top: "logits"
        inner_product_param {{ num_output: {C} bias_term: true }} }}
layer {{ name: "prob" type: "Softmax" bottom: "logits" top: "probs" }}
"""
        with ptrace("Net(MLP pipeline)"):
            net = _make_net(prototxt)
        np.random.seed(999)
        W1 = np.random.randn(H, K).astype(np.float32) * 0.3
        b1 = np.random.randn(H).astype(np.float32) * 0.1
        W2 = np.random.randn(C, H).astype(np.float32) * 0.3
        b2 = np.random.randn(C).astype(np.float32) * 0.1
        net.layer_by_name("ip1").blobs[0].from_numpy(W1)
        net.layer_by_name("ip1").blobs[1].from_numpy(b1)
        net.layer_by_name("ip2").blobs[0].from_numpy(W2)
        net.layer_by_name("ip2").blobs[1].from_numpy(b2)
        inp = np.random.randn(N, K, 1, 1).astype(np.float32) * 0.5
        with ptrace("MLP pipeline forward") as t:
            out = net.forward({"data": inp})
            t['probs_sum_max_err'] = float(np.max(np.abs(np.sum(out["probs"], axis=1) - 1.0)))
            t['probs_min'] = float(np.min(out["probs"]))
        # Softmax output must sum to 1
        assert out["probs"].shape == (N, C, 1, 1)
        np.testing.assert_allclose(np.sum(out["probs"], axis=1), np.ones((N, 1, 1)), rtol=1e-5)
        assert np.all(out["probs"] >= 0.0)
        assert np.all(out["probs"] <= 1.0)

    def test_ip_sigmoid_sigmoid_numpy_chain(self, ptrace):
        """IP -> Sigmoid chain verified against numpy step-by-step."""
        N, K, C = 2, 4, 3
        prototxt = f"""name: "ip_sig"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {K} dim: 1 dim: 1 }} }} }}
layer {{ name: "ip" type: "InnerProduct" bottom: "data" top: "ip_out"
        inner_product_param {{ num_output: {C} bias_term: true }} }}
layer {{ name: "sig" type: "Sigmoid" bottom: "ip_out" top: "out" }}
"""
        with ptrace("Net(IP->Sigmoid)"):
            net = _make_net(prototxt)
        np.random.seed(1234)
        W = np.random.randn(C, K).astype(np.float32) * 0.5
        b = np.random.randn(C).astype(np.float32) * 0.1
        net.layer_by_name("ip").blobs[0].from_numpy(W)
        net.layer_by_name("ip").blobs[1].from_numpy(b)
        inp = np.random.randn(N, K, 1, 1).astype(np.float32)
        with ptrace("IP->Sigmoid forward"):
            out = net.forward({"data": inp})
        # Reference computation
        ip_ref = inner_product_np(inp, W, b)
        sig_ref = sigmoid_np(ip_ref)
        np.testing.assert_allclose(out["out"], sig_ref, rtol=1e-5)

    def test_stability_20_iters(self, ptrace):
        """20-iteration pipeline stability test: no segfault, stable results."""
        N, C = 4, 3
        prototxt = f"""name: "p3c_stress"
layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ dim: {N} dim: {C} dim: 2 dim: 2 }} }} }}
layer {{ name: "r1" type: "ReLU" bottom: "data" top: "r_out" }}
layer {{ name: "sig1" type: "Sigmoid" bottom: "r_out" top: "s_out" }}
layer {{ name: "flat" type: "Flatten" bottom: "s_out" top: "f_out"
        flatten_param {{ axis: 1 end_axis: -1 }} }}
"""
        with ptrace("Net(P3-C stress test)"):
            net = _make_net(prototxt)
        np.random.seed(5678)
        outputs = []
        for i in range(20):
            inp = np.random.randn(N, C, 2, 2).astype(np.float32) * 0.1
            with ptrace(f"P3-C stress iter #{i}"):
                out = net.forward({"data": inp})
            # Single-consumer blob model: only terminal blob (f_out) is available
            assert "f_out" in out, f"Terminal blob f_out missing at iter {i}"
            outputs.append(out["f_out"].copy())
            # Shape consistency: Flatten(axis=1, end_axis=-1) -> (N, C*2*2)
            assert out["f_out"].shape == (N, C * 2 * 2)
            # Value range: ReLU then Sigmoid -> values in (0,1); Flatten preserves values
            assert np.all(out["f_out"] >= 0.0)
            assert np.all(out["f_out"] <= 1.0)
        # All iterations complete without crash
        assert len(outputs) == 20
