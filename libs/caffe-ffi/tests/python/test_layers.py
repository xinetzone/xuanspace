from __future__ import annotations

import numpy as np
import pytest

import caffe_ffi
from caffe_ffi import caffe_pb2, net_param_from_string, net_from_param, Blob, Layer
from .conftest import require_cpp_extension


class TestLayerType:
    def test_layer_type_property(self):
        layer = Layer()
        assert layer.type == ""

    def test_layer_repr(self):
        layer = Layer()
        r = repr(layer)
        assert "Layer" in r


class TestInputLayer:
    def test_input_layer_in_net(self):
        prototxt = """name: "input_test"
input: "data"
input_shape { dim: 1 dim: 3 }
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 3 } }
}
"""
        param = net_param_from_string(prototxt)
        assert len(param.layer) == 1
        assert param.layer[0].type == "Input"

    @require_cpp_extension
    def test_input_layer_forward(self):
        prototxt = """name: "input_test"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 2 dim: 3 } }
}
"""
        net = net_from_param(net_param_from_string(prototxt))
        inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        out = net.forward({"data": inp})
        np.testing.assert_array_equal(net["data"].data, inp)


class TestReLU:
    def relu_np(self, x, negative_slope=0.0):
        return np.where(x > 0, x, x * negative_slope)

    @require_cpp_extension
    def test_relu_positive_unchanged(self):
        prototxt = """name: "relu_test"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 } } }
layer { name: "relu" type: "ReLU" bottom: "data" top: "out" }
"""
        net = net_from_param(net_param_from_string(prototxt))
        inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        out = net.forward({"data": inp})
        np.testing.assert_array_equal(out["out"], self.relu_np(inp))

    @require_cpp_extension
    def test_relu_negative_zero(self):
        prototxt = """name: "relu_test"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 } } }
layer { name: "relu" type: "ReLU" bottom: "data" top: "out" }
"""
        net = net_from_param(net_param_from_string(prototxt))
        inp = np.array([[-1, -2, 3], [-4, 5, -6]], dtype=np.float32)
        out = net.forward({"data": inp})
        expected = self.relu_np(inp)
        np.testing.assert_array_equal(out["out"], expected)

    def test_relu_numpy_reference_positive(self):
        x = np.array([1, 2, 3], dtype=np.float32)
        assert np.all(self.relu_np(x) == x)

    def test_relu_numpy_reference_negative(self):
        x = np.array([-1, -2, 3], dtype=np.float32)
        expected = np.array([0, 0, 3], dtype=np.float32)
        np.testing.assert_array_equal(self.relu_np(x), expected)

    def test_relu_numpy_reference_negative_slope(self):
        x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        expected = np.array([-0.2, -0.1, 0, 1, 2], dtype=np.float32)
        np.testing.assert_allclose(self.relu_np(x, negative_slope=0.1), expected)


class TestInnerProduct:
    def inner_product_np(self, x, W, b):
        return x @ W.T + b

    @require_cpp_extension
    def test_inner_product_matmul_bias(self):
        prototxt = """name: "ip_test"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 } } }
layer {
  name: "ip"
  type: "InnerProduct"
  bottom: "data"
  top: "ip"
  inner_product_param { num_output: 2 bias_term: true }
}
"""
        net = net_from_param(net_param_from_string(prototxt))
        W = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        b = np.array([0.1, 0.2], dtype=np.float32)
        net.layers_array()[1].blobs[0].from_numpy(W)
        net.layers_array()[1].blobs[1].from_numpy(b.reshape(-1))
        inp = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        out = net.forward({"data": inp})
        expected = self.inner_product_np(inp, W, b)
        np.testing.assert_allclose(out["ip"], expected, rtol=1e-5)

    def test_inner_product_numpy_reference(self):
        x = np.array([[1, 2, 3]], dtype=np.float32)
        W = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        b = np.array([0, 0], dtype=np.float32)
        expected = np.array([[1, 2]], dtype=np.float32)
        np.testing.assert_array_equal(self.inner_product_np(x, W, b), expected)


class TestSoftmax:
    def softmax_np(self, x, axis=1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    @require_cpp_extension
    def test_softmax_sums_to_one(self):
        prototxt = """name: "softmax_test"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 } } }
layer { name: "prob" type: "Softmax" bottom: "data" top: "prob" }
"""
        net = net_from_param(net_param_from_string(prototxt))
        inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        out = net.forward({"data": inp})
        np.testing.assert_allclose(out["prob"].sum(axis=1), np.array([1.0, 1.0]), rtol=1e-5)

    @require_cpp_extension
    def test_softmax_all_zero_uniform(self):
        prototxt = """name: "softmax_test"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 1 dim: 3 } } }
layer { name: "prob" type: "Softmax" bottom: "data" top: "prob" }
"""
        net = net_from_param(net_param_from_string(prototxt))
        inp = np.zeros((1, 3), dtype=np.float32)
        out = net.forward({"data": inp})
        expected = np.ones((1, 3), dtype=np.float32) / 3.0
        np.testing.assert_allclose(out["prob"], expected, rtol=1e-5)

    def test_softmax_numpy_reference_sum_one(self):
        x = np.array([[1, 2, 3]], dtype=np.float32)
        s = self.softmax_np(x)
        np.testing.assert_allclose(s.sum(), 1.0, rtol=1e-6)

    def test_softmax_numpy_reference_uniform(self):
        x = np.zeros((1, 4), dtype=np.float32)
        s = self.softmax_np(x)
        np.testing.assert_allclose(s, np.full((1, 4), 0.25), rtol=1e-6)


class TestFlatten:
    def flatten_np(self, x, axis=1, end_axis=-1):
        shape = x.shape
        if end_axis < 0:
            end_axis = len(shape) + end_axis
        new_shape = list(shape[:axis])
        new_shape.append(-1)
        new_shape.extend(shape[end_axis + 1:])
        return x.reshape(new_shape)

    @require_cpp_extension
    def test_flatten_shape(self):
        prototxt = """name: "flatten_test"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 dim: 4 dim: 5 } } }
layer { name: "flat" type: "Flatten" bottom: "data" top: "flat" }
"""
        net = net_from_param(net_param_from_string(prototxt))
        inp = np.random.randn(2, 3, 4, 5).astype(np.float32)
        out = net.forward({"data": inp})
        assert out["flat"].shape == (2, 60)

    def test_flatten_numpy_reference(self):
        x = np.zeros((2, 3, 4, 5), dtype=np.float32)
        f = self.flatten_np(x)
        assert f.shape == (2, 60)

    def test_flatten_numpy_reference_end_axis(self):
        x = np.zeros((2, 3, 4, 5), dtype=np.float32)
        f = self.flatten_np(x, axis=1, end_axis=2)
        assert f.shape == (2, 12, 5)


class TestSigmoid:
    def sigmoid_np(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def test_sigmoid_numpy_output_range(self):
        x = np.array([-10, -1, 0, 1, 10], dtype=np.float32)
        y = self.sigmoid_np(x)
        assert np.all(y > 0) and np.all(y < 1)

    def test_sigmoid_numpy_zero(self):
        x = np.array([0.0], dtype=np.float32)
        np.testing.assert_allclose(self.sigmoid_np(x), np.array([0.5]), rtol=1e-6)

    def test_sigmoid_numpy_symmetric(self):
        x = np.array([1.0, -1.0], dtype=np.float32)
        y = self.sigmoid_np(x)
        np.testing.assert_allclose(y[0] + y[1], 1.0, rtol=1e-6)


class TestTanH:
    def tanh_np(self, x):
        return np.tanh(x)

    def test_tanh_numpy_output_range(self):
        x = np.array([-1, -0.5, 0, 0.5, 1], dtype=np.float32)
        y = self.tanh_np(x)
        assert np.all(y >= -1.0) and np.all(y <= 1.0)

    def test_tanh_numpy_zero(self):
        x = np.array([0.0], dtype=np.float32)
        np.testing.assert_allclose(self.tanh_np(x), np.array([0.0]), rtol=1e-6, atol=1e-7)

    def test_tanh_numpy_symmetric(self):
        x = np.array([1.0, -1.0], dtype=np.float32)
        y = self.tanh_np(x)
        np.testing.assert_allclose(y[0], -y[1], rtol=1e-6)


class TestPReLU:
    def prelu_np(self, x, slope=0.25, channel_shared=True, channels=None):
        if channel_shared:
            return np.where(x > 0, x, slope * x)
        else:
            assert channels is not None
            slope_arr = np.asarray(slope, dtype=np.float32).reshape(channels)
            shape = [1] * x.ndim
            shape[1] = channels
            slope_bc = slope_arr.reshape(shape)
            return np.where(x > 0, x, slope_bc * x)

    def test_prelu_numpy_channel_shared(self):
        x = np.array([[-2, -1, 0, 1, 2]], dtype=np.float32)
        y = self.prelu_np(x, slope=0.25)
        expected = np.array([[-0.5, -0.25, 0, 1, 2]], dtype=np.float32)
        np.testing.assert_allclose(y, expected, rtol=1e-5)

    def test_prelu_numpy_positive_unchanged(self):
        x = np.array([[1, 2, 3]], dtype=np.float32)
        y = self.prelu_np(x)
        np.testing.assert_array_equal(y, x)

    def test_prelu_numpy_per_channel(self):
        x = np.ones((2, 3, 4, 4), dtype=np.float32)
        x[:, 0] = -1.0
        x[:, 2] = -2.0
        slopes = np.array([0.1, 0.0, 0.5], dtype=np.float32)
        y = self.prelu_np(x, slope=slopes, channel_shared=False, channels=3)
        assert y[0, 0, 0, 0] == pytest.approx(-0.1, abs=1e-5)
        assert y[0, 1, 0, 0] == pytest.approx(1.0, abs=1e-5)
        assert y[0, 2, 0, 0] == pytest.approx(-1.0, abs=1e-5)


class TestELU:
    def elu_np(self, x, alpha=1.0):
        return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

    def test_elu_numpy_positive_identity(self):
        x = np.array([0, 1, 2, 3], dtype=np.float32)
        np.testing.assert_array_equal(self.elu_np(x), x)

    def test_elu_numpy_negative(self):
        x = np.array([-1.0], dtype=np.float32)
        y = self.elu_np(x, alpha=1.0)
        expected = np.exp(np.array([-1.0], dtype=np.float32)) - 1.0
        np.testing.assert_allclose(y, expected, rtol=1e-5)

    def test_elu_numpy_alpha(self):
        x = np.array([-1.0], dtype=np.float32)
        y = self.elu_np(x, alpha=0.5)
        expected = 0.5 * (np.exp(np.array([-1.0], dtype=np.float32)) - 1.0)
        np.testing.assert_allclose(y, expected, rtol=1e-5)


class TestDropout:
    def dropout_np(self, x):
        return x.copy()

    def test_dropout_numpy_identity(self):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y = self.dropout_np(x)
        np.testing.assert_array_equal(y, x)


class TestConcat:
    def concat_np(self, arrays, axis=0):
        return np.concatenate(arrays, axis=axis)

    def test_concat_numpy_axis0(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        b = np.array([[5, 6], [7, 8]], dtype=np.float32)
        y = self.concat_np([a, b], axis=0)
        assert y.shape == (4, 2)
        expected = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        np.testing.assert_array_equal(y, expected)

    def test_concat_numpy_axis1(self):
        a = np.ones((2, 3), dtype=np.float32)
        b = np.zeros((2, 4), dtype=np.float32)
        y = self.concat_np([a, b], axis=1)
        assert y.shape == (2, 7)
        np.testing.assert_array_equal(y[:, :3], a)
        np.testing.assert_array_equal(y[:, 3:], b)

    def test_concat_numpy_three_inputs(self):
        a = np.ones((1, 2), dtype=np.float32)
        b = np.zeros((1, 3), dtype=np.float32)
        c = np.full((1, 4), 2.0, dtype=np.float32)
        y = self.concat_np([a, b, c], axis=1)
        assert y.shape == (1, 9)

    def test_concat_numpy_axis2_3d(self):
        a = np.ones((2, 3, 4), dtype=np.float32)
        b = np.zeros((2, 3, 5), dtype=np.float32)
        y = self.concat_np([a, b], axis=2)
        assert y.shape == (2, 3, 9)


class TestEltwise:
    def eltwise_sum_np(self, arrays, coeffs=None):
        if coeffs is None:
            coeffs = [1.0] * len(arrays)
        result = coeffs[0] * arrays[0]
        for i in range(1, len(arrays)):
            result = result + coeffs[i] * arrays[i]
        return result

    def eltwise_prod_np(self, arrays, coeffs=None):
        if coeffs is None:
            coeffs = [1.0] * len(arrays)
        result = coeffs[0] * arrays[0]
        for i in range(1, len(arrays)):
            result = result * (coeffs[i] * arrays[i])
        return result

    def eltwise_max_np(self, arrays, coeffs=None):
        if coeffs is None:
            coeffs = [1.0] * len(arrays)
        result = coeffs[0] * arrays[0]
        for i in range(1, len(arrays)):
            result = np.maximum(result, coeffs[i] * arrays[i])
        return result

    def test_eltwise_sum_numpy(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)
        y = self.eltwise_sum_np([a, b])
        np.testing.assert_array_equal(y, np.array([5, 7, 9], dtype=np.float32))

    def test_eltwise_sum_with_coeffs(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)
        y = self.eltwise_sum_np([a, b], coeffs=[2.0, 0.5])
        expected = 2.0 * a + 0.5 * b
        np.testing.assert_allclose(y, expected, rtol=1e-6)

    def test_eltwise_prod_numpy(self):
        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)
        y = self.eltwise_prod_np([a, b])
        np.testing.assert_array_equal(y, np.array([4, 10, 18], dtype=np.float32))

    def test_eltwise_max_numpy(self):
        a = np.array([1, 5, 3], dtype=np.float32)
        b = np.array([4, 2, 6], dtype=np.float32)
        y = self.eltwise_max_np([a, b])
        np.testing.assert_array_equal(y, np.array([4, 5, 6], dtype=np.float32))


class TestReshape:
    def reshape_np(self, x, new_shape):
        return x.reshape(new_shape)

    def test_reshape_numpy_basic(self):
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        y = self.reshape_np(x, (6, 4))
        assert y.shape == (6, 4)
        np.testing.assert_array_equal(y.flatten(), x.flatten())

    def test_reshape_numpy_infer(self):
        x = np.zeros((2, 3, 4), dtype=np.float32)
        y = self.reshape_np(x, (-1,))
        assert y.shape == (24,)

    def test_reshape_numpy_infer_2d(self):
        x = np.zeros((2, 12), dtype=np.float32)
        y = self.reshape_np(x, (4, -1))
        assert y.shape == (4, 6)

    def test_reshape_numpy_data_unchanged(self):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y = self.reshape_np(x, (3, 2))
        np.testing.assert_array_equal(y.flatten(), x.flatten())

    def test_reshape_numpy_same_data(self):
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        y = self.reshape_np(x, (2, 12))
        np.testing.assert_array_equal(y.ravel(), x.ravel())
