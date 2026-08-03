"""Flatten layer backward tests.

Flatten is a pure shape-transformation layer:
  - Forward: data is copied (reshape is a view change)
  - Backward: top_diff is copied back to bottom_diff (identity pass-through)
"""
import textwrap
import numpy as np
import pytest
from caffe_ffi import Net

from .conftest import require_cpp_extension


def _make_flatten_proto(input_shape, axis=1, end_axis=-1):
    dims_str = " ".join(str(d) for d in input_shape)
    return textwrap.dedent(f"""\
        name: "flatten_test"
        input: "data"
        input_shape {{ dim: {dims_str.replace(' ', ' dim: ')} }}
        layer {{
          name: "flatten"
          type: "Flatten"
          bottom: "data"
          top: "flat"
          flatten_param {{ axis: {axis} end_axis: {end_axis} }}
        }}
    """)


def _flatten_shape(shape, axis, end_axis):
    ndim = len(shape)
    if axis < 0:
        axis = ndim + axis
    if end_axis < 0:
        end_axis = ndim + end_axis
    out = list(shape[:axis])
    flat_dim = 1
    for i in range(axis, end_axis + 1):
        flat_dim *= shape[i]
    out.append(flat_dim)
    out.extend(shape[end_axis + 1:])
    return tuple(out)


# Common 4D shapes used in CNNs (N, C, H, W)
_4D_SHAPES = [
    # Small batches, common channels
    (1, 1, 1, 1), (1, 1, 2, 2), (1, 1, 3, 3), (1, 1, 4, 4), (1, 1, 5, 5),
    (1, 1, 7, 7), (1, 1, 8, 8), (1, 1, 14, 14), (1, 1, 28, 28),
    (1, 3, 1, 1), (1, 3, 2, 2), (1, 3, 3, 3), (1, 3, 4, 4), (1, 3, 8, 8),
    (1, 3, 16, 16), (1, 3, 32, 32), (1, 3, 56, 56),
    (1, 8, 1, 1), (1, 8, 2, 2), (1, 8, 3, 3), (1, 8, 7, 7), (1, 8, 14, 14),
    (1, 16, 1, 1), (1, 16, 2, 2), (1, 16, 3, 3), (1, 16, 7, 7), (1, 16, 14, 14),
    (1, 32, 1, 1), (1, 32, 2, 2), (1, 32, 3, 3), (1, 32, 7, 7),
    (1, 64, 1, 1), (1, 64, 2, 2), (1, 64, 3, 3), (1, 64, 7, 7),
    (1, 128, 1, 1), (1, 128, 2, 2), (1, 128, 3, 3), (1, 128, 7, 7),
    # Batch > 1
    (2, 1, 1, 1), (2, 1, 28, 28), (2, 3, 4, 4), (2, 3, 8, 8), (2, 3, 32, 32),
    (2, 8, 2, 2), (2, 8, 3, 3), (2, 8, 7, 7),
    (2, 16, 3, 3), (2, 16, 7, 7), (2, 32, 3, 3), (2, 64, 3, 3),
    (4, 1, 28, 28), (4, 3, 4, 4), (4, 3, 8, 8), (4, 3, 32, 32),
    (4, 8, 3, 3), (4, 16, 3, 3), (4, 32, 7, 7), (4, 64, 7, 7),
    (8, 1, 28, 28), (8, 3, 8, 8), (8, 16, 3, 3), (8, 32, 3, 3),
    (16, 1, 28, 28), (16, 3, 8, 8), (16, 16, 3, 3), (16, 32, 7, 7),
    (32, 1, 28, 28), (32, 3, 8, 8), (32, 16, 3, 3), (32, 64, 7, 7),
]

# Filter out shapes too large for numerical gradient (>200 elements)
_SMALL_4D_SHAPES = [s for s in _4D_SHAPES if int(np.prod(s)) <= 200]

# 2D shapes (batch, features)
_2D_SHAPES = [
    (1, 1), (1, 2), (1, 10), (1, 100), (1, 784), (1, 1000),
    (2, 1), (2, 2), (2, 10), (2, 100), (2, 784),
    (4, 10), (4, 100), (4, 500), (8, 10), (8, 100),
    (16, 10), (16, 100), (32, 10), (32, 100), (64, 10), (64, 100),
    (100, 1), (100, 10), (100, 100), (1, 4096),
]

# 1D shapes
_1D_SHAPES = [1, 2, 5, 10, 100, 128, 256, 512, 784, 1000, 4096]

# 5D shapes
_5D_SHAPES = [
    (1, 1, 1, 1, 1), (1, 3, 2, 2, 2), (2, 8, 2, 2, 2),
    (1, 16, 2, 3, 4), (2, 3, 4, 5, 6), (4, 8, 2, 2, 2),
]

# 3D shapes
_3D_SHAPES = [
    (1, 1, 1), (1, 10, 10), (2, 3, 4), (2, 8, 8), (4, 16, 3),
    (4, 3, 32), (8, 16, 7), (16, 64, 7), (32, 1, 28),
]


@require_cpp_extension
class TestFlattenBackwardIdentity:

    @pytest.mark.parametrize("input_shape", _4D_SHAPES)
    def test_flatten_default_axis_passthrough(self, input_shape):
        proto = _make_flatten_proto(input_shape)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        expected = _flatten_shape(input_shape, 1, -1)
        assert out["flat"].shape == expected
        dy = np.random.randn(*expected).astype(np.float32)
        net.backward({"flat": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dx, dy.reshape(input_shape))
        assert dx.dtype == np.float32
        assert dx.shape == input_shape
        assert np.all(np.isfinite(dx))

    @pytest.mark.parametrize("input_shape", _2D_SHAPES)
    def test_flatten_2d_input(self, input_shape):
        proto = _make_flatten_proto(input_shape)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        assert out["flat"].shape == input_shape
        dy = np.random.randn(*input_shape).astype(np.float32)
        net.backward({"flat": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dx, dy)

    @pytest.mark.parametrize("n", _1D_SHAPES)
    def test_flatten_1d_input(self, n):
        """1D input flatten with axis=0 is identity."""
        input_shape = (n,)
        proto = _make_flatten_proto(input_shape, axis=0, end_axis=-1)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        assert out["flat"].shape == input_shape
        dy = np.random.randn(*input_shape).astype(np.float32)
        net.backward({"flat": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dx, dy)

    @pytest.mark.parametrize("input_shape", _5D_SHAPES)
    def test_flatten_5d_input(self, input_shape):
        """5D input with default axis=1."""
        proto = _make_flatten_proto(input_shape)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        expected = _flatten_shape(input_shape, 1, -1)
        assert out["flat"].shape == expected
        dy = np.random.randn(*expected).astype(np.float32)
        net.backward({"flat": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dx, dy.reshape(input_shape))

    @pytest.mark.parametrize("input_shape", _3D_SHAPES)
    def test_flatten_3d_input(self, input_shape):
        """3D input with default axis=1."""
        proto = _make_flatten_proto(input_shape)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        expected = _flatten_shape(input_shape, 1, -1)
        assert out["flat"].shape == expected
        dy = np.random.randn(*expected).astype(np.float32)
        net.backward({"flat": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dx, dy.reshape(input_shape))

    @pytest.mark.parametrize("axis,end_axis", [
        (0, -1), (1, -1), (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3),
    ])
    def test_flatten_custom_axes_4d(self, axis, end_axis):
        input_shape = (2, 3, 4, 5)
        proto = _make_flatten_proto(input_shape, axis=axis, end_axis=end_axis)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        expected = _flatten_shape(input_shape, axis, end_axis)
        assert out["flat"].shape == expected
        dy = np.random.randn(*expected).astype(np.float32)
        net.backward({"flat": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dx, dy.reshape(input_shape))

    @pytest.mark.parametrize("axis,end_axis", [
        (0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (1, -1), (1, 1), (1, 2), (1, 3), (1, 4),
        (2, -1), (2, 2), (2, 3), (2, 4),
    ])
    def test_flatten_custom_axes_5d(self, axis, end_axis):
        input_shape = (2, 3, 4, 5, 6)
        proto = _make_flatten_proto(input_shape, axis=axis, end_axis=end_axis)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        expected = _flatten_shape(input_shape, axis, end_axis)
        assert out["flat"].shape == expected
        dy = np.random.randn(*expected).astype(np.float32)
        net.backward({"flat": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dx, dy.reshape(input_shape))


@require_cpp_extension
class TestFlattenBackwardNumerical:

    def _num_grad(self, net, x, h=1e-3):
        grad = np.zeros_like(x)
        for idx in np.ndindex(x.shape):
            xp = x.copy(); xp[idx] += h
            fp = float(np.sum(net.forward({"data": xp})["flat"] ** 2))
            xm = x.copy(); xm[idx] -= h
            fm = float(np.sum(net.forward({"data": xm})["flat"] ** 2))
            grad[idx] = (fp - fm) / (2 * h)
        return grad

    @pytest.mark.parametrize("input_shape", [s for s in _SMALL_4D_SHAPES if int(np.prod(s)) <= 100])
    def test_flatten_numerical_gradient(self, input_shape):
        proto = _make_flatten_proto(input_shape)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32) * 0.5
        out = net.forward({"data": x})
        dy = (2 * out["flat"]).astype(np.float32)
        net.backward({"flat": dy})
        dx_an = net.blob_by_name("data").diff
        dx_num = self._num_grad(net, x)
        np.testing.assert_allclose(dx_an, dx_num, rtol=1e-2, atol=2e-3)

    @pytest.mark.parametrize("input_shape", [s for s in _2D_SHAPES if int(np.prod(s)) <= 50])
    def test_flatten_numerical_gradient_2d(self, input_shape):
        proto = _make_flatten_proto(input_shape)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32) * 0.5
        out = net.forward({"data": x})
        dy = (2 * out["flat"]).astype(np.float32)
        net.backward({"flat": dy})
        dx_an = net.blob_by_name("data").diff
        dx_num = self._num_grad(net, x)
        np.testing.assert_allclose(dx_an, dx_num, rtol=1e-2, atol=2e-3)

    @pytest.mark.parametrize("input_shape,axis,end_axis", [
        ((2, 3, 2, 2), 0, -1),
        ((2, 3, 2, 2), 1, 2),
        ((2, 3, 2, 2), 2, 3),
        ((2, 3, 4), 0, -1),
        ((2, 3, 4), 1, 2),
        ((2, 3, 2, 2, 2), 0, -1),
        ((2, 3, 2, 2, 2), 1, 3),
    ])
    def test_flatten_numerical_gradient_custom_axes(self, input_shape, axis, end_axis):
        proto = _make_flatten_proto(input_shape, axis=axis, end_axis=end_axis)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32) * 0.5
        out = net.forward({"data": x})
        dy = (2 * out["flat"]).astype(np.float32)
        net.backward({"flat": dy})
        dx_an = net.blob_by_name("data").diff
        dx_num = self._num_grad(net, x)
        np.testing.assert_allclose(dx_an, dx_num, rtol=1e-2, atol=2e-3)


@require_cpp_extension
class TestFlattenBackwardEdgeCases:

    @pytest.mark.parametrize("input_shape", [s for s in _SMALL_4D_SHAPES if int(np.prod(s)) <= 100])
    def test_zero_dy_zero_dx(self, input_shape):
        proto = _make_flatten_proto(input_shape)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        net.forward({"data": x})
        expected = _flatten_shape(input_shape, 1, -1)
        net.backward({"flat": np.zeros(expected, dtype=np.float32)})
        np.testing.assert_array_equal(net.blob_by_name("data").diff, 0.0)

    def test_deterministic(self):
        input_shape = (2, 4, 3, 3)
        proto1 = _make_flatten_proto(input_shape)
        proto2 = _make_flatten_proto(input_shape)
        net1, net2 = Net(proto1), Net(proto2)
        x = np.random.randn(*input_shape).astype(np.float32)
        expected = _flatten_shape(input_shape, 1, -1)
        dy = np.random.randn(*expected).astype(np.float32)
        for n in (net1, net2):
            n.forward({"data": x})
            n.backward({"flat": dy})
        np.testing.assert_array_equal(
            net1.blob_by_name("data").diff, net2.blob_by_name("data").diff)

    @pytest.mark.parametrize("input_shape", [
        (3, 4, 5, 6), (1, 1, 1, 1), (8, 3, 16, 16), (2, 3, 4, 5),
        (4, 16, 3, 3), (1, 3, 56, 56),
    ])
    def test_shapes_dtypes(self, input_shape):
        proto = _make_flatten_proto(input_shape)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        assert out["flat"].dtype == np.float32
        dy = np.random.randn(*out["flat"].shape).astype(np.float32)
        net.backward({"flat": dy})
        dx = net.blob_by_name("data").diff
        assert dx.dtype == np.float32
        assert dx.shape == input_shape
        assert np.all(np.isfinite(dx))

    def test_forward_preserved_after_backward(self):
        input_shape = (2, 3, 4, 5)
        proto = _make_flatten_proto(input_shape)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out0 = net.forward({"data": x})["flat"].copy()
        dy = np.random.randn(*out0.shape).astype(np.float32)
        net.backward({"flat": dy})
        out1 = net.forward({"data": x})["flat"]
        np.testing.assert_array_equal(out1, out0)

    @pytest.mark.parametrize("conv_ch,kernel,ip_out", [
        (4, 3, 2), (8, 3, 4), (2, 1, 2), (16, 3, 10),
    ])
    def test_flatten_in_chain(self, conv_ch, kernel, ip_out):
        """Conv -> ReLU -> Flatten -> IP: gradient flows through Flatten."""
        proto = textwrap.dedent(f"""\
            name: "flatten_chain"
            input: "data"
            input_shape {{ dim: 2 dim: 3 dim: 4 dim: 4 }}
            layer {{
              name: "conv" type: "Convolution" bottom: "data" top: "conv"
              convolution_param {{ num_output: {conv_ch} kernel_size: {kernel} pad: 1 stride: 1 bias_term: false
                weight_filler {{ type: "gaussian" std: 0.5 }} }}
            }}
            layer {{ name: "relu" type: "ReLU" bottom: "conv" top: "conv" }}
            layer {{ name: "flat" type: "Flatten" bottom: "conv" top: "flat" flatten_param {{ axis: 1 }} }}
            layer {{
              name: "ip" type: "InnerProduct" bottom: "flat" top: "ip"
              inner_product_param {{ num_output: {ip_out} bias_term: false
                weight_filler {{ type: "gaussian" std: 0.5 }} }}
            }}
        """)
        net = Net(proto)
        k = kernel
        spatial_out = 4 + 2*1 - k + 1  # = 4 for k=3,pad=1
        conv_w = np.random.randn(conv_ch, 3, k, k).astype(np.float32) * 0.5
        ip_in_dim = conv_ch * spatial_out * spatial_out
        ip_w = np.random.randn(ip_out, ip_in_dim).astype(np.float32) * 0.5
        net.layer_by_name("conv").blobs[0].from_numpy(conv_w)
        net.layer_by_name("ip").blobs[0].from_numpy(ip_w)
        x = np.random.randn(2, 3, 4, 4).astype(np.float32)
        out = net.forward({"data": x})
        assert out["ip"].shape == (2, ip_out)
        dy = np.random.randn(2, ip_out).astype(np.float32) + 0.1
        net.backward({"ip": dy})
        dx = net.blob_by_name("conv").diff
        assert dx.shape == (2, conv_ch, spatial_out, spatial_out)
        assert np.all(np.isfinite(dx))

    def test_no_learnable_params(self):
        proto = _make_flatten_proto((2, 3, 4, 5))
        net = Net(proto)
        assert len(net.layer_by_name("flatten").blobs) == 0

    @pytest.mark.parametrize("fill_val", [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 100.0, -100.0])
    def test_special_values_forward_backward(self, fill_val):
        """Forward/backward with special input values (zeros, ones, large)."""
        input_shape = (2, 3, 4, 5)
        proto = _make_flatten_proto(input_shape)
        net = Net(proto)
        x = np.full(input_shape, fill_val, dtype=np.float32)
        out = net.forward({"data": x})
        expected_shape = _flatten_shape(input_shape, 1, -1)
        assert out["flat"].shape == expected_shape
        np.testing.assert_allclose(out["flat"].flat[0], fill_val, rtol=1e-6)
        dy = np.full(expected_shape, fill_val * 2, dtype=np.float32)
        net.backward({"flat": dy})
        dx = net.blob_by_name("data").diff
        assert dx.shape == input_shape
        np.testing.assert_allclose(dx.flat[0], fill_val * 2, rtol=1e-6)

    @pytest.mark.parametrize("input_shape", [
        (1, 1, 1, 1), (1, 1, 1, 100), (100, 1, 1, 1),
        (1, 1, 28, 28), (1, 100, 1, 1), (100, 1, 1, 100),
        (2, 1, 1, 1), (1, 2, 1, 1), (1, 1, 2, 1), (1, 1, 1, 2),
    ])
    def test_degenerate_shapes(self, input_shape):
        """Degenerate shapes with dim=1."""
        proto = _make_flatten_proto(input_shape)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        expected = _flatten_shape(input_shape, 1, -1)
        assert out["flat"].shape == expected
        dy = np.random.randn(*expected).astype(np.float32)
        net.backward({"flat": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dx, dy.reshape(input_shape))

    @pytest.mark.parametrize("input_shape,dy_val", [
        ((2, 3, 4, 5), 0.001), ((2, 3, 4, 5), -0.001),
        ((4, 8, 3, 3), 1e-6), ((1, 1, 1, 1), 3.14159),
    ])
    def test_small_and_large_dy(self, input_shape, dy_val):
        proto = _make_flatten_proto(input_shape)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        net.forward({"data": x})
        expected = _flatten_shape(input_shape, 1, -1)
        dy = np.full(expected, dy_val, dtype=np.float32)
        net.backward({"flat": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_allclose(dx, dy_val, rtol=1e-6)


@require_cpp_extension
class TestFlattenInplaceProtection:

    def test_inplace_forbidden(self):
        proto = textwrap.dedent("""\
            name: "flatten_inplace"
            input: "data"
            input_shape { dim: 2 dim: 3 dim: 4 dim: 5 }
            layer { name: "flat" type: "Flatten" bottom: "data" top: "data" }
        """)
        with pytest.raises(Exception):
            Net(proto)
