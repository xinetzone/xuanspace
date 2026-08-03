"""Reshape layer backward tests.

Reshape is a pure shape-transformation layer:
  - Forward: data is memcpy'd (or same blob if in-place); only shape metadata changes
  - Backward: top_diff is memcpy'd back (identity pass-through, shape restored)
"""
import textwrap
import numpy as np
import pytest
from caffe_ffi import Net

from .conftest import require_cpp_extension


def _make_reshape_proto(input_shape, shape_dims, axis=0, num_axes=-1, inplace=False):
    dims_str = " ".join(str(d) for d in input_shape)
    shape_str = " ".join(f"dim: {d}" for d in shape_dims)
    top_name = "data" if inplace else "reshaped"
    return textwrap.dedent(f"""\
        name: "reshape_test"
        input: "data"
        input_shape {{ dim: {dims_str.replace(' ', ' dim: ')} }}
        layer {{
          name: "reshape"
          type: "Reshape"
          bottom: "data"
          top: "{top_name}"
          reshape_param {{ shape {{ {shape_str} }} axis: {axis} num_axes: {num_axes} }}
        }}
    """)


def _resolve_reshape(input_shape, shape_dims, axis=0, num_axes=-1):
    ndim = len(input_shape)
    if axis < 0:
        axis = ndim + axis
    if axis > ndim:
        axis = ndim
    end_axis = ndim if num_axes == -1 else min(axis + num_axes, ndim)

    out = list(input_shape[:axis])
    region_count = 1
    for i in range(axis, end_axis):
        region_count *= input_shape[i]

    constant_count = 1
    inferred_idx = -1
    resolved = []
    for i, d in enumerate(shape_dims):
        if d == 0:
            if axis + i < len(input_shape):
                resolved.append(input_shape[axis + i])
            else:
                resolved.append(0)
            constant_count *= resolved[-1] if resolved[-1] != 0 else 1
        elif d == -1:
            inferred_idx = len(resolved)
            resolved.append(0)
        else:
            resolved.append(d)
            constant_count *= d

    if inferred_idx >= 0 and constant_count > 0:
        resolved[inferred_idx] = region_count // constant_count
    out.extend(resolved)
    out.extend(input_shape[end_axis:])
    return tuple(out)


# Common 4D shapes used in CNNs
_4D_SHAPES = [
    (1, 1, 1, 1), (1, 1, 2, 2), (1, 1, 3, 3), (1, 1, 4, 4), (1, 1, 5, 5),
    (1, 1, 7, 7), (1, 1, 8, 8), (1, 1, 14, 14), (1, 1, 28, 28),
    (1, 3, 1, 1), (1, 3, 2, 2), (1, 3, 3, 3), (1, 3, 4, 4), (1, 3, 8, 8),
    (1, 3, 16, 16), (1, 3, 32, 32),
    (1, 8, 1, 1), (1, 8, 2, 2), (1, 8, 3, 3), (1, 8, 7, 7), (1, 8, 14, 14),
    (1, 16, 1, 1), (1, 16, 2, 2), (1, 16, 3, 3), (1, 16, 7, 7),
    (1, 32, 1, 1), (1, 32, 2, 2), (1, 32, 3, 3), (1, 32, 7, 7),
    (1, 64, 1, 1), (1, 64, 2, 2), (1, 64, 3, 3), (1, 64, 7, 7),
    (2, 1, 1, 1), (2, 1, 28, 28), (2, 3, 4, 4), (2, 3, 8, 8),
    (2, 8, 2, 2), (2, 8, 3, 3), (2, 8, 7, 7),
    (2, 16, 3, 3), (2, 16, 7, 7), (2, 32, 3, 3),
    (4, 1, 28, 28), (4, 3, 4, 4), (4, 3, 8, 8),
    (4, 8, 3, 3), (4, 16, 3, 3), (4, 32, 7, 7),
    (8, 1, 28, 28), (8, 3, 8, 8), (8, 16, 3, 3),
    (16, 1, 28, 28), (16, 3, 8, 8), (16, 16, 3, 3),
    (32, 1, 28, 28), (32, 3, 8, 8), (32, 16, 3, 3),
]

_SMALL_SHAPES = [s for s in _4D_SHAPES if int(np.prod(s)) <= 200]

_2D_SHAPES = [
    (1, 1), (1, 2), (1, 10), (1, 100), (1, 784),
    (2, 1), (2, 2), (2, 10), (2, 100), (2, 784),
    (4, 10), (4, 100), (8, 10), (8, 100),
    (16, 10), (16, 100), (32, 10), (32, 100),
    (64, 10), (64, 100), (100, 1), (100, 10),
]

_3D_SHAPES = [
    (1, 1, 1), (1, 10, 10), (2, 3, 4), (2, 8, 8), (4, 16, 3),
    (4, 3, 32), (8, 16, 7), (16, 64, 7), (32, 1, 28),
]

_1D_SHAPES = [1, 2, 5, 10, 100, 128, 256, 512, 784]

_5D_SHAPES = [
    (1, 1, 1, 1, 1), (1, 3, 2, 2, 2), (2, 8, 2, 2, 2),
    (1, 16, 2, 3, 4), (2, 3, 4, 5, 6),
]


def _flatten_target_dims(shape):
    """Generate reshape dims that flatten the entire shape."""
    return (-1,)


def _batch_feat_dims(shape):
    """Generate reshape dims that flatten to (batch, features) from 4D."""
    if len(shape) >= 2:
        return (0, -1)
    return (-1,)


@require_cpp_extension
class TestReshapeBackwardIdentity:

    @pytest.mark.parametrize("input_shape", _4D_SHAPES)
    def test_reshape_flatten_all(self, input_shape):
        """Flatten entire 4D tensor to 1D."""
        shape_dims = (-1,)
        proto = _make_reshape_proto(input_shape, shape_dims)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        expected = _resolve_reshape(input_shape, shape_dims)
        assert out["reshaped"].shape == expected
        dy = np.random.randn(*expected).astype(np.float32)
        net.backward({"reshaped": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dx, dy.reshape(input_shape))
        assert dx.shape == input_shape
        assert dx.dtype == np.float32

    @pytest.mark.parametrize("input_shape", _4D_SHAPES)
    def test_reshape_to_batch_features(self, input_shape):
        """Reshape 4D (N,C,H,W) to (N, C*H*W) using dim=0 copy."""
        shape_dims = (0, -1)
        proto = _make_reshape_proto(input_shape, shape_dims)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        expected = _resolve_reshape(input_shape, shape_dims)
        assert out["reshaped"].shape == expected
        dy = np.random.randn(*expected).astype(np.float32)
        net.backward({"reshaped": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dx, dy.reshape(input_shape))

    @pytest.mark.parametrize("input_shape", _2D_SHAPES)
    def test_reshape_2d_flatten(self, input_shape):
        shape_dims = (-1,)
        proto = _make_reshape_proto(input_shape, shape_dims)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        total = int(np.prod(input_shape))
        assert out["reshaped"].shape == (total,)
        dy = np.random.randn(total).astype(np.float32)
        net.backward({"reshaped": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dx, dy.reshape(input_shape))

    @pytest.mark.parametrize("n", _1D_SHAPES)
    def test_reshape_1d_identity(self, n):
        """1D -> (-1,) is identity."""
        input_shape = (n,)
        shape_dims = (-1,)
        proto = _make_reshape_proto(input_shape, shape_dims)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        assert out["reshaped"].shape == input_shape
        dy = np.random.randn(*input_shape).astype(np.float32)
        net.backward({"reshaped": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dx, dy)

    @pytest.mark.parametrize("input_shape", _5D_SHAPES)
    def test_reshape_5d_flatten(self, input_shape):
        shape_dims = (-1,)
        proto = _make_reshape_proto(input_shape, shape_dims)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        total = int(np.prod(input_shape))
        assert out["reshaped"].shape == (total,)
        dy = np.random.randn(total).astype(np.float32)
        net.backward({"reshaped": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dx, dy.reshape(input_shape))

    @pytest.mark.parametrize("input_shape", _3D_SHAPES)
    def test_reshape_3d_flatten(self, input_shape):
        shape_dims = (-1,)
        proto = _make_reshape_proto(input_shape, shape_dims)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        total = int(np.prod(input_shape))
        assert out["reshaped"].shape == (total,)
        dy = np.random.randn(total).astype(np.float32)
        net.backward({"reshaped": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dx, dy.reshape(input_shape))

    @pytest.mark.parametrize("input_shape", _3D_SHAPES)
    def test_reshape_3d_to_batch_features(self, input_shape):
        shape_dims = (0, -1)
        proto = _make_reshape_proto(input_shape, shape_dims)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        expected = _resolve_reshape(input_shape, shape_dims)
        assert out["reshaped"].shape == expected
        dy = np.random.randn(*expected).astype(np.float32)
        net.backward({"reshaped": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dx, dy.reshape(input_shape))

    @pytest.mark.parametrize("input_shape,shape_dims", [
        ((2, 3, 4, 5), (2, 3, 4, 5)),
        ((4, 6, 3, 3), (4, 6, 9)),
        ((8, 16, 3, 3), (8, 144)),
        ((2, 3, 4, 5), (2, 60)),
        ((4, 3, 8, 8), (4, 192)),
    ])
    def test_reshape_partial(self, input_shape, shape_dims):
        """Reshape to multi-dim target."""
        proto = _make_reshape_proto(input_shape, shape_dims)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        expected = _resolve_reshape(input_shape, shape_dims)
        assert out["reshaped"].shape == expected
        dy = np.random.randn(*expected).astype(np.float32)
        net.backward({"reshaped": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dx, dy.reshape(input_shape))

    @pytest.mark.parametrize("input_shape,shape_dims", [
        ((120,), (2, 3, 4, 5)),
        ((60,), (3, 4, 5)),
        ((54,), (6, 3, 3)),
        ((784,), (1, 28, 28)),
        ((100,), (10, 10)),
    ])
    def test_reshape_1d_to_nd(self, input_shape, shape_dims):
        """1D -> multi-dim unflatten."""
        proto = _make_reshape_proto(input_shape, shape_dims)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        assert out["reshaped"].shape == shape_dims
        dy = np.random.randn(*shape_dims).astype(np.float32)
        net.backward({"reshaped": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dx, dy.reshape(input_shape))


@require_cpp_extension
class TestReshapeBackwardNumerical:

    def _num_grad(self, net, x, h=1e-3):
        grad = np.zeros_like(x)
        for idx in np.ndindex(x.shape):
            xp = x.copy(); xp[idx] += h
            fp = float(np.sum(net.forward({"data": xp})["reshaped"] ** 2))
            xm = x.copy(); xm[idx] -= h
            fm = float(np.sum(net.forward({"data": xm})["reshaped"] ** 2))
            grad[idx] = (fp - fm) / (2 * h)
        return grad

    @pytest.mark.parametrize("input_shape", [s for s in _SMALL_SHAPES if int(np.prod(s)) <= 100])
    def test_reshape_numerical_gradient_flatten(self, input_shape):
        proto = _make_reshape_proto(input_shape, (-1,))
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32) * 0.5
        out = net.forward({"data": x})
        dy = (2 * out["reshaped"]).astype(np.float32)
        net.backward({"reshaped": dy})
        dx_an = net.blob_by_name("data").diff
        dx_num = self._num_grad(net, x)
        np.testing.assert_allclose(dx_an, dx_num, rtol=1e-2, atol=2e-3)

    @pytest.mark.parametrize("input_shape", [s for s in _SMALL_SHAPES if int(np.prod(s)) <= 100][:10])
    def test_reshape_numerical_gradient_batch_feat(self, input_shape):
        proto = _make_reshape_proto(input_shape, (0, -1))
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32) * 0.5
        out = net.forward({"data": x})
        dy = (2 * out["reshaped"]).astype(np.float32)
        net.backward({"reshaped": dy})
        dx_an = net.blob_by_name("data").diff
        dx_num = self._num_grad(net, x)
        np.testing.assert_allclose(dx_an, dx_num, rtol=1e-2, atol=2e-3)

    @pytest.mark.parametrize("input_shape", [s for s in _2D_SHAPES if int(np.prod(s)) <= 50])
    def test_reshape_2d_numerical_gradient(self, input_shape):
        proto = _make_reshape_proto(input_shape, (-1,))
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32) * 0.5
        out = net.forward({"data": x})
        dy = (2 * out["reshaped"]).astype(np.float32)
        net.backward({"reshaped": dy})
        dx_an = net.blob_by_name("data").diff
        dx_num = self._num_grad(net, x)
        np.testing.assert_allclose(dx_an, dx_num, rtol=1e-2, atol=2e-3)

    @pytest.mark.parametrize("input_shape,shape_dims", [
        ((1, 8, 2, 2), (1, -1)),
        ((2, 3, 4), (2, 12)),
        ((2, 6, 2, 2), (4, -1)),
        ((2, 3, 2, 2), (-1,)),
        ((4, 3, 3), (2, -1)),
    ])
    def test_reshape_numerical_gradient_multi_dim(self, input_shape, shape_dims):
        proto = _make_reshape_proto(input_shape, shape_dims)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32) * 0.5
        out = net.forward({"data": x})
        dy = (2 * out["reshaped"]).astype(np.float32)
        net.backward({"reshaped": dy})
        dx_an = net.blob_by_name("data").diff
        dx_num = self._num_grad(net, x)
        np.testing.assert_allclose(dx_an, dx_num, rtol=1e-2, atol=2e-3)


@require_cpp_extension
class TestReshapeBackwardEdgeCases:

    @pytest.mark.parametrize("input_shape", [s for s in _SMALL_SHAPES if int(np.prod(s)) <= 100])
    def test_zero_dy_zero_dx(self, input_shape):
        proto = _make_reshape_proto(input_shape, (-1,))
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        net.forward({"data": x})
        total = int(np.prod(input_shape))
        net.backward({"reshaped": np.zeros((total,), dtype=np.float32)})
        np.testing.assert_array_equal(net.blob_by_name("data").diff, 0.0)

    def test_deterministic(self):
        input_shape = (2, 3, 4, 5)
        n1 = Net(_make_reshape_proto(input_shape, (2, 60)))
        n2 = Net(_make_reshape_proto(input_shape, (2, 60)))
        x = np.random.randn(*input_shape).astype(np.float32)
        dy = np.random.randn(2, 60).astype(np.float32)
        for n in (n1, n2):
            n.forward({"data": x})
            n.backward({"reshaped": dy})
        np.testing.assert_array_equal(
            n1.blob_by_name("data").diff, n2.blob_by_name("data").diff)

    @pytest.mark.parametrize("input_shape,shape_dims", [
        ((3, 4, 5), (-1,)),
        ((2, 3, 4, 5), (-1,)),
        ((1, 1, 1, 1), (-1,)),
        ((8, 3, 16, 16), (0, -1)),
    ])
    def test_shapes_dtypes(self, input_shape, shape_dims):
        proto = _make_reshape_proto(input_shape, shape_dims)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        assert out["reshaped"].dtype == np.float32
        dy = np.random.randn(*out["reshaped"].shape).astype(np.float32)
        net.backward({"reshaped": dy})
        dx = net.blob_by_name("data").diff
        assert dx.dtype == np.float32
        assert dx.shape == input_shape
        assert np.all(np.isfinite(dx))

    def test_forward_preserved_after_backward(self):
        input_shape = (2, 3, 4, 5)
        proto = _make_reshape_proto(input_shape, (2, 60))
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out0 = net.forward({"data": x})["reshaped"].copy()
        dy = np.random.randn(2, 60).astype(np.float32)
        net.backward({"reshaped": dy})
        out1 = net.forward({"data": x})["reshaped"]
        np.testing.assert_array_equal(out1, out0)

    @pytest.mark.parametrize("conv_ch,kernel,ip_out", [
        (4, 3, 2), (8, 3, 4), (2, 1, 2), (16, 3, 10),
    ])
    def test_reshape_in_chain(self, conv_ch, kernel, ip_out):
        proto = textwrap.dedent(f"""\
            name: "reshape_chain"
            input: "data"
            input_shape {{ dim: 2 dim: 3 dim: 4 dim: 4 }}
            layer {{
              name: "conv" type: "Convolution" bottom: "data" top: "conv"
              convolution_param {{ num_output: {conv_ch} kernel_size: {kernel} pad: 1 stride: 1 bias_term: false
                weight_filler {{ type: "gaussian" std: 0.5 }} }}
            }}
            layer {{ name: "relu" type: "ReLU" bottom: "conv" top: "conv" }}
            layer {{
              name: "reshape" type: "Reshape" bottom: "conv" top: "flat"
              reshape_param {{ shape {{ dim: 0 dim: -1 }} }}
            }}
            layer {{
              name: "ip" type: "InnerProduct" bottom: "flat" top: "ip"
              inner_product_param {{ num_output: {ip_out} bias_term: false
                weight_filler {{ type: "gaussian" std: 0.5 }} }}
            }}
        """)
        net = Net(proto)
        k = kernel
        spatial_out = 4 + 2*1 - k + 1
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
        proto = _make_reshape_proto((2, 3, 4, 5), (-1,))
        net = Net(proto)
        assert len(net.layer_by_name("reshape").blobs) == 0

    def test_reshape_flatten_equivalence(self):
        """Reshape{shape: dim: -1} should behave like Flatten axis=0."""
        input_shape = (2, 3, 4, 5)
        total = int(np.prod(input_shape))
        proto_r = _make_reshape_proto(input_shape, (-1,))
        proto_f = textwrap.dedent(f"""\
            name: "flat"
            input: "data"
            input_shape {{ dim: 2 dim: 3 dim: 4 dim: 5 }}
            layer {{ name: "flat" type: "Flatten" bottom: "data" top: "flat" flatten_param {{ axis: 0 }} }}
        """)
        nr = Net(proto_r)
        nf = Net(proto_f)
        x = np.random.randn(*input_shape).astype(np.float32)
        or_ = nr.forward({"data": x})["reshaped"]
        of_ = nf.forward({"data": x})["flat"]
        assert or_.shape == of_.shape == (total,)
        np.testing.assert_array_equal(or_, of_)
        dy = np.random.randn(total).astype(np.float32)
        nr.backward({"reshaped": dy})
        nf.backward({"flat": dy})
        dx_r = nr.blob_by_name("data").diff
        dx_f = nf.blob_by_name("data").diff
        np.testing.assert_array_equal(dx_r.ravel(), dx_f.ravel())

    @pytest.mark.parametrize("fill_val", [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 100.0, -100.0])
    def test_special_values_forward_backward(self, fill_val):
        input_shape = (2, 3, 4, 5)
        proto = _make_reshape_proto(input_shape, (-1,))
        net = Net(proto)
        x = np.full(input_shape, fill_val, dtype=np.float32)
        out = net.forward({"data": x})
        total = int(np.prod(input_shape))
        assert out["reshaped"].shape == (total,)
        dy = np.full((total,), fill_val * 2, dtype=np.float32)
        net.backward({"reshaped": dy})
        dx = net.blob_by_name("data").diff
        assert dx.shape == input_shape
        np.testing.assert_allclose(dx.flat[0], fill_val * 2, rtol=1e-6)

    @pytest.mark.parametrize("input_shape", [
        (5,), (1,), (100,), (256,), (512,),
        (2, 3, 4, 5, 6), (1, 2, 3, 4, 5), (1, 3, 2, 2, 2),
    ])
    def test_1d_5d_inputs(self, input_shape):
        total = int(np.prod(input_shape))
        proto = _make_reshape_proto(input_shape, (-1,))
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        assert out["reshaped"].shape == (total,)
        dy = np.random.randn(total).astype(np.float32)
        net.backward({"reshaped": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dx, dy.reshape(input_shape))

    @pytest.mark.parametrize("input_shape,axis,num_axes,shape_dims,expected_shape", [
        ((2, 3, 4, 5), 1, 2, (12,), (2, 12, 5)),
        ((2, 3, 4, 5), 2, -1, (20,), (2, 3, 20)),
        ((4, 6, 3, 3), 1, -1, (0, -1), (4, 6, 9)),
        ((2, 3, 4, 5), 0, 2, (6,), (6, 4, 5)),
        ((2, 3, 4, 5), 1, 3, (0, -1), (2, 3, 20)),
        ((2, 3, 4), 1, 2, (12,), (2, 12)),
        ((4, 6, 3), 1, -1, (0, -1), (4, 6, 3)),
        ((2, 3, 4, 5), 2, 3, (-1,), (2, 3, 20)),
    ])
    def test_reshape_with_axis_num_axes(self, input_shape, axis, num_axes, shape_dims, expected_shape):
        proto = _make_reshape_proto(input_shape, shape_dims, axis=axis, num_axes=num_axes)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        assert out["reshaped"].shape == expected_shape
        dy = np.random.randn(*expected_shape).astype(np.float32)
        net.backward({"reshaped": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dx, dy.reshape(input_shape))

    @pytest.mark.parametrize("input_shape,shape_dims", [
        ((1, 1, 1, 1), (1,)),
        ((2, 1, 1, 1), (2,)),
        ((1, 100, 1, 1), (1, -1)),
        ((100, 1, 1, 1), (100,)),
    ])
    def test_degenerate_shapes(self, input_shape, shape_dims):
        proto = _make_reshape_proto(input_shape, shape_dims)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        expected = _resolve_reshape(input_shape, shape_dims)
        assert out["reshaped"].shape == expected
        dy = np.random.randn(*expected).astype(np.float32)
        net.backward({"reshaped": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dx, dy.reshape(input_shape))

    @pytest.mark.parametrize("input_shape,dy_val", [
        ((2, 3, 4, 5), 0.001), ((2, 3, 4, 5), -0.001),
        ((4, 8, 3, 3), 1e-6), ((1, 1, 1, 1), 3.14159),
    ])
    def test_small_and_large_dy(self, input_shape, dy_val):
        total = int(np.prod(input_shape))
        proto = _make_reshape_proto(input_shape, (-1,))
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        net.forward({"data": x})
        dy = np.full((total,), dy_val, dtype=np.float32)
        net.backward({"reshaped": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_allclose(dx, dy_val, rtol=1e-6)

    def test_copy_dim_zero_preserves_axis(self):
        """dim=0 copies the corresponding axis from input."""
        input_shape = (2, 3, 4, 5)
        for axis_i, dim_i in [(1, 3), (2, 4), (3, 5)]:
            shape_dims = [0, -1]
            proto = _make_reshape_proto(input_shape, tuple(shape_dims))
            net = Net(proto)
            x = np.random.randn(*input_shape).astype(np.float32)
            out = net.forward({"data": x})
            expected = (2, 60)
            assert out["reshaped"].shape == expected

    def test_reshape_inferred_dim_middle(self):
        """-1 infers dimension in the middle."""
        input_shape = (2, 3, 4, 5)
        shape_dims = (2, -1, 10)
        proto = _make_reshape_proto(input_shape, shape_dims)
        net = Net(proto)
        x = np.random.randn(*input_shape).astype(np.float32)
        out = net.forward({"data": x})
        assert out["reshaped"].shape == (2, 6, 10)
        dy = np.random.randn(2, 6, 10).astype(np.float32)
        net.backward({"reshaped": dy})
        dx = net.blob_by_name("data").diff
        np.testing.assert_array_equal(dx, dy.reshape(input_shape))
