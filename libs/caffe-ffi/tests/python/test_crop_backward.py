"""Crop layer Backward gradient tests.

Crop has two bottoms: ``data`` (the input to be cropped) and ``crop_ref``
(which defines the output size). For every dim ``i >= axis`` the output takes
``crop_ref.shape[i]`` elements starting at ``offsets[i]`` from ``data``.

Forward :  out[i >= axis] = data[offset_i : offset_i + crop_size_i]
Backward:  d_data = 0, then d_data[offset region] = dy  (zero-pad outside crop)

Covers:
  1. Known-value hand verification (axis=0, offset=1 on 1D; axis=2 NCHW)
  2. Analytical gradient (numpy reference vs caffe-ffi)
  3. Numerical gradient check (central finite differences w.r.t. data)
  4. Zero-pad outside crop region (untouched pixels get 0 gradient)
  5. Zero dy -> zero gradients
  6. Shape/finite/determinism checks
  7. Forward-Backward round-trip
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension
from ._grad_check_utils import numerical_gradient

EPS = 1e-3
RTOL = 1e-3
ATOL = 1e-4


# ---------------------------------------------------------------------------
# Numpy reference for Crop forward/backward
# ---------------------------------------------------------------------------

def _build_crop_slices(in_shape, crop_shape, axis, offset):
    """Build the slice tuple selecting the cropped region of an input.

    Offsets follow the layer convention: they are *relative to axis* and
    length either 1 (broadcast to every dim >= axis) or ``num_dims >= axis``
    (one offset per dim starting at axis).
    """
    ndim = len(in_shape)
    if offset is None:
        offset = [0] * (ndim - axis)
    sl = []
    for i in range(ndim):
        if i >= axis:
            off = offset[0] if len(offset) == 1 else offset[i - axis]
            sl.append(slice(off, off + crop_shape[i]))
        else:
            sl.append(slice(None))
    return tuple(sl)


def crop_forward_np(x, crop_shape, axis=2, offset=None):
    """Numpy reference: select the cropped region of x."""
    sl = _build_crop_slices(x.shape, crop_shape, axis, offset)
    return x[sl].astype(np.float32)


def crop_backward_np(dy, in_shape, crop_shape, axis=2, offset=None):
    """Numpy reference for Crop backward.

    Returns a full-size dX with ``dy`` copied into the cropped region and
    zeros everywhere else (the region of ``data`` excluded by the crop).
    """
    dX = np.zeros(in_shape, dtype=np.float32)
    sl = _build_crop_slices(in_shape, crop_shape, axis, offset)
    dX[sl] = dy
    return dX


# ---------------------------------------------------------------------------
# Prototxt builder
# ---------------------------------------------------------------------------

def _make_crop_prototxt(in_shape, crop_shape, axis=2, offset=None):
    """Create Input(data) + Input(crop_ref) -> Crop prototxt.

    Args:
        in_shape: shape of bottom[0] (the data to crop).
        crop_shape: shape of bottom[1] (defines output size for dims >= axis).
        axis: crop axis (default caffe Crop axis=2).
        offset: crop offset. If None, offset param omitted (defaults to 0).
    """
    dims_in = "\n".join(f"          dim: {d}" for d in in_shape)
    dims_ref = "\n".join(f"          dim: {d}" for d in crop_shape)

    # Protobuf text format requires each repeated `offset` field on its own line.
    offset_str = ""
    if offset is not None:
        off_lines = "\n".join(f"          offset: {int(o)}" for o in offset)
        offset_str = f"\n{off_lines}"

    return textwrap.dedent(f"""\
        name: "test_crop_bw"
        layer {{
          name: "data"
          type: "Input"
          top: "data"
          input_param {{ shape {{
{dims_in}
          }} }}
        }}
        layer {{
          name: "crop_ref"
          type: "Input"
          top: "crop_ref"
          input_param {{ shape {{
{dims_ref}
          }} }}
        }}
        layer {{
          name: "crop"
          type: "Crop"
          bottom: "data"
          bottom: "crop_ref"
          top: "out"
          crop_param {{ axis: {axis}{offset_str}
          }}
        }}
    """)


def _make_crop_net(in_shape, crop_shape, axis=2, offset=None):
    return Net(_make_crop_prototxt(in_shape, crop_shape, axis, offset))


# ---------------------------------------------------------------------------
# Helper: run forward + backward
# ---------------------------------------------------------------------------

def _run_crop_backward(net, x, crop_ref, dy):
    """Run forward then backward, return (dX, out)."""
    out = net.forward({"data": x.astype(np.float32), "crop_ref": crop_ref.astype(np.float32)})
    net.backward({"out": dy.astype(np.float32)})
    dX = net.blob_by_name("data").diff
    return dX, out["out"]


def _numerical_grad_input(net, x, crop_ref, dy, h=EPS):
    """Numerical gradient w.r.t. the (cropped) data input."""
    current = x.astype(np.float32).copy()

    def _forward():
        out = net.forward({"data": current, "crop_ref": crop_ref.astype(np.float32)})
        return out["out"]

    def _get():
        return current.copy()

    def _set(arr):
        nonlocal current
        np.copyto(current, arr)

    return numerical_gradient(_forward, _get, _set, dy, h=h,
                              name="input:data", verbose=False)


# ---------------------------------------------------------------------------
# Tests: Known values (L1)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestCropBackwardKnownValues:
    """Hand-computed known value tests."""

    def test_crop_1d_axis0_offset1(self):
        """Crop axis=0 offset=1: data[0..4], crop_ref size 3 -> out=data[1:4].

        dX = zeros(5), dX[1:4] = dy.
        """
        net = _make_crop_net((5,), (3,), axis=0, offset=[1])
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        crop_ref = np.zeros((3,), dtype=np.float32)
        dy = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        dX, out = _run_crop_backward(net, x, crop_ref, dy)
        np.testing.assert_allclose(out, x[1:4], rtol=1e-5)
        expected = np.array([0.0, 10.0, 20.0, 30.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(dX, expected, rtol=1e-5)

    def test_crop_1d_axis0_offset0(self):
        """Crop axis=0 offset=0 (default): out=data[0:3], dX[0:3]=dy."""
        net = _make_crop_net((5,), (3,), axis=0)
        x = np.arange(5, dtype=np.float32)
        crop_ref = np.zeros((3,), dtype=np.float32)
        dy = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        dX, out = _run_crop_backward(net, x, crop_ref, dy)
        np.testing.assert_allclose(out, x[0:3], rtol=1e-5)
        expected = np.array([100.0, 200.0, 300.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(dX, expected, rtol=1e-5)

    def test_crop_nchw_axis2_offset(self):
        """Crop axis=2 on (1,1,4,4) with offset=[1,1] -> out (1,1,2,3).

        dX is zero except the cropped sub-block.
        """
        net = _make_crop_net((1, 1, 4, 4), (1, 1, 2, 3), axis=2, offset=[1, 1])
        x = np.arange(1 * 1 * 4 * 4, dtype=np.float32).reshape(1, 1, 4, 4)
        crop_ref = np.zeros((1, 1, 2, 3), dtype=np.float32)
        dy = np.full((1, 1, 2, 3), 7.0, dtype=np.float32)
        dX, out = _run_crop_backward(net, x, crop_ref, dy)
        np.testing.assert_allclose(out, x[0, 0, 1:3, 1:4].reshape(1, 1, 2, 3), rtol=1e-5)
        expected = np.zeros((1, 1, 4, 4), dtype=np.float32)
        expected[0, 0, 1:3, 1:4] = 7.0
        np.testing.assert_allclose(dX, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Analytical gradient vs numpy (L2)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestCropBackwardNumpy:
    """Compare caffe-ffi backward against numpy reference."""

    @pytest.mark.parametrize("in_shape,crop_shape,axis,offset", [
        ((5,),       (3,),       0,  [1]),        # 1D, offset=1
        ((6, 6),     (4, 4),     0,  [1, 1]),     # 2D, both dims cropped
        ((1, 4, 4),  (1, 4, 3),  2,  None),       # 3D, crop last dim, default offset
        ((1, 4, 4),  (1, 4, 3),  2,  [1]),        # 3D, crop last dim, offset=1
        ((1, 3, 5, 5), (1, 3, 3, 4), 2, None),    # 4D NCHW, default offset
        ((1, 3, 5, 5), (1, 3, 3, 4), 2, [1, 1]),  # 4D NCHW, offset on H,W
    ])
    def test_crop_vs_numpy(self, in_shape, crop_shape, axis, offset):
        """Crop backward matches numpy zero-pad reference."""
        rng = np.random.RandomState(42)
        x = rng.randn(*in_shape).astype(np.float32) * 0.5
        crop_ref = np.zeros(crop_shape, dtype=np.float32)
        dy = rng.randn(*crop_shape).astype(np.float32) * 0.1
        net = _make_crop_net(in_shape, crop_shape, axis=axis, offset=offset)
        dX, out = _run_crop_backward(net, x, crop_ref, dy)
        np.testing.assert_allclose(out, crop_forward_np(x, crop_shape, axis, offset),
                                   rtol=RTOL, atol=ATOL)
        dx_ref = crop_backward_np(dy, in_shape, crop_shape, axis, offset)
        np.testing.assert_allclose(dX, dx_ref, rtol=RTOL, atol=ATOL,
                                   err_msg="Crop backward zero-pad mismatch")


# ---------------------------------------------------------------------------
# Tests: Numerical gradient (L3)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestCropBackwardNumerical:
    """Central finite difference numerical gradient checks."""

    @pytest.mark.parametrize("in_shape,crop_shape,axis,offset", [
        ((5,),       (3,),       0,  [1]),
        ((1, 4, 4),  (1, 4, 3),  2,  None),
        ((1, 3, 5, 5), (1, 3, 3, 4), 2, [1, 1]),
    ])
    def test_numerical_grad(self, in_shape, crop_shape, axis, offset):
        """Numerical gradient matches analytical dX."""
        rng = np.random.RandomState(789)
        x = rng.randn(*in_shape).astype(np.float32) * 0.5
        crop_ref = np.zeros(crop_shape, dtype=np.float32)
        dy = rng.randn(*crop_shape).astype(np.float32) * 0.1
        net = _make_crop_net(in_shape, crop_shape, axis=axis, offset=offset)
        dX, _ = _run_crop_backward(net, x, crop_ref, dy)
        num_dx = _numerical_grad_input(net, x, crop_ref, dy, h=EPS)
        np.testing.assert_allclose(dX, num_dx, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Tests: Properties (L4)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestCropBackwardProperties:
    """Property tests: zero-pad, zero dy, shapes, determinism, round-trip."""

    def test_zero_gradient_outside_crop_region(self):
        """Pixels excluded by the crop get exactly 0 gradient."""
        in_shape, crop_shape, axis, offset = (1, 1, 5, 5), (1, 1, 2, 3), 2, [1, 1]
        net = _make_crop_net(in_shape, crop_shape, axis=axis, offset=offset)
        rng = np.random.RandomState(202)
        x = rng.randn(*in_shape).astype(np.float32)
        crop_ref = np.zeros(crop_shape, dtype=np.float32)
        dy = rng.randn(*crop_shape).astype(np.float32)
        dX, _ = _run_crop_backward(net, x, crop_ref, dy)
        # Outside the cropped block [0,0,1:3,1:4] must be exactly 0.
        untouched = dX.copy()
        untouched[0, 0, 1:3, 1:4] = 0.0
        np.testing.assert_array_equal(untouched, 0.0)

    def test_zero_dy_gives_zero_gradients(self):
        """dy=0 -> dX=0 everywhere."""
        in_shape, crop_shape, axis = (5,), (3,), 0
        net = _make_crop_net(in_shape, crop_shape, axis=axis, offset=[1])
        x = np.random.RandomState(303).randn(*in_shape).astype(np.float32)
        crop_ref = np.zeros(crop_shape, dtype=np.float32)
        dy = np.zeros(crop_shape, dtype=np.float32)
        dX, _ = _run_crop_backward(net, x, crop_ref, dy)
        np.testing.assert_array_equal(dX, 0.0)

    def test_gradient_shape(self):
        """dX has the same shape as the (uncropped) input data."""
        in_shape, crop_shape, axis = (1, 3, 5, 5), (1, 3, 3, 4), 2
        net = _make_crop_net(in_shape, crop_shape, axis=axis, offset=None)
        rng = np.random.RandomState(404)
        x = rng.randn(*in_shape).astype(np.float32)
        crop_ref = np.zeros(crop_shape, dtype=np.float32)
        dy = rng.randn(*crop_shape).astype(np.float32)
        dX, _ = _run_crop_backward(net, x, crop_ref, dy)
        assert dX.shape == in_shape

    def test_determinism(self):
        """Same inputs -> same gradients."""
        in_shape, crop_shape, axis = (5,), (3,), 0
        offset = [1]
        rng = np.random.RandomState(505)
        x = rng.randn(*in_shape).astype(np.float32)
        crop_ref = np.zeros(crop_shape, dtype=np.float32)
        dy = rng.randn(*crop_shape).astype(np.float32)
        net1 = _make_crop_net(in_shape, crop_shape, axis=axis, offset=offset)
        dX1, _ = _run_crop_backward(net1, x, crop_ref, dy)
        net2 = _make_crop_net(in_shape, crop_shape, axis=axis, offset=offset)
        dX2, _ = _run_crop_backward(net2, x, crop_ref, dy)
        np.testing.assert_array_equal(dX1, dX2)

    def test_forward_preserved_after_backward(self):
        """Backward doesn't change forward output."""
        in_shape, crop_shape, axis = (1, 2, 4, 4), (1, 2, 2, 3), 2
        net = _make_crop_net(in_shape, crop_shape, axis=axis, offset=None)
        rng = np.random.RandomState(606)
        x = rng.randn(*in_shape).astype(np.float32)
        crop_ref = np.zeros(crop_shape, dtype=np.float32)
        dy = rng.randn(*crop_shape).astype(np.float32)
        out1 = net.forward({"data": x, "crop_ref": crop_ref})["out"].copy()
        net.backward({"out": dy})
        out2 = net.forward({"data": x, "crop_ref": crop_ref})["out"]
        np.testing.assert_allclose(out1, out2, rtol=1e-6)

    def test_finite_values(self):
        """Gradients are finite (no NaN/Inf)."""
        in_shape, crop_shape, axis = (1, 3, 5, 5), (1, 3, 3, 4), 2
        net = _make_crop_net(in_shape, crop_shape, axis=axis, offset=None)
        rng = np.random.RandomState(707)
        x = rng.randn(*in_shape).astype(np.float32) * 0.5
        crop_ref = np.zeros(crop_shape, dtype=np.float32)
        dy = rng.randn(*crop_shape).astype(np.float32)
        dX, _ = _run_crop_backward(net, x, crop_ref, dy)
        assert np.all(np.isfinite(dX))

    def test_round_trip_crop_backward(self):
        """Crop -> (insert zeros) recovers the original data region.

        The backward zero-pads the crop: dX = dy placed at the crop offset.
        Building the crop from dX's cropped region must reproduce dy.
        """
        in_shape, crop_shape, axis, offset = (1, 1, 5, 5), (1, 1, 2, 3), 2, [1, 1]
        net = _make_crop_net(in_shape, crop_shape, axis=axis, offset=offset)
        rng = np.random.RandomState(808)
        x = rng.randn(*in_shape).astype(np.float32)
        crop_ref = np.zeros(crop_shape, dtype=np.float32)
        dy = rng.randn(*crop_shape).astype(np.float32)
        dX, _ = _run_crop_backward(net, x, crop_ref, dy)
        # Re-extract the crop region from dX -> should equal dy.
        np.testing.assert_allclose(dX[0, 0, 1:3, 1:4], dy[0, 0], rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])