"""Slice layer Backward gradient tests.

Slice splits the bottom along an axis into N tops (reverse of Concat).
Backward: each top's diff is copied back to its slice of the bottom diff,
i.e. bottom_diff is the concatenation of top diffs along the slice axis.

Covers:
  1. Known-value hand verification (axis=0, axis=1, slice_points)
  2. Analytical gradient (numpy reference vs caffe-ffi)
  3. Numerical gradient check (central finite differences)
  4. N=1 identity passthrough
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
# Numpy reference
# ---------------------------------------------------------------------------

def slice_backward_np(dys, axis):
    """Numpy reference for Slice backward: concatenate top diffs along axis."""
    return np.concatenate([d.astype(np.float32) for d in dys], axis=axis)


# ---------------------------------------------------------------------------
# Prototxt builder
# ---------------------------------------------------------------------------

def _make_slice_prototxt(shape, num_tops, axis, slice_points=None):
    dims = "\n".join(f"          dim: {d}" for d in shape)
    tops = "\n".join(f'  top: "t{i}"' for i in range(num_tops))
    sp = ""
    if slice_points:
        sp_lines = "\n".join(f"    slice_point: {p}" for p in slice_points)
        sp = f"\n{sp_lines}"
    return textwrap.dedent(f"""\
        name: "test_slice_bw"
        layer {{
          name: "data"
          type: "Input"
          top: "data"
          input_param {{ shape {{
{dims}
          }} }}
        }}
        layer {{
          name: "slice"
          type: "Slice"
          bottom: "data"
{tops}
          slice_param {{
            axis: {axis}{sp}
          }}
        }}
    """)


def _make_slice_net(shape, num_tops, axis, slice_points=None):
    return Net(_make_slice_prototxt(shape, num_tops, axis, slice_points))


def _split_dys_along_axis(dy, axis, slice_points):
    """Split a full dy into per-top diffs along axis (for the test's dy generation)."""
    out = []
    prev = 0
    for p in list(slice_points) + [dy.shape[axis]]:
        s = [slice(None)] * dy.ndim
        s[axis] = slice(prev, p)
        out.append(dy[tuple(s)].copy())
        prev = p
    return out


def _run_slice_backward(net, x, dys, num_tops):
    """Run forward then backward, return (dX, out_dict)."""
    out = net.forward({"data": x.astype(np.float32)})
    diff_dict = {f"t{i}": dys[i].astype(np.float32) for i in range(num_tops)}
    net.backward(diff_dict)
    dX = net.blob_by_name("data").diff
    return dX, out


def _numerical_grad_input(net, x, dys, num_tops, h=EPS):
    """Numerical gradient w.r.t. input via flattened-concatenation.

    L = sum_i sum(dy_i * out_i). Flatten and concatenate the outputs and dy;
    the total length equals the input count (Slice tiles the input exactly).
    """
    current = x.astype(np.float32).copy()
    dy_flat = np.concatenate([d.astype(np.float32).ravel() for d in dys])

    def _forward():
        out = net.forward({"data": current})
        return np.concatenate([out[f"t{i}"].ravel() for i in range(num_tops)])

    def _get():
        return current.copy()

    def _set(arr):
        nonlocal current
        np.copyto(current, arr)

    return numerical_gradient(_forward, _get, _set, dy_flat, h=h,
                              name="input:data", verbose=False)


# ---------------------------------------------------------------------------
# Tests: Known values (L1)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestSliceBackwardKnownValues:
    """Hand-computed known value tests."""

    def test_slice_axis0_2way(self):
        """Slice axis=0, N=2: x=[1,2,3,4] -> t0=[1,2], t1=[3,4]; dX=[dy0,dy1]."""
        net = _make_slice_net((4,), 2, 0)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        dy0 = np.array([10.0, 20.0], dtype=np.float32)
        dy1 = np.array([30.0, 40.0], dtype=np.float32)
        dX, out = _run_slice_backward(net, x, [dy0, dy1], 2)
        np.testing.assert_allclose(out["t0"], x[0:2], rtol=1e-6)
        np.testing.assert_allclose(out["t1"], x[2:4], rtol=1e-6)
        np.testing.assert_allclose(dX, np.concatenate([dy0, dy1]), rtol=1e-5)

    def test_slice_axis1_2d(self):
        """Slice axis=1: x shape(1,6) -> t0 shape(1,3), t1 shape(1,3)."""
        net = _make_slice_net((1, 6), 2, 1)
        x = np.arange(6, dtype=np.float32).reshape(1, 6)
        dy0 = np.full((1, 3), 1.0, dtype=np.float32)
        dy1 = np.full((1, 3), 2.0, dtype=np.float32)
        dX, _ = _run_slice_backward(net, x, [dy0, dy1], 2)
        np.testing.assert_allclose(dX, np.concatenate([dy0, dy1], axis=1), rtol=1e-5)

    def test_slice_axis1_nchw(self):
        """Slice axis=1 (channels): 6ch -> 3ch+3ch; dX = concat(dy0, dy1)."""
        net = _make_slice_net((1, 6, 2, 2), 2, 1)
        rng = np.random.RandomState(0)
        x = rng.randn(1, 6, 2, 2).astype(np.float32)
        dy0 = rng.randn(1, 3, 2, 2).astype(np.float32)
        dy1 = rng.randn(1, 3, 2, 2).astype(np.float32)
        dX, _ = _run_slice_backward(net, x, [dy0, dy1], 2)
        np.testing.assert_allclose(dX, np.concatenate([dy0, dy1], axis=1), rtol=1e-5)

    def test_slice_explicit_points(self):
        """Slice with slice_points=[1,3]: 6ch -> 1ch+2ch+3ch; dX = concat(dy0,dy1,dy2)."""
        net = _make_slice_net((1, 6, 2, 2), 3, 1, slice_points=[1, 3])
        rng = np.random.RandomState(7)
        x = rng.randn(1, 6, 2, 2).astype(np.float32)
        dy0 = rng.randn(1, 1, 2, 2).astype(np.float32)
        dy1 = rng.randn(1, 2, 2, 2).astype(np.float32)
        dy2 = rng.randn(1, 3, 2, 2).astype(np.float32)
        dX, _ = _run_slice_backward(net, x, [dy0, dy1, dy2], 3)
        np.testing.assert_allclose(dX, np.concatenate([dy0, dy1, dy2], axis=1), rtol=1e-5)

    def test_slice_n1_identity(self):
        """Slice N=1 is identity passthrough: dX = dy."""
        net = _make_slice_net((2, 3), 1, 1)
        x = np.random.RandomState(99).randn(2, 3).astype(np.float32)
        dy0 = np.random.RandomState(1).randn(2, 3).astype(np.float32)
        dX, out = _run_slice_backward(net, x, [dy0], 1)
        np.testing.assert_allclose(out["t0"], x, rtol=1e-6)
        np.testing.assert_allclose(dX, dy0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Analytical gradient vs numpy (L2)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestSliceBackwardNumpy:
    """Compare caffe-ffi backward against numpy reference (concatenation)."""

    @pytest.mark.parametrize("shape,num_tops,axis", [
        ((2, 6), 2, 1),
        ((2, 6), 3, 1),
        ((1, 6, 4), 2, 1),
        ((1, 2, 6, 4), 2, 2),
        ((1, 2, 3, 6), 3, 3),
    ])
    def test_slice_vs_numpy(self, shape, num_tops, axis):
        """Slice backward matches numpy concatenation for various shapes/axes."""
        rng = np.random.RandomState(42)
        x = rng.randn(*shape).astype(np.float32) * 0.5
        dim = shape[axis]
        assert dim % num_tops == 0, "axis dim must be divisible by num_tops"
        part = dim // num_tops
        # Generate a full dy then split it into per-top diffs
        full_dy = rng.randn(*shape).astype(np.float32) * 0.1
        slice_points = [part * (i + 1) for i in range(num_tops - 1)]
        dys = _split_dys_along_axis(full_dy, axis, slice_points)
        net = _make_slice_net(shape, num_tops, axis)
        dX, out = _run_slice_backward(net, x, dys, num_tops)
        # Forward: slice x
        s = [slice(None)] * x.ndim
        s[axis] = slice(0, part)
        np.testing.assert_allclose(out["t0"], x[tuple(s)], rtol=RTOL, atol=ATOL)
        ref = slice_backward_np(dys, axis)
        np.testing.assert_allclose(dX, ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Tests: Numerical gradient (L3)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestSliceBackwardNumerical:
    """Central finite difference numerical gradient checks."""

    @pytest.mark.parametrize("shape,num_tops,axis", [
        ((4,), 2, 0),
        ((2, 6), 2, 1),
        ((2, 6), 3, 1),
        ((1, 2, 6, 2), 2, 2),
    ])
    def test_numerical_grad(self, shape, num_tops, axis):
        """Numerical gradient matches analytical dX."""
        rng = np.random.RandomState(789)
        x = rng.randn(*shape).astype(np.float32) * 0.5
        dim = shape[axis]
        full_dy = rng.randn(*shape).astype(np.float32) * 0.1
        part = dim // num_tops
        slice_points = [part * (i + 1) for i in range(num_tops - 1)]
        dys = _split_dys_along_axis(full_dy, axis, slice_points)
        net = _make_slice_net(shape, num_tops, axis)
        dX, _ = _run_slice_backward(net, x, dys, num_tops)
        num_dx = _numerical_grad_input(net, x, dys, num_tops, h=EPS)
        np.testing.assert_allclose(dX, num_dx, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Tests: Properties (L4)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestSliceBackwardProperties:
    """Property tests: zero gradients, shapes, determinism, round-trip."""

    def test_zero_dy_gives_zero_gradients(self):
        """dy=0 for all tops -> dX=0."""
        shape = (2, 6)
        net = _make_slice_net(shape, 2, 1)
        x = np.random.RandomState(202).randn(*shape).astype(np.float32)
        dys = [np.zeros((2, 3), dtype=np.float32) for _ in range(2)]
        dX, _ = _run_slice_backward(net, x, dys, 2)
        np.testing.assert_array_equal(dX, 0.0)

    def test_gradient_shape(self):
        """dX has same shape as input."""
        shape = (2, 6)
        net = _make_slice_net(shape, 3, 1)
        rng = np.random.RandomState(303)
        x = rng.randn(*shape).astype(np.float32)
        dys = [rng.randn(2, 2).astype(np.float32) for _ in range(3)]
        dX, _ = _run_slice_backward(net, x, dys, 3)
        assert dX.shape == shape

    def test_determinism(self):
        """Same inputs -> same gradients."""
        shape = (2, 6)
        rng = np.random.RandomState(404)
        x = rng.randn(*shape).astype(np.float32)
        dy0 = rng.randn(2, 3).astype(np.float32)
        dy1 = rng.randn(2, 3).astype(np.float32)
        net1 = _make_slice_net(shape, 2, 1)
        dX1, _ = _run_slice_backward(net1, x, [dy0, dy1], 2)
        net2 = _make_slice_net(shape, 2, 1)
        dX2, _ = _run_slice_backward(net2, x, [dy0, dy1], 2)
        np.testing.assert_array_equal(dX1, dX2)

    def test_forward_preserved_after_backward(self):
        """Backward doesn't change forward output."""
        shape = (1, 6, 2, 2)
        net = _make_slice_net(shape, 2, 1)
        rng = np.random.RandomState(505)
        x = rng.randn(*shape).astype(np.float32)
        dy0 = rng.randn(1, 3, 2, 2).astype(np.float32)
        dy1 = rng.randn(1, 3, 2, 2).astype(np.float32)
        out1 = net.forward({"data": x})["t0"].copy()
        net.backward({"t0": dy0, "t1": dy1})
        out2 = net.forward({"data": x})["t0"]
        np.testing.assert_allclose(out1, out2, rtol=1e-6)

    def test_finite_values(self):
        """Gradients are finite (no NaN/Inf)."""
        shape = (3, 6)
        net = _make_slice_net(shape, 2, 1)
        rng = np.random.RandomState(606)
        x = rng.randn(*shape).astype(np.float32) * 0.5
        dy0 = rng.randn(3, 3).astype(np.float32)
        dy1 = rng.randn(3, 3).astype(np.float32)
        dX, _ = _run_slice_backward(net, x, [dy0, dy1], 2)
        assert np.all(np.isfinite(dX))

    def test_round_trip_slice_concat(self):
        """Slice -> Concat round-trip: backward of slice reconstructs the full dy.

        If we slice dy into per-top diffs and run Slice backward, we should
        recover the original full dy (Slice backward is the inverse of Forward).
        """
        shape = (2, 6)
        axis = 1
        net = _make_slice_net(shape, 2, axis)
        rng = np.random.RandomState(707)
        x = rng.randn(*shape).astype(np.float32)
        full_dy = rng.randn(*shape).astype(np.float32)
        dys = _split_dys_along_axis(full_dy, axis, [3])
        dX, _ = _run_slice_backward(net, x, dys, 2)
        np.testing.assert_allclose(dX, full_dy, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])