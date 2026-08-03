"""Concat layer Backward gradient tests.

Concat concatenates multiple input blobs along a specified axis.
Gradient (Backward): each bottom receives the slice of top_diff corresponding
to its position along the concat axis (reverse of Forward's memcpy).

Covers:
  1. Known-value hand verification for axis=0 and axis=1
  2. Analytical gradient (numpy reference vs caffe-ffi)
  3. Numerical gradient check (central finite differences for each input)
  4. Multiple axes (0, 1, 2, 3) and varying sizes
  5. Three or more inputs
  6. Zero dy -> zero gradients
  7. Shape/finite/determinism checks
  8. Forward-Backward round-trip (split -> concat -> split recovers original)
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension
from ._grad_check_utils import (
    numerical_gradient,
)

EPS = 1e-3
RTOL = 1e-3
ATOL = 1e-4


# ---------------------------------------------------------------------------
# Numpy reference for Concat forward/backward
# ---------------------------------------------------------------------------

def concat_forward_np(inputs, axis=1):
    """Numpy reference: concatenate inputs along axis."""
    return np.concatenate(inputs, axis=axis).astype(np.float32)


def concat_backward_np(dy, input_shapes, axis=1):
    """Numpy reference for Concat backward.

    Returns list of dX arrays, each being the slice of dy along axis.
    """
    dXs = []
    offset = 0
    ndim = dy.ndim
    for shape in input_shapes:
        dim = shape[axis]
        # Build slice tuple
        sl = [slice(None)] * ndim
        sl[axis] = slice(offset, offset + dim)
        dXs.append(dy[tuple(sl)].copy().astype(np.float32))
        offset += dim
    return dXs


# ---------------------------------------------------------------------------
# Prototxt builder
# ---------------------------------------------------------------------------

def _make_concat_prototxt(shapes, axis=1):
    """Create multi-Input -> Concat prototxt.

    Args:
        shapes: list of shape tuples (one per input). All dims except `axis` must match.
        axis: concat axis (default=1, i.e. channel axis for NCHW).
    """
    num_inputs = len(shapes)
    input_names = [chr(ord('a') + i) for i in range(num_inputs)]

    input_layers = []
    for name, shape in zip(input_names, shapes):
        dims_lines = "\n".join(f"          dim: {d}" for d in shape)
        input_layers.append(textwrap.dedent(f"""\
            layer {{
              name: "{name}"
              type: "Input"
              top: "{name}"
              input_param {{ shape {{
{dims_lines}
              }} }}
            }}"""))
    input_layers_str = "\n".join(input_layers)

    bottoms_str = "\n".join(f'  bottom: "{name}"' for name in input_names)

    return textwrap.dedent(f"""\
        name: "test_concat_bw"
{input_layers_str}
        layer {{
          name: "cat"
          type: "Concat"
{bottoms_str}
          top: "out"
          concat_param {{ axis: {axis} }}
        }}
    """)


def _make_concat_net(shapes, axis=1):
    proto = _make_concat_prototxt(shapes, axis)
    return Net(proto)


# ---------------------------------------------------------------------------
# Helper: run forward+backward
# ---------------------------------------------------------------------------

def _run_concat_backward(net, inputs, dy, num_inputs=2):
    """Run forward then backward, return list of dX arrays and output."""
    input_names = [chr(ord('a') + i) for i in range(num_inputs)]
    input_dict = {name: x.astype(np.float32) for name, x in zip(input_names, inputs)}
    out = net.forward(input_dict)
    net.backward({"out": dy.astype(np.float32)})
    dXs = [net.blob_by_name(name).diff for name in input_names]
    return dXs, out["out"]


def _numerical_grad_input(net, inputs, input_idx, dy, output_name="out", h=EPS):
    """Compute numerical gradient w.r.t. one input."""
    num_inputs = len(inputs)
    input_names = [chr(ord('a') + i) for i in range(num_inputs)]
    target_name = input_names[input_idx]
    current = inputs[input_idx].astype(np.float32).copy()

    def _forward():
        feed = {}
        for i, name in enumerate(input_names):
            if i == input_idx:
                feed[name] = current
            else:
                feed[name] = inputs[i].astype(np.float32)
        out = net.forward(feed)
        return out[output_name]

    def _get():
        return current.copy()

    def _set(arr):
        nonlocal current
        np.copyto(current, arr)

    return numerical_gradient(_forward, _get, _set, dy, h=h,
                              name=f"input:{target_name}", verbose=False)


# ---------------------------------------------------------------------------
# Tests: Known values (L1)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestConcatBackwardKnownValues:
    """Hand-computed known value tests."""

    def test_concat_axis0_two_inputs(self):
        """Concat axis=0: a=[1,2], b=[3,4,5] -> out=[1,2,3,4,5], dA=dy[0:2], dB=dy[2:5]."""
        shapes = [(2,), (3,)]
        net = _make_concat_net(shapes, axis=0)
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        dy = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
        dXs, out = _run_concat_backward(net, [a, b], dy, num_inputs=2)
        np.testing.assert_allclose(out, np.concatenate([a, b], axis=0), rtol=1e-5)
        np.testing.assert_allclose(dXs[0], dy[0:2], rtol=1e-5)
        np.testing.assert_allclose(dXs[1], dy[2:5], rtol=1e-5)

    def test_concat_axis1_2d(self):
        """Concat axis=1: a shape(1,2), b shape(1,3) -> out shape(1,5)."""
        shapes = [(1, 2), (1, 3)]
        net = _make_concat_net(shapes, axis=1)
        a = np.array([[1.0, 2.0]], dtype=np.float32)
        b = np.array([[3.0, 4.0, 5.0]], dtype=np.float32)
        dy = np.array([[10.0, 20.0, 30.0, 40.0, 50.0]], dtype=np.float32)
        dXs, out = _run_concat_backward(net, [a, b], dy, num_inputs=2)
        np.testing.assert_allclose(out, np.concatenate([a, b], axis=1), rtol=1e-5)
        np.testing.assert_allclose(dXs[0], dy[:, 0:2], rtol=1e-5)
        np.testing.assert_allclose(dXs[1], dy[:, 2:5], rtol=1e-5)

    def test_concat_axis1_nchw(self):
        """Concat axis=1 (channels): two (1,2,2,2) blobs -> (1,4,2,2)."""
        shapes = [(1, 2, 2, 2), (1, 2, 2, 2)]
        net = _make_concat_net(shapes, axis=1)
        rng = np.random.RandomState(0)
        a = rng.randn(1, 2, 2, 2).astype(np.float32)
        b = rng.randn(1, 2, 2, 2).astype(np.float32)
        dy = rng.randn(1, 4, 2, 2).astype(np.float32)
        dXs, out = _run_concat_backward(net, [a, b], dy, num_inputs=2)
        np.testing.assert_allclose(out, np.concatenate([a, b], axis=1), rtol=1e-5)
        np.testing.assert_allclose(dXs[0], dy[:, :2, :, :], rtol=1e-5)
        np.testing.assert_allclose(dXs[1], dy[:, 2:, :, :], rtol=1e-5)

    def test_concat_axis3_nchw(self):
        """Concat axis=3 (width): (1,1,2,2) + (1,1,2,3) -> (1,1,2,5)."""
        shapes = [(1, 1, 2, 2), (1, 1, 2, 3)]
        net = _make_concat_net(shapes, axis=3)
        a = np.ones((1, 1, 2, 2), dtype=np.float32)
        b = np.ones((1, 1, 2, 3), dtype=np.float32) * 2.0
        dy = np.arange(1*1*2*5, dtype=np.float32).reshape(1, 1, 2, 5)
        dXs, out = _run_concat_backward(net, [a, b], dy, num_inputs=2)
        np.testing.assert_allclose(out, np.concatenate([a, b], axis=3), rtol=1e-5)
        np.testing.assert_allclose(dXs[0], dy[:, :, :, :2], rtol=1e-5)
        np.testing.assert_allclose(dXs[1], dy[:, :, :, 2:], rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Analytical gradient vs numpy (L2)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestConcatBackwardNumpy:
    """Compare caffe-ffi backward against numpy reference."""

    @pytest.mark.parametrize("shapes,axis", [
        ([(2, 3), (2, 4)], 1),           # axis=1, 2D
        ([(1, 3, 4), (1, 5, 4)], 1),      # axis=1, 3D
        ([(2, 3, 4), (2, 3, 6)], 2),      # axis=2, 3D
        ([(1, 2, 3, 4), (1, 3, 3, 4)], 1), # axis=1, 4D NCHW
        ([(1, 2, 3, 4), (1, 2, 5, 4)], 2), # axis=2 (H), 4D
        ([(1, 2, 3, 4), (1, 2, 3, 6)], 3), # axis=3 (W), 4D
    ])
    def test_concat_vs_numpy(self, shapes, axis):
        """Concat backward matches numpy slicing for various shapes/axes."""
        rng = np.random.RandomState(42)
        inputs = [rng.randn(*s).astype(np.float32) * 0.5 for s in shapes]
        out_shape = list(shapes[0])
        out_shape[axis] = sum(s[axis] for s in shapes)
        dy = rng.randn(*out_shape).astype(np.float32)
        net = _make_concat_net(shapes, axis=axis)
        dXs, out = _run_concat_backward(net, inputs, dy, num_inputs=len(shapes))
        np.testing.assert_allclose(out, concat_forward_np(inputs, axis), rtol=RTOL, atol=ATOL)
        dXs_ref = concat_backward_np(dy, shapes, axis=axis)
        for j, (dx, dx_ref) in enumerate(zip(dXs, dXs_ref)):
            np.testing.assert_allclose(dx, dx_ref, rtol=RTOL, atol=ATOL,
                                       err_msg=f"bottom[{j}] gradient mismatch")

    def test_three_inputs(self):
        """Concat with 3 inputs along axis=1."""
        shapes = [(2, 2), (2, 3), (2, 1)]
        axis = 1
        rng = np.random.RandomState(123)
        inputs = [rng.randn(*s).astype(np.float32) for s in shapes]
        out_shape = list(shapes[0])
        out_shape[axis] = sum(s[axis] for s in shapes)
        dy = rng.randn(*out_shape).astype(np.float32)
        net = _make_concat_net(shapes, axis=axis)
        dXs, out = _run_concat_backward(net, inputs, dy, num_inputs=3)
        np.testing.assert_allclose(out, concat_forward_np(inputs, axis), rtol=RTOL, atol=ATOL)
        dXs_ref = concat_backward_np(dy, shapes, axis=axis)
        for j, (dx, dx_ref) in enumerate(zip(dXs, dXs_ref)):
            np.testing.assert_allclose(dx, dx_ref, rtol=RTOL, atol=ATOL)

    def test_four_inputs_axis0(self):
        """Concat with 4 inputs along axis=0."""
        shapes = [(1, 3), (2, 3), (3, 3), (1, 3)]
        axis = 0
        rng = np.random.RandomState(456)
        inputs = [rng.randn(*s).astype(np.float32) * 0.3 for s in shapes]
        out_shape = list(shapes[0])
        out_shape[axis] = sum(s[axis] for s in shapes)
        dy = rng.randn(*out_shape).astype(np.float32)
        net = _make_concat_net(shapes, axis=axis)
        dXs, out = _run_concat_backward(net, inputs, dy, num_inputs=4)
        np.testing.assert_allclose(out, concat_forward_np(inputs, axis), rtol=RTOL, atol=ATOL)
        dXs_ref = concat_backward_np(dy, shapes, axis=axis)
        for j, (dx, dx_ref) in enumerate(zip(dXs, dXs_ref)):
            np.testing.assert_allclose(dx, dx_ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Tests: Numerical gradient (L3)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestConcatBackwardNumerical:
    """Central finite difference numerical gradient checks."""

    @pytest.mark.parametrize("shapes,axis", [
        ([(2, 3), (2, 4)], 1),
        ([(2, 3, 2), (2, 3, 3)], 2),
        ([(1, 2, 2, 2), (1, 3, 2, 2)], 1),
    ])
    def test_numerical_grad(self, shapes, axis):
        """Numerical gradient for each input matches analytical dX."""
        rng = np.random.RandomState(789)
        inputs = [rng.randn(*s).astype(np.float32) * 0.5 for s in shapes]
        out_shape = list(shapes[0])
        out_shape[axis] = sum(s[axis] for s in shapes)
        dy = rng.randn(*out_shape).astype(np.float32) * 0.1
        net = _make_concat_net(shapes, axis=axis)
        dXs, _ = _run_concat_backward(net, inputs, dy, num_inputs=len(shapes))
        for j in range(len(shapes)):
            net2 = _make_concat_net(shapes, axis=axis)
            num_dx = _numerical_grad_input(net2, inputs, j, dy, h=EPS)
            np.testing.assert_allclose(dXs[j], num_dx, rtol=RTOL, atol=ATOL,
                                       err_msg=f"Numerical grad mismatch for bottom[{j}]")

    def test_numerical_grad_three_inputs(self):
        """Numerical gradient for 3-input concat."""
        shapes = [(1, 2, 2), (1, 3, 2), (1, 1, 2)]
        axis = 1
        rng = np.random.RandomState(101)
        inputs = [rng.randn(*s).astype(np.float32) * 0.5 for s in shapes]
        out_shape = list(shapes[0])
        out_shape[axis] = sum(s[axis] for s in shapes)
        dy = rng.randn(*out_shape).astype(np.float32) * 0.1
        net = _make_concat_net(shapes, axis=axis)
        dXs, _ = _run_concat_backward(net, inputs, dy, num_inputs=3)
        for j in range(3):
            net2 = _make_concat_net(shapes, axis=axis)
            num_dx = _numerical_grad_input(net2, inputs, j, dy, h=EPS)
            np.testing.assert_allclose(dXs[j], num_dx, rtol=RTOL, atol=ATOL,
                                       err_msg=f"Numerical grad mismatch for bottom[{j}]")


# ---------------------------------------------------------------------------
# Tests: Properties (L4)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestConcatBackwardProperties:
    """Property tests: zero gradients, shapes, determinism, round-trip."""

    def test_zero_dy_gives_zero_gradients(self):
        """dy=0 -> all dX=0."""
        shapes = [(2, 3), (2, 4)]
        axis = 1
        rng = np.random.RandomState(202)
        inputs = [rng.randn(*s).astype(np.float32) for s in shapes]
        dy = np.zeros((2, 7), dtype=np.float32)
        net = _make_concat_net(shapes, axis=axis)
        dXs, _ = _run_concat_backward(net, inputs, dy, num_inputs=2)
        for dx in dXs:
            np.testing.assert_array_equal(dx, 0.0)

    def test_gradient_shapes(self):
        """Each dX has same shape as corresponding input."""
        shapes = [(2, 3), (2, 5), (2, 2)]
        axis = 1
        rng = np.random.RandomState(303)
        inputs = [rng.randn(*s).astype(np.float32) for s in shapes]
        out_shape = list(shapes[0])
        out_shape[axis] = sum(s[axis] for s in shapes)
        dy = rng.randn(*out_shape).astype(np.float32)
        net = _make_concat_net(shapes, axis=axis)
        dXs, _ = _run_concat_backward(net, inputs, dy, num_inputs=3)
        for j, (dx, s) in enumerate(zip(dXs, shapes)):
            assert dx.shape == s, f"dX[{j}] shape {dx.shape} != input shape {s}"

    def test_determinism(self):
        """Same inputs -> same gradients."""
        shapes = [(2, 3), (2, 4)]
        axis = 1
        rng = np.random.RandomState(404)
        inputs = [rng.randn(*s).astype(np.float32) for s in shapes]
        dy = rng.randn(2, 7).astype(np.float32)
        net1 = _make_concat_net(shapes, axis=axis)
        dXs1, _ = _run_concat_backward(net1, inputs, dy, num_inputs=2)
        net2 = _make_concat_net(shapes, axis=axis)
        dXs2, _ = _run_concat_backward(net2, inputs, dy, num_inputs=2)
        for j in range(2):
            np.testing.assert_array_equal(dXs1[j], dXs2[j])

    def test_forward_preserved_after_backward(self):
        """Backward doesn't change forward output."""
        shapes = [(1, 3, 2, 2), (1, 2, 2, 2)]
        axis = 1
        rng = np.random.RandomState(505)
        inputs = [rng.randn(*s).astype(np.float32) for s in shapes]
        dy = rng.randn(1, 5, 2, 2).astype(np.float32)
        net = _make_concat_net(shapes, axis=axis)
        out1 = net.forward({chr(ord('a')+i): x for i, x in enumerate(inputs)})["out"].copy()
        net.backward({"out": dy})
        out2 = net.forward({chr(ord('a')+i): x for i, x in enumerate(inputs)})["out"]
        np.testing.assert_allclose(out1, out2, rtol=1e-6)

    def test_finite_values(self):
        """Gradients are finite (no NaN/Inf)."""
        shapes = [(3, 4), (3, 2)]
        axis = 1
        rng = np.random.RandomState(606)
        inputs = [rng.randn(*s).astype(np.float32) * 0.5 for s in shapes]
        dy = rng.randn(3, 6).astype(np.float32)
        net = _make_concat_net(shapes, axis=axis)
        dXs, _ = _run_concat_backward(net, inputs, dy, num_inputs=2)
        for j, dx in enumerate(dXs):
            assert np.all(np.isfinite(dx)), f"dX[{j}] has NaN/Inf"

    def test_round_trip_split_concat(self):
        """Concat -> Backward splits dy exactly; Backward->Forward->concat should recover dy.

        Forward concatenates inputs, backward slices dy. If we forward the backward
        results (treating dXs as new inputs), we should recover dy (since Concat backward
        is just slicing, forward is just concatenation).
        """
        shapes = [(2, 3), (2, 4), (2, 2)]
        axis = 1
        rng = np.random.RandomState(707)
        inputs = [rng.randn(*s).astype(np.float32) for s in shapes]
        out_shape = list(shapes[0])
        out_shape[axis] = sum(s[axis] for s in shapes)
        dy = rng.randn(*out_shape).astype(np.float32)
        net = _make_concat_net(shapes, axis=axis)
        dXs, _ = _run_concat_backward(net, inputs, dy, num_inputs=3)
        # Concat the dXs back together -> should recover dy
        reconstructed = concat_forward_np(list(dXs), axis=axis)
        np.testing.assert_allclose(reconstructed, dy, rtol=1e-6)

    def test_axis2_3d_numerical(self):
        """3D concat along axis=2 numerical gradient."""
        shapes = [(1, 2, 3), (1, 2, 4)]
        axis = 2
        rng = np.random.RandomState(808)
        inputs = [rng.randn(*s).astype(np.float32) * 0.5 for s in shapes]
        out_shape = list(shapes[0])
        out_shape[axis] = sum(s[axis] for s in shapes)
        dy = rng.randn(*out_shape).astype(np.float32) * 0.1
        net = _make_concat_net(shapes, axis=axis)
        dXs, _ = _run_concat_backward(net, inputs, dy, num_inputs=2)
        for j in range(2):
            net2 = _make_concat_net(shapes, axis=axis)
            num_dx = _numerical_grad_input(net2, inputs, j, dy, h=EPS)
            np.testing.assert_allclose(dXs[j], num_dx, rtol=RTOL, atol=ATOL)

    def test_unequal_sizes_axis0(self):
        """Concat axis=0 with unequal sizes."""
        shapes = [(3, 2), (5, 2)]
        axis = 0
        rng = np.random.RandomState(909)
        inputs = [rng.randn(*s).astype(np.float32) for s in shapes]
        out_shape = list(shapes[0])
        out_shape[axis] = sum(s[axis] for s in shapes)
        dy = rng.randn(*out_shape).astype(np.float32)
        net = _make_concat_net(shapes, axis=axis)
        dXs, out = _run_concat_backward(net, inputs, dy, num_inputs=2)
        np.testing.assert_allclose(out, concat_forward_np(inputs, axis), rtol=RTOL, atol=ATOL)
        dXs_ref = concat_backward_np(dy, shapes, axis=axis)
        np.testing.assert_allclose(dXs[0], dXs_ref[0], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(dXs[1], dXs_ref[1], rtol=RTOL, atol=ATOL)
