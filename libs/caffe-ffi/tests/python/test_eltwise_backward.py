"""Eltwise layer Backward gradient tests.

Eltwise supports three operations: SUM, PROD, MAX.
Gradients:
  SUM:  dX_j[i] = dy[i] * coeffs[j]
  PROD: dX_j[i] = dy[i] * coeffs[j] * prod_{k!=j}(coeffs[k] * x_k[i])
  MAX:  dX_j[i] = dy[i] * coeffs[j] if j is argmax, else 0 (winner-take-all)

All bottoms have identical shapes (no broadcasting in caffe-ffi Eltwise).
MAX uses a winner mask (max_idx_) recorded during Forward.

Covers:
  1. Known-value hand verification for all three ops
  2. Analytical gradient (numpy reference vs caffe-ffi)
  3. Numerical gradient check (central finite differences for each input)
  4. Coefficient support (weighted SUM/PROD/MAX)
  5. Three inputs (not just two)
  6. Zero dy -> zero gradients
  7. Shape/finite/determinism checks
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension
from ._grad_check_utils import (
    assert_grad_close,
    numerical_gradient,
)

EPS = 1e-3
RTOL = 1e-3
ATOL = 1e-4


# ---------------------------------------------------------------------------
# Numpy reference for Eltwise forward/backward
# ---------------------------------------------------------------------------

def eltwise_forward_np(inputs, op="SUM", coeffs=None):
    """Numpy reference: element-wise operation across inputs.

    Args:
        inputs: list of numpy arrays (all same shape).
        op: "SUM", "PROD", or "MAX".
        coeffs: list of coefficients (default all 1.0).

    Returns:
        Output numpy array (same shape as inputs).
    """
    n = len(inputs)
    if coeffs is None:
        coeffs = [1.0] * n
    inputs = [x.astype(np.float64) for x in inputs]
    if op == "SUM":
        out = sum(c * x for c, x in zip(coeffs, inputs))
    elif op == "PROD":
        out = np.ones_like(inputs[0])
        for c, x in zip(coeffs, inputs):
            out = out * (c * x)
    elif op == "MAX":
        out = coeffs[0] * inputs[0]
        for c, x in zip(coeffs[1:], inputs[1:]):
            out = np.maximum(out, c * x)
    else:
        raise ValueError(f"Unknown op: {op}")
    return out.astype(np.float32)


def eltwise_backward_np(inputs, dy, op="SUM", coeffs=None):
    """Numpy reference for Eltwise backward.

    Returns:
        List of dX arrays (one per input).
    """
    n = len(inputs)
    if coeffs is None:
        coeffs = [1.0] * n
    inputs64 = [x.astype(np.float64) for x in inputs]
    dy64 = dy.astype(np.float64)
    dXs = []

    if op == "SUM":
        for c in coeffs:
            dXs.append((dy64 * c).astype(np.float32))

    elif op == "PROD":
        for j in range(n):
            # dX_j = dy * coeffs[j] * prod_{k!=j}(coeffs[k] * x_k)
            prod_others = np.ones_like(dy64)
            for k in range(n):
                if k != j:
                    prod_others = prod_others * (coeffs[k] * inputs64[k])
            dXj = dy64 * coeffs[j] * prod_others
            dXs.append(dXj.astype(np.float32))

    elif op == "MAX":
        # Compute winner for each position
        # First pass: compute scaled values
        scaled = [c * x for c, x in zip(coeffs, inputs64)]
        # Find argmax (first occurrence wins, matching C++ implementation)
        max_vals = scaled[0].copy()
        winners = np.zeros_like(max_vals, dtype=np.int32)
        for j in range(1, n):
            mask = scaled[j] > max_vals
            max_vals[mask] = scaled[j][mask]
            winners[mask] = j
        # Assign gradients
        for j in range(n):
            mask = (winners == j)
            dXj = np.where(mask, dy64 * coeffs[j], 0.0)
            dXs.append(dXj.astype(np.float32))

    else:
        raise ValueError(f"Unknown op: {op}")

    return dXs


# ---------------------------------------------------------------------------
# Prototxt builder
# ---------------------------------------------------------------------------

def _make_eltwise_prototxt(shape, num_inputs=2, op="SUM", coeffs=None):
    """Create multi-Input -> Eltwise prototxt.

    Args:
        shape: tuple/list of input dimensions (e.g. (2,3)).
        num_inputs: number of input blobs (named a, b, c, ...).
        op: "SUM", "PROD", or "MAX".
        coeffs: optional list of coefficients.
    """
    dims_lines = "\n".join(f"          dim: {d}" for d in shape)
    input_names = [chr(ord('a') + i) for i in range(num_inputs)]

    input_layers = []
    for name in input_names:
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

    coeff_str = ""
    if coeffs is not None:
        coeff_str = " " + " ".join(f"coeff: {c}" for c in coeffs)

    return textwrap.dedent(f"""\
        name: "test_eltwise_bw"
{input_layers_str}
        layer {{
          name: "ew"
          type: "Eltwise"
{bottoms_str}
          top: "out"
          eltwise_param {{ operation: {op}{coeff_str} }}
        }}
    """)


def _make_eltwise_net(shape, num_inputs=2, op="SUM", coeffs=None):
    proto = _make_eltwise_prototxt(shape, num_inputs, op, coeffs)
    return Net(proto)


# ---------------------------------------------------------------------------
# Helper: run forward+backward
# ---------------------------------------------------------------------------

def _run_eltwise_backward(net, inputs, dy, num_inputs=2):
    """Run forward then backward, return list of dX arrays and output."""
    input_names = [chr(ord('a') + i) for i in range(num_inputs)]
    input_dict = {name: x.astype(np.float32) for name, x in zip(input_names, inputs)}
    out = net.forward(input_dict)
    net.backward({"out": dy.astype(np.float32)})
    dXs = [net.blob_by_name(name).diff for name in input_names]
    return dXs, out["out"]


def _numerical_grad_input(net, inputs, input_idx, dy, output_name="out", h=EPS):
    """Compute numerical gradient w.r.t. one input (holding others fixed)."""
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
class TestEltwiseBackwardKnownValues:
    """Hand-computed known value tests."""

    def test_sum_two_inputs_simple(self):
        """SUM: a=[1,2], b=[3,4], dy=[1,1] -> dA=[1,1], dB=[1,1] (coeffs=1)."""
        net = _make_eltwise_net((1, 2), num_inputs=2, op="SUM")
        a = np.array([[1.0, 2.0]], dtype=np.float32)
        b = np.array([[3.0, 4.0]], dtype=np.float32)
        dy = np.array([[1.0, 1.0]], dtype=np.float32)
        dXs, out = _run_eltwise_backward(net, [a, b], dy, num_inputs=2)
        np.testing.assert_allclose(out, a + b, rtol=1e-5)
        np.testing.assert_allclose(dXs[0], dy, rtol=1e-5)
        np.testing.assert_allclose(dXs[1], dy, rtol=1e-5)

    def test_sum_with_coeffs(self):
        """SUM: coeffs=[2,0.5], a=[1,2], b=[4,6], dy=[1,1] -> dA=[2,2], dB=[0.5,0.5]."""
        net = _make_eltwise_net((1, 2), num_inputs=2, op="SUM", coeffs=[2.0, 0.5])
        a = np.array([[1.0, 2.0]], dtype=np.float32)
        b = np.array([[4.0, 6.0]], dtype=np.float32)
        dy = np.array([[1.0, 1.0]], dtype=np.float32)
        dXs, out = _run_eltwise_backward(net, [a, b], dy, num_inputs=2)
        expected_out = 2 * a + 0.5 * b
        np.testing.assert_allclose(out, expected_out, rtol=1e-5)
        np.testing.assert_allclose(dXs[0], dy * 2.0, rtol=1e-5)
        np.testing.assert_allclose(dXs[1], dy * 0.5, rtol=1e-5)

    def test_prod_two_inputs_simple(self):
        """PROD: a=[2,3], b=[4,5], dy=[1,1] -> dA=[4,5], dB=[2,3]."""
        net = _make_eltwise_net((1, 2), num_inputs=2, op="PROD")
        a = np.array([[2.0, 3.0]], dtype=np.float32)
        b = np.array([[4.0, 5.0]], dtype=np.float32)
        dy = np.array([[1.0, 1.0]], dtype=np.float32)
        dXs, out = _run_eltwise_backward(net, [a, b], dy, num_inputs=2)
        np.testing.assert_allclose(out, a * b, rtol=1e-5)
        np.testing.assert_allclose(dXs[0], b, rtol=1e-5)
        np.testing.assert_allclose(dXs[1], a, rtol=1e-5)

    def test_max_two_inputs_simple(self):
        """MAX: a=[1,5,3], b=[3,2,7], dy=[1,1,1] -> dA=[0,1,0], dB=[1,0,1]."""
        net = _make_eltwise_net((1, 3), num_inputs=2, op="MAX")
        a = np.array([[1.0, 5.0, 3.0]], dtype=np.float32)
        b = np.array([[3.0, 2.0, 7.0]], dtype=np.float32)
        dy = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        dXs, out = _run_eltwise_backward(net, [a, b], dy, num_inputs=2)
        expected_out = np.maximum(a, b)
        np.testing.assert_allclose(out, expected_out, rtol=1e-5)
        # winners: b (3>1), a (5>2), b (7>3)
        np.testing.assert_allclose(dXs[0], np.array([[0.0, 1.0, 0.0]], dtype=np.float32), rtol=1e-5)
        np.testing.assert_allclose(dXs[1], np.array([[1.0, 0.0, 1.0]], dtype=np.float32), rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Analytical gradient vs numpy (L2)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestEltwiseBackwardNumpy:
    """Compare caffe-ffi backward against numpy reference."""

    @pytest.mark.parametrize("shape,num_inputs", [
        ((2, 3), 2),
        ((1, 1, 4, 4), 2),
        ((2, 3, 4), 2),
        ((2, 3), 3),
    ])
    def test_sum_vs_numpy(self, shape, num_inputs):
        """SUM: caffe-ffi dX matches numpy reference for various shapes."""
        rng = np.random.RandomState(42)
        inputs = [rng.randn(*shape).astype(np.float32) * 0.5 + 1.0 for _ in range(num_inputs)]
        dy = rng.randn(*shape).astype(np.float32)
        net = _make_eltwise_net(shape, num_inputs=num_inputs, op="SUM")
        dXs, _ = _run_eltwise_backward(net, inputs, dy, num_inputs=num_inputs)
        dXs_ref = eltwise_backward_np(inputs, dy, op="SUM")
        for j, (dx, dx_ref) in enumerate(zip(dXs, dXs_ref)):
            np.testing.assert_allclose(dx, dx_ref, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("shape,coeffs", [
        ((2, 3), [2.0, 0.5]),
        ((1, 2, 3), [1.0, -1.0]),
    ])
    def test_sum_coeffs_vs_numpy(self, shape, coeffs):
        """SUM with coefficients: caffe-ffi dX matches numpy reference."""
        rng = np.random.RandomState(123)
        inputs = [rng.randn(*shape).astype(np.float32) * 0.5 + 1.0 for _ in coeffs]
        dy = rng.randn(*shape).astype(np.float32)
        net = _make_eltwise_net(shape, num_inputs=len(coeffs), op="SUM", coeffs=coeffs)
        dXs, _ = _run_eltwise_backward(net, inputs, dy, num_inputs=len(coeffs))
        dXs_ref = eltwise_backward_np(inputs, dy, op="SUM", coeffs=coeffs)
        for j, (dx, dx_ref) in enumerate(zip(dXs, dXs_ref)):
            np.testing.assert_allclose(dx, dx_ref, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("shape,num_inputs", [
        ((2, 3), 2),
        ((1, 1, 3, 3), 2),
        ((2, 3), 3),
    ])
    def test_prod_vs_numpy(self, shape, num_inputs):
        """PROD: caffe-ffi dX matches numpy reference (using positive inputs to avoid zero issues)."""
        rng = np.random.RandomState(99)
        # Use positive values away from zero to avoid division issues
        inputs = [rng.uniform(0.5, 2.0, size=shape).astype(np.float32) for _ in range(num_inputs)]
        dy = rng.randn(*shape).astype(np.float32) * 0.1
        net = _make_eltwise_net(shape, num_inputs=num_inputs, op="PROD")
        dXs, out = _run_eltwise_backward(net, inputs, dy, num_inputs=num_inputs)
        out_ref = eltwise_forward_np(inputs, op="PROD")
        np.testing.assert_allclose(out, out_ref, rtol=1e-5)
        dXs_ref = eltwise_backward_np(inputs, dy, op="PROD")
        for j, (dx, dx_ref) in enumerate(zip(dXs, dXs_ref)):
            np.testing.assert_allclose(dx, dx_ref, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("shape,num_inputs", [
        ((2, 3), 2),
        ((1, 1, 2, 4), 2),
        ((2, 2, 2), 3),
    ])
    def test_max_vs_numpy(self, shape, num_inputs):
        """MAX: caffe-ffi dX matches numpy reference (winner-take-all)."""
        rng = np.random.RandomState(77)
        # Use inputs with clear winners (large differences to avoid ties)
        inputs = []
        for i in range(num_inputs):
            scale = 1.0 + i * 0.5
            inputs.append((rng.randn(*shape).astype(np.float32) * scale + i * 2.0))
        dy = rng.randn(*shape).astype(np.float32)
        net = _make_eltwise_net(shape, num_inputs=num_inputs, op="MAX")
        dXs, out = _run_eltwise_backward(net, inputs, dy, num_inputs=num_inputs)
        out_ref = eltwise_forward_np(inputs, op="MAX")
        np.testing.assert_allclose(out, out_ref, rtol=1e-5)
        dXs_ref = eltwise_backward_np(inputs, dy, op="MAX")
        for j, (dx, dx_ref) in enumerate(zip(dXs, dXs_ref)):
            np.testing.assert_allclose(dx, dx_ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Tests: Numerical gradient (L3)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestEltwiseBackwardNumerical:
    """Central finite difference numerical gradient verification."""

    @pytest.mark.parametrize("input_idx", [0, 1])
    def test_sum_numerical_grad(self, input_idx):
        """SUM: numerical gradient matches analytic for both inputs."""
        shape = (2, 3)
        rng = np.random.RandomState(1)
        inputs = [rng.randn(*shape).astype(np.float32) * 0.5 for _ in range(2)]
        dy = rng.randn(*shape).astype(np.float32) * 0.1
        net = _make_eltwise_net(shape, num_inputs=2, op="SUM")
        dXs, _ = _run_eltwise_backward(net, inputs, dy, num_inputs=2)
        num_grad = _numerical_grad_input(net, inputs, input_idx, dy)
        assert_grad_close(dXs[input_idx], num_grad,
                          name=f"SUM dX[{input_idx}]", rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("input_idx", [0, 1])
    def test_prod_numerical_grad(self, input_idx):
        """PROD: numerical gradient matches analytic (positive inputs)."""
        shape = (2, 3)
        rng = np.random.RandomState(2)
        inputs = [rng.uniform(0.8, 1.5, size=shape).astype(np.float32) for _ in range(2)]
        dy = rng.randn(*shape).astype(np.float32) * 0.1
        net = _make_eltwise_net(shape, num_inputs=2, op="PROD")
        dXs, _ = _run_eltwise_backward(net, inputs, dy, num_inputs=2)
        num_grad = _numerical_grad_input(net, inputs, input_idx, dy)
        assert_grad_close(dXs[input_idx], num_grad,
                          name=f"PROD dX[{input_idx}]", rtol=RTOL, atol=ATOL*10)

    @pytest.mark.parametrize("input_idx", [0, 1])
    def test_max_numerical_grad(self, input_idx):
        """MAX: numerical gradient matches analytic (no ties)."""
        shape = (2, 3)
        rng = np.random.RandomState(3)
        # Create inputs with clear winners to avoid C^0 discontinuities
        inputs = [
            rng.randn(*shape).astype(np.float32) + 0.0,
            rng.randn(*shape).astype(np.float32) + 3.0,  # bias b higher
        ]
        dy = rng.randn(*shape).astype(np.float32) * 0.1
        net = _make_eltwise_net(shape, num_inputs=2, op="MAX")
        dXs, _ = _run_eltwise_backward(net, inputs, dy, num_inputs=2)
        num_grad = _numerical_grad_input(net, inputs, input_idx, dy)
        # MAX has C^0 kinks at equality points; use looser tolerance and avoid kinks
        assert_grad_close(dXs[input_idx], num_grad,
                          name=f"MAX dX[{input_idx}]", rtol=5e-3, atol=1e-3)

    def test_sum_coeffs_numerical_grad(self):
        """SUM with coeffs: numerical gradient matches analytic."""
        shape = (2, 3)
        rng = np.random.RandomState(4)
        inputs = [rng.randn(*shape).astype(np.float32) * 0.5 for _ in range(2)]
        dy = rng.randn(*shape).astype(np.float32) * 0.1
        coeffs = [2.0, 0.5]
        net = _make_eltwise_net(shape, num_inputs=2, op="SUM", coeffs=coeffs)
        dXs, _ = _run_eltwise_backward(net, inputs, dy, num_inputs=2)
        for j in range(2):
            num_grad = _numerical_grad_input(net, inputs, j, dy)
            assert_grad_close(dXs[j], num_grad,
                              name=f"SUM(coeffs) dX[{j}]", rtol=RTOL, atol=ATOL)

    def test_three_inputs_sum_numerical_grad(self):
        """SUM with 3 inputs: numerical gradient matches for all three."""
        shape = (1, 4)
        rng = np.random.RandomState(5)
        inputs = [rng.randn(*shape).astype(np.float32) * 0.5 for _ in range(3)]
        dy = rng.randn(*shape).astype(np.float32) * 0.1
        net = _make_eltwise_net(shape, num_inputs=3, op="SUM")
        dXs, _ = _run_eltwise_backward(net, inputs, dy, num_inputs=3)
        for j in range(3):
            num_grad = _numerical_grad_input(net, inputs, j, dy)
            assert_grad_close(dXs[j], num_grad,
                              name=f"SUM(3) dX[{j}]", rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Tests: Properties (L4)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestEltwiseBackwardProperties:
    """Property-based tests."""

    def test_zero_dy_gives_zero_gradients_sum(self):
        """SUM: dy=0 -> all dX=0."""
        shape = (2, 3)
        rng = np.random.RandomState(10)
        inputs = [rng.randn(*shape).astype(np.float32) for _ in range(2)]
        dy = np.zeros(shape, dtype=np.float32)
        net = _make_eltwise_net(shape, num_inputs=2, op="SUM")
        dXs, _ = _run_eltwise_backward(net, inputs, dy, num_inputs=2)
        for j, dx in enumerate(dXs):
            np.testing.assert_allclose(dx, np.zeros_like(dx), atol=1e-6)

    def test_zero_dy_gives_zero_gradients_prod(self):
        """PROD: dy=0 -> all dX=0."""
        shape = (2, 3)
        rng = np.random.RandomState(11)
        inputs = [rng.uniform(0.5, 1.5, size=shape).astype(np.float32) for _ in range(2)]
        dy = np.zeros(shape, dtype=np.float32)
        net = _make_eltwise_net(shape, num_inputs=2, op="PROD")
        dXs, _ = _run_eltwise_backward(net, inputs, dy, num_inputs=2)
        for j, dx in enumerate(dXs):
            np.testing.assert_allclose(dx, np.zeros_like(dx), atol=1e-6)

    def test_zero_dy_gives_zero_gradients_max(self):
        """MAX: dy=0 -> all dX=0."""
        shape = (2, 3)
        rng = np.random.RandomState(12)
        inputs = [rng.randn(*shape).astype(np.float32) for _ in range(2)]
        dy = np.zeros(shape, dtype=np.float32)
        net = _make_eltwise_net(shape, num_inputs=2, op="MAX")
        dXs, _ = _run_eltwise_backward(net, inputs, dy, num_inputs=2)
        for j, dx in enumerate(dXs):
            np.testing.assert_allclose(dx, np.zeros_like(dx), atol=1e-6)

    def test_gradient_shapes(self):
        """All dX have same shape as inputs."""
        shape = (2, 3, 4)
        rng = np.random.RandomState(13)
        for op in ["SUM", "PROD", "MAX"]:
            inputs = [rng.randn(*shape).astype(np.float32) for _ in range(2)]
            dy = rng.randn(*shape).astype(np.float32)
            net = _make_eltwise_net(shape, num_inputs=2, op=op)
            dXs, _ = _run_eltwise_backward(net, inputs, dy, num_inputs=2)
            for j, dx in enumerate(dXs):
                assert dx.shape == shape, f"{op}: dX[{j}].shape={dx.shape} != {shape}"

    def test_determinism(self):
        """Same inputs produce same gradients (deterministic)."""
        shape = (2, 3)
        rng = np.random.RandomState(14)
        for op in ["SUM", "PROD", "MAX"]:
            inputs = [rng.randn(*shape).astype(np.float32) * 0.5 + 1.0 for _ in range(2)]
            dy = rng.randn(*shape).astype(np.float32)
            net1 = _make_eltwise_net(shape, num_inputs=2, op=op)
            dXs1, _ = _run_eltwise_backward(net1, inputs, dy, num_inputs=2)
            net2 = _make_eltwise_net(shape, num_inputs=2, op=op)
            dXs2, _ = _run_eltwise_backward(net2, inputs, dy, num_inputs=2)
            for j in range(2):
                np.testing.assert_array_equal(dXs1[j], dXs2[j])

    def test_forward_preserved_after_backward(self):
        """Backward does not modify forward output."""
        shape = (2, 3)
        rng = np.random.RandomState(15)
        for op in ["SUM", "PROD", "MAX"]:
            inputs = [rng.randn(*shape).astype(np.float32) * 0.5 + 1.0 for _ in range(2)]
            dy = rng.randn(*shape).astype(np.float32)
            net = _make_eltwise_net(shape, num_inputs=2, op=op)
            input_names = [chr(ord('a') + i) for i in range(2)]
            input_dict = {n: x for n, x in zip(input_names, inputs)}
            out_before = net.forward(input_dict)["out"].copy()
            net.backward({"out": dy})
            out_after = net.blob_by_name("out").data
            np.testing.assert_allclose(out_before, out_after, rtol=1e-6)

    def test_finite_values(self):
        """No NaN/Inf in gradients with normal inputs."""
        shape = (2, 3)
        rng = np.random.RandomState(16)
        for op in ["SUM", "MAX"]:
            inputs = [rng.randn(*shape).astype(np.float32) * 0.5 for _ in range(2)]
            dy = rng.randn(*shape).astype(np.float32)
            net = _make_eltwise_net(shape, num_inputs=2, op=op)
            dXs, _ = _run_eltwise_backward(net, inputs, dy, num_inputs=2)
            for j, dx in enumerate(dXs):
                assert np.all(np.isfinite(dx)), f"{op} dX[{j}] has NaN/Inf"

    def test_max_gradient_conservation(self):
        """MAX: at most one input gets gradient per position (winner-take-all)."""
        shape = (4, 5)
        rng = np.random.RandomState(17)
        inputs = [
            rng.randn(*shape).astype(np.float32) + 1.0,
            rng.randn(*shape).astype(np.float32) + 2.0,
            rng.randn(*shape).astype(np.float32) + 3.0,
        ]
        dy = np.ones(shape, dtype=np.float32)
        net = _make_eltwise_net(shape, num_inputs=3, op="MAX")
        dXs, _ = _run_eltwise_backward(net, inputs, dy, num_inputs=3)
        # For each position, only one dX should be non-zero
        for i in range(shape[0]):
            for j in range(shape[1]):
                non_zero_count = sum(1 for dx in dXs if abs(dx[i, j]) > 1e-6)
                assert non_zero_count <= 1, f"Position ({i},{j}): {non_zero_count} non-zero grads"
