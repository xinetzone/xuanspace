"""LRN layer Backward gradient tests (ACROSS_CHANNELS mode).

LRN forward: y[n,c,h,w] = x[n,c,h,w] / scale[n,c,h,w]^beta
where scale[n,c,h,w] = k + (alpha/size) * sum_{c' in window(c)} x[n,c',h,w]^2
and window(c) spans `size` adjacent channels (pre_pad = (size-1)//2).

Backward (derived via chain rule):
  dx[c] = dy[c] * scale[c]^{-beta}
        - cache_ratio * x[c] * sum_{c' in window(c)} dy[c'] * y[c'] / scale[c']
where cache_ratio = 2 * alpha * beta / size.

Covers:
  1. Analytical gradient (numpy reference vs caffe-ffi)
  2. Numerical gradient check (central finite differences)
  3. Zero dy -> zero gradients
  4. Shape/finite/determinism checks
  5. Forward-Backward consistency
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension
from ._grad_check_utils import numerical_gradient

EPS = 1e-3
RTOL = 2e-3
ATOL = 1e-4


# ---------------------------------------------------------------------------
# Numpy reference
# ---------------------------------------------------------------------------

def lrn_forward_np(x, size=5, alpha=1e-4, beta=0.75, k=1.0):
    """Numpy reference for LRN forward (ACROSS_CHANNELS)."""
    N, C, H, W = x.shape
    pre_pad = (size - 1) // 2
    padded = np.zeros((N, C + size - 1, H, W), dtype=np.float64)
    padded[:, pre_pad:pre_pad + C, :, :] = x.astype(np.float64)
    scale = np.ones_like(x, dtype=np.float64) * k
    for c in range(C):
        window = padded[:, c:c + size, :, :]
        scale[:, c, :, :] += (alpha / size) * np.sum(window ** 2, axis=1)
    return x.astype(np.float64) / (scale ** beta), scale


def lrn_backward_np(dy, x, size=5, alpha=1e-4, beta=0.75, k=1.0):
    """Numpy reference for LRN backward (ACROSS_CHANNELS)."""
    N, C, H, W = x.shape
    pre_pad = (size - 1) // 2
    y, scale = lrn_forward_np(x, size, alpha, beta, k)
    cache_ratio = 2.0 * alpha * beta / size

    # Direct term: dy * scale^{-beta}
    dx = dy.astype(np.float64) * (scale ** (-beta))

    # Accumulate term: -cache_ratio * x[c] * sum_{c' in window(c)} dy[c'] * y[c'] / scale[c']
    padded = np.zeros((N, C + size - 1, H, W), dtype=np.float64)
    padded[:, pre_pad:pre_pad + C, :, :] = (dy.astype(np.float64) * y / scale)
    for c in range(C):
        window = padded[:, c:c + size, :, :]
        dx[:, c, :, :] -= cache_ratio * x[:, c, :, :] * np.sum(window, axis=1)
    return dx.astype(np.float32)


# ---------------------------------------------------------------------------
# Prototxt builder
# ---------------------------------------------------------------------------

def _make_lrn_prototxt(batch=1, channels=6, h=4, w=4, size=5, alpha=1e-4, beta=0.75, k=1.0):
    return textwrap.dedent(f"""\
        name: "test_lrn_bw"
        layer {{
          name: "data"
          type: "Input"
          top: "data"
          input_param {{ shape {{ dim: {batch} dim: {channels} dim: {h} dim: {w} }} }}
        }}
        layer {{
          name: "lrn"
          type: "LRN"
          bottom: "data"
          top: "lrn_out"
          lrn_param {{
            local_size: {size}
            alpha: {alpha}
            beta: {beta}
            k: {k}
          }}
        }}
    """)


def _make_lrn_net(batch=1, channels=6, h=4, w=4, size=5, alpha=1e-4, beta=0.75, k=1.0):
    return Net(_make_lrn_prototxt(batch, channels, h, w, size, alpha, beta, k))


def _run_lrn_backward(net, x, dy):
    """Run forward then backward, return (dX, y)."""
    out = net.forward({"data": x.astype(np.float32)})
    net.backward({"lrn_out": dy.astype(np.float32)})
    dX = net.blob_by_name("data").diff
    return dX, out["lrn_out"]


# ---------------------------------------------------------------------------
# Tests: Analytical gradient vs numpy (L2)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestLRNBackwardNumpy:
    """Compare caffe-ffi backward against numpy reference."""

    @pytest.mark.parametrize("channels,size,alpha,beta,k", [
        (4, 3, 0.001, 0.5, 2.0),
        (6, 5, 1e-4, 0.75, 1.0),
        (8, 3, 5e-3, 0.5, 1.0),
        (5, 5, 1e-4, 0.75, 1.0),
    ])
    def test_lrn_vs_numpy(self, channels, size, alpha, beta, k):
        """LRN backward matches numpy reference for various params."""
        rng = np.random.RandomState(42)
        x = rng.randn(1, channels, 3, 3).astype(np.float32) * 0.3
        dy = rng.randn(1, channels, 3, 3).astype(np.float32) * 0.1
        net = _make_lrn_net(channels=channels, size=size, alpha=alpha, beta=beta, k=k)
        dX, y = _run_lrn_backward(net, x, dy)
        ref = lrn_backward_np(dy, x, size=size, alpha=alpha, beta=beta, k=k)
        # LRN involves pow/sum; use relaxed tolerance for float32
        np.testing.assert_allclose(dX, ref, rtol=RTOL, atol=ATOL)

    def test_lrn_multibatch(self):
        """LRN backward with batch > 1."""
        rng = np.random.RandomState(123)
        x = rng.randn(2, 6, 3, 3).astype(np.float32) * 0.3
        dy = rng.randn(2, 6, 3, 3).astype(np.float32) * 0.1
        net = _make_lrn_net(batch=2, channels=6, size=5, alpha=1e-4, beta=0.75, k=1.0)
        dX, _ = _run_lrn_backward(net, x, dy)
        ref = lrn_backward_np(dy, x, size=5, alpha=1e-4, beta=0.75, k=1.0)
        np.testing.assert_allclose(dX, ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Tests: Numerical gradient (L3)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestLRNBackwardNumerical:
    """Central finite difference numerical gradient checks."""

    @pytest.mark.parametrize("channels,size,alpha,beta,k", [
        (4, 3, 0.001, 0.5, 2.0),
        (6, 5, 1e-4, 0.75, 1.0),
    ])
    def test_numerical_grad(self, channels, size, alpha, beta, k):
        """Numerical gradient matches analytical dX."""
        rng = np.random.RandomState(789)
        x = rng.randn(1, channels, 3, 3).astype(np.float32) * 0.3
        dy = rng.randn(1, channels, 3, 3).astype(np.float32) * 0.1
        net = _make_lrn_net(channels=channels, size=size, alpha=alpha, beta=beta, k=k)
        dX, _ = _run_lrn_backward(net, x, dy)

        current = x.astype(np.float32).copy()

        def _forward():
            return net.forward({"data": current})["lrn_out"]

        def _get():
            return current.copy()

        def _set(arr):
            nonlocal current
            np.copyto(current, arr)

        num_dx = numerical_gradient(_forward, _get, _set, dy, h=EPS,
                                    name="input:data", verbose=False)
        np.testing.assert_allclose(dX, num_dx, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# Tests: Properties (L4)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestLRNBackwardProperties:
    """Property tests: zero gradients, shapes, finite, determinism."""

    def test_zero_dy_gives_zero_gradients(self):
        """dy=0 -> dX=0."""
        net = _make_lrn_net(channels=4, size=3)
        x = np.random.RandomState(202).randn(1, 4, 3, 3).astype(np.float32) * 0.3
        dy = np.zeros((1, 4, 3, 3), dtype=np.float32)
        dX, _ = _run_lrn_backward(net, x, dy)
        np.testing.assert_array_equal(dX, 0.0)

    def test_gradient_shape(self):
        """dX has same shape as input."""
        net = _make_lrn_net(channels=6, size=5)
        rng = np.random.RandomState(303)
        x = rng.randn(1, 6, 4, 4).astype(np.float32) * 0.3
        dy = rng.randn(1, 6, 4, 4).astype(np.float32)
        dX, _ = _run_lrn_backward(net, x, dy)
        assert dX.shape == x.shape

    def test_determinism(self):
        """Same inputs -> same gradients."""
        rng = np.random.RandomState(404)
        x = rng.randn(1, 6, 3, 3).astype(np.float32) * 0.3
        dy = rng.randn(1, 6, 3, 3).astype(np.float32)
        net1 = _make_lrn_net(channels=6, size=5)
        dX1, _ = _run_lrn_backward(net1, x, dy)
        net2 = _make_lrn_net(channels=6, size=5)
        dX2, _ = _run_lrn_backward(net2, x, dy)
        np.testing.assert_array_equal(dX1, dX2)

    def test_forward_preserved_after_backward(self):
        """Backward doesn't change forward output."""
        rng = np.random.RandomState(505)
        x = rng.randn(1, 6, 3, 3).astype(np.float32) * 0.3
        dy = rng.randn(1, 6, 3, 3).astype(np.float32)
        net = _make_lrn_net(channels=6, size=5)
        y1 = net.forward({"data": x})["lrn_out"].copy()
        net.backward({"lrn_out": dy})
        y2 = net.forward({"data": x})["lrn_out"]
        np.testing.assert_allclose(y1, y2, rtol=1e-6)

    def test_finite_values(self):
        """Gradients are finite (no NaN/Inf)."""
        rng = np.random.RandomState(606)
        x = rng.randn(1, 6, 3, 3).astype(np.float32) * 0.3
        dy = rng.randn(1, 6, 3, 3).astype(np.float32)
        net = _make_lrn_net(channels=6, size=5)
        dX, _ = _run_lrn_backward(net, x, dy)
        assert np.all(np.isfinite(dX))

    def test_forward_consistency(self):
        """Forward output matches numpy reference (prerequisite for backward)."""
        rng = np.random.RandomState(707)
        x = rng.randn(1, 6, 3, 3).astype(np.float32) * 0.3
        net = _make_lrn_net(channels=6, size=5)
        y = net.forward({"data": x})["lrn_out"]
        y_ref, _ = lrn_forward_np(x, size=5, alpha=1e-4, beta=0.75, k=1.0)
        np.testing.assert_allclose(y, y_ref.astype(np.float32), rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])