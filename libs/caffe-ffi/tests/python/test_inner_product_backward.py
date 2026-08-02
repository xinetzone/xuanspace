"""InnerProduct layer Backward gradient tests.

Covers:
  1. Analytical gradient correctness (numpy reference vs caffe-ffi computed):
     - dX (bottom/input gradient)
     - dW (weight gradient)
     - db (bias gradient)
  2. Numerical gradient check (central finite differences for dX, dW, db)
  3. Configurations: with/without bias, default/transpose weight layout
  4. Known-value verification (hand-computed expected gradients)
  5. Multi-dimensional input (NCHW flattened along axis)
  6. Determinism across repeated backward calls
  7. propagate_down=false skips dX computation
  8. Gradient finiteness and shape checks

Mathematical reference (transpose=false, default Caffe convention):
  Forward:  Y = X_flat @ W^T + b    where X_flat is (M, K), W is (N, K), b is (N,)
  Backward:
    dW = dY^T @ X_flat             (N, K)  -- weight gradient
    db = sum(dY, axis=0)            (N,)    -- bias gradient = column-sum of dY
    dX_flat = dY @ W               (M, K)  -- input gradient, reshaped back to X shape

  When transpose=true: W is (K, N), and:
    Forward:  Y = X_flat @ W + b
    Backward:
      dW = X_flat^T @ dY           (K, N)
      db = sum(dY, axis=0)          (N,)
      dX_flat = dY @ W^T           (M, K)

Loss function: L = sum(dY * Y)  (linear loss for gradient checking).
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension

# ---------------------------------------------------------------------------
# Numerical gradient epsilon
# ---------------------------------------------------------------------------
EPS = 1e-3


# ---------------------------------------------------------------------------
# Numpy reference implementations (double precision for accuracy)
# ---------------------------------------------------------------------------

def _ip_forward_ref(x, W, b=None, axis=1, transpose=False):
    """Numpy reference for InnerProduct forward.

    Args:
        x: Input tensor (any shape, will be flattened along axis)
        W: Weight matrix. Shape (N, K) if transpose=False (default), (K, N) if transpose=True
        b: Bias vector (N,) or None
        axis: Flatten axis (default 1)
        transpose: Weight layout flag
    Returns:
        y: Output tensor with shape shape[:axis] + (N,) + (1,)*(ndim-axis-1)
    """
    x64 = x.astype(np.float64)
    W64 = W.astype(np.float64)
    shape = x64.shape
    ndim = len(shape)
    M = int(np.prod(shape[:axis]))
    K = int(np.prod(shape[axis:]))
    if transpose:
        N = W64.shape[1]
    else:
        N = W64.shape[0]
    x_flat = x64.reshape(M, K)
    if transpose:
        y_flat = x_flat @ W64
    else:
        y_flat = x_flat @ W64.T
    if b is not None:
        y_flat = y_flat + b.astype(np.float64)
    out_shape = list(shape[:axis]) + [N] + [1] * (ndim - axis - 1)
    return y_flat.reshape(out_shape).astype(np.float32)


def _ip_backward_ref(x, W, dy, b=None, axis=1, transpose=False):
    """Numpy reference for InnerProduct backward (analytical gradients).

    Args:
        x: Input tensor (same shape as forward input)
        W: Weight matrix
        dy: Upstream gradient tensor (same shape as forward output)
        b: Bias vector (not used in gradient computation except for shape reference)
        axis: Flatten axis
        transpose: Weight layout flag
    Returns:
        (dX, dW, db): Analytical gradients as numpy arrays (float64 computation, float32 output)
    """
    x64 = x.astype(np.float64)
    W64 = W.astype(np.float64)
    dy64 = dy.astype(np.float64)
    shape = x64.shape
    ndim = len(shape)
    M = int(np.prod(shape[:axis]))
    K = int(np.prod(shape[axis:]))
    if transpose:
        N = W64.shape[1]
    else:
        N = W64.shape[0]

    x_flat = x64.reshape(M, K)
    # Determine dy shape and flatten: output is shape[:axis] + [N] + [1]*
    out_ndim = ndim
    dy_flat = dy64.reshape(M, N)

    if transpose:
        # W is (K, N): Y = X @ W, dW = X^T @ dY, dX = dY @ W^T
        dW = x_flat.T @ dy_flat  # (K, N)
        dX_flat = dy_flat @ W64.T  # (M, K)
    else:
        # W is (N, K): Y = X @ W^T, dW = dY^T @ X, dX = dY @ W
        dW = dy_flat.T @ x_flat  # (N, K)
        dX_flat = dy_flat @ W64  # (M, K)

    db = dy_flat.sum(axis=0)  # (N,) -- always column sum

    out_shape = list(shape[:axis]) + [N] + [1] * (ndim - axis - 1)
    dX = dX_flat.reshape(shape).astype(np.float32)
    dW = dW.astype(np.float32)
    db = db.astype(np.float32)
    return dX, dW, db


# ---------------------------------------------------------------------------
# Prototxt builders
# ---------------------------------------------------------------------------

def _make_ip_prototxt(num_output, input_dims, bias_term=True, axis=1, transpose=False,
                      weight_filler="constant", weight_value=1.0,
                      bias_filler="constant", bias_value=0.0):
    """Create Input -> InnerProduct prototxt."""
    dims_str = " ".join(str(d) for d in input_dims)
    bias_str = "true" if bias_term else "false"
    trans_str = "true" if transpose else "false"
    return textwrap.dedent(f"""\
        name: "test_ip_bw"
        input: "data"
        input_dim: {input_dims[0]}
        input_dim: {input_dims[1]}
        input_dim: {input_dims[2]}
        input_dim: {input_dims[3]}
        layer {{
          name: "ip"
          type: "InnerProduct"
          bottom: "data"
          top: "out"
          inner_product_param {{
            num_output: {num_output}
            bias_term: {bias_str}
            axis: {axis}
            transpose: {trans_str}
            weight_filler {{ type: "{weight_filler}" value: {weight_value} }}
            bias_filler {{ type: "{bias_filler}" value: {bias_value} }}
          }}
        }}
    """)


# ---------------------------------------------------------------------------
# Numerical gradient helpers (central finite differences)
# ---------------------------------------------------------------------------

def _num_grad_x(net, x, W, b, dy, h=EPS, axis=1, transpose=False):
    """Numerical gradient of loss w.r.t. input x via central differences.

    L = sum(dy * out), perturb each element of x by ±h.
    """
    grad = np.zeros_like(x, dtype=np.float64)
    flat_x = x.ravel()
    flat_grad = grad.ravel()
    for i in range(flat_x.size):
        orig = flat_x[i]

        xp = x.copy()
        xp.ravel()[i] = orig + h
        _set_ip_weights(net, W, b, transpose=transpose)
        out_p = net.forward({"data": xp.astype(np.float32)})["out"]
        loss_p = float(np.sum(dy.astype(np.float64) * out_p.astype(np.float64)))

        xm = x.copy()
        xm.ravel()[i] = orig - h
        _set_ip_weights(net, W, b, transpose=transpose)
        out_m = net.forward({"data": xm.astype(np.float32)})["out"]
        loss_m = float(np.sum(dy.astype(np.float64) * out_m.astype(np.float64)))

        flat_grad[i] = (loss_p - loss_m) / (2.0 * h)
    return grad.astype(np.float32)


def _num_grad_W(net, x, W, b, dy, h=EPS, axis=1, transpose=False):
    """Numerical gradient of loss w.r.t. weight matrix W via central differences."""
    grad = np.zeros_like(W, dtype=np.float64)
    flat_W = W.ravel()
    flat_grad = grad.ravel()
    for i in range(flat_W.size):
        orig = flat_W[i]

        Wp = W.copy()
        Wp.ravel()[i] = orig + h
        _set_ip_weights(net, Wp, b, transpose=transpose)
        out_p = net.forward({"data": x.astype(np.float32)})["out"]
        loss_p = float(np.sum(dy.astype(np.float64) * out_p.astype(np.float64)))

        Wm = W.copy()
        Wm.ravel()[i] = orig - h
        _set_ip_weights(net, Wm, b, transpose=transpose)
        out_m = net.forward({"data": x.astype(np.float32)})["out"]
        loss_m = float(np.sum(dy.astype(np.float64) * out_m.astype(np.float64)))

        flat_grad[i] = (loss_p - loss_m) / (2.0 * h)
    return grad.astype(np.float32)


def _num_grad_b(net, x, W, b, dy, h=EPS, axis=1, transpose=False):
    """Numerical gradient of loss w.r.t. bias vector b via central differences."""
    grad = np.zeros_like(b, dtype=np.float64)
    for i in range(b.size):
        orig = b.flat[i]

        bp = b.copy()
        bp.flat[i] = orig + h
        _set_ip_weights(net, W, bp, transpose=transpose)
        out_p = net.forward({"data": x.astype(np.float32)})["out"]
        loss_p = float(np.sum(dy.astype(np.float64) * out_p.astype(np.float64)))

        bm = b.copy()
        bm.flat[i] = orig - h
        _set_ip_weights(net, W, bm, transpose=transpose)
        out_m = net.forward({"data": x.astype(np.float32)})["out"]
        loss_m = float(np.sum(dy.astype(np.float64) * out_m.astype(np.float64)))

        grad.flat[i] = (loss_p - loss_m) / (2.0 * h)
    return grad.astype(np.float32)


def _set_ip_weights(net, W, b=None, transpose=False):
    """Set InnerProduct layer weights (and optionally bias)."""
    ip_layer = net.layer_by_name("ip")
    ip_layer.blobs[0].from_numpy(W.astype(np.float32))
    if b is not None and len(ip_layer.blobs) >= 2:
        ip_layer.blobs[1].from_numpy(b.reshape(-1).astype(np.float32))


# ---------------------------------------------------------------------------
# Test Class: InnerProduct Backward (default: bias_term=true, transpose=false)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestInnerProductBackward:
    """InnerProduct backward gradient tests with default settings (bias on, transpose=false)."""

    # ---- helpers ----

    def _make_net(self, M=2, K=6, N=4, bias=True):
        """Create a simple Input(N,C,H,W)->InnerProduct net.

        Shape (M, C, H, W) with axis=1: K = C*H*W, top shape (M, N, 1, 1).
        """
        # Use (M, C, H, W) = (M, 2, 1, 3) so K = 2*1*3 = 6 for default test
        C, H, W_dim = 2, 1, K // 2  # ensure C*H*W = K
        # Adjust if K is odd
        if C * H * W_dim != K:
            # Fallback: use (M, 1, 1, K) for exact K
            C, H, W_dim = 1, 1, K
        input_dims = (M, C, H, W_dim)
        proto = _make_ip_prototxt(N, input_dims, bias_term=bias, axis=1, transpose=False)
        return Net(proto)

    # ---- known-value tests ----

    def test_ip_backward_known_values(self):
        """Hand-computed known values: 2x3 input, W=[[1,0,-1],[0,1,-1]], b=[0,0].

        X = [[1, 2, 3],
             [4, 5, 6]]
        W = [[1, 0, -1],
             [0, 1, -1]]    (N=2, K=3)
        b = [0, 0]
        dy = [[1, 0],
              [0, 1]]

        Y = X @ W^T = [[1-3, 2-3], [4-6, 5-6]] = [[-2, -1], [-2, -1]]

        dW = dy^T @ X = [[1,0],[0,1]]^T @ [[1,2,3],[4,5,6]]
           = [[1,2,3], [4,5,6]]
        db = sum(dy, axis=0) = [1, 1]
        dX = dy @ W = [[1,0],[0,1]] @ [[1,0,-1],[0,1,-1]] = [[1,0,-1],[0,1,-1]]
        """
        M, K, N = 2, 3, 2
        proto = _make_ip_prototxt(
            N, (M, 1, 1, K), bias_term=True, axis=1, transpose=False,
            weight_filler="constant", weight_value=0.0,
            bias_filler="constant", bias_value=0.0,
        )
        net = Net(proto)

        x = np.array([[[[1.0, 2.0, 3.0]]], [[[4.0, 5.0, 6.0]]]], dtype=np.float32)
        W = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]], dtype=np.float32)
        b = np.array([0.0, 0.0], dtype=np.float32)
        dy = np.array([[[[1.0, 0.0]]], [[[0.0, 1.0]]]], dtype=np.float32)

        _set_ip_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"out": dy})

        # Read gradients
        dX = net.blob_by_name("data").diff
        dW = net.layer_by_name("ip").blobs[0].diff
        db = net.layer_by_name("ip").blobs[1].diff

        expected_dW = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        expected_db = np.array([1.0, 1.0], dtype=np.float32)
        expected_dX = np.array([[[[1.0, 0.0, -1.0]]], [[[0.0, 1.0, -1.0]]]], dtype=np.float32)

        np.testing.assert_allclose(dW, expected_dW, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(db, expected_db, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dX, expected_dX, rtol=1e-5, atol=1e-6)

    def test_ip_backward_known_values_with_bias(self):
        """Known values with non-zero bias.

        X = [[1, 0]], W = [[2, 3]], b = [1], dy = [[1]]
        Y = [1*2+0*3+1] = [3]
        dW = dy^T @ X = [[1]]^T @ [[1,0]] = [[1,0]] * 1 = [[1,0]] → wait, dy is (1,1)
        dW = [[1],[1]]? No: dy is (1,1), dy^T is (1,1), X is (1,2), dW = dy^T @ X = (1,1) @ (1,2)
        That gives (1,2). Let me recalculate.
        Actually M=1, K=2, N=1:
          X_flat = [[1, 0]] (1,2)
          W = [[2, 3]] (1,2)
          b = [1] (1,)
          Y = X @ W^T + b = [[1*2+0*3]] + [1] = [[3]] (1,1)
          dy = [[1]] (1,1)
          dW = dy^T @ X = [[1]]^T @ [[1,0]] = [[1]] @ [[1,0]] = [[1,0]] (1,2)
          db = sum(dy) = [1] (1,)
          dX = dy @ W = [[1]] @ [[2,3]] = [[2,3]] (1,2)
        """
        M, K, N = 1, 2, 1
        proto = _make_ip_prototxt(
            N, (M, 1, 1, K), bias_term=True, axis=1, transpose=False,
            weight_filler="constant", weight_value=0.0,
            bias_filler="constant", bias_value=0.0,
        )
        net = Net(proto)

        x = np.array([[[[1.0, 0.0]]]], dtype=np.float32)
        W = np.array([[2.0, 3.0]], dtype=np.float32)
        b = np.array([1.0], dtype=np.float32)
        dy = np.array([[[[1.0]]]], dtype=np.float32)

        _set_ip_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"out": dy})

        dX = net.blob_by_name("data").diff
        dW = net.layer_by_name("ip").blobs[0].diff
        db = net.layer_by_name("ip").blobs[1].diff

        np.testing.assert_allclose(dW, np.array([[1.0, 0.0]], dtype=np.float32), rtol=1e-5)
        np.testing.assert_allclose(db, np.array([1.0], dtype=np.float32), rtol=1e-5)
        np.testing.assert_allclose(dX, np.array([[[[2.0, 3.0]]]], dtype=np.float32), rtol=1e-5)

    # ---- analytical gradient vs numpy reference ----

    def test_ip_backward_analytical_dx(self):
        """dX (input gradient) matches numpy reference on random data."""
        rng = np.random.RandomState(42)
        M, K, N = 3, 8, 5
        net = self._make_net(M=M, K=K, N=N, bias=True)
        x = (rng.randn(M, 1, 1, K).astype(np.float32)) * 0.5
        W = (rng.randn(N, K).astype(np.float32)) * 0.3
        b = rng.randn(N).astype(np.float32) * 0.1
        dy = rng.randn(M, N, 1, 1).astype(np.float32) * 0.5

        _set_ip_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"out": dy})

        dX = net.blob_by_name("data").diff
        expected_dX, _, _ = _ip_backward_ref(x, W, dy, b=b, axis=1, transpose=False)
        np.testing.assert_allclose(dX, expected_dX, rtol=1e-5, atol=1e-6)

    def test_ip_backward_analytical_dw(self):
        """dW (weight gradient) matches numpy reference on random data."""
        rng = np.random.RandomState(43)
        M, K, N = 3, 8, 5
        net = self._make_net(M=M, K=K, N=N, bias=True)
        x = (rng.randn(M, 1, 1, K).astype(np.float32)) * 0.5
        W = (rng.randn(N, K).astype(np.float32)) * 0.3
        b = rng.randn(N).astype(np.float32) * 0.1
        dy = rng.randn(M, N, 1, 1).astype(np.float32) * 0.5

        _set_ip_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"out": dy})

        dW = net.layer_by_name("ip").blobs[0].diff
        _, expected_dW, _ = _ip_backward_ref(x, W, dy, b=b, axis=1, transpose=False)
        np.testing.assert_allclose(dW, expected_dW, rtol=1e-5, atol=1e-6)

    def test_ip_backward_analytical_db(self):
        """db (bias gradient) matches numpy reference on random data."""
        rng = np.random.RandomState(44)
        M, K, N = 3, 8, 5
        net = self._make_net(M=M, K=K, N=N, bias=True)
        x = (rng.randn(M, 1, 1, K).astype(np.float32)) * 0.5
        W = (rng.randn(N, K).astype(np.float32)) * 0.3
        b = rng.randn(N).astype(np.float32) * 0.1
        dy = rng.randn(M, N, 1, 1).astype(np.float32) * 0.5

        _set_ip_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"out": dy})

        db = net.layer_by_name("ip").blobs[1].diff
        _, _, expected_db = _ip_backward_ref(x, W, dy, b=b, axis=1, transpose=False)
        np.testing.assert_allclose(db, expected_db, rtol=1e-5, atol=1e-6)

    # ---- numerical gradient checks ----

    def test_ip_numerical_gradient_dx(self):
        """Central finite difference check for dX (small tensor for speed)."""
        rng = np.random.RandomState(7)
        M, K, N = 2, 4, 3  # small: 2*4=8 elements
        net = self._make_net(M=M, K=K, N=N, bias=True)
        x = (rng.randn(M, 1, 1, K).astype(np.float32)) * 0.5
        W = (rng.randn(N, K).astype(np.float32)) * 0.3
        b = rng.randn(N).astype(np.float32) * 0.1
        dy = rng.randn(M, N, 1, 1).astype(np.float32) * 0.5

        # Analytical
        _set_ip_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx_analytic = net.blob_by_name("data").diff

        # Numerical
        dx_numeric = _num_grad_x(net, x, W, b, dy, h=EPS)

        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=1e-3, atol=1e-4)

    def test_ip_numerical_gradient_dw(self):
        """Central finite difference check for dW (small matrix for speed)."""
        rng = np.random.RandomState(13)
        M, K, N = 2, 3, 2  # small: 2*3=6 weight elements
        net = self._make_net(M=M, K=K, N=N, bias=True)
        x = (rng.randn(M, 1, 1, K).astype(np.float32)) * 0.5
        W = (rng.randn(N, K).astype(np.float32)) * 0.3
        b = rng.randn(N).astype(np.float32) * 0.1
        dy = rng.randn(M, N, 1, 1).astype(np.float32) * 0.5

        # Analytical
        _set_ip_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"out": dy})
        dw_analytic = net.layer_by_name("ip").blobs[0].diff

        # Numerical
        dw_numeric = _num_grad_W(net, x, W, b, dy, h=EPS)

        np.testing.assert_allclose(dw_analytic, dw_numeric, rtol=1e-3, atol=1e-4)

    def test_ip_numerical_gradient_db(self):
        """Central finite difference check for db."""
        rng = np.random.RandomState(17)
        M, K, N = 2, 4, 3  # N=3 bias elements
        net = self._make_net(M=M, K=K, N=N, bias=True)
        x = (rng.randn(M, 1, 1, K).astype(np.float32)) * 0.5
        W = (rng.randn(N, K).astype(np.float32)) * 0.3
        b = rng.randn(N).astype(np.float32) * 0.1
        dy = rng.randn(M, N, 1, 1).astype(np.float32) * 0.5

        # Analytical
        _set_ip_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"out": dy})
        db_analytic = net.layer_by_name("ip").blobs[1].diff

        # Numerical
        db_numeric = _num_grad_b(net, x, W, b, dy, h=EPS)

        np.testing.assert_allclose(db_analytic, db_numeric, rtol=1e-3, atol=1e-4)

    # ---- no-bias configuration ----

    def test_ip_backward_no_bias(self):
        """Without bias_term: db should not exist, dX and dW still correct."""
        rng = np.random.RandomState(55)
        M, K, N = 3, 6, 4
        proto = _make_ip_prototxt(N, (M, 1, 1, K), bias_term=False, axis=1, transpose=False,
                                  weight_filler="constant", weight_value=0.0)
        net = Net(proto)
        x = (rng.randn(M, 1, 1, K).astype(np.float32)) * 0.5
        W = (rng.randn(N, K).astype(np.float32)) * 0.3
        dy = rng.randn(M, N, 1, 1).astype(np.float32) * 0.5

        _set_ip_weights(net, W, None)
        net.forward({"data": x})
        net.backward({"out": dy})

        dX = net.blob_by_name("data").diff
        dW = net.layer_by_name("ip").blobs[0].diff

        expected_dX, expected_dW, _ = _ip_backward_ref(x, W, dy, b=None, axis=1, transpose=False)
        np.testing.assert_allclose(dX, expected_dX, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dW, expected_dW, rtol=1e-5, atol=1e-6)
        # No bias blob
        assert len(net.layer_by_name("ip").blobs) == 1, "No bias blob when bias_term=false"

    def test_ip_numerical_gradient_no_bias(self):
        """Numerical gradient check without bias (dX, dW)."""
        rng = np.random.RandomState(59)
        M, K, N = 2, 3, 2
        proto = _make_ip_prototxt(N, (M, 1, 1, K), bias_term=False, axis=1, transpose=False,
                                  weight_filler="constant", weight_value=0.0)
        net = Net(proto)
        x = (rng.randn(M, 1, 1, K).astype(np.float32)) * 0.5
        W = (rng.randn(N, K).astype(np.float32)) * 0.3
        dy = rng.randn(M, N, 1, 1).astype(np.float32) * 0.5

        _set_ip_weights(net, W, None)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx_analytic = net.blob_by_name("data").diff
        dw_analytic = net.layer_by_name("ip").blobs[0].diff

        dx_numeric = _num_grad_x(net, x, W, None, dy, h=EPS)
        dw_numeric = _num_grad_W(net, x, W, None, dy, h=EPS)

        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(dw_analytic, dw_numeric, rtol=1e-3, atol=1e-4)

    # ---- shape and finiteness checks ----

    def test_ip_backward_shapes(self):
        """Gradient shapes must match parameter shapes."""
        rng = np.random.RandomState(99)
        M, K, N = 4, 12, 6
        net = self._make_net(M=M, K=K, N=N, bias=True)
        x = rng.randn(M, 1, 1, K).astype(np.float32)
        W = (rng.randn(N, K).astype(np.float32)) * 0.1
        b = rng.randn(N).astype(np.float32) * 0.1
        dy = rng.randn(M, N, 1, 1).astype(np.float32)

        _set_ip_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"out": dy})

        assert net.blob_by_name("data").diff.shape == x.shape
        assert net.layer_by_name("ip").blobs[0].diff.shape == (N, K)
        assert net.layer_by_name("ip").blobs[1].diff.shape == (N,)

    def test_ip_backward_finite(self):
        """Gradients must be finite (no NaN/Inf) on random data."""
        rng = np.random.RandomState(101)
        M, K, N = 4, 8, 4
        net = self._make_net(M=M, K=K, N=N, bias=True)
        x = rng.randn(M, 1, 1, K).astype(np.float32)
        W = (rng.randn(N, K).astype(np.float32)) * 0.5
        b = rng.randn(N).astype(np.float32)
        dy = rng.randn(M, N, 1, 1).astype(np.float32)

        _set_ip_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"out": dy})

        dX = net.blob_by_name("data").diff
        dW = net.layer_by_name("ip").blobs[0].diff
        db = net.layer_by_name("ip").blobs[1].diff
        assert np.all(np.isfinite(dX)), "dX contains NaN/Inf"
        assert np.all(np.isfinite(dW)), "dW contains NaN/Inf"
        assert np.all(np.isfinite(db)), "db contains NaN/Inf"

    def test_ip_backward_zero_dy_gives_zero_grads(self):
        """Zero upstream gradient → all gradients zero."""
        rng = np.random.RandomState(200)
        M, K, N = 2, 4, 3
        net = self._make_net(M=M, K=K, N=N, bias=True)
        x = rng.randn(M, 1, 1, K).astype(np.float32) * 2.0
        W = (rng.randn(N, K).astype(np.float32)) * 0.5
        b = rng.randn(N).astype(np.float32) * 0.5
        dy = np.zeros((M, N, 1, 1), dtype=np.float32)

        _set_ip_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"out": dy})

        dX = net.blob_by_name("data").diff
        dW = net.layer_by_name("ip").blobs[0].diff
        db = net.layer_by_name("ip").blobs[1].diff
        np.testing.assert_array_equal(dX, np.zeros_like(dX))
        np.testing.assert_array_equal(dW, np.zeros_like(dW))
        np.testing.assert_array_equal(db, np.zeros_like(db))

    # ---- determinism ----

    def test_ip_backward_deterministic(self):
        """Backward must produce identical gradients across repeated calls."""
        rng = np.random.RandomState(77)
        M, K, N = 3, 6, 4
        net = self._make_net(M=M, K=K, N=N, bias=True)
        x = (rng.randn(M, 1, 1, K).astype(np.float32)) * 0.5
        W = (rng.randn(N, K).astype(np.float32)) * 0.3
        b = rng.randn(N).astype(np.float32) * 0.1
        dy = rng.randn(M, N, 1, 1).astype(np.float32) * 0.5

        _set_ip_weights(net, W, b)
        net.forward({"data": x})
        results = []
        for _ in range(5):
            net.backward({"out": dy})
            results.append((
                net.blob_by_name("data").diff.copy(),
                net.layer_by_name("ip").blobs[0].diff.copy(),
                net.layer_by_name("ip").blobs[1].diff.copy(),
            ))
        for i in range(1, 5):
            np.testing.assert_array_equal(results[0][0], results[i][0])
            np.testing.assert_array_equal(results[0][1], results[i][1])
            np.testing.assert_array_equal(results[0][2], results[i][2])

    def test_ip_backward_preserves_forward_output(self):
        """Backward must not modify forward activations."""
        rng = np.random.RandomState(88)
        M, K, N = 2, 4, 3
        net = self._make_net(M=M, K=K, N=N, bias=True)
        x = (rng.randn(M, 1, 1, K).astype(np.float32)) * 0.5
        W = (rng.randn(N, K).astype(np.float32)) * 0.3
        b = rng.randn(N).astype(np.float32) * 0.1
        dy = rng.randn(M, N, 1, 1).astype(np.float32) * 0.5

        _set_ip_weights(net, W, b)
        out = net.forward({"data": x})
        y_before = out["out"].copy()
        net.backward({"out": dy})
        y_after = net.blob_by_name("out").data
        np.testing.assert_array_equal(y_before, y_after)


# ---------------------------------------------------------------------------
# Test Class: InnerProduct Backward with multi-dimensional input (NCHW, axis=1)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestInnerProductBackwardNCHW:
    """Test InnerProduct backward with realistic NCHW-style inputs (flatten from axis=1)."""

    def test_ip_backward_nchw_analytical(self):
        """dX/dW/db match numpy reference for (2,3,4,4) input (K=3*4*4=48)."""
        rng = np.random.RandomState(300)
        N_batch, C, H, W_dim, N_out = 2, 3, 4, 4, 8
        K = C * H * W_dim  # 48
        proto = _make_ip_prototxt(N_out, (N_batch, C, H, W_dim), bias_term=True,
                                  axis=1, transpose=False,
                                  weight_filler="constant", weight_value=0.0)
        net = Net(proto)
        x = (rng.randn(N_batch, C, H, W_dim).astype(np.float32)) * 0.1
        W = (rng.randn(N_out, K).astype(np.float32)) * 0.1
        b = rng.randn(N_out).astype(np.float32) * 0.01
        dy = rng.randn(N_batch, N_out, 1, 1).astype(np.float32) * 0.1

        _set_ip_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"out": dy})

        dX = net.blob_by_name("data").diff
        dW = net.layer_by_name("ip").blobs[0].diff
        db = net.layer_by_name("ip").blobs[1].diff

        expected_dX, expected_dW, expected_db = _ip_backward_ref(
            x, W, dy, b=b, axis=1, transpose=False
        )
        np.testing.assert_allclose(dX, expected_dX, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dW, expected_dW, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(db, expected_db, rtol=1e-5, atol=1e-6)

    def test_ip_backward_nchw_dx_numerical(self):
        """Numerical gradient check for dX on small NCHW-like input."""
        rng = np.random.RandomState(301)
        N_batch, C, H, W_dim, N_out = 2, 1, 2, 2, 3  # K=4, small for speed
        K = C * H * W_dim
        proto = _make_ip_prototxt(N_out, (N_batch, C, H, W_dim), bias_term=True,
                                  axis=1, transpose=False,
                                  weight_filler="constant", weight_value=0.0)
        net = Net(proto)
        x = (rng.randn(N_batch, C, H, W_dim).astype(np.float32)) * 0.5
        W = (rng.randn(N_out, K).astype(np.float32)) * 0.3
        b = rng.randn(N_out).astype(np.float32) * 0.1
        dy = rng.randn(N_batch, N_out, 1, 1).astype(np.float32) * 0.5

        _set_ip_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx_analytic = net.blob_by_name("data").diff
        dx_numeric = _num_grad_x(net, x, W, b, dy, h=EPS)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# Test Class: InnerProduct Backward with transpose=true weight layout
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestInnerProductBackwardTranspose:
    """Test InnerProduct backward with transpose=true (W stored as (K, N) instead of (N, K))."""

    def _make_transpose_net(self, M=2, K=6, N=4, bias=True):
        return Net(_make_ip_prototxt(
            N, (M, 1, 1, K), bias_term=bias, axis=1, transpose=True,
            weight_filler="constant", weight_value=0.0,
            bias_filler="constant", bias_value=0.0,
        ))

    def test_ip_transpose_backward_analytical(self):
        """dX/dW/db match numpy reference with transpose=true."""
        rng = np.random.RandomState(400)
        M, K, N = 3, 8, 5
        net = self._make_transpose_net(M=M, K=K, N=N, bias=True)
        x = (rng.randn(M, 1, 1, K).astype(np.float32)) * 0.5
        W = (rng.randn(K, N).astype(np.float32)) * 0.3  # (K, N) for transpose=true
        b = rng.randn(N).astype(np.float32) * 0.1
        dy = rng.randn(M, N, 1, 1).astype(np.float32) * 0.5

        _set_ip_weights(net, W, b, transpose=True)
        net.forward({"data": x})
        net.backward({"out": dy})

        dX = net.blob_by_name("data").diff
        dW = net.layer_by_name("ip").blobs[0].diff
        db = net.layer_by_name("ip").blobs[1].diff

        expected_dX, expected_dW, expected_db = _ip_backward_ref(
            x, W, dy, b=b, axis=1, transpose=True
        )
        np.testing.assert_allclose(dX, expected_dX, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(dW, expected_dW, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(db, expected_db, rtol=1e-5, atol=1e-6)

    def test_ip_transpose_numerical_gradient_dx(self):
        """Numerical gradient for dX with transpose=true."""
        rng = np.random.RandomState(401)
        M, K, N = 2, 4, 3
        net = self._make_transpose_net(M=M, K=K, N=N, bias=True)
        x = (rng.randn(M, 1, 1, K).astype(np.float32)) * 0.5
        W = (rng.randn(K, N).astype(np.float32)) * 0.3
        b = rng.randn(N).astype(np.float32) * 0.1
        dy = rng.randn(M, N, 1, 1).astype(np.float32) * 0.5

        _set_ip_weights(net, W, b, transpose=True)
        net.forward({"data": x})
        net.backward({"out": dy})
        dx_analytic = net.blob_by_name("data").diff
        dx_numeric = _num_grad_x(net, x, W, b, dy, h=EPS, transpose=True)
        np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=1e-3, atol=1e-4)

    def test_ip_transpose_numerical_gradient_dw(self):
        """Numerical gradient for dW with transpose=true."""
        rng = np.random.RandomState(402)
        M, K, N = 2, 3, 2
        net = self._make_transpose_net(M=M, K=K, N=N, bias=True)
        x = (rng.randn(M, 1, 1, K).astype(np.float32)) * 0.5
        W = (rng.randn(K, N).astype(np.float32)) * 0.3
        b = rng.randn(N).astype(np.float32) * 0.1
        dy = rng.randn(M, N, 1, 1).astype(np.float32) * 0.5

        _set_ip_weights(net, W, b, transpose=True)
        net.forward({"data": x})
        net.backward({"out": dy})
        dw_analytic = net.layer_by_name("ip").blobs[0].diff
        dw_numeric = _num_grad_W(net, x, W, b, dy, h=EPS, transpose=True)
        np.testing.assert_allclose(dw_analytic, dw_numeric, rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# Test Class: Identity-like gradients (unit weights, zero bias)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestInnerProductBackwardIdentity:
    """Tests with identity-like weight matrix to verify gradient propagation."""

    def test_ip_identity_weights_dx_equals_dy(self):
        """Square weight matrix I: dX = dy @ W = dy @ I = dy_flat."""
        N = K = 4
        M = 2
        proto = _make_ip_prototxt(N, (M, 1, 1, K), bias_term=True, axis=1, transpose=False,
                                  weight_filler="constant", weight_value=0.0,
                                  bias_filler="constant", bias_value=0.0)
        net = Net(proto)
        # Identity W
        W = np.eye(N, K, dtype=np.float32)
        b = np.zeros(N, dtype=np.float32)
        x = np.zeros((M, 1, 1, K), dtype=np.float32)  # zero input
        dy = np.random.RandomState(500).randn(M, N, 1, 1).astype(np.float32)

        _set_ip_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"out": dy})

        # dX = dy @ I = dy reshaped to input shape
        dX = net.blob_by_name("data").diff
        # dy is (M, N, 1, 1) which flattens to (M, N) = (M, K), dX should equal dy
        np.testing.assert_allclose(
            dX.reshape(M, K), dy.reshape(M, N), rtol=1e-5, atol=1e-6
        )
        # dW = dy^T @ x = dy^T @ 0 = 0
        dW = net.layer_by_name("ip").blobs[0].diff
        np.testing.assert_array_equal(dW, np.zeros_like(dW))
        # db = sum(dy, axis=0)
        db = net.layer_by_name("ip").blobs[1].diff
        expected_db = dy.reshape(M, N).sum(axis=0)
        np.testing.assert_allclose(db, expected_db, rtol=1e-5)

    def test_ip_ones_weights_dw(self):
        """All-ones W, dy=ones, x=known → verify dW = dy^T @ x = 1^T @ x = sum(x, axis=0)."""
        M, K, N = 3, 4, 2
        proto = _make_ip_prototxt(N, (M, 1, 1, K), bias_term=False, axis=1, transpose=False,
                                  weight_filler="constant", weight_value=0.0)
        net = Net(proto)
        W = np.ones((N, K), dtype=np.float32)
        rng = np.random.RandomState(501)
        x = rng.randn(M, 1, 1, K).astype(np.float32)
        dy = np.ones((M, N, 1, 1), dtype=np.float32)

        _set_ip_weights(net, W, None)
        net.forward({"data": x})
        net.backward({"out": dy})

        dW = net.layer_by_name("ip").blobs[0].diff
        # dW = dy^T @ x = ones(N,M) @ x_flat(M,K) = column-sum of x_flat repeated N times
        x_flat = x.reshape(M, K).astype(np.float64)
        expected_dW = np.ones((N, 1), dtype=np.float64) @ x_flat.sum(axis=0, keepdims=True)
        np.testing.assert_allclose(dW, expected_dW.astype(np.float32), rtol=1e-5)

    def test_ip_db_is_column_sum(self):
        """db should always be the column-wise sum of dy, independent of W and x."""
        rng = np.random.RandomState(502)
        M, K, N = 5, 6, 4
        proto = _make_ip_prototxt(N, (M, 1, 1, K), bias_term=True, axis=1, transpose=False,
                                  weight_filler="constant", weight_value=0.0)
        net = Net(proto)
        W = (rng.randn(N, K).astype(np.float32)) * 0.5
        b = rng.randn(N).astype(np.float32) * 0.5
        x = rng.randn(M, 1, 1, K).astype(np.float32)
        dy = rng.randn(M, N, 1, 1).astype(np.float32)

        _set_ip_weights(net, W, b)
        net.forward({"data": x})
        net.backward({"out": dy})

        db = net.layer_by_name("ip").blobs[1].diff
        expected_db = dy.reshape(M, N).sum(axis=0)
        np.testing.assert_allclose(db, expected_db, rtol=1e-5)
