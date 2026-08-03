"""Self-test for _grad_check_utils.py using pure numpy functions (no C++ extension needed).

Verifies:
1. compare_gradients correctly identifies matching/non-matching gradients
2. numerical_gradient correctly computes central finite differences
3. assert_grad_close raises/doesn't raise as expected
4. assert_backward_matches_reference regression utility works (mock net)
5. Logging works as expected
"""
from __future__ import annotations

import sys
import logging
import numpy as np
from pathlib import Path

# Add tests/python to path
_test_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_test_dir))

# Configure logging to see GRAD logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

from _grad_check_utils import (
    compare_gradients,
    numerical_gradient,
    assert_grad_close,
    assert_backward_matches_reference,
)


def test_compare_gradients_matching():
    """Identical arrays should pass."""
    a = np.random.randn(10, 10).astype(np.float32)
    info = compare_gradients(a, a, name="identical", rtol=1e-5, atol=1e-6)
    assert info["passed"], f"Identical arrays should pass, got {info}"
    assert info["max_abs_err"] == 0.0
    print("  ✓ test_compare_gradients_matching passed")


def test_compare_gradients_noisy():
    """Small noise within tolerance should pass."""
    a = np.random.randn(5, 5).astype(np.float32)
    n = a + np.float32(1e-5)
    info = compare_gradients(a, n, name="small_noise", rtol=1e-3, atol=1e-4)
    assert info["passed"], f"Small noise within tol should pass, got {info}"
    print("  ✓ test_compare_gradients_noisy passed")


def test_compare_gradients_mismatch():
    """Large difference should fail."""
    a = np.ones((3, 3), dtype=np.float32)
    n = a + np.float32(1.0)
    info = compare_gradients(a, n, name="large_diff", rtol=1e-3, atol=1e-4, verbose=False)
    assert not info["passed"], "Large difference should fail"
    assert info["max_abs_err"] > 0.9
    print("  ✓ test_compare_gradients_mismatch passed")


def test_numerical_gradient_quadratic():
    """Test numerical gradient on f(x) = sum(x^2), df/dx = 2x."""
    x0 = np.array([1.0, 2.0, 3.0, 0.5, -1.0], dtype=np.float32)
    x_work = x0.copy()

    def forward():
        return (x_work ** 2).astype(np.float32)

    def get_param():
        return x_work.copy()

    def set_param(arr):
        np.copyto(x_work, arr)

    # dy is all ones because L = sum(1 * output) = sum(x^2)
    dy = np.ones_like(x0, dtype=np.float32)
    # Use h=1e-3 (standard for float32 numerical gradient)
    grad_numeric = numerical_gradient(forward, get_param, set_param, dy, h=1e-3, name="quadratic")
    grad_analytic = 2.0 * x0

    info = compare_gradients(grad_analytic, grad_numeric, name="quadratic_grad", rtol=5e-3, atol=5e-4)
    assert info["passed"], f"Quadratic gradient should match: {info}"
    print("  ✓ test_numerical_gradient_quadratic passed")


def test_numerical_gradient_matmul():
    """Test numerical gradient on matrix multiply Y = X @ W, dL/dW = X^T @ dy."""
    np.random.seed(42)
    X = np.random.randn(2, 3).astype(np.float32) * 0.5
    W0 = np.random.randn(3, 4).astype(np.float32) * 0.3
    W_work = W0.copy()

    def forward():
        return (X @ W_work).astype(np.float32)

    def get_param():
        return W_work.copy()

    def set_param(arr):
        np.copyto(W_work, arr)

    dy = np.random.randn(2, 4).astype(np.float32) * 0.2
    grad_numeric = numerical_gradient(forward, get_param, set_param, dy, h=1e-3, name="matmul_W", verbose=False)
    grad_analytic = X.T @ dy

    info = compare_gradients(grad_analytic, grad_numeric, name="matmul_dW", rtol=2e-2, atol=2e-3)
    assert info["passed"], f"Matmul dW should match: {info}"
    print("  ✓ test_numerical_gradient_matmul passed")


def test_assert_grad_close_passes():
    """assert_grad_close should not raise on matching arrays."""
    a = np.random.randn(4, 4).astype(np.float32)
    assert_grad_close(a, a, name="assert_pass", rtol=1e-5, atol=1e-6, verbose=False)
    print("  ✓ test_assert_grad_close_passes passed")


def test_assert_grad_close_raises():
    """assert_grad_close should raise AssertionError on mismatch."""
    a = np.ones((2, 2), dtype=np.float32)
    b = a + np.float32(10.0)
    try:
        assert_grad_close(a, b, name="assert_fail", rtol=1e-3, atol=1e-4, verbose=False)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        msg = str(e)
        assert "gradient check FAILED" in msg
        assert "max|a-n|" in msg
    print("  ✓ test_assert_grad_close_raises passed")


# ---------------------------------------------------------------------------
# Mock net for testing assert_backward_matches_reference without C++ extension
# ---------------------------------------------------------------------------

class _MockPoolingNet:
    """Minimal mock Net that implements 2x2 stride-2 MAX pooling in pure numpy.

    Simulates the caffe_ffi Net interface (forward, backward, blob_by_name)
    using the same mathematical rules as C++ MAX pooling:
      - Forward: 2x2 MAX pool with stride 2
      - Backward: Winner-Takes-All gradient routing
    """

    BLOB_DATA = {}
    BLOB_DIFF = {}
    CURRENT_INPUT = None

    class _Blob:
        def __init__(self):
            self.data = None
            self.diff = None

    def __init__(self, inject_error: float = 0.0):
        """Create mock net.

        Args:
            inject_error: If non-zero, add this error to dX at position (0,0,1,1)
                to simulate a C++ backward bug for regression testing.
        """
        self._blobs = {"data": self._Blob(), "pool": self._Blob()}
        self._inject_error = inject_error

    def forward(self, input_dict):
        x = input_dict["data"].astype(np.float32)
        self.CURRENT_INPUT = x.copy()
        N, C, H, W = x.shape
        # 2x2 s=2 MAX pool
        H_out, W_out = H // 2, W // 2
        y = np.zeros((N, C, H_out, W_out), dtype=np.float32)
        for n in range(N):
            for c in range(C):
                for ph in range(H_out):
                    for pw in range(W_out):
                        patch = x[n, c, ph*2:ph*2+2, pw*2:pw*2+2]
                        y[n, c, ph, pw] = patch.max()
        self._blobs["data"].data = x.copy()
        self._blobs["pool"].data = y
        return {"pool": y}

    def backward(self, dy_dict):
        dy = dy_dict["pool"].astype(np.float32)
        x = self.CURRENT_INPUT
        N, C, H, W = x.shape
        H_out, W_out = dy.shape[2], dy.shape[3]
        dX = np.zeros_like(x, dtype=np.float32)
        for n in range(N):
            for c in range(C):
                for ph in range(H_out):
                    for pw in range(W_out):
                        patch = x[n, c, ph*2:ph*2+2, pw*2:pw*2+2]
                        flat_idx = int(np.argmax(patch))
                        wh = ph*2 + flat_idx // 2
                        ww = pw*2 + flat_idx % 2
                        dX[n, c, wh, ww] += dy[n, c, ph, pw]
        # Inject deliberate error for negative testing
        if self._inject_error != 0.0:
            dX[0, 0, 1, 1] += np.float32(self._inject_error)
        self._blobs["data"].diff = dX
        self._blobs["pool"].diff = dy

    def blob_by_name(self, name):
        return self._blobs[name]


def _mock_maxpool_backward_ref(dy, x, kernel_size=2, stride=2, pool_type='MAX'):
    """Numpy reference for 2x2 s=2 MAX pooling backward (same logic as _MockPoolingNet)."""
    N, C, H, W = x.shape
    H_out, W_out = H // stride, W // stride
    dx = np.zeros_like(x, dtype=np.float64)
    for n in range(N):
        for c in range(C):
            for ph in range(H_out):
                for pw in range(W_out):
                    patch = x[n, c, ph*stride:ph*stride+kernel_size, pw*stride:pw*stride+kernel_size]
                    flat_idx = int(np.argmax(patch))
                    wh = ph*stride + flat_idx // kernel_size
                    ww = pw*stride + flat_idx % kernel_size
                    dx[n, c, wh, ww] += dy[n, c, ph, pw]
    return dx.astype(np.float32)


def test_assert_backward_matches_reference_passes():
    """assert_backward_matches_reference should pass when C++ matches reference."""
    rng = np.random.RandomState(42)
    x = rng.randn(1, 1, 4, 4).astype(np.float32) * 2.0  # scaled to avoid ties
    dy = rng.randn(1, 1, 2, 2).astype(np.float32) * 0.5

    net = _MockPoolingNet(inject_error=0.0)
    result = assert_backward_matches_reference(
        net, _mock_maxpool_backward_ref,
        input_name="data", output_name="pool",
        x=x, dy=dy,
        name="mock_maxpool",
        rtol=1e-5, atol=1e-6,
        skip_numerical=True,  # skip numerical since mock doesn't support perturbation
        verbose=True,
    )
    assert result["ref_passed"], "Matching net should pass reference check"
    assert result["numerical_passed"] is None  # skipped
    np.testing.assert_array_equal(result["y"].shape, (1, 1, 2, 2))
    assert result["dX_cpp"].shape == (1, 1, 4, 4)
    print("  ✓ test_assert_backward_matches_reference_passes passed")


def test_assert_backward_matches_reference_fails():
    """assert_backward_matches_reference should raise AssertionError on mismatch."""
    rng = np.random.RandomState(123)
    x = rng.randn(1, 1, 4, 4).astype(np.float32) * 2.0
    dy = rng.randn(1, 1, 2, 2).astype(np.float32) * 0.5

    # Inject a large error (1.0) at position (0,0,1,1)
    net = _MockPoolingNet(inject_error=1.0)
    try:
        assert_backward_matches_reference(
            net, _mock_maxpool_backward_ref,
            input_name="data", output_name="pool",
            x=x, dy=dy,
            name="mock_buggy",
            rtol=1e-5, atol=1e-6,
            skip_numerical=True,
            verbose=False,
        )
        assert False, "Should have raised AssertionError for mismatched backward"
    except AssertionError as e:
        msg = str(e)
        assert "BACKWARD REGRESSION FAILED" in msg
        assert "cpp vs ref" in msg.lower() or "cpp-ref" in msg
    print("  ✓ test_assert_backward_matches_reference_fails passed")


def test_assert_backward_matches_reference_skip_numerical():
    """skip_numerical=True should skip numerical gradient (fast CI mode)."""
    rng = np.random.RandomState(99)
    x = rng.randn(1, 2, 4, 4).astype(np.float32) * 2.0
    dy = rng.randn(1, 2, 2, 2).astype(np.float32) * 0.5

    net = _MockPoolingNet(inject_error=0.0)
    import time
    t0 = time.perf_counter()
    result = assert_backward_matches_reference(
        net, _mock_maxpool_backward_ref,
        input_name="data", output_name="pool",
        x=x, dy=dy,
        name="mock_skip_num",
        rtol=1e-5, atol=1e-6,
        skip_numerical=True,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    assert result["ref_passed"]
    assert result["numerical_passed"] is None
    assert elapsed < 1.0, f"skip_numerical should be fast, took {elapsed:.2f}s"
    print(f"  ✓ test_assert_backward_matches_reference_skip_numerical passed ({elapsed*1000:.1f}ms)")


if __name__ == "__main__":
    print("Running _grad_check_utils self-tests...\n")
    test_compare_gradients_matching()
    test_compare_gradients_noisy()
    test_compare_gradients_mismatch()
    test_numerical_gradient_quadratic()
    test_numerical_gradient_matmul()
    test_assert_grad_close_passes()
    test_assert_grad_close_raises()
    test_assert_backward_matches_reference_passes()
    test_assert_backward_matches_reference_fails()
    test_assert_backward_matches_reference_skip_numerical()
    print("\n✅ All 10 self-tests passed!")
