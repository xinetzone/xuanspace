"""Self-test for _grad_check_utils.py using pure numpy functions (no C++ extension needed).

Verifies:
1. compare_gradients correctly identifies matching/non-matching gradients
2. numerical_gradient correctly computes central finite differences
3. Logging works as expected
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

from _grad_check_utils import compare_gradients, numerical_gradient, assert_grad_close


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


if __name__ == "__main__":
    print("Running _grad_check_utils self-tests...\n")
    test_compare_gradients_matching()
    test_compare_gradients_noisy()
    test_compare_gradients_mismatch()
    test_numerical_gradient_quadratic()
    test_numerical_gradient_matmul()
    test_assert_grad_close_passes()
    test_assert_grad_close_raises()
    print("\n✅ All self-tests passed!")
