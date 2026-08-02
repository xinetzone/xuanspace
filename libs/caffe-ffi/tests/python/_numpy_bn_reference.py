"""Numpy reference implementation of caffe-ffi BatchNorm layer (inference mode).

This script provides:
1. Exact numpy replica of the C++ Forward_cpu computation
2. Analytical Backward gradient computation
3. Central finite-difference numerical gradient verification
4. Self-tests to validate the backward formula dX = dy / sqrt(var*sf + eps)

C++ Forward formula (from batch_norm_layer.cpp):
    sf = 1/blobs[2][0]   if blobs[2][0] != 0 else 1.0
    y[n,c,h,w] = (x[n,c,h,w] - mean[c]*sf) / sqrt(max(var[c]*sf, 0) + eps)

C++ channel indexing (NCHW layout):
    spatial_dim = H * W
    c = (flat_index / spatial_dim) % C

Analytical Backward (inference mode, mu/sigma^2 are constants):
    dX = dy * inv_std[c]
    where inv_std[c] = 1 / sqrt(max(var[c]*sf, 0) + eps)

NOTE: Caffe's BatchNorm layer stores running statistics in blobs_[0]/[1]/[2]:
    - blobs_[0] = sum of x (per channel), NOT the mean directly
    - blobs_[1] = sum of (x-mu)^2 (per channel), NOT variance directly
    - blobs_[2][0] = count (scalar), scale_factor = 1/count
    When blobs_[2] = 1 (default initialization), mean = blobs_[0], var = blobs_[1].
    The Scale layer (separate) provides learnable gamma/beta.

Usage:
    python _numpy_bn_reference.py          # run self-tests
    from _numpy_bn_reference import bn_forward, bn_backward  # import
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Core implementations (float64 for numerical accuracy)
# ---------------------------------------------------------------------------

def bn_forward(
    x: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    count: float = 1.0,
    eps: float = 1e-5,
    axis: int = 1,
) -> np.ndarray:
    """Numpy reference for BatchNorm forward (inference/global-stats mode).

    Exactly replicates C++ BatchNormLayer::Forward_cpu logic.

    Args:
        x: Input tensor, shape (N, C, ...) or (N, C) or 1D
        mean: Per-channel mean (sum), shape (C,). Blob[0] from Caffe.
        variance: Per-channel variance (sum_sq), shape (C,). Blob[1] from Caffe.
        count: Blob[2][0] scalar. If count=0, treated as 1.0 scale factor.
        eps: Epsilon for numerical stability (default 1e-5, matches Caffe default).
        axis: Channel axis (default 1 for NCHW).

    Returns:
        y: Normalized output, same shape as x.
    """
    x64 = x.astype(np.float64)
    mean64 = mean.astype(np.float64)
    var64 = variance.astype(np.float64)

    # Compute scale_factor exactly as C++ does
    scale_factor = 0.0 if count == 0.0 else 1.0 / count
    sf = 1.0 if scale_factor == 0.0 else scale_factor

    # Effective mean and variance
    eff_mean = mean64 * sf          # (C,)
    eff_var = np.maximum(var64 * sf, 0.0)  # (C,), clamped to >= 0
    inv_std = 1.0 / np.sqrt(eff_var + eps)  # (C,)

    # Reshape for broadcasting along channel axis
    ndim = x64.ndim
    shape = [1] * ndim
    C = mean64.shape[0]
    # Determine channel axis and reshape
    if ndim == 1:
        # 1D: treat entire tensor as one channel
        # But C++ sets channels_ = 1 when num_axes == 1
        eff_mean_bc = eff_mean.reshape(1)
        inv_std_bc = inv_std.reshape(1)
    else:
        shape[axis] = C
        eff_mean_bc = eff_mean.reshape(shape)
        inv_std_bc = inv_std.reshape(shape)

    y = (x64 - eff_mean_bc) * inv_std_bc
    return y.astype(np.float32)


def bn_backward(
    dy: np.ndarray,
    variance: np.ndarray,
    count: float = 1.0,
    eps: float = 1e-5,
    axis: int = 1,
) -> np.ndarray:
    """Numpy reference for BatchNorm backward (inference mode).

    Analytical gradient: dX = dy * inv_std[c]
    where inv_std[c] = 1 / sqrt(max(var[c]*sf, 0) + eps)

    In inference mode, mean and variance are constants (running statistics),
    so there is no gradient flow through them. No blob gradients needed
    (running stats are not learnable params; learnable gamma/beta are in
    the separate Scale layer).

    Args:
        dy: Upstream gradient, same shape as forward output.
        variance: Per-channel variance (blob[1]), shape (C,).
        count: Blob[2][0] scalar.
        eps: Epsilon.
        axis: Channel axis.

    Returns:
        dx: Input gradient, same shape as dy.
    """
    dy64 = dy.astype(np.float64)
    var64 = variance.astype(np.float64)

    scale_factor = 0.0 if count == 0.0 else 1.0 / count
    sf = 1.0 if scale_factor == 0.0 else scale_factor

    eff_var = np.maximum(var64 * sf, 0.0)
    inv_std = 1.0 / np.sqrt(eff_var + eps)  # (C,)

    ndim = dy64.ndim
    C = var64.shape[0]
    shape = [1] * ndim
    if ndim == 1:
        inv_std_bc = inv_std.reshape(1)
    else:
        shape[axis] = C
        inv_std_bc = inv_std.reshape(shape)

    dx = dy64 * inv_std_bc
    return dx.astype(np.float32)


def bn_get_inv_std(
    variance: np.ndarray,
    count: float = 1.0,
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute per-channel inverse standard deviation used by both fwd and bwd.

    Returns inv_std of shape (C,), float32.
    """
    var64 = variance.astype(np.float64)
    scale_factor = 0.0 if count == 0.0 else 1.0 / count
    sf = 1.0 if scale_factor == 0.0 else scale_factor
    eff_var = np.maximum(var64 * sf, 0.0)
    return (1.0 / np.sqrt(eff_var + eps)).astype(np.float32)


# ---------------------------------------------------------------------------
# Numerical gradient check (central finite differences)
# ---------------------------------------------------------------------------

def numerical_grad_x(x, mean, variance, dy, count=1.0, eps_bn=1e-5, h=1e-3, axis=1):
    """Compute numerical gradient dL/dx via central finite differences.

    L = sum(dy * y_hat), perturb each element of x by +/-h.
    """
    grad = np.zeros_like(x, dtype=np.float64)
    flat_x = x.ravel()
    flat_grad = grad.ravel()
    for i in range(flat_x.size):
        orig = flat_x[i]

        xp = x.copy()
        xp.ravel()[i] = orig + h
        yp = bn_forward(xp, mean, variance, count=count, eps=eps_bn, axis=axis)
        loss_p = float(np.sum(dy.astype(np.float64) * yp.astype(np.float64)))

        xm = x.copy()
        xm.ravel()[i] = orig - h
        ym = bn_forward(xm, mean, variance, count=count, eps=eps_bn, axis=axis)
        loss_m = float(np.sum(dy.astype(np.float64) * ym.astype(np.float64)))

        flat_grad[i] = (loss_p - loss_m) / (2.0 * h)
    return grad.astype(np.float32)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _self_test():
    """Run self-tests to verify forward and backward correctness."""
    print("=== numpy BatchNorm forward/backward self-test ===\n")
    np.random.seed(42)
    passed = 0
    total = 0

    # [1] Known value: zero mean, unit variance, count=1 → y = x / sqrt(1+eps)
    total += 1
    print("[1] Known value: zero mean, unit var, count=1 ...", end=" ")
    N, C, H, W = 1, 2, 1, 3
    x = np.array([[[[1.0, 2.0, 3.0]], [[0.0, -1.0, 1.0]]]], dtype=np.float32)
    mean = np.zeros(C, dtype=np.float32)
    var = np.ones(C, dtype=np.float32)
    count = 1.0
    eps = 1e-5
    y = bn_forward(x, mean, var, count=count, eps=eps)
    expected = x / np.sqrt(1.0 + eps)
    np.testing.assert_allclose(y, expected.astype(np.float32), rtol=1e-6)
    print("OK")
    passed += 1

    # [2] Known value: non-zero mean and variance
    total += 1
    print("[2] Known value: mean=2, var=4, count=1 ...", end=" ")
    x2 = np.array([[[[4.0]]]], dtype=np.float32)  # N=1,C=1,H=1,W=1
    mean2 = np.array([2.0], dtype=np.float32)
    var2 = np.array([4.0], dtype=np.float32)
    y2 = bn_forward(x2, mean2, var2, count=1.0, eps=0.0)  # eps=0 for clean value
    # y = (4 - 2) / sqrt(4) = 2/2 = 1.0
    np.testing.assert_allclose(y2, np.array([[[[1.0]]]], dtype=np.float32), rtol=1e-6)
    print("OK")
    passed += 1

    # [3] Backward known value: dy=ones, var=4 → dx = 1/sqrt(4) = 0.5
    total += 1
    print("[3] Backward known value: var=4, dy=1 → dx=0.5 ...", end=" ")
    dy3 = np.ones((1, 1, 1, 1), dtype=np.float32)
    var3 = np.array([4.0], dtype=np.float32)
    dx3 = bn_backward(dy3, var3, count=1.0, eps=0.0)
    np.testing.assert_allclose(dx3, np.full_like(dy3, 0.5), rtol=1e-6)
    print("OK")
    passed += 1

    # [4] Backward: count != 1.0 (sf = 1/count)
    total += 1
    print("[4] Backward with count=10, var_stored=40 (eff_var=4) ...", end=" ")
    # If count=10 and var stored=40, eff_var = 40*(1/10) = 4
    dy4 = np.ones((1, 1, 2, 2), dtype=np.float32)
    var4 = np.array([40.0], dtype=np.float32)
    dx4 = bn_backward(dy4, var4, count=10.0, eps=0.0)
    # inv_std = 1/sqrt(4) = 0.5
    np.testing.assert_allclose(dx4, np.full_like(dy4, 0.5), rtol=1e-6)
    print("OK")
    passed += 1

    # [5] Backward: zero dy gives zero gradients
    total += 1
    print("[5] Zero dy → zero dx ...", end=" ")
    C5 = 4
    x5 = np.random.randn(2, C5, 3, 3).astype(np.float32) * 2.0
    mean5 = np.random.randn(C5).astype(np.float32) * 0.5
    var5 = np.abs(np.random.randn(C5).astype(np.float32)) + 0.5
    dy5 = np.zeros((2, C5, 3, 3), dtype=np.float32)
    dx5 = bn_backward(dy5, var5)
    np.testing.assert_array_equal(dx5, np.zeros_like(dx5))
    print("OK")
    passed += 1

    # [6] Numerical gradient check (small tensor for speed)
    total += 1
    print("[6] Numerical gradient check (NCHW, 2x2x2x2) ...", end=" ")
    N6, C6, H6, W6 = 1, 2, 2, 2  # 8 elements
    x6 = np.random.randn(N6, C6, H6, W6).astype(np.float32) * 0.5
    mean6 = np.random.randn(C6).astype(np.float32) * 0.3
    var6 = (np.random.rand(C6).astype(np.float32) + 0.5) * 2.0  # positive var
    dy6 = np.random.randn(N6, C6, H6, W6).astype(np.float32) * 0.5
    dx_analytic = bn_backward(dy6, var6, count=1.0, eps=1e-5)
    dx_numeric = numerical_grad_x(x6, mean6, var6, dy6, count=1.0, eps_bn=1e-5, h=1e-3)
    np.testing.assert_allclose(dx_analytic, dx_numeric, rtol=1e-3, atol=1e-4)
    print("OK")
    passed += 1

    # [7] Numerical gradient check with non-unit count
    total += 1
    print("[7] Numerical gradient with count=5 (sf=0.2) ...", end=" ")
    N7, C7 = 1, 2
    x7 = np.random.randn(N7, C7, 1, 3).astype(np.float32) * 0.5
    mean7 = np.random.randn(C7).astype(np.float32) * 0.5
    var7 = (np.random.rand(C7).astype(np.float32) + 1.0) * 5.0
    dy7 = np.random.randn(N7, C7, 1, 3).astype(np.float32) * 0.5
    count7 = 5.0
    dx_analytic7 = bn_backward(dy7, var7, count=count7, eps=1e-5)
    dx_numeric7 = numerical_grad_x(x7, mean7, var7, dy7, count=count7, eps_bn=1e-5, h=1e-3)
    np.testing.assert_allclose(dx_analytic7, dx_numeric7, rtol=1e-3, atol=1e-4)
    print("OK")
    passed += 1

    # [8] Multi-channel shape test: per-channel inv_std correctly applied
    total += 1
    print("[8] Per-channel scaling verification ...", end=" ")
    N8, C8 = 2, 3
    x8 = np.random.randn(N8, C8, 2, 2).astype(np.float32)
    mean8 = np.zeros(C8, dtype=np.float32)
    var8 = np.array([1.0, 4.0, 9.0], dtype=np.float32)  # inv_std: 1, 0.5, 1/3
    dy8 = np.ones((N8, C8, 2, 2), dtype=np.float32)
    dx8 = bn_backward(dy8, var8, count=1.0, eps=0.0)
    # Channel 0: inv_std = 1/1 = 1.0
    np.testing.assert_allclose(dx8[:, 0, :, :], np.ones((N8, 2, 2), dtype=np.float32), rtol=1e-6)
    # Channel 1: inv_std = 1/2 = 0.5
    np.testing.assert_allclose(dx8[:, 1, :, :], np.full((N8, 2, 2), 0.5, dtype=np.float32), rtol=1e-6)
    # Channel 2: inv_std = 1/3 ≈ 0.333...
    np.testing.assert_allclose(dx8[:, 2, :, :], np.full((N8, 2, 2), 1.0/3, dtype=np.float32), rtol=1e-5)
    print("OK")
    passed += 1

    # [9] 1D input edge case
    total += 1
    print("[9] 1D input edge case ...", end=" ")
    x9 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mean9 = np.array([0.0], dtype=np.float32)
    var9 = np.array([1.0], dtype=np.float32)
    dy9 = np.ones(3, dtype=np.float32)
    y9 = bn_forward(x9, mean9, var9, count=1.0, eps=0.0)
    dx9 = bn_backward(dy9, var9, count=1.0, eps=0.0)
    np.testing.assert_allclose(y9, x9, rtol=1e-6)  # mean=0, var=1 → y=x
    np.testing.assert_allclose(dx9, dy9, rtol=1e-6)  # dx = dy*1 = dy
    print("OK")
    passed += 1

    # [10] Forward-backward composition: verify forward output matches expected normalization
    total += 1
    print("[10] Forward-backward composition on random NCHW ...", end=" ")
    N10, C10, H10, W10 = 2, 4, 3, 3
    x10 = np.random.randn(N10, C10, H10, W10).astype(np.float32) * 3.0
    mean10 = np.random.randn(C10).astype(np.float32) * 0.5
    var10 = (np.random.rand(C10).astype(np.float32) + 0.5) * 4.0
    count10 = 1.0
    eps10 = 1e-5
    y10 = bn_forward(x10, mean10, var10, count=count10, eps=eps10)
    # Manual verification of each channel
    inv_std_10 = bn_get_inv_std(var10, count=count10, eps=eps10)
    eff_mean_10 = mean10.astype(np.float64) * (1.0 / count10 if count10 != 0 else 1.0)
    for c in range(C10):
        y_c_expected = ((x10[:, c, :, :].astype(np.float64) - eff_mean_10[c]) * inv_std_10[c]).astype(np.float32)
        np.testing.assert_allclose(y10[:, c, :, :], y_c_expected, rtol=1e-5)
    print("OK")
    passed += 1

    # [11] Determinism: same input → same output
    total += 1
    print("[11] Determinism check ...", end=" ")
    x11 = np.random.randn(2, 3, 4, 4).astype(np.float32)
    mean11 = np.random.randn(3).astype(np.float32)
    var11 = np.abs(np.random.randn(3).astype(np.float32)) + 0.1
    dy11 = np.random.randn(2, 3, 4, 4).astype(np.float32)
    results_fwd = [bn_forward(x11, mean11, var11) for _ in range(5)]
    results_bwd = [bn_backward(dy11, var11) for _ in range(5)]
    for i in range(1, 5):
        np.testing.assert_array_equal(results_fwd[0], results_fwd[i])
        np.testing.assert_array_equal(results_bwd[0], results_bwd[i])
    print("OK")
    passed += 1

    # [12] eps effect: larger eps → smaller inv_std → smaller gradient magnitude
    total += 1
    print("[12] Epsilon effect on gradient magnitude ...", end=" ")
    var12 = np.array([0.0], dtype=np.float32)  # zero variance
    dy12 = np.ones((1, 1, 1, 1), dtype=np.float32)
    dx_eps_small = bn_backward(dy12, var12, count=1.0, eps=1e-5)
    dx_eps_large = bn_backward(dy12, var12, count=1.0, eps=1.0)
    # With var=0: inv_std = 1/sqrt(0+eps). Larger eps → smaller inv_std.
    assert dx_eps_large[0, 0, 0, 0] < dx_eps_small[0, 0, 0, 0], \
        f"Expected smaller grad with larger eps, got {dx_eps_large[0,0,0,0]} vs {dx_eps_small[0,0,0,0]}"
    print("OK")
    passed += 1

    print(f"\n=== {passed}/{total} self-tests PASSED ===")
    return passed == total


if __name__ == "__main__":
    success = _self_test()
    if not success:
        raise SystemExit(1)
