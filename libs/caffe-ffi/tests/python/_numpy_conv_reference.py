"""Numpy reference for Convolution layer forward/backward.

Implements standard 2D convolution with groups, supporting:
  - im2col / col2im for patch extraction and gradient accumulation
  - Forward: Y = W (*) X + b
  - Backward: dX, dW, db via col2im/GEMM

All math done in float64 for numerical stability; returned in float32.
"""
from __future__ import annotations

import numpy as np


def _get_conv_output_shape(H, W, Kh, Kw, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w):
    """Compute output spatial dimensions."""
    Ho = (H + 2 * pad_h - dilation_h * (Kh - 1) - 1) // stride_h + 1
    Wo = (W + 2 * pad_w - dilation_w * (Kw - 1) - 1) // stride_w + 1
    return Ho, Wo


def _im2col(x, C, H, W, Kh, Kw, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, Ho, Wo):
    """im2col: extract patches from single image x (C, H, W) -> (C*Kh*Kw, Ho*Wo)."""
    # Pad input
    if pad_h > 0 or pad_w > 0:
        x_pad = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    else:
        x_pad = x
    cols = np.zeros((C * Kh * Kw, Ho * Wo), dtype=np.float64)
    idx = 0
    for oh in range(Ho):
        for ow in range(Wo):
            h_start = oh * stride_h
            w_start = ow * stride_w
            patch = x_pad[:, h_start:h_start + dilation_h * (Kh - 1) + 1:dilation_h,
                          w_start:w_start + dilation_w * (Kw - 1) + 1:dilation_w]
            # patch shape: (C, Kh, Kw) -> flatten to (C*Kh*Kw,)
            cols[:, idx] = patch.reshape(-1)
            idx += 1
    return cols


def _col2im(cols, C, H, W, Kh, Kw, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, Ho, Wo):
    """col2im: accumulate column patches back into image gradient (C, H, W)."""
    H_pad = H + 2 * pad_h
    W_pad = W + 2 * pad_w
    x_pad = np.zeros((C, H_pad, W_pad), dtype=np.float64)
    idx = 0
    for oh in range(Ho):
        for ow in range(Wo):
            h_start = oh * stride_h
            w_start = ow * stride_w
            patch = cols[:, idx].reshape(C, Kh, Kw)
            x_pad[:, h_start:h_start + dilation_h * (Kh - 1) + 1:dilation_h,
                  w_start:w_start + dilation_w * (Kw - 1) + 1:dilation_w] += patch
            idx += 1
    if pad_h > 0 or pad_w > 0:
        return x_pad[:, pad_h:pad_h + H, pad_w:pad_w + W]
    return x_pad


def conv_forward(x, W, b=None, stride=1, pad=0, dilation=1, groups=1):
    """Conv forward (NCHW).

    Args:
        x: input (N, Ci, H, W) float32/float64
        W: weights (Co, Ci/g, Kh, Kw)
        b: bias (Co,) or None
        stride: int or (sh, sw)
        pad: int or (ph, pw)
        dilation: int or (dh, dw)
        groups: int
    Returns:
        y: output (N, Co, Ho, Wo)
    """
    x64 = x.astype(np.float64)
    W64 = W.astype(np.float64)

    N, Ci, H, W_dim = x64.shape
    Co, Ci_per_g, Kh, Kw = W64.shape
    assert Ci == Ci_per_g * groups, f"Ci={Ci} must equal Ci/g*g={Ci_per_g*groups}"
    assert Co % groups == 0

    if isinstance(stride, int):
        sh = sw = stride
    else:
        sh, sw = stride
    if isinstance(pad, int):
        ph = pw = pad
    else:
        ph, pw = pad
    if isinstance(dilation, int):
        dh = dw = dilation
    else:
        dh, dw = dilation

    Ho, Wo = _get_conv_output_shape(H, W_dim, Kh, Kw, ph, pw, sh, sw, dh, dw)
    Co_per_g = Co // groups

    y = np.zeros((N, Co, Ho, Wo), dtype=np.float64)

    for n in range(N):
        for g in range(groups):
            x_g = x64[n, g * Ci_per_g:(g + 1) * Ci_per_g]  # (Ci/g, H, W)
            W_g = W64[g * Co_per_g:(g + 1) * Co_per_g]  # (Co/g, Ci/g, Kh, Kw)
            cols = _im2col(x_g, Ci_per_g, H, W_dim, Kh, Kw, ph, pw, sh, sw, dh, dw, Ho, Wo)
            # W_g reshaped to (Co/g, Ci/g*Kh*Kw), cols is (Ci/g*Kh*Kw, Ho*Wo)
            out_g = W_g.reshape(Co_per_g, -1) @ cols  # (Co/g, Ho*Wo)
            y[n, g * Co_per_g:(g + 1) * Co_per_g] = out_g.reshape(Co_per_g, Ho, Wo)

    if b is not None:
        y += b.reshape(1, -1, 1, 1).astype(np.float64)

    return y.astype(np.float32)


def conv_backward(dy, x, W, b=None, stride=1, pad=0, dilation=1, groups=1):
    """Conv backward: compute dX, dW, db from dy.

    Args:
        dy: top diff (N, Co, Ho, Wo)
        x: original input (N, Ci, H, W)
        W: weights (Co, Ci/g, Kh, Kw)
        b: bias (Co,) or None
        stride, pad, dilation, groups: same as forward
    Returns:
        (dX, dW, db) as float32 arrays
    """
    dy64 = dy.astype(np.float64)
    x64 = x.astype(np.float64)
    W64 = W.astype(np.float64)

    N, Ci, H, W_dim = x64.shape
    Co, Ci_per_g, Kh, Kw = W64.shape
    _, _, Ho, Wo = dy64.shape
    Co_per_g = Co // groups

    if isinstance(stride, int):
        sh = sw = stride
    else:
        sh, sw = stride
    if isinstance(pad, int):
        ph = pw = pad
    else:
        ph, pw = pad
    if isinstance(dilation, int):
        dh = dw = dilation
    else:
        dh, dw = dilation

    dX = np.zeros_like(x64)
    dW = np.zeros_like(W64)
    db = None
    if b is not None:
        db = np.zeros(Co, dtype=np.float64)

    for n in range(N):
        for g in range(groups):
            x_g = x64[n, g * Ci_per_g:(g + 1) * Ci_per_g]
            dy_g = dy64[n, g * Co_per_g:(g + 1) * Co_per_g]  # (Co/g, Ho, Wo)
            W_g = W64[g * Co_per_g:(g + 1) * Co_per_g]

            cols = _im2col(x_g, Ci_per_g, H, W_dim, Kh, Kw, ph, pw, sh, sw, dh, dw, Ho, Wo)
            # (Ci/g*Kh*Kw, Ho*Wo)

            # dW: W_g.reshape(Co/g, Ci/g*Kh*Kw) @ cols^T -> but accumulate over n
            dW_g = dy_g.reshape(Co_per_g, -1) @ cols.T  # (Co/g, Ci/g*Kh*Kw)
            dW[g * Co_per_g:(g + 1) * Co_per_g] += dW_g.reshape(Co_per_g, Ci_per_g, Kh, Kw)

            # dX: col2im of W_g^T @ dy_g_flat
            dcol = W_g.reshape(Co_per_g, -1).T @ dy_g.reshape(Co_per_g, -1)  # (Ci/g*Kh*Kw, Ho*Wo)
            dX_g = _col2im(dcol, Ci_per_g, H, W_dim, Kh, Kw, ph, pw, sh, sw, dh, dw, Ho, Wo)
            dX[n, g * Ci_per_g:(g + 1) * Ci_per_g] += dX_g

        if db is not None:
            db += dy64[n].sum(axis=(1, 2))  # sum over Ho, Wo

    dX = dX.astype(np.float32)
    dW = dW.astype(np.float32)
    db = db.astype(np.float32) if db is not None else None
    return dX, dW, db


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # Test 1: 1x1 conv no bias, N=1, Ci=2, H=W=3, Co=4
    N, Ci, H, W_dim, Co = 1, 2, 3, 3, 4
    x = np.random.randn(N, Ci, H, W_dim).astype(np.float32) * 0.5
    W = np.random.randn(Co, Ci, 1, 1).astype(np.float32) * 0.3
    y = conv_forward(x, W, b=None, stride=1, pad=0, groups=1)
    assert y.shape == (N, Co, H, W_dim), f"Expected {(N,Co,H,W_dim)}, got {y.shape}"

    # Numerical gradient check for 1x1 conv
    h = 1e-4
    dy = np.random.randn(*y.shape).astype(np.float32) * 0.3
    dX, dW, _ = conv_backward(dy, x, W, stride=1, pad=0, groups=1)

    # Check dX
    dX_num = np.zeros_like(x, dtype=np.float64)
    for i in range(x.size):
        xp = x.copy(); xp.flat[i] += h
        xm = x.copy(); xm.flat[i] -= h
        yp = conv_forward(xp, W, stride=1, pad=0).astype(np.float64)
        ym = conv_forward(xm, W, stride=1, pad=0).astype(np.float64)
        dX_num.flat[i] = float(np.sum(dy.astype(np.float64) * (yp - ym))) / (2 * h)
    np.testing.assert_allclose(dX, dX_num.astype(np.float32), rtol=1e-3, atol=1e-3)

    # Check dW
    dW_num = np.zeros_like(W, dtype=np.float64)
    for i in range(W.size):
        Wp = W.copy(); Wp.flat[i] += h
        Wm = W.copy(); Wm.flat[i] -= h
        yp = conv_forward(x, Wp, stride=1, pad=0).astype(np.float64)
        ym = conv_forward(x, Wm, stride=1, pad=0).astype(np.float64)
        dW_num.flat[i] = float(np.sum(dy.astype(np.float64) * (yp - ym))) / (2 * h)
    np.testing.assert_allclose(dW, dW_num.astype(np.float32), rtol=1e-3, atol=1e-3)

    # Test 2: 3x3 conv with pad=1, stride=1
    Kh = Kw = 3
    N, Ci, H, W_dim, Co = 1, 1, 4, 4, 1
    x = np.random.randn(N, Ci, H, W_dim).astype(np.float32) * 0.5
    W = np.random.randn(Co, Ci, Kh, Kw).astype(np.float32) * 0.3
    b = np.array([0.1], dtype=np.float32)
    y = conv_forward(x, W, b=b, pad=1)
    Ho, Wo = H, W_dim
    assert y.shape == (N, Co, Ho, Wo)

    dy = np.random.randn(*y.shape).astype(np.float32) * 0.3
    dX, dW, db = conv_backward(dy, x, W, b=b, pad=1)
    # numerical dX
    dX_num = np.zeros_like(x, dtype=np.float64)
    for i in range(x.size):
        xp = x.copy(); xp.flat[i] += h
        xm = x.copy(); xm.flat[i] -= h
        yp = conv_forward(xp, W, b=b, pad=1).astype(np.float64)
        ym = conv_forward(xm, W, b=b, pad=1).astype(np.float64)
        dX_num.flat[i] = float(np.sum(dy.astype(np.float64) * (yp - ym))) / (2 * h)
    np.testing.assert_allclose(dX, dX_num.astype(np.float32), rtol=5e-3, atol=5e-3)

    # numerical db
    db_num = np.zeros_like(b, dtype=np.float64)
    for i in range(b.size):
        bp = b.copy(); bp.flat[i] += h
        bm = b.copy(); bm.flat[i] -= h
        yp = conv_forward(x, W, b=bp, pad=1).astype(np.float64)
        ym = conv_forward(x, W, b=bm, pad=1).astype(np.float64)
        db_num.flat[i] = float(np.sum(dy.astype(np.float64) * (yp - ym))) / (2 * h)
    np.testing.assert_allclose(db, db_num.astype(np.float32), rtol=1e-3, atol=1e-3)

    print("All _numpy_conv_reference self-tests passed!")
