"""Pooling layer Backward gradient tests.

Covers:
  1. Analytical gradient correctness (numpy reference vs caffe-ffi computed dX)
     - MAX pooling: gradient routes to argmax winner in each window
     - AVE pooling: gradient evenly distributed across pooling window
  2. Numerical gradient check (central finite differences via _grad_check_utils)
  3. Known-value verification (2x2 stride=2 hand-computed)
  4. Configurations: 2x2 s2, 3x3 s1 pad=1, 3x3 s2, global pooling
  5. Zero dy -> zero dX
  6. Shape/dtype/finite/determinism checks
  7. Forward output preserved after backward

Pooling layers have NO learnable parameters (no weight/bias blobs), so only
dX (input gradient) needs verification.

Mathematical reference:
  Forward MAX:   y[n,c,ph,pw] = max_{(h,w) in window(ph,pw)} x[n,c,h,w]
  Forward AVE:   y[n,c,ph,pw] = mean_{(h,w) in window(ph,pw)} x[n,c,h,w]
  Backward MAX:  dx[n,c,h,w] += dy[n,c,ph,pw] if (h,w) is the argmax winner
  Backward AVE:  dx[n,c,h,w] += dy[n,c,ph,pw] / pool_size for each (h,w) in window

Note: Overlapping windows (stride < kernel) cause gradient accumulation in
overlap regions; boundary windows have smaller pool_size when padding=0.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension
from ._grad_check_utils import (
    assert_grad_close,
    numerical_grad_for_input,
)

EPS_NUMERICAL = 1e-3


# ---------------------------------------------------------------------------
# Numpy reference implementation for pooling backward
# ---------------------------------------------------------------------------

def pooling_backward_np(dy, x, kernel_size, stride=None, pad=0,
                        pool_type='MAX', ceil_mode=True, global_pooling=False):
    """Numpy reference for 2D pooling backward (NCHW format).

    Args:
        dy: Upstream gradient (N, C, H_out, W_out)
        x: Input tensor from forward pass (N, C, H, W) — needed for MAX argmax
        kernel_size, stride, pad, pool_type, ceil_mode, global_pooling:
            same parameters as pooling2d_np

    Returns:
        dx: Gradient w.r.t. input (N, C, H, W)
    """
    N, C, H, W = x.shape

    if global_pooling:
        kH, kW = H, W
        stride_h, stride_w = 1, 1
        pad_h, pad_w = 0, 0
    else:
        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size
        if stride is None:
            stride = kernel_size
        if isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride
        if isinstance(pad, int):
            pad_h = pad_w = pad
        else:
            pad_h, pad_w = pad

    if ceil_mode:
        H_out = int(np.ceil(float(H + 2 * pad_h - kH) / stride_h)) + 1
        W_out = int(np.ceil(float(W + 2 * pad_w - kW) / stride_w)) + 1
    else:
        H_out = int(np.floor(float(H + 2 * pad_h - kH) / stride_h)) + 1
        W_out = int(np.floor(float(W + 2 * pad_w - kW) / stride_w)) + 1

    if pad_h > 0 or pad_w > 0:
        if (H_out - 1) * stride_h >= H + pad_h:
            H_out -= 1
        if (W_out - 1) * stride_w >= W + pad_w:
            W_out -= 1

    if global_pooling:
        H_out = W_out = 1

    dx = np.zeros_like(x, dtype=np.float64)
    dy64 = dy.astype(np.float64)
    x64 = x.astype(np.float64)

    for n in range(N):
        for c in range(C):
            for ph in range(H_out):
                for pw in range(W_out):
                    hstart = ph * stride_h - pad_h
                    wstart = pw * stride_w - pad_w
                    hend = min(hstart + kH, H)
                    wend = min(wstart + kW, W)
                    hstart = max(hstart, 0)
                    wstart = max(wstart, 0)

                    if hend <= hstart or wend <= wstart:
                        continue

                    pool_size = (hend - hstart) * (wend - wstart)
                    dyi = dy64[n, c, ph, pw]

                    if pool_type == 'MAX':
                        patch = x64[n, c, hstart:hend, wstart:wend]
                        if patch.size > 0:
                            flat_idx = int(np.argmax(patch))
                            winner_h = hstart + flat_idx // (wend - wstart)
                            winner_w = wstart + flat_idx % (wend - wstart)
                            dx[n, c, winner_h, winner_w] += dyi
                    elif pool_type == 'AVE':
                        scale = dyi / pool_size if pool_size > 0 else 0.0
                        dx[n, c, hstart:hend, wstart:wend] += scale

    return dx.astype(np.float32)


# ---------------------------------------------------------------------------
# Prototxt builder
# ---------------------------------------------------------------------------

def _make_pool_prototxt(input_dims, kernel_size, stride=None, pad=0,
                        pool='MAX', global_pooling=False):
    """Create Input -> Pooling prototxt."""
    dims_str = " ".join(str(d) for d in input_dims)
    if isinstance(kernel_size, int):
        ks_str = f"kernel_size: {kernel_size}"
    else:
        ks_str = f"kernel_h: {kernel_size[0]}\n    kernel_w: {kernel_size[1]}"
    if stride is None:
        stride_str = ""
    elif isinstance(stride, int):
        stride_str = f"\n    stride: {stride}"
    else:
        stride_str = f"\n    stride_h: {stride[0]}\n    stride_w: {stride[1]}"
    pad_str = ""
    if isinstance(pad, int):
        if pad != 0:
            pad_str = f"\n    pad: {pad}"
    else:
        pad_str = f"\n    pad_h: {pad[0]}\n    pad_w: {pad[1]}"
    global_str = "\n    global_pooling: true" if global_pooling else ""
    return textwrap.dedent(f"""\
        name: "test_pool_bw"
        input: "data"
        input_dim: {input_dims[0]}
        input_dim: {input_dims[1]}
        input_dim: {input_dims[2]}
        input_dim: {input_dims[3]}
        layer {{
          name: "pool"
          type: "Pooling"
          bottom: "data"
          top: "pool"
          pooling_param {{
            pool: {pool}
            {ks_str}{stride_str}{pad_str}{global_str}
          }}
        }}
    """)


def _make_pool_net(N, C, H, W, kernel_size, stride=None, pad=0,
                   pool='MAX', global_pooling=False):
    input_dims = (N, C, H, W)
    proto = _make_pool_prototxt(input_dims, kernel_size, stride, pad, pool, global_pooling)
    return Net(proto)


def _run_pool_backward(net, x, dy):
    """Run forward then backward, return dX."""
    out = net.forward({"data": x.astype(np.float32)})
    net.backward({"pool": dy.astype(np.float32)})
    dX = net.blob_by_name("data").diff
    return out["pool"], dX


# ---------------------------------------------------------------------------
# Test Class 1: MAX Pooling 2x2 stride=2 (non-overlapping, simplest case)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestMaxPoolBackward2x2:
    """MAX pooling 2x2 stride=2 backward tests (non-overlapping windows)."""

    def test_maxpool_2x2_known_values(self):
        """2x2 MAX pool s2: known 4x4 input, hand-computed gradients."""
        N, C, H, W = 1, 1, 4, 4
        net = _make_pool_net(N, C, H, W, kernel_size=2, stride=2, pool='MAX')
        x = np.array([[[[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9,10,11,12],
                        [13,14,15,16]]]], dtype=np.float32)
        dy = np.array([[[[10, 20],
                         [30, 40]]]], dtype=np.float32)

        y, dX = _run_pool_backward(net, x, dy)

        assert y.shape == (1, 1, 2, 2)
        assert y[0, 0, 0, 0] == 6.0
        assert y[0, 0, 0, 1] == 8.0
        assert y[0, 0, 1, 0] == 14.0
        assert y[0, 0, 1, 1] == 16.0

        expected_dx = np.array([[[[0, 0, 0, 0],
                                  [0,10, 0,20],
                                  [0, 0, 0, 0],
                                  [0,30, 0,40]]]], dtype=np.float32)
        np.testing.assert_array_equal(dX, expected_dx)

    def test_maxpool_2x2_analytical_dx(self):
        """MAX pool 2x2 s2: caffe-ffi dX vs numpy reference."""
        rng = np.random.RandomState(42)
        N, C, H, W = 2, 3, 4, 4
        net = _make_pool_net(N, C, H, W, kernel_size=2, stride=2, pool='MAX')
        x = rng.randn(N, C, H, W).astype(np.float32)
        dy = rng.randn(N, C, 2, 2).astype(np.float32)

        y, dX = _run_pool_backward(net, x, dy)
        expected_dx = pooling_backward_np(dy, x, kernel_size=2, stride=2, pool_type='MAX')
        np.testing.assert_allclose(dX, expected_dx, rtol=1e-5, atol=1e-6)

    def test_maxpool_2x2_numerical_dx(self):
        """MAX pool 2x2 s2: numerical gradient check for dX (small input)."""
        rng = np.random.RandomState(123)
        N, C, H, W = 1, 2, 4, 4
        net = _make_pool_net(N, C, H, W, kernel_size=2, stride=2, pool='MAX')
        x = rng.randn(N, C, H, W).astype(np.float32) * 0.5 + 1.0
        dy = rng.randn(N, C, 2, 2).astype(np.float32) * 0.1

        out = net.forward({"data": x})
        net.backward({"pool": dy})
        analytic_dX = net.blob_by_name("data").diff

        numerical_dX = numerical_grad_for_input(
            net, "data", x, "pool", dy, h=EPS_NUMERICAL,
            name="pool_max_2x2_dx", verbose=True,
        )
        assert_grad_close(analytic_dX, numerical_dX, name="dX(MAX pool 2x2 s2)",
                          rtol=1e-2, atol=1e-3)

    def test_maxpool_zero_dy_zero_dx(self):
        """Zero dy should produce zero dX."""
        N, C, H, W = 1, 2, 4, 4
        net = _make_pool_net(N, C, H, W, kernel_size=2, stride=2, pool='MAX')
        x = np.random.RandomState(0).randn(N, C, H, W).astype(np.float32)
        dy = np.zeros((N, C, 2, 2), dtype=np.float32)
        _, dX = _run_pool_backward(net, x, dy)
        np.testing.assert_array_equal(dX, np.zeros_like(dX))


# ---------------------------------------------------------------------------
# Test Class 2: AVE Pooling 2x2 stride=2
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestAvePoolBackward2x2:
    """AVE pooling 2x2 stride=2 backward tests."""

    def test_avepool_2x2_known_values(self):
        """2x2 AVE pool s2: uniform dy=4 → dx=1 each (scale=1/4)."""
        N, C, H, W = 1, 1, 4, 4
        net = _make_pool_net(N, C, H, W, kernel_size=2, stride=2, pool='AVE')
        x = np.ones((N, C, H, W), dtype=np.float32)
        dy = np.full((N, C, 2, 2), 4.0, dtype=np.float32)

        y, dX = _run_pool_backward(net, x, dy)
        expected_dx = np.ones((N, C, H, W), dtype=np.float32)
        np.testing.assert_allclose(dX, expected_dx, rtol=1e-6)

    def test_avepool_2x2_analytical_dx(self):
        """AVE pool 2x2 s2: caffe-ffi dX vs numpy reference."""
        rng = np.random.RandomState(43)
        N, C, H, W = 2, 3, 4, 4
        net = _make_pool_net(N, C, H, W, kernel_size=2, stride=2, pool='AVE')
        x = rng.randn(N, C, H, W).astype(np.float32)
        dy = rng.randn(N, C, 2, 2).astype(np.float32)

        _, dX = _run_pool_backward(net, x, dy)
        expected_dx = pooling_backward_np(dy, x, kernel_size=2, stride=2, pool_type='AVE')
        np.testing.assert_allclose(dX, expected_dx, rtol=1e-5, atol=1e-6)

    def test_avepool_2x2_numerical_dx(self):
        """AVE pool 2x2 s2: numerical gradient check for dX."""
        rng = np.random.RandomState(456)
        N, C, H, W = 1, 2, 4, 4
        net = _make_pool_net(N, C, H, W, kernel_size=2, stride=2, pool='AVE')
        x = rng.randn(N, C, H, W).astype(np.float32) * 0.5
        dy = rng.randn(N, C, 2, 2).astype(np.float32) * 0.1

        net.forward({"data": x})
        net.backward({"pool": dy})
        analytic_dX = net.blob_by_name("data").diff

        numerical_dX = numerical_grad_for_input(
            net, "data", x, "pool", dy, h=EPS_NUMERICAL,
            name="pool_ave_2x2_dx", verbose=True,
        )
        assert_grad_close(analytic_dX, numerical_dX, name="dX(AVE pool 2x2 s2)",
                          rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# Test Class 3: Overlapping MAX Pooling (3x3, stride=1, pad=1, same HxW)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestMaxPoolBackwardOverlapping:
    """MAX pooling 3x3 stride=1 pad=1 (overlapping windows, same-size output)."""

    def test_maxpool_3x3_pad1_analytical_dx(self):
        """MAX pool 3x3 s1 pad1: caffe-ffi dX vs numpy reference."""
        rng = np.random.RandomState(44)
        N, C, H, W = 1, 2, 5, 5
        net = _make_pool_net(N, C, H, W, kernel_size=3, stride=1, pad=1, pool='MAX')
        x = rng.randn(N, C, H, W).astype(np.float32)
        dy = rng.randn(N, C, H, W).astype(np.float32)

        _, dX = _run_pool_backward(net, x, dy)
        expected_dx = pooling_backward_np(dy, x, kernel_size=3, stride=1, pad=1, pool_type='MAX')
        np.testing.assert_allclose(dX, expected_dx, rtol=1e-5, atol=1e-6)

    def test_maxpool_3x3_pad1_numerical_dx(self):
        """MAX pool 3x3 s1 pad1: numerical gradient check (small input)."""
        rng = np.random.RandomState(789)
        N, C, H, W = 1, 1, 4, 4
        net = _make_pool_net(N, C, H, W, kernel_size=3, stride=1, pad=1, pool='MAX')
        x = rng.randn(N, C, H, W).astype(np.float32) * 0.5 + 0.5
        dy = rng.randn(N, C, H, W).astype(np.float32) * 0.1

        net.forward({"data": x})
        net.backward({"pool": dy})
        analytic_dX = net.blob_by_name("data").diff

        numerical_dX = numerical_grad_for_input(
            net, "data", x, "pool", dy, h=EPS_NUMERICAL,
            name="pool_max_3x3_s1p1_dx", verbose=True,
        )
        assert_grad_close(analytic_dX, numerical_dX, name="dX(MAX pool 3x3 s1p1)",
                          rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# Test Class 4: Overlapping AVE Pooling (3x3, stride=2, pad=0)
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestAvePoolBackwardOverlapping:
    """AVE pooling 3x3 stride=2 pad=0 (overlapping windows, boundary effects)."""

    def test_avepool_3x3_s2_analytical_dx(self):
        """AVE pool 3x3 s2: caffe-ffi dX vs numpy reference (boundary windows smaller)."""
        rng = np.random.RandomState(45)
        N, C, H, W = 1, 2, 7, 7
        net = _make_pool_net(N, C, H, W, kernel_size=3, stride=2, pool='AVE')
        x = rng.randn(N, C, H, W).astype(np.float32)
        # Output: floor((7-3)/2)+1 = 3x3
        dy = rng.randn(N, C, 3, 3).astype(np.float32)

        _, dX = _run_pool_backward(net, x, dy)
        expected_dx = pooling_backward_np(dy, x, kernel_size=3, stride=2, pool_type='AVE')
        np.testing.assert_allclose(dX, expected_dx, rtol=1e-5, atol=1e-6)

    def test_avepool_3x3_s2_numerical_dx(self):
        """AVE pool 3x3 s2: numerical gradient check (tiny input)."""
        rng = np.random.RandomState(101)
        N, C, H, W = 1, 1, 5, 5
        net = _make_pool_net(N, C, H, W, kernel_size=3, stride=2, pool='AVE')
        x = rng.randn(N, C, H, W).astype(np.float32) * 0.5
        # floor((5-3)/2)+1 = 2x2 output
        dy = rng.randn(N, C, 2, 2).astype(np.float32) * 0.1

        net.forward({"data": x})
        net.backward({"pool": dy})
        analytic_dX = net.blob_by_name("data").diff

        numerical_dX = numerical_grad_for_input(
            net, "data", x, "pool", dy, h=EPS_NUMERICAL,
            name="pool_ave_3x3_s2_dx", verbose=True,
        )
        assert_grad_close(analytic_dX, numerical_dX, name="dX(AVE pool 3x3 s2)",
                          rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# Test Class 5: Global Pooling
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestGlobalPoolBackward:
    """Global MAX/AVE pooling backward tests."""

    def test_global_maxpool_analytical_dx(self):
        """Global MAX pool: gradient routes to single winner per (n,c)."""
        rng = np.random.RandomState(46)
        N, C, H, W = 2, 3, 4, 4
        net = _make_pool_net(N, C, H, W, kernel_size=0, global_pooling=True, pool='MAX')
        x = rng.randn(N, C, H, W).astype(np.float32)
        dy = rng.randn(N, C, 1, 1).astype(np.float32)

        y, dX = _run_pool_backward(net, x, dy)
        assert y.shape == (N, C, 1, 1)
        expected_dx = pooling_backward_np(dy, x, kernel_size=0, pool_type='MAX', global_pooling=True)
        np.testing.assert_allclose(dX, expected_dx, rtol=1e-5, atol=1e-6)

    def test_global_avepool_analytical_dx(self):
        """Global AVE pool: gradient = dy/(H*W) everywhere."""
        N, C, H, W = 1, 2, 3, 3
        net = _make_pool_net(N, C, H, W, kernel_size=0, global_pooling=True, pool='AVE')
        x = np.random.RandomState(47).randn(N, C, H, W).astype(np.float32)
        dy = np.full((N, C, 1, 1), 9.0, dtype=np.float32)

        y, dX = _run_pool_backward(net, x, dy)
        expected_dx = np.full((N, C, H, W), 1.0, dtype=np.float32)
        np.testing.assert_allclose(dX, expected_dx, rtol=1e-6)

    def test_global_maxpool_numerical_dx(self):
        """Global MAX pool: numerical gradient check (tiny)."""
        rng = np.random.RandomState(202)
        N, C, H, W = 1, 1, 3, 3
        net = _make_pool_net(N, C, H, W, kernel_size=0, global_pooling=True, pool='MAX')
        x = rng.randn(N, C, H, W).astype(np.float32) * 0.5 + 1.0
        dy = rng.randn(N, C, 1, 1).astype(np.float32) * 0.1

        net.forward({"data": x})
        net.backward({"pool": dy})
        analytic_dX = net.blob_by_name("data").diff

        numerical_dX = numerical_grad_for_input(
            net, "data", x, "pool", dy, h=EPS_NUMERICAL,
            name="global_maxpool_dx", verbose=True,
        )
        assert_grad_close(analytic_dX, numerical_dX, name="dX(Global MAX pool)",
                          rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# Test Class 6: Determinism and edge cases
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestPoolBackwardDeterminism:
    """Determinism, shape, dtype, and forward-preservation checks."""

    def test_deterministic(self):
        """Same input → same dX (determinism)."""
        rng = np.random.RandomState(99)
        N, C, H, W = 2, 2, 4, 4
        net1 = _make_pool_net(N, C, H, W, kernel_size=2, stride=2, pool='MAX')
        net2 = _make_pool_net(N, C, H, W, kernel_size=2, stride=2, pool='MAX')
        x = rng.randn(N, C, H, W).astype(np.float32)
        dy = rng.randn(N, C, 2, 2).astype(np.float32)
        _, dX1 = _run_pool_backward(net1, x, dy)
        _, dX2 = _run_pool_backward(net2, x, dy)
        np.testing.assert_array_equal(dX1, dX2)

    def test_dx_shape_dtype(self):
        """dX has correct shape and dtype."""
        N, C, H, W = 2, 3, 6, 6
        for pool in ('MAX', 'AVE'):
            net = _make_pool_net(N, C, H, W, kernel_size=2, stride=2, pool=pool)
            x = np.random.RandomState(0).randn(N, C, H, W).astype(np.float32)
            dy = np.random.RandomState(1).randn(N, C, 3, 3).astype(np.float32)
            _, dX = _run_pool_backward(net, x, dy)
            assert dX.shape == (N, C, H, W), f"{pool}: wrong dX shape"
            assert dX.dtype == np.float32, f"{pool}: wrong dtype"
            assert np.all(np.isfinite(dX)), f"{pool}: non-finite values in dX"

    def test_forward_preserved_after_backward(self):
        """Forward output blob data is unchanged after Backward."""
        N, C, H, W = 1, 1, 4, 4
        net = _make_pool_net(N, C, H, W, kernel_size=2, stride=2, pool='MAX')
        x = np.random.RandomState(50).randn(N, C, H, W).astype(np.float32)
        out = net.forward({"data": x})
        y_before = out["pool"].copy()
        dy = np.random.RandomState(51).randn(N, C, 2, 2).astype(np.float32)
        net.backward({"pool": dy})
        y_after = net.blob_by_name("pool").data
        np.testing.assert_array_equal(y_before, y_after)


# ---------------------------------------------------------------------------
# Test Class 7: Tie-breaking (平局)专项测试
# ---------------------------------------------------------------------------
# 陷阱描述: 当MAX pooling窗口内存在多个相等最大值时,argmax只返回第一个
#           最大值位置。C++实现必须与numpy参考的平局打破规则一致
#           (flatten order: row-major,即按行优先遍历的第一个最大值)。
#           若C++实现用了不同的平局策略(如最后一个最大值),会导致梯度
#           路由到错误位置,在平局密集的输入(如ReLU后全0/全常数区域)
#           产生系统性偏差。
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestMaxPoolTieBreaking:
    """MAX pooling平局处理专项测试 — 验证C++与numpy argmax平局规则一致。"""

    def test_tie_2x2_s2_all_equal(self):
        """窗口内所有值相等: 梯度应路由到第一个(左上角)位置。"""
        N, C, H, W = 1, 1, 2, 2
        net = _make_pool_net(N, C, H, W, kernel_size=2, stride=2, pool='MAX')
        x = np.full((N, C, H, W), 3.0, dtype=np.float32)
        dy = np.array([[[[7.0]]]], dtype=np.float32)

        _, dX = _run_pool_backward(net, x, dy)

        # numpy argmax在全相等时返回索引0 → (0,0)位置
        expected_dx = np.array([[[[7.0, 0.0],
                                  [0.0, 0.0]]]], dtype=np.float32)
        np.testing.assert_array_equal(dX, expected_dx)

    def test_tie_2x2_s2_partial_equal(self):
        """窗口内部分值相等: 梯度路由到第一个最大值位置。"""
        N, C, H, W = 1, 1, 4, 4
        net = _make_pool_net(N, C, H, W, kernel_size=2, stride=2, pool='MAX')
        # 构造: 左上窗口[[5,5],[5,2]]最大值5,argmax返回0→(0,0)
        #        右上窗口[[8,8],[3,7]]最大值8,argmax返回0→(0,2)
        #        左下窗口[[6,4],[3,6]]最大值6,argmax返回0→(2,0)
        #        右下窗口[[9,9],[9,9]]最大值9,argmax返回0→(2,2)
        x = np.array([[[[5, 5, 8, 8],
                        [5, 2, 3, 7],
                        [6, 4, 9, 9],
                        [3, 6, 9, 9]]]], dtype=np.float32)
        dy = np.array([[[[10, 20],
                         [30, 40]]]], dtype=np.float32)

        y, dX = _run_pool_backward(net, x, dy)

        # 前向输出验证
        assert y[0, 0, 0, 0] == 5.0
        assert y[0, 0, 0, 1] == 8.0
        assert y[0, 0, 1, 0] == 6.0
        assert y[0, 0, 1, 1] == 9.0

        # 梯度路由位置验证(第一个argmax位置接收梯度)
        expected_dx = np.array([[[[10,  0, 20,  0],
                                  [ 0,  0,  0,  0],
                                  [30,  0, 40,  0],
                                  [ 0,  0,  0,  0]]]], dtype=np.float32)
        np.testing.assert_array_equal(dX, expected_dx)

    def test_tie_vs_numpy_reference_random_ties(self):
        """随机平局输入: C++ dX必须与numpy参考实现完全一致。"""
        rng = np.random.RandomState(777)
        N, C, H, W = 2, 2, 4, 4
        net = _make_pool_net(N, C, H, W, kernel_size=2, stride=2, pool='MAX')

        # 构造含平局的输入: 将随机输入量化到{-1,0,1}三值,制造大量平局
        x = rng.choice([-1.0, 0.0, 1.0], size=(N, C, H, W)).astype(np.float32)
        dy = rng.randn(N, C, 2, 2).astype(np.float32)

        _, dX = _run_pool_backward(net, x, dy)
        expected_dx = pooling_backward_np(dy, x, kernel_size=2, stride=2, pool_type='MAX')
        np.testing.assert_array_equal(dX, expected_dx)

    def test_tie_deterministic_across_runs(self):
        """含平局输入的梯度必须确定性可复现。"""
        N, C, H, W = 1, 1, 4, 4
        x = np.array([[[[2, 2, 1, 1],
                        [2, 2, 1, 1],
                        [3, 3, 0, 0],
                        [3, 3, 0, 0]]]], dtype=np.float32)
        dy = np.array([[[[5, 6],
                         [7, 8]]]], dtype=np.float32)

        dX_runs = []
        for _ in range(5):
            net = _make_pool_net(N, C, H, W, kernel_size=2, stride=2, pool='MAX')
            _, dX = _run_pool_backward(net, x, dy)
            dX_runs.append(dX.copy())

        for i in range(1, 5):
            np.testing.assert_array_equal(dX_runs[0], dX_runs[i])


# ---------------------------------------------------------------------------
# Test Class 8: Overlap Accumulation (重叠累加)专项测试
# ---------------------------------------------------------------------------
# 陷阱描述: 当stride < kernel时,窗口发生重叠。此时:
#   - AVE pooling: 重叠区域的像素从多个窗口接收梯度贡献,必须累加
#   - MAX pooling: 若同一像素是多个窗口的winner,其梯度是多个dy之和
#   - 边界窗口pool_size小于kernel*kernel时,必须用实际pool_size归一化
# 常见错误:
#   1. 覆盖而非累加(= 代替 +=),导致重叠区域梯度丢失
#   2. 边界窗口仍用kernel*kernel归一化而非实际pool_size
#   3. AVE pooling在stride=1的完全重叠场景中,中心像素梯度远大于边角
# ---------------------------------------------------------------------------

@require_cpp_extension
class TestPoolOverlapAccumulation:
    """Overlap accumulation专项测试 — stride < kernel时梯度累加正确性。"""

    def test_ave_3x3_s1_overlap_accumulation_known_values(self):
        """AVE 3x3 s1 pad=0 (3x3输入→1x1输出,单窗口,无重叠基础验证)。"""
        N, C, H, W = 1, 1, 3, 3
        net = _make_pool_net(N, C, H, W, kernel_size=3, stride=1, pool='AVE')
        x = np.zeros((N, C, H, W), dtype=np.float32)
        dy = np.array([[[[9.0]]]], dtype=np.float32)

        _, dX = _run_pool_backward(net, x, dy)
        # 单窗口,dy=9,平均分到9个位置 → 每个位置=1.0
        expected_dx = np.ones((N, C, H, W), dtype=np.float32)
        np.testing.assert_allclose(dX, expected_dx, rtol=1e-6)

    def test_ave_3x3_s1_pad1_overlap_center_accumulates_more(self):
        """AVE 3x3 s1 pad=1 (5x5输入): 中心像素从最多窗口接收梯度。"""
        N, C, H, W = 1, 1, 5, 5
        net = _make_pool_net(N, C, H, W, kernel_size=3, stride=1, pad=1, pool='AVE')
        x = np.zeros((N, C, H, W), dtype=np.float32)
        # 所有dy=1,验证梯度累加模式:
        # - 角像素(0,0): 仅属于1个窗口(ph=0,pw=0),贡献=1/4=0.25 (pad导致实际2x2窗口)
        # - 边像素(0,2): 属于2个窗口(ph=0,pw=1; ph=0,pw=2),贡献=0.25+1/9+...
        # - 中心像素(2,2): 属于9个窗口(3x3),每个贡献1/9,总贡献=1.0
        dy = np.ones((N, C, H, W), dtype=np.float32)

        _, dX = _run_pool_backward(net, x, dy)
        expected_dx = pooling_backward_np(dy, x, kernel_size=3, stride=1, pad=1, pool_type='AVE')
        np.testing.assert_allclose(dX, expected_dx, rtol=1e-5, atol=1e-6)

        # 中心像素(2,2)属于9个窗口,每个贡献1/9 → 期望=1.0
        assert abs(dX[0, 0, 2, 2] - 1.0) < 1e-5, \
            f"Center pixel should accumulate ~1.0 from 9 windows, got {dX[0,0,2,2]}"

    def test_ave_boundary_pool_size_correction(self):
        """AVE pooling边界窗口pool_size小于kernel²时必须用实际大小归一化（CEIL模式）。"""
        # C++默认使用CEIL round_mode（Caffe原生默认行为）
        # 4x5输入,3x3 kernel,stride=2,pad=0 → CEIL输出2x2:
        #   ph=0,pw=0: h[0:3],w[0:3] → pool_size=9 (完整窗口)
        #   ph=0,pw=1: h[0:3],w[2:5] → pool_size=9 (完整窗口)
        #   ph=1,pw=0: h[2:4],w[0:3] → pool_size=6 (边界裁剪: 2行×3列)
        #   ph=1,pw=1: h[2:4],w[2:5] → pool_size=6 (边界裁剪: 2行×3列)
        N, C, H, W = 1, 1, 4, 5
        net = _make_pool_net(N, C, H, W, kernel_size=3, stride=2, pad=0, pool='AVE')
        x = np.zeros((N, C, H, W), dtype=np.float32)
        # CEIL模式输出: H_out=ceil((4-3)/2)+1=2, W_out=ceil((5-3)/2)+1=2
        dy = np.array([[[[9.0, 9.0], [9.0, 9.0]]]], dtype=np.float32)

        _, dX = _run_pool_backward(net, x, dy)
        expected_dx = pooling_backward_np(dy, x, kernel_size=3, stride=2, pad=0, pool_type='AVE', ceil_mode=True)
        np.testing.assert_allclose(dX, expected_dx, rtol=1e-5, atol=1e-6)

        # 边界窗口(ph=1)pool_size=6: dy=9 → 每个元素=9/6=1.5
        # 完整窗口(ph=0)pool_size=9: dy=9 → 每个元素=9/9=1.0
        # 非边界区域(0:2,0:2)只属于(0,0)窗口 → =1.0
        np.testing.assert_allclose(dX[0, 0, 0:2, 0:2], 1.0, rtol=1e-5)
        # 非边界区域(0:2,3:5)只属于(0,1)窗口 → =1.0
        np.testing.assert_allclose(dX[0, 0, 0:2, 3:5], 1.0, rtol=1e-5)
        # 边界区域(3:4,0:2)只属于(1,0)窗口 → =1.5
        np.testing.assert_allclose(dX[0, 0, 3:4, 0:2], 1.5, rtol=1e-5)
        # 边界区域(3:4,3:5)只属于(1,1)窗口 → =1.5
        np.testing.assert_allclose(dX[0, 0, 3:4, 3:5], 1.5, rtol=1e-5)
        # 重叠区域(2:3,2:3)属于所有4个窗口 → 1.0+1.0+1.5+1.5=5.0
        assert abs(dX[0, 0, 2, 2] - 5.0) < 1e-5, \
            f"Overlap pixel (2,2) should accumulate 5.0 from 4 windows, got {dX[0,0,2,2]}"

    def test_max_overlap_same_pixel_wins_multiple_windows(self):
        """MAX pooling重叠: 同一像素是多个窗口winner时梯度应累加。"""
        N, C, H, W = 1, 1, 3, 3
        # 3x3 s1 pad=0 → 输出1x1 (无重叠)
        # 用 5x5 s1 pad=0,3x3 kernel → 输出3x3
        net = _make_pool_net(N, C, 5, 5, kernel_size=3, stride=1, pad=0, pool='MAX')
        # 中心像素(2,2)=100,周围都是小值 → 中心像素是所有9个窗口(3x3输出)的winner
        x = np.zeros((N, C, 5, 5), dtype=np.float32)
        x[0, 0, 2, 2] = 100.0
        dy = np.ones((N, C, 3, 3), dtype=np.float32)  # 所有dy=1

        _, dX = _run_pool_backward(net, x, dy)
        expected_dx = pooling_backward_np(dy, x, kernel_size=3, stride=1, pad=0, pool_type='MAX')
        np.testing.assert_allclose(dX, expected_dx, rtol=1e-5, atol=1e-6)

        # 中心像素累加了9个窗口的dy=1 → dX[2,2]=9.0
        assert abs(dX[0, 0, 2, 2] - 9.0) < 1e-5, \
            f"Center winner should accumulate 9.0 from 9 windows, got {dX[0,0,2,2]}"
        # 其他位置都是0
        mask = np.ones((5, 5), dtype=bool)
        mask[2, 2] = False
        np.testing.assert_array_equal(dX[0, 0][mask], 0.0)

    def test_ave_stride1_full_overlap_random_vs_numpy(self):
        """AVE 3x3 s1 pad=1随机输入: C++ dX与numpy参考对比(包含所有重叠+边界情况)。"""
        rng = np.random.RandomState(888)
        N, C, H, W = 1, 2, 6, 6
        net = _make_pool_net(N, C, H, W, kernel_size=3, stride=1, pad=1, pool='AVE')
        x = rng.randn(N, C, H, W).astype(np.float32)
        dy = rng.randn(N, C, H, W).astype(np.float32)

        _, dX = _run_pool_backward(net, x, dy)
        expected_dx = pooling_backward_np(dy, x, kernel_size=3, stride=1, pad=1, pool_type='AVE')
        np.testing.assert_allclose(dX, expected_dx, rtol=1e-5, atol=1e-6)

    def test_max_stride1_overlap_random_vs_numpy(self):
        """MAX 3x3 s1 pad=1随机输入: C++ dX与numpy参考对比(重叠winner累加)。"""
        rng = np.random.RandomState(999)
        N, C, H, W = 1, 2, 6, 6
        net = _make_pool_net(N, C, H, W, kernel_size=3, stride=1, pad=1, pool='MAX')
        x = rng.randn(N, C, H, W).astype(np.float32) * 2.0
        dy = rng.randn(N, C, H, W).astype(np.float32)

        _, dX = _run_pool_backward(net, x, dy)
        expected_dx = pooling_backward_np(dy, x, kernel_size=3, stride=1, pad=1, pool_type='MAX')
        np.testing.assert_allclose(dX, expected_dx, rtol=1e-5, atol=1e-6)

    def test_gradient_sum_conservation_ave(self):
        """AVE pooling梯度和守恒: sum(dX) == sum(dy) (不含pad的情况下)。"""
        rng = np.random.RandomState(111)
        for H, W, ks, stride, pad in [
            (4, 4, 2, 2, 0),   # 非重叠
            (5, 5, 3, 1, 1),   # 重叠+pad
            (7, 7, 3, 2, 0),   # 重叠无pad,边界pool_size变化
        ]:
            net = _make_pool_net(1, 1, H, W, kernel_size=ks, stride=stride, pad=pad, pool='AVE')
            x = rng.randn(1, 1, H, W).astype(np.float32)
            dy = rng.randn(*net.forward({"data": x})["pool"].shape).astype(np.float32)
            _, dX = _run_pool_backward(net, x, dy)
            # AVE pooling中,每个dy_i平均分配到pool_size个位置,总和守恒: sum(dX) = sum(dy)
            np.testing.assert_allclose(
                dX.sum(), dy.sum(), rtol=1e-4,
                err_msg=f"AVE gradient sum not conserved for H={H},W={W},ks={ks},s={stride},p={pad}"
            )
