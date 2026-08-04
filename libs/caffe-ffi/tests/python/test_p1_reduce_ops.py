"""P1 归一化/规约/复制类算子单元测试：MVN、Reduction、Tile、Im2col。

覆盖内容：
  - 前向数值正确性（对比 numpy 参考实现）
  - 特异性分支（MVN 的 normalize_variance / across_channels 组合、
    Reduction 的 SUM/ASUM/SUMSQ/MEAN + axis + coeff、Tile 的 tiles/axis、
    Im2col 的 kernel/pad/stride/dilation）
  - 输出形状断言
  - 注册/实例化断言
  - 数值梯度（backward）校验

numpy 参考语义均从对应 C++ 实现（src/caffe_ffi/layers/*.cpp）确认。
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from caffe_ffi import Net
from .conftest import require_cpp_extension
from .caffe_test_helpers import make_net
from ._grad_check_utils import (
    assert_grad_close,
    numerical_grad_for_input,
)


# ──────────────────────────────────────────────────────────────────────
# numpy 参考实现（语义与 C++ 一一对应）
# ──────────────────────────────────────────────────────────────────────

def mvn_reference(x, normalize_variance=True, across_channels=False, eps=1e-9):
    """MVN 参考：对每个 group 做 (x - mean) / (sqrt(var) + eps)。

    across_channels=False: group = 每个 (n, c) 通道平面（num = N*C）
    across_channels=True:  group = 每个样本（num = N）
    """
    x64 = np.asarray(x, dtype=np.float64)
    n, c = x64.shape[0], x64.shape[1]
    num = n if across_channels else n * c
    dim = x64.size // num
    xf = x64.reshape(num, dim)
    mean = xf.mean(axis=1, keepdims=True)
    d = xf - mean
    if normalize_variance:
        var = (d * d).mean(axis=1, keepdims=True)
        std = np.sqrt(var) + eps
        out = d / std
    else:
        out = d
    return out.reshape(x64.shape).astype(np.float32)


def reduction_reference(x, operation="SUM", axis=0, coeff=1.0):
    """Reduction 参考：沿 axis 之后的所有维度规约，输出 shape = x.shape[:axis]."""
    x64 = np.asarray(x, dtype=np.float64)
    axis = axis if axis >= 0 else axis + x64.ndim
    num_ = int(np.prod(x64.shape[:axis])) if axis > 0 else 1
    dim_ = int(np.prod(x64.shape[axis:]))
    xf = x64.reshape(num_, dim_)
    if operation == "SUM":
        s = xf.sum(axis=1)
    elif operation == "ASUM":
        s = np.abs(xf).sum(axis=1)
    elif operation == "SUMSQ":
        s = (xf * xf).sum(axis=1)
    elif operation == "MEAN":
        s = xf.sum(axis=1)
    else:
        raise ValueError(operation)
    # C++ 中 MEAN 会将 coeff 先除以 dim_
    c = (coeff / dim_) if operation == "MEAN" else coeff
    out = c * s
    return out.reshape(x64.shape[:axis]).astype(np.float32)


def tile_reference(x, axis=1, tiles=1):
    """Tile 参考：按 C++ 语义——对每个前导组（outer_dim_）复制整个尾块 count(axis_) 份。

    axis=1 时即把每个样本的整个 [C,H,W] 尾块复制 tiles 次（不是逐元素 np.repeat）。
    """
    x = np.asarray(x, dtype=np.float32)
    axis = axis if axis >= 0 else axis + x.ndim
    outer_dim_ = int(np.prod(x.shape[:axis])) if axis > 0 else 1
    inner_dim_ = int(np.prod(x.shape[axis:]))
    xf = x.reshape(outer_dim_, inner_dim_)
    rep = np.repeat(xf, tiles, axis=0)  # 每个 outer 行复制 tiles 次（块复制）
    out_shape = list(x.shape[:axis]) + [x.shape[axis] * tiles] + list(x.shape[axis + 1:])
    return rep.reshape(out_shape)


def im2col_reference(x, kernel_h, kernel_w, pad_h=0, pad_w=0,
                     stride_h=1, stride_w=1, dilation_h=1, dilation_w=1):
    """Im2col 参考：NCHW -> [N, C*kh*kw, out_h, out_w]（Caffe 布局）。"""
    x = np.asarray(x, dtype=np.float32)
    n, c, h, w = x.shape
    out_h = (h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    out_w = (w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    out = np.zeros((n, c * kernel_h * kernel_w, out_h, out_w), dtype=np.float32)
    for cc in range(c):
        for kh in range(kernel_h):
            for kw in range(kernel_w):
                for oh in range(out_h):
                    ih = oh * stride_h - pad_h + kh * dilation_h
                    for ow in range(out_w):
                        iw = ow * stride_w - pad_w + kw * dilation_w
                        if 0 <= ih < h and 0 <= iw < w:
                            out[:, cc * kernel_h * kernel_w + kh * kernel_w + kw,
                                oh, ow] = x[:, cc, ih, iw]
    return out


# ──────────────────────────────────────────────────────────────────────
# prototxt 构造辅助
# ──────────────────────────────────────────────────────────────────────

def _input_layer(shape):
    dims = " ".join(f"dim: {d}" for d in shape)
    return f'layer {{ name: "data" type: "Input" top: "data" input_param {{ shape {{ {dims} }} }} }}'


def _mvn_proto(shape, normalize_variance=True, across_channels=False, eps=1e-9):
    return textwrap.dedent(f"""\
        name: "mvn_test"
        {_input_layer(shape)}
        layer {{
          name: "mvn" type: "MVN" bottom: "data" top: "mvn"
          mvn_param {{ normalize_variance: {str(normalize_variance).lower()}
                      across_channels: {str(across_channels).lower()} eps: {eps} }}
        }}
    """)


def _reduction_proto(shape, operation="SUM", axis=1, coeff=1.0):
    return textwrap.dedent(f"""\
        name: "reduction_test"
        {_input_layer(shape)}
        layer {{
          name: "red" type: "Reduction" bottom: "data" top: "red"
          reduction_param {{ operation: {operation} axis: {axis} coeff: {coeff} }}
        }}
    """)


def _tile_proto(shape, axis=1, tiles=1):
    return textwrap.dedent(f"""\
        name: "tile_test"
        {_input_layer(shape)}
        layer {{
          name: "tile" type: "Tile" bottom: "data" top: "tile"
          tile_param {{ axis: {axis} tiles: {tiles} }}
        }}
    """)


def _im2col_proto(shape, kernel_size=None, kernel_h=None, kernel_w=None,
                  pad=None, stride=None, dilation=None,
                  pad_h=None, pad_w=None, stride_h=None, stride_w=None,
                  dilation_h=None, dilation_w=None):
    body = []
    if kernel_size is not None:
        body.append(f"kernel_size: {kernel_size}")
    if kernel_h is not None:
        body.append(f"kernel_h: {kernel_h}")
    if kernel_w is not None:
        body.append(f"kernel_w: {kernel_w}")
    if pad is not None:
        body.append(f"pad: {pad}")
    if pad_h is not None:
        body.append(f"pad_h: {pad_h}")
    if pad_w is not None:
        body.append(f"pad_w: {pad_w}")
    if stride is not None:
        body.append(f"stride: {stride}")
    if stride_h is not None:
        body.append(f"stride_h: {stride_h}")
    if stride_w is not None:
        body.append(f"stride_w: {stride_w}")
    if dilation is not None:
        body.append(f"dilation: {dilation}")
    if dilation_h is not None:
        body.append(f"dilation_h: {dilation_h}")
    if dilation_w is not None:
        body.append(f"dilation_w: {dilation_w}")
    params = " ".join(body)
    return textwrap.dedent(f"""\
        name: "im2col_test"
        {_input_layer(shape)}
        layer {{
          name: "col" type: "Im2col" bottom: "data" top: "col"
          im2col_param {{ {params} }}
        }}
    """)


# ──────────────────────────────────────────────────────────────────────
# MVN
# ──────────────────────────────────────────────────────────────────────

@require_cpp_extension
class TestMVNLayer:
    SHAPE = (2, 3, 4, 4)

    def _run(self, net, x):
        return net.Forward({"data": x})["mvn"].data

    @pytest.mark.parametrize("normalize_variance,across_channels", [
        (True, False), (True, True), (False, False), (False, True),
    ])
    def test_mvn_forward_matches_reference(self, normalize_variance, across_channels):
        proto = _mvn_proto(self.SHAPE, normalize_variance, across_channels)
        net = make_net(proto)
        np.random.seed(0)
        x = np.random.randn(*self.SHAPE).astype(np.float32)
        out = self._run(net, x)
        expected = mvn_reference(x, normalize_variance, across_channels)
        np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-4)

    def test_mvn_output_shape_identity(self):
        proto = _mvn_proto(self.SHAPE)
        net = make_net(proto)
        x = np.random.randn(*self.SHAPE).astype(np.float32)
        out = self._run(net, x)
        assert out.shape == self.SHAPE
        assert out.dtype == np.float32

    def test_mvn_constant_input_finite(self):
        """常量输入 -> 方差为 0，std = eps，输出全 0（不会除零/Nan）。"""
        proto = _mvn_proto(self.SHAPE, normalize_variance=True, across_channels=False)
        net = make_net(proto)
        x = np.full(self.SHAPE, 3.0, dtype=np.float32)
        out = self._run(net, x)
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out, 0.0, atol=1e-7)

    def test_mvn_center_only_no_variance(self):
        """normalize_variance=False：输出 = x - mean（仅中心化）。"""
        proto = _mvn_proto(self.SHAPE, normalize_variance=False, across_channels=False)
        net = make_net(proto)
        x = np.random.randn(*self.SHAPE).astype(np.float32)
        out = self._run(net, x)
        expected = mvn_reference(x, normalize_variance=False, across_channels=False)
        np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-4)

    def test_mvn_registration(self):
        proto = _mvn_proto(self.SHAPE)
        net = make_net(proto)
        layer = net.layer_by_name("mvn")
        assert layer.type == "MVN"
        assert len(layer.blobs) == 0  # 无学习参数

    def test_mvn_backward_numerical(self):
        proto = _mvn_proto(self.SHAPE, normalize_variance=True, across_channels=False)
        net = make_net(proto)
        x = np.random.randn(*self.SHAPE).astype(np.float32)
        out = net.Forward({"data": x})["mvn"].data
        dy = np.random.randn(*out.shape).astype(np.float32)
        net.backward({"mvn": dy})
        dX_an = net.blob_by_name("data").diff
        dX_num = numerical_grad_for_input(
            net, "data", x, "mvn", dy, h=1e-3, name="mvn/dx", verbose=False)
        assert_grad_close(dX_an, dX_num, name="mvn/dx", rtol=1e-2, atol=1e-3)


# ──────────────────────────────────────────────────────────────────────
# Reduction
# ──────────────────────────────────────────────────────────────────────

@require_cpp_extension
class TestReductionLayer:
    SHAPE = (2, 3, 4, 4)

    def _run(self, net, x):
        return net.Forward({"data": x})["red"].data

    @pytest.mark.parametrize("operation", ["SUM", "ASUM", "SUMSQ", "MEAN"])
    def test_reduction_forward_operations(self, operation):
        proto = _reduction_proto(self.SHAPE, operation=operation, axis=1)
        net = make_net(proto)
        x = np.random.randn(*self.SHAPE).astype(np.float32) * 2.0
        out = self._run(net, x)
        expected = reduction_reference(x, operation=operation, axis=1)
        np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("axis", [1, 2, 3, -1])
    def test_reduction_forward_axis(self, axis):
        proto = _reduction_proto(self.SHAPE, operation="SUM", axis=axis)
        net = make_net(proto)
        x = np.random.randn(*self.SHAPE).astype(np.float32)
        out = self._run(net, x)
        expected = reduction_reference(x, operation="SUM", axis=axis)
        np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-4)
        assert out.shape == expected.shape

    def test_reduction_output_shape(self):
        proto = _reduction_proto(self.SHAPE, operation="SUM", axis=1)
        net = make_net(proto)
        x = np.random.randn(*self.SHAPE).astype(np.float32)
        out = self._run(net, x)
        assert out.shape == (2,)
        assert out.dtype == np.float32

    def test_reduction_coeff_scaling(self):
        proto = _reduction_proto(self.SHAPE, operation="SUM", axis=1, coeff=2.5)
        net = make_net(proto)
        x = np.random.randn(*self.SHAPE).astype(np.float32)
        out = self._run(net, x)
        expected = reduction_reference(x, operation="SUM", axis=1, coeff=2.5)
        np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-4)

    def test_reduction_registration(self):
        proto = _reduction_proto(self.SHAPE, operation="SUM", axis=1)
        net = make_net(proto)
        layer = net.layer_by_name("red")
        assert layer.type == "Reduction"
        assert len(layer.blobs) == 0

    @pytest.mark.parametrize("operation", ["SUM", "ASUM", "SUMSQ", "MEAN"])
    def test_reduction_backward_numerical(self, operation):
        proto = _reduction_proto(self.SHAPE, operation=operation, axis=1)
        net = make_net(proto)
        x = np.random.randn(*self.SHAPE).astype(np.float32) * 2.0
        out = net.Forward({"data": x})["red"].data
        dy = np.random.randn(*out.shape).astype(np.float32)
        net.backward({"red": dy})
        dX_an = net.blob_by_name("data").diff
        dX_num = numerical_grad_for_input(
            net, "data", x, "red", dy, h=1e-3, name="reduction/dx", verbose=False)
        # SUMSQ 在 x≈0 处梯度接近 0，有限差分相对误差偏大，放宽容差
        assert_grad_close(dX_an, dX_num, name="reduction/dx", rtol=0.05, atol=1e-2)


# ──────────────────────────────────────────────────────────────────────
# Tile
# ──────────────────────────────────────────────────────────────────────

@require_cpp_extension
class TestTileLayer:
    SHAPE = (2, 3, 4, 4)

    def _run(self, net, x):
        return net.Forward({"data": x})["tile"].data

    @pytest.mark.parametrize("axis,tiles", [
        (1, 3), (2, 2), (3, 4), (0, 2), (-1, 3),
    ])
    def test_tile_forward_matches_reference(self, axis, tiles):
        proto = _tile_proto(self.SHAPE, axis=axis, tiles=tiles)
        net = make_net(proto)
        x = np.random.randn(*self.SHAPE).astype(np.float32)
        out = self._run(net, x)
        expected = tile_reference(x, axis=axis, tiles=tiles)
        np.testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize("axis,tiles", [(1, 3), (2, 2), (3, 4)])
    def test_tile_output_shape(self, axis, tiles):
        proto = _tile_proto(self.SHAPE, axis=axis, tiles=tiles)
        net = make_net(proto)
        x = np.random.randn(*self.SHAPE).astype(np.float32)
        out = self._run(net, x)
        expected_shape = list(self.SHAPE)
        expected_shape[axis] *= tiles
        assert out.shape == tuple(expected_shape)
        assert out.dtype == np.float32

    def test_tile_default_tiles_one(self):
        proto = _tile_proto(self.SHAPE, axis=1, tiles=1)
        net = make_net(proto)
        x = np.random.randn(*self.SHAPE).astype(np.float32)
        out = self._run(net, x)
        np.testing.assert_array_equal(out, x)

    def test_tile_registration(self):
        proto = _tile_proto(self.SHAPE, axis=1, tiles=2)
        net = make_net(proto)
        layer = net.layer_by_name("tile")
        assert layer.type == "Tile"
        assert len(layer.blobs) == 0

    def test_tile_backward_numerical(self):
        proto = _tile_proto(self.SHAPE, axis=1, tiles=3)
        net = make_net(proto)
        x = np.random.randn(*self.SHAPE).astype(np.float32)
        out = net.Forward({"data": x})["tile"].data
        dy = np.random.randn(*out.shape).astype(np.float32)
        net.backward({"tile": dy})
        dX_an = net.blob_by_name("data").diff
        dX_num = numerical_grad_for_input(
            net, "data", x, "tile", dy, h=1e-3, name="tile/dx", verbose=False)
        assert_grad_close(dX_an, dX_num, name="tile/dx", rtol=1e-2, atol=1e-3)


# ──────────────────────────────────────────────────────────────────────
# Im2col
# ──────────────────────────────────────────────────────────────────────

@require_cpp_extension
class TestIm2colLayer:
    SHAPE = (2, 3, 5, 5)

    def _run(self, net, x):
        return net.Forward({"data": x})["col"].data

    def test_im2col_default_kernel(self):
        net = make_net(_im2col_proto(self.SHAPE, kernel_size=3))
        x = np.random.randn(*self.SHAPE).astype(np.float32)
        out = self._run(net, x)
        expected = im2col_reference(x, kernel_h=3, kernel_w=3)
        np.testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize("kernel,pad,stride", [
        (3, 1, 1), (3, 0, 1), (2, 1, 2), (3, 1, 2), (1, 0, 1),
    ])
    def test_im2col_kernel_pad_stride(self, kernel, pad, stride):
        net = make_net(_im2col_proto(self.SHAPE, kernel_size=kernel, pad=pad, stride=stride))
        x = np.random.randn(*self.SHAPE).astype(np.float32)
        out = self._run(net, x)
        expected = im2col_reference(x, kernel_h=kernel, kernel_w=kernel,
                                    pad_h=pad, pad_w=pad, stride_h=stride, stride_w=stride)
        np.testing.assert_array_equal(out, expected)

    def test_im2col_dilation(self):
        net = make_net(_im2col_proto(self.SHAPE, kernel_size=3, pad=2, stride=1, dilation=2))
        x = np.random.randn(*self.SHAPE).astype(np.float32)
        out = self._run(net, x)
        expected = im2col_reference(x, kernel_h=3, kernel_w=3,
                                    pad_h=2, pad_w=2, stride_h=1, stride_w=1,
                                    dilation_h=2, dilation_w=2)
        np.testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize("kernel_h,kernel_w,pad_h,pad_w,stride_h,stride_w", [
        (3, 5, 1, 2, 1, 1), (2, 3, 0, 1, 2, 1), (1, 1, 0, 0, 1, 1),
    ])
    def test_im2col_rect_kernel(self, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w):
        net = make_net(_im2col_proto(
            self.SHAPE, kernel_h=kernel_h, kernel_w=kernel_w,
            pad_h=pad_h, pad_w=pad_w, stride_h=stride_h, stride_w=stride_w))
        x = np.random.randn(*self.SHAPE).astype(np.float32)
        out = self._run(net, x)
        expected = im2col_reference(x, kernel_h=kernel_h, kernel_w=kernel_w,
                                    pad_h=pad_h, pad_w=pad_w,
                                    stride_h=stride_h, stride_w=stride_w)
        np.testing.assert_array_equal(out, expected)

    def test_im2col_output_shape(self):
        net = make_net(_im2col_proto(self.SHAPE, kernel_size=3, pad=1, stride=1))
        x = np.random.randn(*self.SHAPE).astype(np.float32)
        out = self._run(net, x)
        # out_h = out_w = (5 + 2*1 - 3) // 1 + 1 = 5
        assert out.shape == (2, 3 * 3 * 3, 5, 5)
        assert out.dtype == np.float32

    def test_im2col_registration(self):
        net = make_net(_im2col_proto(self.SHAPE, kernel_size=3))
        layer = net.layer_by_name("col")
        assert layer.type == "Im2col"
        assert len(layer.blobs) == 0

    def test_im2col_backward_numerical(self):
        net = make_net(_im2col_proto(self.SHAPE, kernel_size=3, pad=1, stride=1))
        x = np.random.randn(*self.SHAPE).astype(np.float32)
        out = net.Forward({"data": x})["col"].data
        dy = np.random.randn(*out.shape).astype(np.float32)
        net.backward({"col": dy})
        dX_an = net.blob_by_name("data").diff
        dX_num = numerical_grad_for_input(
            net, "data", x, "col", dy, h=1e-3, name="im2col/dx", verbose=False)
        assert_grad_close(dX_an, dX_num, name="im2col/dx", rtol=1e-2, atol=1e-3)