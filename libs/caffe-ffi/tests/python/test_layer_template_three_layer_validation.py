"""P? 阶段：<LayerName> 层 forward 测试模板（三层验证法）。

Comprehensive tests covering the actual forward computation of:
- <LayerName> layers (<parameter_variant_1>, <parameter_variant_2>, ...)

遵循三层测试验证法（three-layer-test-validation模式）：
  L1 - Known Values: 手工构造极小输入，手算期望值精确验证
  L2 - Random Numpy Match: 固定seed随机输入，与numpy参考实现对比
  L3 - Repeated Determinism: 重复forward验证确定性 + weights不变性

Each test includes numpy reference implementations and detailed perf_trace
logging of forward time, memory peaks, and exception details.

Covered scenarios:
1.  Identity case (已知值：输入=输出)
2.  <known_value_scenario_2>
3.  <known_value_scenario_3>
4.  Random input numpy match (不同参数组合)
5.  Repeated forward determinism
6.  Weights不变性验证
7.  （可选）边界与极端输入

Run with:
    pytest tests/python/test_layer_template_three_layer_validation.py -v
    pytest tests/python/test_layer_template_three_layer_validation.py -v -s  # verbose with [PERF] logs
"""
from __future__ import annotations

import numpy as np
import pytest

import caffe_ffi
from caffe_ffi import net_param_from_string, net_from_param
from .conftest import require_cpp_extension, perf_trace


# ═══════════════════════════════════════════════════════════════════════
# Numpy Reference Implementations（numpy-reference-first模式）
# ═══════════════════════════════════════════════════════════════════════

def relu_np(x: np.ndarray, negative_slope: float = 0.0) -> np.ndarray:
    """Numpy reference for ReLU activation (NCHW format).

    ReLU(x) = max(x, 0)  当 negative_slope=0
    LeakyReLU(x) = max(x, 0) + negative_slope * min(x, 0)

    Args:
        x: Input tensor (any shape, typically NCHW)
        negative_slope: Slope for negative values (0 = standard ReLU)
    Returns:
        Output tensor (same shape as input)
    """
    return np.where(x > 0, x, x * negative_slope).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Prototxt helpers（separate-nets-independent-ops模式）
# ═══════════════════════════════════════════════════════════════════════

def _make_relu_net(negative_slope: float = 0.0,
                   input_shape: tuple = (1, 1, 2, 2)) -> tuple:
    """Create a Net with a single ReLU layer.

    Args:
        negative_slope: LeakyReLU slope parameter
        input_shape: Input blob shape (N, C, H, W)
    Returns:
        (net, input_blob_name, output_blob_name)
    """
    n, c, h, w = input_shape
    prototxt = f"""name: 'test_relu'
input: 'data'
input_shape {{ dim: {n} dim: {c} dim: {h} dim: {w} }}
layer {{
  name: 'relu'
  type: 'ReLU'
  bottom: 'data'
  top: 'relu_out'
  relu_param {{ negative_slope: {negative_slope} }}
}}
"""
    param = net_param_from_string(prototxt)
    net = net_from_param(param)
    return net, "data", "relu_out"


# ═══════════════════════════════════════════════════════════════════════
# Test Class
# ═══════════════════════════════════════════════════════════════════════

@require_cpp_extension
class TestReLULayers:
    """ReLU/LeakyReLU 层 forward 测试（三层验证法）。"""

    # ── L1: Known Values 已知值精确验证 ──────────────────────────────

    def test_known_values_identity(self, ptrace):
        """L1-1: 正数输入 → ReLU输出=输入（identity for positive values）。"""
        with ptrace("ReLU known: identity positive") as t:
            net, in_name, out_name = _make_relu_net(negative_slope=0.0,
                                                    input_shape=(1, 1, 2, 2))
            inp = np.array([[[[1.0, 2.0],
                              [3.0, 4.0]]]], dtype=np.float32)
            t["shape"] = str(inp.shape)
            t["negative_slope"] = 0.0

            outputs = net.Forward({in_name: inp})
            result = outputs[out_name]

            t["output_shape"] = str(result.shape)

        # 手算期望值：正数原样通过
        expected = np.array([[[[1.0, 2.0],
                               [3.0, 4.0]]]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_known_values_zero_negative(self, ptrace):
        """L1-2: 负数输入 → 标准ReLU输出=0。"""
        with ptrace("ReLU known: zero negative") as t:
            net, in_name, out_name = _make_relu_net(negative_slope=0.0,
                                                    input_shape=(1, 1, 2, 2))
            inp = np.array([[[[-1.0, -2.0],
                              [-3.0, -0.5]]]], dtype=np.float32)
            t["shape"] = str(inp.shape)

            outputs = net.Forward({in_name: inp})
            result = outputs[out_name]

        expected = np.zeros_like(inp)
        np.testing.assert_array_equal(result, expected)

    def test_known_values_leaky_relu(self, ptrace):
        """L1-3: LeakyReLU negative_slope=0.1 → 负数按斜率缩放。"""
        with ptrace("ReLU known: leaky slope=0.1") as t:
            net, in_name, out_name = _make_relu_net(negative_slope=0.1,
                                                    input_shape=(1, 1, 2, 2))
            inp = np.array([[[[-10.0, 1.0],
                              [-1.0, 0.0]]]], dtype=np.float32)
            t["shape"] = str(inp.shape)
            t["negative_slope"] = 0.1

            outputs = net.Forward({in_name: inp})
            result = outputs[out_name]

        # 手算：x<0时输出 x*0.1
        expected = np.array([[[[-1.0, 1.0],
                               [-0.1, 0.0]]]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)

    def test_known_values_mixed_signs(self, ptrace):
        """L1-4: 混合正负值 + 零值精确验证。"""
        with ptrace("ReLU known: mixed signs+zero") as t:
            net, in_name, out_name = _make_relu_net(negative_slope=0.0,
                                                    input_shape=(1, 1, 1, 4))
            inp = np.array([[[[-2.0, 0.0, 0.5, 100.0]]]], dtype=np.float32)
            t["shape"] = str(inp.shape)

            outputs = net.Forward({in_name: inp})
            result = outputs[out_name]

        expected = np.array([[[[0.0, 0.0, 0.5, 100.0]]]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    # ── L2: Random Data Numpy Match 随机数据对比 ─────────────────────

    @pytest.mark.parametrize("negative_slope", [0.0, 0.01, 0.1, 0.5])
    @pytest.mark.parametrize("shape", [
        (1, 1, 3, 3),      # 最小可行
        (2, 4, 5, 5),      # 标准验证
        (1, 3, 1, 1),      # 1x1 spatial
        (2, 8, 16, 16),    # 稍大张量
    ])
    def test_random_numpy_match(self, ptrace, negative_slope, shape):
        """L2: 固定seed随机输入，与numpy参考实现对比。"""
        rng = np.random.RandomState(42)  # 固定seed，避免偶发失败
        inp = rng.randn(*shape).astype(np.float32) * 2.0  # σ=2，覆盖正负范围

        with ptrace(f"ReLU random slope={negative_slope}") as t:
            net, in_name, out_name = _make_relu_net(
                negative_slope=negative_slope, input_shape=shape)
            t["shape"] = str(shape)
            t["negative_slope"] = negative_slope
            t["input_range"] = f"[{inp.min():.2f}, {inp.max():.2f}]"

            outputs = net.Forward({in_name: inp})
            result = outputs[out_name]

        # Numpy参考实现对比
        expected = relu_np(inp, negative_slope=negative_slope)
        # 激活层使用较严容差（逐元素操作，无乘加累积误差）
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
        assert result.shape == shape, f"Shape mismatch: {result.shape} vs {shape}"
        assert not np.any(np.isnan(result)), "Output contains NaN"
        assert not np.any(np.isinf(result)), "Output contains Inf"

    # ── L3: Repeated Determinism 重复前向确定性验证 ──────────────────

    def test_repeated_forward_determinism(self, ptrace):
        """L3-1: 相同输入连续forward两次，输出完全一致。"""
        rng = np.random.RandomState(123)
        inp = rng.randn(2, 4, 6, 6).astype(np.float32)

        with ptrace("ReLU determinism: 2x forward") as t:
            net, in_name, out_name = _make_relu_net(
                negative_slope=0.01, input_shape=(2, 4, 6, 6))
            t["shape"] = str(inp.shape)

            out1 = net.Forward({in_name: inp})[out_name]
            out2 = net.Forward({in_name: inp})[out_name]

        # 确定性验证：两次输出完全相等（逐元素激活无随机因素）
        np.testing.assert_array_equal(out1, out2)

    def test_weights_invariance(self, ptrace):
        """L3-2: forward不修改权重（ReLU无权重，验证Net状态不被污染）。"""
        # ReLU无权重blob，但验证网络可多次forward无状态累积
        rng = np.random.RandomState(456)
        inp1 = rng.randn(1, 2, 3, 3).astype(np.float32)
        inp2 = rng.randn(1, 2, 3, 3).astype(np.float32)

        with ptrace("ReLU weights invariance") as t:
            net, in_name, out_name = _make_relu_net(input_shape=(1, 2, 3, 3))
            t["shape"] = "(1,2,3,3)"

            # 第一次forward
            out_a = net.Forward({in_name: inp1})[out_name]
            # 第二次用不同输入
            out_b = net.Forward({in_name: inp2})[out_name]
            # 第三次回到第一次输入，结果应与第一次相同
            out_a2 = net.Forward({in_name: inp1})[out_name]

        np.testing.assert_array_equal(out_a, out_a2,
                                      err_msg="Forward is stateful!")
        # 验证两次不同输入结果不同（非平凡测试）
        assert not np.array_equal(out_a, out_b), "Different inputs give same output?"

    def test_multi_round_stability(self, ptrace):
        """L3-3: 20轮交替输入稳定性测试（无内存泄漏/状态污染）。"""
        rng = np.random.RandomState(789)
        shapes = [(1, 1, 2, 2), (2, 3, 4, 4)]
        nets = []
        for shape in shapes:
            net, in_name, out_name = _make_relu_net(input_shape=shape)
            nets.append((net, in_name, out_name, shape))

        with ptrace("ReLU 20-round stability") as t:
            t["rounds"] = 20
            t["nets"] = len(nets)

            for round_i in range(20):
                for net, in_name, out_name, shape in nets:
                    inp = rng.randn(*shape).astype(np.float32)
                    out = net.Forward({in_name: inp})[out_name]
                    expected = relu_np(inp)
                    np.testing.assert_allclose(
                        out, expected, rtol=1e-6, atol=1e-7,
                        err_msg=f"Round {round_i} failed")

    # ── L4 (Optional): Boundary & Edge Cases 边界与极端输入 ──────────

    def test_edge_all_zeros(self, ptrace):
        """L4-1: 全零输入 → 全零输出。"""
        with ptrace("ReLU edge: all zeros") as t:
            net, in_name, out_name = _make_relu_net(input_shape=(2, 3, 4, 4))
            inp = np.zeros((2, 3, 4, 4), dtype=np.float32)
            t["shape"] = str(inp.shape)

            outputs = net.Forward({in_name: inp})
            result = outputs[out_name]

        np.testing.assert_array_equal(result, inp)

    def test_edge_large_values(self, ptrace):
        """L4-2: 极大值输入（验证无溢出）。"""
        with ptrace("ReLU edge: large values") as t:
            net, in_name, out_name = _make_relu_net(input_shape=(1, 1, 1, 4))
            inp = np.array([[[[-1e6, 1e6, -1e-6, 1e-6]]]], dtype=np.float32)
            t["shape"] = str(inp.shape)
            t["value_range"] = "[-1e6, 1e6]"

            outputs = net.Forward({in_name: inp})
            result = outputs[out_name]

        expected = np.array([[[[0.0, 1e6, 0.0, 1e-6]]]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        assert not np.any(np.isnan(result)), "Large values caused NaN"
        assert not np.any(np.isinf(result)), "Large values caused Inf"

    def test_edge_1d_input(self, ptrace):
        """L4-3: 1D输入（非典型shape验证）。"""
        # 注意：根据实际层的shape要求调整，此例假设支持任意shape
        with ptrace("ReLU edge: 1D input") as t:
            n, c = 2, 8
            prototxt = f"""name: 'test_relu_1d'
input: 'data'
input_shape {{ dim: {n} dim: {c} }}
layer {{ name: 'relu' type: 'ReLU' bottom: 'data' top: 'out' }}
"""
            param = net_param_from_string(prototxt)
            net = net_from_param(param)
            t["shape"] = f"({n}, {c})"

            rng = np.random.RandomState(999)
            inp = rng.randn(n, c).astype(np.float32)
            outputs = net.Forward({"data": inp})
            result = outputs["out"]

        expected = relu_np(inp)
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)


# ═══════════════════════════════════════════════════════════════════════
# 注册到性能追踪（添加到conftest.py的_P?B_TEST_CLASSES集合）
# ═══════════════════════════════════════════════════════════════════════
# 在conftest.py中添加：
# _P?__TEST_CLASSES = {
#     "TestReLULayers",  # <-- 添加你的测试类名
# }
# _PERF_TEST_CLASSES = _P1_TEST_CLASSES | ... | _P?__TEST_CLASSES
