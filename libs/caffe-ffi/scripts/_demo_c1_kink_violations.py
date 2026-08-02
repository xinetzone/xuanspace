#!/usr/bin/env python3
"""
模拟违规案例测试：验证 check_c1_kink_protection.py 能否正确捕获并报错。

测试策略：
1. 在 tests/python/ 下创建带有 _violation_demo_ 前缀的临时测试文件
2. 每个文件模拟一种典型违规场景或正确场景
3. 调用 check_c1_kink_protection.py 扫描
4. 验证退出码和输出是否符合预期
5. 清理所有临时文件

参考经验：测试用例必须在扫描器工作区内落盘，用例结束后立即清理。
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CHECK_SCRIPT = SCRIPT_DIR / "check_c1_kink_protection.py"
TEST_DIR = PROJECT_ROOT / "tests" / "python"

PREFIX = "test__violation_demo_"
RESULTS: list[tuple[str, bool, str]] = []
TEMP_FILES: list[Path] = []


def run_check(target: Path | None = None) -> tuple[int, str]:
    """运行 check_c1_kink_protection.py，返回 (exit_code, stdout+stderr)。"""
    cmd = [sys.executable, str(CHECK_SCRIPT), str(target) if target else str(TEST_DIR)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    return proc.returncode, proc.stdout + proc.stderr


def create_demo(filename: str, content: str) -> Path:
    """在 tests/python/ 下创建一个临时演示文件。"""
    path = TEST_DIR / f"{PREFIX}{filename}"
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    TEMP_FILES.append(path)
    return path


def cleanup():
    """清理所有临时文件。"""
    for p in TEMP_FILES:
        if p.exists():
            p.unlink()


def test_case(name: str, filename: str, content: str, expect_fail: bool, required_substrings: list[str] | None = None):
    """执行单个测试案例。"""
    path = create_demo(filename, content)
    exit_code, output = run_check(path)
    detected_violation = exit_code != 0

    # 检查结果
    passed = True
    reasons = []

    if expect_fail and not detected_violation:
        passed = False
        reasons.append("预期检测到违规但检查通过了（漏报）")
    if not expect_fail and detected_violation:
        passed = False
        reasons.append("预期通过但检查报告了违规（误报）")

    # 检查必需的输出子串
    if required_substrings:
        for substr in required_substrings:
            if substr not in output:
                passed = False
                reasons.append(f"输出中缺少关键信息: '{substr}'")

    status = "PASS" if passed else "FAIL"
    RESULTS.append((name, passed, "; ".join(reasons) if reasons else "符合预期"))

    # 打印详细信息
    print(f"  [{status}] {name}")
    if not passed:
        print(f"         原因: {'; '.join(reasons)}")
        print(f"         退出码: {exit_code}, 预期: {'非0' if expect_fail else '0'}")
        # 打印输出的关键行
        for line in output.splitlines():
            if "FAIL" in line or "FAIL" in line or "violation" in line.lower() or "missing" in line.lower():
                print(f"         输出: {line.strip()}")


# ══════════════════════════════════════════════════════════════════════
# 违规案例（应被检测到，exit_code != 0）
# ══════════════════════════════════════════════════════════════════════

VIOLATION_CASES = [
    # ── V1: LeakyReLU 数值梯度无防护 ──
    (
        "V1: LeakyReLU(negative_slope=0.1) 数值梯度缺少拐点防护",
        "v1_leakyrelu_unprotected.py",
        """
        import numpy as np

        def test_leakyrelu_numeric_grad_unprotected():
            net = Net('''
                name: 'bad'
                input: 'data'
                input_dim: 1 input_dim: 1 input_dim: 3 input_dim: 4
                layer { name: 'r' type: 'ReLU' bottom: 'data' top: 'out'
                        relu_param { negative_slope: 0.1 } }
            ''')
            rng = np.random.RandomState(42)
            x = rng.randn(1, 1, 3, 4).astype(np.float32) * 2.0
            # ❌ 没有调用 avoid_c1_discontinuity！
            dy = np.ones_like(x)
            dx_num = _num_grad(net, x, dy)
        """,
        True,
        ["LeakyReLU", "missing C¹ kink protection"],
    ),
    # ── V2: PReLU 数值梯度无防护（prototxt方式）──
    (
        "V2: PReLU 数值梯度缺少拐点防护（prototxt方式）",
        "v2_prelu_unprotected.py",
        """
        import numpy as np

        def test_prelu_numeric_grad_no_protection():
            proto = '''
                name: 'bad_prelu'
                input: 'data'
                input_dim: 1 input_dim: 1 input_dim: 2 input_dim: 3
                layer { name: 'p' type: 'PReLU' bottom: 'data' top: 'out'
                        prelu_param { channel_shared: true filler { type: 'constant' value: 0.25 } } }
            '''
            net = Net(proto)
            rng = np.random.RandomState(99)
            x = rng.randn(1, 1, 2, 3).astype(np.float32) * 1.5
            # ❌ 没有调用 avoid_c1_discontinuity！
            dy = rng.randn(*x.shape).astype(np.float32)
            dx_num = _num_grad(net, x, dy)
        """,
        True,
        ["PReLU", "missing C¹ kink protection"],
    ),
    # ── V3: ELU(alpha=0.5) 数值梯度无防护 ──
    (
        "V3: ELU(alpha=0.5) 数值梯度缺少拐点防护（C¹不连续因为alpha≠1）",
        "v3_elu_alpha05_unprotected.py",
        """
        import numpy as np

        def test_elu_alpha05_numeric_grad():
            net = _make_elu_net(alpha=0.5)
            rng = np.random.RandomState(55)
            x = rng.randn(1, 1, 3, 4).astype(np.float32) * 2.0
            # ❌ ELU alpha=0.5 是C¹不连续的，没有调用 avoid_c1_discontinuity！
            dy = rng.randn(*x.shape).astype(np.float32)
            dx_num = _num_grad(net, x, dy)
        """,
        True,
        ["ELU(alpha≠1)", "missing C¹ kink protection"],
    ),
    # ── V4: LeakyReLU 手动中心差分循环无防护 ──
    (
        "V4: LeakyReLU 手动中心差分循环（无_num_grad调用）缺少防护",
        "v4_manual_fd_unprotected.py",
        """
        import numpy as np

        def test_leakyrelu_manual_fd():
            proto = '''
                layer { name: 'r' type: 'ReLU' bottom: 'data' top: 'out'
                        relu_param { negative_slope: 0.2 } }
            '''
            net = Net(proto)
            rng = np.random.RandomState(7)
            x = rng.randn(1, 1, 2, 2).astype(np.float32) * 2.0
            h = 1e-3
            # ❌ 手动中心差分但没有拐点防护
            dx = np.zeros_like(x)
            for i in range(x.size):
                xp = x.copy(); xp.flat[i] += h
                xm = x.copy(); xm.flat[i] -= h
                lp = np.sum(net.forward({'data': xp})['out'])
                lm = np.sum(net.forward({'data': xm})['out'])
                dx.flat[i] = (lp - lm) / (2*h)
        """,
        True,
        ["LeakyReLU"],
    ),
    # ── V5: LeakyReLU（alternative param format: negative_slope=0.3）──
    (
        "V5: LeakyReLU 内联negative_slope参数无防护",
        "v5_inline_slope_unprotected.py",
        """
        import numpy as np
        # 直接设置negative_slope=0.3（非prototxt块格式）
        def test_leaky_inline():
            net = make_net(negative_slope=0.3)
            x = np.random.randn(1,1,3,4).astype(np.float32) * 2.0
            # ❌ 无防护
            dx_n = _num_grad(net, x, dy)
        """,
        True,
        ["LeakyReLU", "missing"],
    ),
]

# ══════════════════════════════════════════════════════════════════════
# 正确/豁免案例（应通过检查，exit_code == 0）
# ══════════════════════════════════════════════════════════════════════

PASS_CASES = [
    # ── P1: LeakyReLU 正确使用 avoid_c1_discontinuity ──
    (
        "P1: LeakyReLU 正确使用 avoid_c1_discontinuity",
        "p1_leakyrelu_protected.py",
        """
        import numpy as np
        from .caffe_test_helpers import avoid_c1_discontinuity

        def test_leakyrelu_protected():
            net = Net('''
                layer { name: 'r' type: 'ReLU' relu_param { negative_slope: 0.1 } }
            ''')
            EPS = 1e-3
            x = np.random.randn(1,1,3,4).astype(np.float32) * 2.0
            x = avoid_c1_discontinuity(x, h=EPS)  # ✅ 正确防护
            dx_num = _num_grad(net, x, dy)
        """,
        False,
        None,
    ),
    # ── P2: PReLU 正确使用 avoid_c1_discontinuity ──
    (
        "P2: PReLU 正确使用 avoid_c1_discontinuity",
        "p2_prelu_protected.py",
        """
        from .caffe_test_helpers import avoid_c1_discontinuity
        def test_prelu_ok():
            x = rng.randn(1,1,2,3).astype(np.float32) * 1.5
            h = 1e-3
            x = avoid_c1_discontinuity(x, h=h)  # ✅
            # PReLU layer + numerical grad
            proto = 'layer { type: \"PReLU\" }'
            dx_num = _num_grad(net, x, dy)
        """,
        False,
        None,
    ),
    # ── P3: 豁免注释（故意跨拐点测试误差特性）──
    (
        "P3: c1-kink-ok 豁免注释",
        "p3_suppressed.py",
        """
        # c1-kink-ok: 故意跨拐点验证O(1)误差量级，这是误差特性专项测试
        import numpy as np
        def test_intentional_kink_crossing():
            proto = 'layer { type: \"ReLU\" relu_param { negative_slope: 0.1 } }'
            x = np.array([0.0, 0.0005, -0.0005], dtype=np.float32)
            dx_num = _num_grad(net, x, dy)  # 故意在拐点处采样
        """,
        False,
        None,
    ),
    # ── P4: Sigmoid（C^∞光滑，无数值梯度问题）──
    (
        "P4: Sigmoid（C^∞光滑函数，无需拐点防护）",
        "p4_sigmoid_smooth.py",
        """
        import numpy as np
        def test_sigmoid_grad():
            proto = 'layer { type: \"Sigmoid\" }'
            x = np.random.randn(1,1,3,4).astype(np.float32) * 1.5
            dx_num = _num_grad(net, x, dy)  # ✅ Sigmoid是光滑函数，无需防护
        """,
        False,
        None,
    ),
    # ── P5: TanH（C^∞光滑）──
    (
        "P5: TanH（C^∞光滑函数）",
        "p5_tanh_smooth.py",
        """
        def test_tanh_grad():
            proto = 'layer { type: \"TanH\" }'
            x = rng.randn(1,1,3,4).astype(np.float32) * 2.0
            dx_num = _num_grad(net, x, dy)  # ✅
        """,
        False,
        None,
    ),
    # ── P6: ELU(alpha=1.0) C¹连续，放宽rtol即可（但检查脚本只检测C¹不连续，此处自动通过）──
    (
        "P6: ELU(alpha=1) C¹连续（检查脚本不要求avoid，用户自行放宽rtol）",
        "p6_elu_alpha1.py",
        """
        def test_elu_c1_continuous():
            proto = 'layer { type: \"ELU\" }'  # 默认alpha=1
            x = rng.randn(1,1,3,4).astype(np.float32) * 2.0
            dx_num = _num_grad(net, x, dy)  # ✅ ELU(alpha=1)是C¹连续的，无需avoid
            # 注意：用户需要自行设置 rtol=5e-3
        """,
        False,
        None,
    ),
    # ── P7: PReLU 仅有analytic gradient测试无数值梯度 ──
    (
        "P7: PReLU 仅analytic gradient（无数值梯度检查）",
        "p7_prelu_analytic_only.py",
        """
        def test_prelu_analytic():
            proto = 'layer { type: \"PReLU\" }'
            net.forward({'data': x})
            net.backward({'out': dy})
            dx = net.blob_by_name('data').diff  # ✅ 只有analytic，无数值梯度
        """,
        False,
        None,
    ),
    # ── P8: 标准ReLU(negative_slope=0) 偏移策略（仅警告不报错）──
    (
        "P8: 标准ReLU 偏移到全正（无数值梯度跨拐点风险）",
        "p8_relu_offset.py",
        """
        def test_relu_positive():
            proto = 'layer { type: \"ReLU\" }'  # 默认negative_slope=0
            x = rng.randn(1,1,3,4).astype(np.float32) * 2.0 + 1.0  # 全正偏移
            dx_num = _num_grad(net, x, dy)  # ✅ 偏移到全正，不跨拐点
        """,
        False,
        None,
    ),
]


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  C¹ Kink Protection Check — Violation Detection Demo")
    print("=" * 70)
    print()

    # 确认检查脚本存在
    if not CHECK_SCRIPT.exists():
        print(f"ERROR: 检查脚本不存在: {CHECK_SCRIPT}")
        return 1

    print(f"检查脚本: {CHECK_SCRIPT}")
    print(f"测试目录: {TEST_DIR}")
    print(f"临时文件前缀: {PREFIX}")
    print()

    # ── 违规案例 ──
    print(c_bold("🔴 违规案例（应被检测到）:"))
    print("-" * 50)
    for name, filename, content, expect_fail, required in VIOLATION_CASES:
        test_case(name, filename, content, expect_fail, required)
    print()

    # ── 通过案例 ──
    print(c_bold("🟢 正确/豁免案例（应通过）:"))
    print("-" * 50)
    for name, filename, content, expect_fail, required in PASS_CASES:
        test_case(name, filename, content, expect_fail, required)
    print()

    # ── 汇总 ──
    print("=" * 70)
    passed = sum(1 for _, p, _ in RESULTS if p)
    failed = sum(1 for _, p, _ in RESULTS if not p)
    total = len(RESULTS)

    if failed == 0:
        print(c_green(f"  ✅ 全部 {total} 个测试通过（{passed} passed, 0 failed）！"))
        print("  检查脚本能够正确：")
        print("    • 检测到无防护的LeakyReLU/PReLU/ELU(α≠1)数值梯度测试")
        print("    • 识别手动中心差分循环中的违规")
        print("    • 放行正确使用avoid_c1_discontinuity的测试")
        print("    •  honoring豁免注释(# c1-kink-ok)")
        print("    •  不误报Sigmoid/TanH等C^∞光滑函数")
        print("    •  不误报无数值梯度的analytic-only测试")
    else:
        print(c_red(f"  ❌ {failed}/{total} 个测试失败："))
        for name, p, reason in RESULTS:
            if not p:
                print(f"    - {name}: {reason}")
    print("=" * 70)
    print()

    # ── 清理 ──
    cleanup()
    print("临时文件已清理。")

    return 0 if failed == 0 else 1


def c_bold(s):
    return f"\033[1m{s}\033[0m" if sys.stdout.isatty() else s


def c_green(s):
    return f"\033[92m{s}\033[0m" if sys.stdout.isatty() else s


def c_red(s):
    return f"\033[91m{s}\033[0m" if sys.stdout.isatty() else s


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        cleanup()  # 确保即使异常也清理
