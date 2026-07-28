#!/usr/bin/env python3
"""
demo_ffi.math_ops 模块使用演示脚本

演示如何调用通过 C++ FFI 绑定的 15 个数学函数，
涵盖向量运算、统计、数论、字符串处理、激活函数等功能。

使用方法:
    # 正常运行
    python examples/math_ops_demo.py

    # 开启性能日志（会打印 C++ 侧的详细计时信息到 stderr）
    DEMO_FFI_PERF_LOG=1 python examples/math_ops_demo.py  # Linux/macOS
    $env:DEMO_FFI_PERF_LOG=1; python examples/math_ops_demo.py  # Windows PowerShell
"""
from __future__ import annotations

import os
import sys
import time

# 确保能导入 demo_ffi 包（开发环境下）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from demo_ffi.demo import math as m


def section(title: str) -> None:
    """打印分隔标题"""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_fibonacci() -> None:
    section("1. fibonacci(n) - 斐波那契数列")
    print("计算斐波那契数列第 n 项（n < 0 返回 -1）")
    for n in [0, 1, 2, 5, 10, 20, 30, -1]:
        print(f"  fibonacci({n:>3}) = {m.fibonacci(n)}")


def demo_is_prime() -> None:
    section("2. is_prime(n) - 素数判断")
    print("判断一个整数是否为素数")
    for n in [-5, 0, 1, 2, 3, 4, 9, 17, 97, 100, 101, 997]:
        result = "是素数" if m.is_prime(n) else "非素数"
        print(f"  is_prime({n:>4}) -> {result}")


def demo_vec_operations() -> None:
    section("3. 向量运算: vec_add, vec_scale, vec_dot, vec_l2_norm")

    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [10.0, 20.0, 30.0, 40.0, 50.0]
    print(f"  向量 a = {a}")
    print(f"  向量 b = {b}")
    print()

    c = m.vec_add(a, b)
    print(f"  vec_add(a, b)   = {list(c)}")

    scaled = m.vec_scale(a, 2.5)
    print(f"  vec_scale(a, 2.5) = {[round(x, 2) for x in scaled]}")

    dot = m.vec_dot(a, b)
    print(f"  vec_dot(a, b)   = {dot}")

    norm_a = m.vec_l2_norm(a)
    norm_b = m.vec_l2_norm(b)
    print(f"  vec_l2_norm(a)  = {norm_a:.4f}")
    print(f"  vec_l2_norm(b)  = {norm_b:.4f}")
    print()
    print("  数学恒等式验证: ||v||² == dot(v, v)")
    print(f"    ||a||² = {norm_a**2:.4f}, dot(a,a) = {m.vec_dot(a,a):.4f}")


def demo_vec_stats() -> None:
    section("4. vec_stats(v) - 向量统计 (min, max, mean, stddev)")

    data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0]
    print(f"  输入数据: {data}")
    mn, mx, mean, std = m.vec_stats(data)
    print(f"  最小值  = {mn}")
    print(f"  最大值  = {mx}")
    print(f"  平均值  = {mean:.4f}")
    print(f"  标准差  = {std:.4f}")
    print()

    print("  大数据量性能测试 (100 万元素):")
    big_data = [float(i % 1000) for i in range(1_000_000)]
    t0 = time.perf_counter()
    mn, mx, mean, std = m.vec_stats(big_data)
    t1 = time.perf_counter()
    print(f"    min={mn}, max={mx}, mean={mean:.4f}, std={std:.4f}")
    print(f"    Python 侧耗时: {(t1-t0)*1000:.2f} ms")
    print(f"    提示: 设置 DEMO_FFI_PERF_LOG=1 可查看 C++ 侧逐阶段计时")


def demo_string_functions() -> None:
    section("5. 字符串函数: count_substring, reverse_string")

    text = "hello hello world, hello demo_ffi!"
    sub = "hello"
    print(f"  文本: \"{text}\"")
    print(f"  子串: \"{sub}\"")
    print(f"  count_substring(text, sub) = {m.count_substring(text, sub)}")
    print()

    examples = ["hello", "racecar", "demo_ffi", "", "a"]
    for s in examples:
        rev = m.reverse_string(s)
        print(f"  reverse_string(\"{s}\") = \"{rev}\"")
    print()
    print("  回文验证: reverse(reverse(s)) == s")
    s = "The quick brown fox jumps over the lazy dog"
    double_rev = m.reverse_string(m.reverse_string(s))
    print(f"    原始: \"{s[:40]}...\"")
    print(f"    还原: \"{double_rev[:40]}...\"")
    print(f"    一致: {s == double_rev}")


def demo_gcd_lcm() -> None:
    section("6. gcd(a,b) / lcm(a,b) - 最大公约数 / 最小公倍数")

    pairs = [(12, 8), (7, 5), (100, 75), (0, 5), (-12, 8), (1, 99)]
    print("  验证恒等式: lcm(a,b) * gcd(a,b) == |a * b|")
    print()
    for a, b in pairs:
        g = m.gcd(a, b)
        l = m.lcm(a, b)
        product = abs(a * b)
        check = "✓" if l * g == product else "✗"
        print(f"  gcd({a:>4}, {b:>4}) = {g:>4} | lcm({a:>4}, {b:>4}) = {l:>5} | {check} {l}*{g}={l*g} == |{a*b}|={product}")


def demo_sieve_primes() -> None:
    section("7. sieve_primes(limit) - 埃拉托斯特尼筛法")

    for limit in [10, 30, 100]:
        primes = list(m.sieve_primes(limit))
        print(f"  sieve_primes({limit:>4}) -> {len(primes):>3} 个素数: {primes}")
    print()
    primes_1000 = list(m.sieve_primes(1000))
    print(f"  sieve_primes(1000) -> 共 {len(primes_1000)} 个素数")
    print(f"    前 10 个: {primes_1000[:10]}")
    print(f"    后 10 个: {primes_1000[-10:]}")
    print()
    print("  边界情况:")
    print(f"    sieve_primes(1) = {list(m.sieve_primes(1))}")
    print(f"    sieve_primes(2) = {list(m.sieve_primes(2))}")
    print(f"    sieve_primes(-5) = {list(m.sieve_primes(-5))}")


def demo_sigmoid() -> None:
    section("8. sigmoid(x) / vec_sigmoid(v) - Sigmoid 激活函数")

    print("  标量 sigmoid:")
    for x in [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 100.0, -100.0]:
        s = m.sigmoid(x)
        print(f"    sigmoid({x:>6}) = {s:.10f}  (∈ [0,1]: {'✓' if 0 <= s <= 1 else '✗'})")
    print()
    print("  对称性验证: sigmoid(x) + sigmoid(-x) == 1.0")
    for x in [0.5, 1.0, 2.5]:
        s_sum = m.sigmoid(x) + m.sigmoid(-x)
        print(f"    x={x}: {s_sum:.12f}  {'✓' if abs(s_sum - 1.0) < 1e-12 else '✗'}")
    print()

    vec = [-3.0, -1.0, 0.0, 1.0, 3.0]
    out = list(m.vec_sigmoid(vec))
    print(f"  向量 vec_sigmoid({vec}):")
    print(f"    -> {[round(x, 6) for x in out]}")
    print()
    print("  数值稳定性测试 (极大/极小值):")
    print(f"    sigmoid(500)  = {m.sigmoid(500.0):.12e}")
    print(f"    sigmoid(-500) = {m.sigmoid(-500.0):.12e}")


def demo_binary_search() -> None:
    section("9. binary_search(sorted_arr, target) - 二分查找")

    arr = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0]
    print(f"  有序数组: {arr}")
    print()
    targets = [1.0, 5.0, 13.0, 4.0, 0.0, 100.0]
    for t in targets:
        idx = m.binary_search(arr, t)
        status = f"找到 -> 索引 {idx}" if idx >= 0 else "未找到 -> -1"
        if idx >= 0:
            status += f" (arr[{idx}]={arr[idx]})"
        print(f"    binary_search(..., {t:>5}) -> {status}")
    print()
    print("  大规模数组测试 (10 万元素):")
    big_arr = [float(i) for i in range(100_000)]
    t0 = time.perf_counter()
    idx = m.binary_search(big_arr, 50000.0)
    t1 = time.perf_counter()
    print(f"    查找 50000.0 -> 索引 {idx}")
    print(f"    Python 侧耗时: {(t1-t0)*1000:.4f} ms")


def demo_pipeline() -> None:
    section("10. 组合 Pipeline 示例: 简单数据预处理工作流")

    print("  模拟数据预处理流程:")
    print("    原始数据 -> 缩放 -> 统计 -> Sigmoid 激活 -> 二分类")
    print()

    raw = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    print(f"  原始数据:       {raw}")

    scaled = list(m.vec_scale(raw, 0.5))
    print(f"  缩放 (×0.5):   {[round(x, 2) for x in scaled]}")

    mn, mx, mean, std = m.vec_stats(scaled)
    print(f"  统计:           min={mn}, max={mx}, mean={mean:.2f}, std={std:.2f}")

    normalized = []
    for x in scaled:
        normalized.append((x - mean) / (std if std > 0 else 1.0))
    print(f"  Z-score 标准化: {[round(x, 2) for x in normalized]}")

    activated = list(m.vec_sigmoid(normalized))
    print(f"  Sigmoid 激活:   {[round(x, 4) for x in activated]}")

    classified = [1 if p >= 0.5 else 0 for p in activated]
    print(f"  二分类 (>0.5):  {classified}")


def main() -> None:
    perf_log = os.environ.get("DEMO_FFI_PERF_LOG")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       demo_ffi.math_ops 模块 - C++ FFI 函数演示             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    if perf_log:
        print(f"\n[信息] 性能日志已开启 (DEMO_FFI_PERF_LOG={perf_log})")
        print("       C++ 侧详细计时信息将输出到 stderr\n")

    demo_fibonacci()
    demo_is_prime()
    demo_vec_operations()
    demo_vec_stats()
    demo_string_functions()
    demo_gcd_lcm()
    demo_sieve_primes()
    demo_sigmoid()
    demo_binary_search()
    demo_pipeline()

    section("演示完成")
    print("  所有 15 个 C++ FFI 数学函数已演示完毕。")
    print()
    print("  可用函数清单:")
    funcs = [
        "fibonacci(n)", "is_prime(n)",
        "vec_add(a, b)", "vec_scale(v, factor)",
        "vec_dot(a, b)", "vec_l2_norm(v)",
        "vec_stats(v)",
        "count_substring(text, sub)", "reverse_string(s)",
        "gcd(a, b)", "lcm(a, b)",
        "sieve_primes(limit)",
        "sigmoid(x)", "vec_sigmoid(v)",
        "binary_search(sorted_arr, target)",
    ]
    for f in funcs:
        print(f"    • {f}")
    print()


if __name__ == "__main__":
    main()
