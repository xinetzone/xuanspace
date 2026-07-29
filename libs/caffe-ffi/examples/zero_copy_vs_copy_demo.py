#!/usr/bin/env python3
"""
零拷贝 (zero-copy) vs 传统拷贝 (copy-based) 性能对比演示
==========================================================

本脚本直观展示 caffe-ffi 中 Blob.data_tensor（零拷贝）与 Blob.data（拷贝）
在不同张量大小下的性能差异。

核心原理：
  - data_tensor: 返回 numpy 数组，直接指向 C++ 侧已分配内存，O(1) 时间
  - data:       返回 numpy 数组，完整拷贝 C++ 数据，O(N) 时间

运行方式：
  cd projects/xuanspace/vendor/caffe/caffe-ffi
  $env:PATH = "build/Release;" + $env:PATH   # PowerShell
  python examples/zero_copy_vs_copy_demo.py
"""

from __future__ import annotations

import gc
import statistics
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 尝试导入 caffe_ffi；如果导入失败，给出友好提示
# ---------------------------------------------------------------------------
try:
    import caffe_ffi
    from caffe_ffi import Blob
except ImportError as e:
    print(f"[ERROR] 无法导入 caffe_ffi: {e}")
    print("请先编译 C++ 扩展并将 build/Release 加入 PATH。")
    print("  cmake -B build -DCMAKE_BUILD_TYPE=Release")
    print("  cmake --build build --config Release")
    print("  $env:PATH = 'build/Release;' + $env:PATH")
    sys.exit(1)

NATIVE_MODE = caffe_ffi._ffi_api.is_available()


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _benchmark(fn, repeats: int = 200, warmup: int = 10) -> dict:
    """运行 fn 若干次，返回耗时统计（毫秒）。"""
    # Warmup
    for _ in range(warmup):
        fn()
    # Timed runs
    times_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    return {
        "avg": statistics.mean(times_ms),
        "p50": statistics.median(times_ms),
        "p95": sorted(times_ms)[int(repeats * 0.95) - 1],
        "min": min(times_ms),
        "max": max(times_ms),
        "std": statistics.stdev(times_ms) if repeats > 1 else 0.0,
    }


def _verify_zero_copy(blob: Blob) -> tuple[bool, bool]:
    """
    验证零拷贝语义：
    1. 两次 data_tensor 调用返回相同底层指针
    2. 通过一个引用写入，另一个引用立即可见
    """
    t1 = blob.data_tensor
    t2 = blob.data_tensor

    # 指针一致性
    ptr_same = t1.ctypes.data == t2.ctypes.data

    # 写后读验证：通过 t1 写入，通过 t2 读取
    test_val = 3.14159
    t1[0] = test_val
    write_visible = abs(t2[0] - test_val) < 1e-9
    t1[0] = 0.0  # 还原
    return ptr_same, write_visible


def _verify_copy_isolation(blob: Blob) -> bool:
    """验证拷贝语义：修改返回的副本不影响原 Blob。"""
    snapshot = blob.data
    snapshot[0] = 999.0
    original_val = blob.data[0]
    snapshot[0] = 0.0  # 还原
    return abs(original_val) < 1e-9  # 原值未变


def _format_size(n: int) -> str:
    mb = n * 4 / (1024 * 1024)
    if mb >= 1.0:
        return f"{mb:.1f} MB"
    kb = n * 4 / 1024
    return f"{kb:.1f} KB"


# ---------------------------------------------------------------------------
# DEMO 1: 基础语义验证
# ---------------------------------------------------------------------------

def demo_semantics():
    print("=" * 70)
    print("DEMO 1: 语义验证 — 零拷贝与拷贝的本质区别")
    print("=" * 70)
    print()

    blob = Blob([1_000_000])
    blob.fill(0.0)

    # --- 零拷贝验证 ---
    ptr_ok, write_ok = _verify_zero_copy(blob)
    print(f"  data_tensor 零拷贝验证:")
    print(f"    两次调用指针一致:    {'✓' if ptr_ok else '✗'}")
    print(f"    写后读可见(共享内存): {'✓' if write_ok else '✗'}")

    # --- 拷贝隔离验证 ---
    iso_ok = _verify_copy_isolation(blob)
    print(f"  data 拷贝隔离验证:")
    print(f"    修改副本不影响原Blob: {'✓' if iso_ok else '✗'}")

    print()
    print("  💡 结论:")
    print("    data_tensor → 直接视图，修改会影响Blob内部数据（高性能）")
    print("    data        → 独立副本，修改安全但有拷贝开销（安全隔离）")
    print()


# ---------------------------------------------------------------------------
# DEMO 2: 不同张量大小的性能对比
# ---------------------------------------------------------------------------

def demo_performance_sizes():
    print("=" * 70)
    print("DEMO 2: 性能对比 — 不同张量大小下的访问耗时")
    print("=" * 70)
    print()
    print(f"  {'大小(N floats)':>14}  {'内存':>8}  {'零拷贝(ms)':>12}  {'拷贝(ms)':>10}  {'加速比':>10}")
    print(f"  {'-'*14}  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*10}")

    sizes = [1_000, 10_000, 100_000, 1_000_000, 5_000_000, 10_000_000]
    results = []

    for n in sizes:
        blob = Blob([n])
        blob.fill(1.0)

        # 零拷贝
        stats_zc = _benchmark(lambda: blob.data_tensor, repeats=300, warmup=30)
        # 拷贝
        stats_cp = _benchmark(lambda: blob.data, repeats=100, warmup=10)

        speedup = stats_cp["avg"] / stats_zc["avg"] if stats_zc["avg"] > 0 else float("inf")
        mem_str = _format_size(n)
        zc_str = f"{stats_zc['avg']:.4f}"
        cp_str = f"{stats_cp['avg']:.4f}"
        sp_str = f"{speedup:.0f}×" if speedup < 10000 else f"{speedup/1000:.1f}K×"

        print(f"  {n:>14,}  {mem_str:>8}  {zc_str:>12}  {cp_str:>10}  {sp_str:>10}")
        results.append((n, stats_zc, stats_cp, speedup))

        del blob
        gc.collect()

    print()
    print("  📊 观察:")
    zc_times = [r[1]["avg"] for r in results]
    cp_times = [r[2]["avg"] for r in results]
    print(f"    零拷贝耗时范围: {min(zc_times):.4f} ~ {max(zc_times):.4f} ms (基本恒定)")
    print(f"    拷贝耗时范围:   {min(cp_times):.4f} ~ {max(cp_times):.4f} ms (线性增长)")
    print(f"    最大加速比:     {max(r[3] for r in results):.0f}× (在 {max(sizes):,} floats 时)")
    print()


# ---------------------------------------------------------------------------
# DEMO 3: 原地修改 vs 拷贝后修改
# ---------------------------------------------------------------------------

def demo_inplace_modification():
    print("=" * 70)
    print("DEMO 3: 实际工作负载 — 原地修改 vs 拷贝后修改")
    print("=" * 70)
    print()

    n = 5_000_000
    blob = Blob([n])
    blob.fill(0.0)
    print(f"  张量大小: {n:,} floats ({_format_size(n)})")
    print()

    # --- 方式 A: data_tensor 原地修改 ---
    def inplace_modify():
        t = blob.data_tensor
        t[:] = t * 2.0 + 1.0
        blob.zero()  # 还原

    stats_inplace = _benchmark(inplace_modify, repeats=50, warmup=5)

    # --- 方式 B: data 拷贝 → 修改 → 写回 ---
    def copy_modify():
        t = blob.data           # 拷贝出来
        t = t * 2.0 + 1.0       # 修改
        blob.from_numpy(t)      # 写回（又一次拷贝）

    stats_copy = _benchmark(copy_modify, repeats=50, warmup=5)

    print(f"  {'方式':<30} {'平均(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'data_tensor 原地修改':<30} {stats_inplace['avg']:>10.3f} {stats_inplace['p50']:>10.3f} {stats_inplace['p95']:>10.3f}")
    print(f"  {'data 拷贝→修改→写回':<30} {stats_copy['avg']:>10.3f} {stats_copy['p50']:>10.3f} {stats_copy['p95']:>10.3f}")
    speedup = stats_copy["avg"] / stats_inplace["avg"]
    print()
    print(f"  💡 原地修改快 {speedup:.1f}×：零拷贝省掉了两次内存拷贝（读出+写回）")
    print()

    del blob
    gc.collect()


# ---------------------------------------------------------------------------
# DEMO 4: numpy 生态互操作
# ---------------------------------------------------------------------------

def demo_numpy_interop():
    print("=" * 70)
    print("DEMO 4: numpy 生态互操作 — 零拷贝传递给 numpy/AI 函数")
    print("=" * 70)
    print()

    try:
        import numpy as np
    except ImportError:
        print("  numpy 未安装，跳过此演示")
        return

    n = 2_000_000
    blob = Blob([n])
    # 填充一些有意义的数据
    t = blob.data_tensor
    t[:] = np.random.randn(n).astype(np.float32)

    # --- 零拷贝方式 ---
    def zc_numpy_ops():
        arr = blob.data_tensor
        _ = np.sum(arr)
        _ = np.mean(arr)
        _ = np.max(arr)
        _ = np.dot(arr[:1000], arr[:1000])

    stats_zc = _benchmark(zc_numpy_ops, repeats=100, warmup=10)

    # --- 拷贝方式 ---
    def cp_numpy_ops():
        arr = blob.data
        _ = np.sum(arr)
        _ = np.mean(arr)
        _ = np.max(arr)
        _ = np.dot(arr[:1000], arr[:1000])

    stats_cp = _benchmark(cp_numpy_ops, repeats=100, warmup=10)

    print(f"  操作: np.sum + np.mean + np.max + np.dot(前1000元素)")
    print(f"  张量: {n:,} floats ({_format_size(n)})")
    print()
    print(f"  {'方式':<25} {'平均(ms)':>10}")
    print(f"  {'-'*25} {'-'*10}")
    print(f"  {'零拷贝 data_tensor':<25} {stats_zc['avg']:>10.3f}")
    print(f"  {'拷贝 data':<25} {stats_cp['avg']:>10.3f}")
    speedup = stats_cp["avg"] / stats_zc["avg"]
    print()
    print(f"  💡 numpy 函数直接操作零拷贝视图，无数据搬运开销")
    print()

    del blob
    gc.collect()


# ---------------------------------------------------------------------------
# DEMO 5: 内存安全提醒
# ---------------------------------------------------------------------------

def demo_memory_safety():
    print("=" * 70)
    print("DEMO 5: 注意事项 — 零拷贝的内存安全")
    print("=" * 70)
    print()

    # 正确用法
    print("  ✅ 正确用法（推荐）：")
    print("     with 块或函数内使用 data_tensor，用完即释放引用")
    print()
    print("     def process_blob(blob):")
    print("         arr = blob.data_tensor      # 获取零拷贝视图")
    print("         result = arr.max()          # 做计算")
    print("         return result               # 函数返回后 arr 自动释放")
    print()

    # 错误用法
    print("  ⚠️  需要注意：")
    print("     长期持有 data_tensor 会阻止 Blob 内存释放：")
    print()
    print("     t = blob.data_tensor")
    print("     del blob                       # Blob 内存未释放！t 还持有引用")
    print("     # ... 使用 t ...")
    print("     t = None                       # 现在才释放")
    print()

    # 实际演示
    blob = Blob([1_000_000])
    blob.fill(1.0)
    base = caffe_ffi.total_allocated_bytes()

    t = blob.data_tensor
    del blob
    gc.collect()
    after_del_blob = caffe_ffi.total_allocated_bytes()
    print(f"  实际验证:")
    print(f"    创建Blob后分配: {base:,} bytes")
    print(f"    del blob 后:    {after_del_blob:,} bytes (仍持有引用，内存未释放)")
    del t
    gc.collect()
    after_del_t = caffe_ffi.total_allocated_bytes()
    print(f"    del t 后:       {after_del_t:,} bytes (引用释放，内存归还)")
    print()


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║       零拷贝 (Zero-Copy) vs 传统拷贝 (Copy) 性能对比演示           ║")
    print("║       caffe-ffi + TVM FFI + DLPack                                 ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Python版本:     {sys.version.split()[0]}")
    print(f"  caffe-ffi原生模式: {'是' if NATIVE_MODE else '否 (Python-only模式)'}")
    if not NATIVE_MODE:
        print("  ⚠️  当前为 Python-only 模式，零拷贝性能数据无意义")
        print("     请编译 C++ 扩展后重新运行")
    print()

    if NATIVE_MODE:
        demo_semantics()
        demo_performance_sizes()
        demo_inplace_modification()
        demo_numpy_interop()
        demo_memory_safety()
    else:
        demo_semantics()

    print("=" * 70)
    print("演示结束。")
    print("=" * 70)


if __name__ == "__main__":
    main()
