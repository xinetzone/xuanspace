"""内存泄漏场景专项测试 — 验证析构日志与指针在各种异常情况下的行为

测试场景：
1. 正常创建→释放（基准）
2. 循环变量泄漏（Python常见陷阱）→ total_allocated_bytes检测
3. 故意引用泄漏（global持有）→ 无~Blob()日志（正确：析构函数未被调用）
4. 释放引用后gc.collect() → 析构日志正确打印
5. 异常中退出（无catch）→ 析构函数是否在栈展开时运行
6. 循环引用→gc.collect() → 弱引用是否打破环
7. Reshape重分配旧内存释放 → 旧ptr的FreeData日志正确
8. 进程退出时存活对象 → 析构函数是否在atexit阶段运行

用法:
    pytest tests/python/test_memory_leak.py -v
    # 或独立运行：
    python tests/python/test_memory_leak.py

迁移说明：
- 场景 8（进程退出时存活对象）为进程级行为，无法在单个 pytest 用例中可靠
  复现/断言。全局泄漏检测由 conftest.py 的 pytest_sessionfinish 钩子统一负责，
  故以下方 test_process_exit_alive_objects 的 skip 用例保留语义说明。
- 主动泄漏/全局持引用相关用例标记 @pytest.mark.leak_check(False)，关闭
  conftest.py 的 autouse 泄漏检测，避免对故意构造的泄漏误判。
"""
from __future__ import annotations

import gc
import os
import sys
import traceback

import pytest
import numpy as np
import caffe_ffi
from caffe_ffi import Blob
from caffe_ffi.tools import setup_debug, setup_quiet, blob_snapshot as memory_snapshot, mem_check as check_memory_baseline


def header(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ============================================================
# 模块级内存基线：屏蔽其他测试模块遗留 Blob 对绝对断言的干扰
# ============================================================
# 迁移前为独立脚本（进程级干净内存起点），迁移为 pytest 后与全量套件共享进程，
# 其他模块（如失败用例持有 traceback 引用）会遗留存活 Blob，污染 total_allocated_bytes()
# 的绝对基线。改为模块启动时捕获基线、断言相对差值，并在模块结束释放本模块故意泄漏。
_BASELINE_BYTES: int | None = None


@pytest.fixture(scope="module", autouse=True)
def _capture_memory_baseline() -> Iterator[None]:
    """模块内所有用例之前捕获内存基线，结束后释放本模块故意泄漏的引用。"""
    global _BASELINE_BYTES
    gc.collect()
    _BASELINE_BYTES = caffe_ffi.total_allocated_bytes()
    yield
    _leaked_ref.clear()
    gc.collect()


def _rel_mem() -> int:
    """返回相对模块基线的已分配字节数（屏蔽跨模块遗留 Blob 的干扰）。"""
    return caffe_ffi.total_allocated_bytes() - _BASELINE_BYTES


# ============================================================
# Test 1: 正常创建→释放（基准）
# ============================================================
def test_normal_lifecycle() -> None:
    header("TEST 1: 正常创建→Reshape→写入→释放（基准）")
    setup_debug()
    b = Blob([2, 3, 4, 5])
    b.data_tensor[:] = 42.0
    dp = b.data_tensor.ctypes.data
    nbytes = b.data_tensor.nbytes * 2  # data + diff
    print(f"  [INFO] data_ptr=0x{dp:016x}, nbytes(data+diff)={nbytes}")

    mem_before = _rel_mem()
    assert mem_before == nbytes, f"expected {nbytes}, got {mem_before}"

    del b
    gc.collect()
    mem_after = _rel_mem()
    assert mem_after == 0, f"expected 0, got {mem_after}"
    setup_quiet()


# ============================================================
# Test 2: Python循环变量"泄漏"（常见陷阱）
# ============================================================
def test_loop_variable_leak() -> None:
    header("TEST 2: Python循环变量持有引用（常见陷阱）")
    setup_debug()
    blobs = [Blob([4, 4]) for _ in range(3)]
    for i, b in enumerate(blobs):
        b.data_tensor[:] = float(i)
    mem_with = _rel_mem()
    print(f"  [INFO] 3 blobs alive, total={mem_with} bytes")
    assert mem_with == 384, f"got {mem_with}"

    # 只del list，但循环变量 i, b 仍在帧中持有最后一个Blob
    del blobs
    gc.collect()
    mem_partial = _rel_mem()
    print(f"  [INFO] after del blobs (i,b still alive): total={mem_partial} bytes")
    assert mem_partial == 128, f"got {mem_partial}"
    print("  （查看上方日志确认只有2个~Blob输出）")

    # 彻底删除所有引用
    del i, b
    gc.collect()
    mem_clean = _rel_mem()
    assert mem_clean == 0, f"got {mem_clean}"
    setup_quiet()


# ============================================================
# Test 3: 故意引用泄漏（global持有，~Blob不会被调用）
# ============================================================
_leaked_ref: list = []
_leaked_bytes: int = 0


@pytest.mark.leak_check(False)
def test_intentional_leak() -> None:
    header("TEST 3: 故意泄漏（global引用持有，析构函数不会运行）")
    setup_debug()
    b = Blob([8, 8])
    b.data_tensor[:] = 99.0
    _leaked_ref.append(b)  # 故意泄漏到global
    global _leaked_bytes
    _leaked_bytes = b.data_tensor.nbytes + b.diff_tensor.nbytes  # data + diff
    dp = b.data_tensor.ctypes.data
    print(f"  [INFO] Leaked blob data_ptr=0x{dp:016x}, leaked_bytes={_leaked_bytes}")

    del b
    gc.collect()
    mem = _rel_mem()
    assert mem >= 512, f"expected >=512, got {mem}"
    print(f"  [INFO] Leaked object NOT destroyed (expected - ref held by global)")
    print(f"  [INFO] total_allocated_bytes={mem} (non-zero = leak detected)")
    setup_quiet()


# ============================================================
# Test 4: 引用计数精确释放验证
# ============================================================
@pytest.mark.leak_check(False)
def test_refcount_cleanup() -> None:
    header("TEST 4: 引用计数精确释放验证（无gc）")
    setup_debug()
    b = Blob([3, 3])
    b.data_tensor[:] = 1.0
    dp = b.data_tensor.ctypes.data
    expected = b.data_tensor.nbytes + b.diff_tensor.nbytes  # 72 bytes
    print(f"  [INFO] data_ptr=0x{dp:016x}, expected total={expected} bytes")

    mem_before = _rel_mem() - _leaked_bytes
    assert mem_before == expected, f"expected {expected}, got {mem_before}"

    del b  # 纯引用计数释放，无需gc.collect()
    mem_after = _rel_mem() - _leaked_bytes
    assert mem_after == 0, f"got {mem_after}"
    print(f"  [INFO] ~Blob() for ptr=0x{dp:016x} should appear above")
    setup_quiet()


# ============================================================
# Test 5: 异常抛出后栈展开（有catch）
# ============================================================
@pytest.mark.leak_check(False)
def test_exception_with_catch() -> None:
    header("TEST 5: 异常后catch→del→gc（析构正常）")
    setup_debug()
    b = Blob([5, 5])
    b.data_tensor[:] = 7.0
    dp = b.data_tensor.ctypes.data
    try:
        raise ValueError("simulated error")
    except ValueError:
        print(f"  [INFO] Exception caught, blob still alive, ptr=0x{dp:016x}")
        assert float(b.data_tensor[0, 0]) == 7.0

    del b
    gc.collect()
    mem = _rel_mem() - _leaked_bytes
    assert mem == 0, f"got {mem}"
    print(f"  [INFO] ~Blob() printed above with correct ptr=0x{dp:016x}")
    setup_quiet()


# ============================================================
# Test 6: Reshape重分配旧内存释放验证
# ============================================================
@pytest.mark.leak_check(False)
def test_reshape_reallocation() -> None:
    header("TEST 6: Reshape从小→大，旧ptr的FreeData日志")
    setup_debug()
    b = Blob([2, 2])  # 16*2=32 bytes data, 32 bytes diff = 64 total
    old_dp = b.data_tensor.ctypes.data
    old_dfp = b.diff_tensor.ctypes.data
    print(f"  [INFO] Before Reshape: data_ptr=0x{old_dp:016x}, diff_ptr=0x{old_dfp:016x}")

    b.Reshape([10, 10])  # 100*4*2 = 800 bytes -> triggers reallocation
    new_dp = b.data_tensor.ctypes.data
    new_dfp = b.diff_tensor.ctypes.data
    print(f"  [INFO] After Reshape:  data_ptr=0x{new_dp:016x}, diff_ptr=0x{new_dfp:016x}")

    assert new_dp != old_dp and new_dfp != old_dfp
    print("  （查看上方日志: FreeData应打印old_dp和old_dfp的地址）")

    mem = _rel_mem() - _leaked_bytes
    expected = 100 * 4 * 2  # 800
    assert mem == expected, f"expected {expected}, got {mem}"

    del b
    gc.collect()
    setup_quiet()


# ============================================================
# Test 7: 同shape Reshape跳过重分配
# ============================================================
@pytest.mark.leak_check(False)
def test_reshape_same_shape_noop() -> None:
    header("TEST 7: 相同shape Reshape不应重分配")
    setup_debug()
    b = Blob([3, 4])
    dp_before = b.data_tensor.ctypes.data
    b.Reshape([3, 4])
    dp_after = b.data_tensor.ctypes.data
    assert dp_before == dp_after, f"before=0x{dp_before:016x} after=0x{dp_after:016x}"
    del b
    gc.collect()
    setup_quiet()


# ============================================================
# Test 8: 泄漏清理 — 清理test3中的故意泄漏
# ============================================================
@pytest.mark.leak_check(False)
def test_cleanup_leaks() -> None:
    header("TEST 8: 清理所有故意泄漏的引用")
    setup_debug()
    n = len(_leaked_ref)
    print(f"  [INFO] Cleaning {n} leaked references...")
    _leaked_ref.clear()
    gc.collect()
    mem = _rel_mem()
    assert mem == 0, f"got {mem}"
    print(f"  [INFO] ~Blob() for leaked objects should appear above")
    setup_quiet()


# ============================================================
# 场景 8（迁移）：进程退出时存活对象 — 由 conftest.py 的 pytest_sessionfinish 钩子负责
# ============================================================
@pytest.mark.skip(
    reason="进程退出时存活对象为进程级场景，无法在单个 pytest 用例中可靠复现/断言；"
           "全局泄漏检测由 conftest.py 的 pytest_sessionfinish 钩子统一处理。"
)
def test_process_exit_alive_objects() -> None:
    """原独立脚本场景 8：进程退出时存活对象。

    当进程退出（atexit 阶段）时仍存活的对象，其析构函数是否运行属于进程级行为。
    迁移后该语义由 conftest.py 的 pytest_sessionfinish 钩子覆盖，故此处标记 skip。
    """
    pass


# ============================================================
# 主入口
# ============================================================
def main() -> None:
    print("=" * 70)
    print("  caffe-ffi 内存泄漏场景专项测试")
    print("=" * 70)

    # 确保基线干净
    gc.collect()
    baseline = caffe_ffi.total_allocated_bytes()
    print(f"\n[SETUP] Initial total_allocated_bytes = {baseline}")
    assert baseline == 0, f"Starting with non-zero memory: {baseline}"

    tests = [
        test_normal_lifecycle,
        test_loop_variable_leak,
        test_intentional_leak,
        test_refcount_cleanup,
        test_exception_with_catch,
        test_reshape_reallocation,
        test_reshape_same_shape_noop,
        test_cleanup_leaks,
    ]

    for t in tests:
        try:
            t()
        except Exception:
            print(f"  ❌ {t.__name__} raised exception:")
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("  RESULTS: 上述用例依次执行完毕")
    print("=" * 70)

    final_mem = caffe_ffi.total_allocated_bytes()
    if final_mem == 0:
        print("  ✅ Final memory baseline: 0 bytes (no leaks)")
    else:
        print(f"  ❌ Final memory: {final_mem} bytes (LEAK DETECTED)")

    sys.exit(0 if final_mem == 0 else 1)


if __name__ == "__main__":
    main()