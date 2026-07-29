"""验证 tracked_blob 上下文管理器和 BlobRef 在正常/异常场景下的日志输出。

关键认知：Python的`with ... as b`在调用方作用域创建变量b，它持有Blob引用。
在with块退出时b仍然存活，因此C++析构不会立即触发（这是Python正常行为）。
要验证内存释放，需在del b或函数返回（b超出作用域）后调用mem_check()。
"""
import gc
import os
import sys
import weakref

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from caffe_ffi.tools import (
    setup_debug, setup_quiet,
    tracked_blob, BlobRef, blob_snapshot, mem_check,
)
import caffe_ffi
from caffe_ffi import Blob

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        print(f"  ❌ {name} {detail}")


def header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# Test 1: tracked_blob 正常退出（函数作用域，b随函数返回释放）
# ============================================================
def test_normal_exit():
    header("TEST 1: tracked_blob 正常退出（函数作用域）")
    setup_debug()
    with tracked_blob([10, 10], "normal_exit") as b:
        b.data_tensor[:] = 1.0
        val = float(b.data_tensor[0, 0])
        print(f"  inside with: data[0,0]={val}")
    # with退出时，b仍在函数作用域内存活，tracked_blob报告 NOTE
    # 但函数返回后b超出作用域，C++析构触发
    setup_quiet()


# ============================================================
# Test 2: tracked_blob 异常退出（有catch）
# ============================================================
def test_exception_exit():
    header("TEST 2: tracked_blob 异常退出（catch后函数返回释放）")
    setup_debug()
    try:
        with tracked_blob([5, 5], "exception_exit") as b:
            b.data_tensor[:] = 7.0
            print("  inside with: raising RuntimeError...")
            raise RuntimeError("模拟异常")
    except RuntimeError as e:
        print(f"  outer catch: caught RuntimeError: {e}")
    # 异常被catch后，函数正常返回，b超出作用域→C++析构
    setup_quiet()


# ============================================================
# Test 3: 显式del b验证立即释放
# ============================================================
def test_explicit_del():
    header("TEST 3: 显式del b后C++析构日志立即出现")
    setup_debug()
    mem_before = caffe_ffi.total_allocated_bytes()
    with tracked_blob([6, 6], "explicit_del", verbose=False) as b:
        b.data_tensor[:] = 3.0
    # with退出后显式del b
    print("  calling del b...")
    del b
    gc.collect()
    mem_after = caffe_ffi.total_allocated_bytes()
    check("del b+gc后内存回到基线", mem_after == mem_before,
          f"before={mem_before}, after={mem_after}")
    print("  （上方C++日志中应出现 ~Blob() 打印对应指针）")
    setup_quiet()


# ============================================================
# Test 4: BlobRef weakref回调
# ============================================================
def test_blobref_weakref():
    header("TEST 4: BlobRef 支持weakref和销毁回调")
    setup_debug()
    callback_called = []

    def make_callback(ptr_val, label_val):
        def on_destroy(ref):
            msg = f"[CALLBACK] BlobRef destroyed, data_ptr was 0x{ptr_val:016x}, label={label_val!r}"
            callback_called.append(msg)
            print(msg)
        return on_destroy

    br = BlobRef([3, 3], label="weakref_test")
    print(f"  created: {br}")
    on_destroy = make_callback(br._data_ptr, br._label)
    ref = weakref.ref(br, on_destroy)
    check("weakref指向对象", ref() is br)
    print("  calling del br + gc.collect()...")
    del br
    gc.collect()
    check("del+gc后weakref失效", ref() is None)
    check("回调被调用", len(callback_called) == 1)
    setup_quiet()


# ============================================================
# Test 5: tracked_blob嵌套+内层异常
# ============================================================
def test_nested():
    header("TEST 5: tracked_blob嵌套，内层异常被catch")
    setup_debug()
    with tracked_blob([4, 4], "outer") as outer:
        outer.data_tensor[:] = 2.0
        try:
            with tracked_blob([3, 3], "inner") as inner:
                inner.data_tensor[:] = 3.0
                raise ValueError("inner error")
        except ValueError:
            print("  inner exception caught")
            del inner  # 显式释放内层
            gc.collect()
        print(f"  outer still accessible: {float(outer.data_tensor[0,0])}")
    setup_quiet()


# ============================================================
# Test 6: mem_check / blob_snapshot 工具函数
# ============================================================
def test_snapshot_tools():
    header("TEST 6: blob_snapshot / mem_check 工具")
    mem_check("before_create")
    b1 = Blob([8, 8])
    b2 = Blob([8, 8])
    s1 = blob_snapshot("after_two_blobs")
    check("两个Blob共1024字节", s1 == 1024, f"got {s1}")
    del b2
    gc.collect()
    s2 = blob_snapshot("after_del_one")
    check("删除一个后剩512字节", s2 == 512, f"got {s2}")
    del b1
    gc.collect()
    mem_check("final")


# ============================================================
# 主入口
# ============================================================
def main():
    print("=" * 60)
    print("  BlobRef + tracked_blob 验证测试")
    print("=" * 60)

    gc.collect()
    baseline = caffe_ffi.total_allocated_bytes()
    assert baseline == 0, f"Starting with non-zero memory: {baseline}"
    print(f"\n[SETUP] Initial memory = {baseline} bytes")

    tests = [
        test_normal_exit,
        test_exception_exit,
        test_explicit_del,
        test_blobref_weakref,
        test_nested,
        test_snapshot_tools,
    ]

    for t in tests:
        t()
        gc.collect()

    gc.collect()
    final_mem = caffe_ffi.total_allocated_bytes()

    print(f"\n{'='*60}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    if final_mem == 0:
        print(f"  Final memory: 0 bytes ✅ (no leaks)")
    else:
        print(f"  Final memory: {final_mem} bytes ❌ (LEAK DETECTED)")
    print(f"{'='*60}")

    sys.exit(0 if FAILED == 0 and final_mem == 0 else 1)


if __name__ == "__main__":
    main()
