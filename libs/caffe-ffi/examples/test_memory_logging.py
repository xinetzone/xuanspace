"""
Memory logging verification script for caffe-ffi Blob.
Tests data loading, Reshape, and destruction scenarios to verify
that C++ and Python memory logs are correctly printed.

Usage:
    python examples/test_memory_logging.py
"""
import sys
import os
import gc
import logging
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import caffe_ffi
from caffe_ffi import Blob, set_log_level, get_log_level
from caffe_ffi import LOG_LEVEL_TRACE, LOG_LEVEL_DEBUG, LOG_LEVEL_INFO, LOG_LEVEL_WARN


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='[PY %(levelname)s] %(name)s: %(message)s',
        stream=sys.stdout,
    )
    set_log_level(LOG_LEVEL_DEBUG)
    print(f"[SETUP] C++ log level set to DEBUG ({get_log_level()})")
    print(f"[SETUP] FFI available: {caffe_ffi._ffi_api.is_available()}")
    print(f"[SETUP] lib_path: {caffe_ffi._ffi_api.lib_path()}")
    print("=" * 80)


def test_1_simple_create_and_destroy():
    print("\n" + "=" * 80)
    print("TEST 1: Simple Blob() default constructor -> destruction")
    print("=" * 80)
    b = Blob()
    print(f"[TEST1] Created default Blob, shape={b.shape}")
    dt = b.data_tensor
    print(f"[TEST1] data_tensor shape={dt.shape}, ptr=0x{dt.ctypes.data:016x}")
    del b
    gc.collect()
    print("[TEST1] Blob should be destroyed - check for ~Blob() log above")


def test_2_create_with_shape_and_load():
    print("\n" + "=" * 80)
    print("TEST 2: Blob([2,3,4,5]) -> data load -> zero-copy access")
    print("=" * 80)
    b = Blob([2, 3, 4, 5])
    print(f"[TEST2] Blob shape={b.shape}, count={b.count()}")
    data = np.random.randn(2, 3, 4, 5).astype(np.float32)
    b.data_tensor[:] = data
    readback = b.data_tensor
    max_diff = np.max(np.abs(readback - data))
    print(f"[TEST2] Max diff after write/read: {max_diff:.8f}")
    assert max_diff < 1e-6, "Zero-copy write/read failed!"
    print("[TEST2] Zero-copy verification PASSED")
    del b
    gc.collect()
    print("[TEST2] Blob destroyed")


def test_3_reshape_reallocation():
    print("\n" + "=" * 80)
    print("TEST 3: Reshape triggering reallocation (small -> large -> same -> small)")
    print("=" * 80)
    b = Blob([2, 3])
    ptr1 = b.data_tensor.ctypes.data
    print(f"[TEST3] Initial shape={b.shape}, data_ptr=0x{ptr1:016x}")

    print("[TEST3] Reshaping to [4,5,6] (larger, should REALLOCATE)...")
    b.Reshape([4, 5, 6])
    ptr2 = b.data_tensor.ctypes.data
    print(f"[TEST3] New shape={b.shape}, data_ptr=0x{ptr2:016x}")
    print(f"[TEST3] Pointer changed: {ptr1 != ptr2} (expected: True)")

    print("[TEST3] Reshaping to [4,5,6] (same shape, should SKIP)...")
    b.Reshape([4, 5, 6])
    ptr3 = b.data_tensor.ctypes.data
    print(f"[TEST3] After same-shape reshape: data_ptr=0x{ptr3:016x}")
    print(f"[TEST3] Pointer unchanged: {ptr2 == ptr3} (expected: True)")

    print("[TEST3] Reshaping to [10] (different, should REALLOCATE)...")
    b.Reshape([10])
    ptr4 = b.data_tensor.ctypes.data
    print(f"[TEST3] Final shape={b.shape}, data_ptr=0x{ptr4:016x}")

    del b
    gc.collect()
    print("[TEST3] Blob destroyed")


def test_4_fill_zero_update():
    print("\n" + "=" * 80)
    print("TEST 4: fill(3.14) -> zero() -> diff=1.0 -> Update()")
    print("=" * 80)
    b = Blob([3, 4])
    b.fill(3.14)
    val = float(b.data_tensor[0, 0])
    print(f"[TEST4] After fill(3.14): data[0,0]={val:.4f}")
    assert abs(val - 3.14) < 0.01
    b.zero()
    val = float(b.data_tensor[0, 0])
    print(f"[TEST4] After zero(): data[0,0]={val:.4f}")
    assert abs(val) < 0.001
    b.diff_tensor[0, 0] = 1.0
    b.Update()
    val = float(b.data_tensor[0, 0])
    print(f"[TEST4] After Update: data[0,0]={val:.4f} (expected: -1.0)")
    assert abs(val + 1.0) < 0.001
    del b
    gc.collect()
    print("[TEST4] Blob destroyed")


def test_5_from_numpy_to_numpy():
    print("\n" + "=" * 80)
    print("TEST 5: from_numpy() -> to_numpy() roundtrip")
    print("=" * 80)
    b = Blob()
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b.from_numpy(arr)
    print(f"[TEST5] After from_numpy: shape={b.shape}")
    out = b.to_numpy()
    print(f"[TEST5] to_numpy shape={out.shape}, dtype={out.dtype}")
    assert np.allclose(out, arr), "Roundtrip failed!"
    print("[TEST5] Roundtrip PASSED")
    del b
    gc.collect()
    print("[TEST5] Blob destroyed")


def test_6_copy_from():
    print("\n" + "=" * 80)
    print("TEST 6: copy_from() between Blobs")
    print("=" * 80)
    b1 = Blob([2, 3])
    b1.data_tensor[:] = 42.0
    b2 = Blob([2, 3])
    b2.copy_from(b1)
    val = float(b2.data_tensor[0, 0])
    print(f"[TEST6] After copy_from: b2.data[0,0]={val:.1f}")
    assert abs(val - 42.0) < 0.01
    del b1
    gc.collect()
    print("[TEST6] b1 destroyed (b2 alive)")
    del b2
    gc.collect()
    print("[TEST6] b2 destroyed")


def test_7_multiple_blobs():
    print("\n" + "=" * 80)
    print("TEST 7: Multiple Blobs (distinct memory verification)")
    print("=" * 80)
    blobs = []
    ptrs = []
    for i in range(3):
        b = Blob([100, 100])
        b.data_tensor[:] = float(i + 1)
        p = b.data_tensor.ctypes.data
        blobs.append(b)
        ptrs.append(p)
        print(f"[TEST7] Blob[{i}] shape={b.shape}, ptr=0x{p:016x}, fill_value={i+1}")
    print(f"[TEST7] All pointers unique: {len(set(ptrs)) == len(ptrs)}")
    assert len(set(ptrs)) == len(ptrs)
    for i in range(2, -1, -1):
        print(f"[TEST7] Deleting Blob[{i}]...")
        del blobs[i]
        gc.collect()
    print("[TEST7] All blobs destroyed")


def test_8_exception_during_use():
    print("\n" + "=" * 80)
    print("TEST 8: Exception during Blob lifetime (destructor must still run)")
    print("=" * 80)
    b = Blob([5, 5])
    b.data_tensor[:] = 99.0
    print(f"[TEST8] Blob created, ptr=0x{b.data_tensor.ctypes.data:016x}")
    try:
        print("[TEST8] Raising exception...")
        raise RuntimeError("Simulated error during Blob use")
    except RuntimeError:
        print("[TEST8] Exception caught, Blob still in scope")
        print(f"[TEST8] data accessible: {float(b.data_tensor[0,0]):.1f}")
    del b
    gc.collect()
    print("[TEST8] Blob destroyed after exception handling")


def test_9_diff_tensor_independence():
    print("\n" + "=" * 80)
    print("TEST 9: data_tensor vs diff_tensor pointer independence")
    print("=" * 80)
    b = Blob([100])
    dp = b.data_tensor.ctypes.data
    dfp = b.diff_tensor.ctypes.data
    print(f"[TEST9] data_ptr=0x{dp:016x}, diff_ptr=0x{dfp:016x}")
    print(f"[TEST9] Pointers different: {dp != dfp} (expected: True)")
    assert dp != dfp
    b.data_tensor[:] = 1.0
    b.diff_tensor[:] = 2.0
    d_val = float(b.data_tensor[0])
    df_val = float(b.diff_tensor[0])
    print(f"[TEST9] data[0]={d_val}, diff[0]={df_val}")
    assert abs(d_val - 1.0) < 0.01 and abs(df_val - 2.0) < 0.01
    del b
    gc.collect()
    print("[TEST9] Blob destroyed")


if __name__ == "__main__":
    setup_logging()
    tests = [
        test_1_simple_create_and_destroy,
        test_2_create_with_shape_and_load,
        test_3_reshape_reallocation,
        test_4_fill_zero_update,
        test_5_from_numpy_to_numpy,
        test_6_copy_from,
        test_7_multiple_blobs,
        test_8_exception_during_use,
        test_9_diff_tensor_independence,
    ]
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n[FAIL] {test_fn.__name__} failed: {e}")
            traceback.print_exc()
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 80)
    sys.exit(0 if failed == 0 else 1)
