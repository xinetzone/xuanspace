#!/usr/bin/env python3
"""Phase 2 COW (Copy-on-Write) Regression Test Script.

Validates COW optimization correctness and memory savings across N≥2
Split scenarios. Designed to run standalone (no pytest dependency).

Usage:
    python scripts/run_cow_regression.py              # Run all tests
    python scripts/run_cow_regression.py --quick      # Quick subset (N=1,2,4,8)
    python scripts/run_cow_regression.py --perf-only  # Memory savings only, skip correctness
    python scripts/run_cow_regression.py --csv report.csv  # Export results to CSV

Exit code: 0 if all tests pass, 1 if any test fails.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─── Path setup ──────────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent
_python_dir = _project_root / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))

from caffe_ffi import (
    Blob, Net, LayerParameter, NetParameter,
    total_allocated_bytes, live_blob_count,
    net_param_from_string, net_from_param,
)


# ═══════════════════════════════════════════════════════════════════════
# Test result tracking
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    name: str
    passed: bool = True
    error: str = ""
    mem_saved_bytes: int = 0
    cow_events: int = 0
    elapsed_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


_results: List[TestResult] = []


def _mem_snapshot() -> Tuple[int, int]:
    """Return (allocated_bytes, live_blobs) after aggressive GC."""
    for _ in range(3):
        gc.collect()
    return total_allocated_bytes(), live_blob_count()


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


# ═══════════════════════════════════════════════════════════════════════
# Prototxt builders
# ═══════════════════════════════════════════════════════════════════════

def _build_split_net(num_top: int, feat_dim: int = 8, batch_size: int = 4) -> Tuple[Net, List[str]]:
    """Build a simple Split-only network with N top blobs."""
    lines = [
        f'name: "split_n{num_top}"',
        'layer {',
        '  name: "data"', '  type: "Input"', '  top: "data"',
        f'  input_param {{ shape {{ dim: {batch_size} dim: {feat_dim} }} }}',
        '}',
        'layer {',
        '  name: "split"', '  type: "Split"', '  bottom: "data"',
    ]
    top_names = [f"split_{i}" for i in range(num_top)]
    for tn in top_names:
        lines.append(f'  top: "{tn}"')
    lines.append('}')
    return net_from_param(net_param_from_string("\n".join(lines))), top_names


def _build_split_inplace_net(feat_dim: int = 8, batch_size: int = 4) -> Net:
    """Build a Split + in-place branch network."""
    prototxt = f"""name: "split_inplace"
layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape {{ dim: {batch_size} dim: {feat_dim} }} }}
}}
layer {{
  name: "split"
  type: "Split"
  bottom: "data"
  top: "raw_branch"
  top: "relu_branch"
}}
layer {{
  name: "relu_on_branch"
  type: "ReLU"
  bottom: "relu_branch"
  top: "relu_branch"
}}
layer {{
  name: "raw_fc"
  type: "InnerProduct"
  bottom: "raw_branch"
  top: "raw_out"
  inner_product_param {{ num_output: {feat_dim} bias_term: false }}
}}
layer {{
  name: "relu_fc"
  type: "InnerProduct"
  bottom: "relu_branch"
  top: "relu_out"
  inner_product_param {{ num_output: {feat_dim} bias_term: false }}
}}
"""
    net = net_from_param(net_param_from_string(prototxt))
    rng = np.random.RandomState(42)
    for layer in net.layers_array():
        if layer.type == "InnerProduct" and len(layer.blobs) >= 1:
            W = layer.blobs[0]
            W.from_numpy(rng.randn(*W.shape).astype(np.float32) * 0.1)
            if len(layer.blobs) >= 2:
                layer.blobs[1].from_numpy(np.zeros(layer.blobs[1].shape, dtype=np.float32))
    return net


# ═══════════════════════════════════════════════════════════════════════
# COW API tests
# ═══════════════════════════════════════════════════════════════════════

def test_blob_cow_api():
    """Test Blob-level COW API methods."""
    r = TestResult("BlobCOWAPI")
    t0 = time.perf_counter()
    try:
        # IsDataShared standalone
        b = Blob([4, 4])
        _check(not b.IsDataShared(), "standalone IsDataShared should be False")
        _check(b.DataRefCount() == 1, f"standalone DataRefCount should be 1, got {b.DataRefCount()}")

        # ShareData
        src = Blob([4, 4])
        dst = Blob([4, 4])
        dst.ShareData(src)
        _check(dst.IsDataShared(), "after ShareData, IsDataShared should be True")
        _check(dst.DataRefCount() >= 2, f"after ShareData, refcount should be >=2, got {dst.DataRefCount()}")

        # UnshareData
        src_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        src2 = Blob([4])
        src2.from_numpy(src_data)
        dst2 = Blob([4])
        dst2.ShareData(src2)
        old_ptr = dst2.data_tensor.ctypes.data
        dst2.UnshareData()
        _check(not dst2.IsDataShared(), "after UnshareData, IsDataShared should be False")
        _check(dst2.data_tensor.ctypes.data != old_ptr, "UnshareData must clone to new buffer")
        np.testing.assert_array_equal(dst2.to_numpy(), src_data)

        # mutable_data_tensor triggers COW
        src3 = Blob([4])
        src3.from_numpy(np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))
        dst3 = Blob([4])
        dst3.ShareData(src3)
        mt = dst3.mutable_data_tensor()
        _check(not dst3.IsDataShared(), "mutable_data_tensor should trigger COW")
        _check(dst3.DataRefCount() == 1, "after COW, refcount should be 1")
        np.testing.assert_array_equal(np.asarray(mt), [10.0, 20.0, 30.0, 40.0])

        # const data_tensor does NOT trigger COW
        src4 = Blob([4])
        dst4 = Blob([4])
        dst4.ShareData(src4)
        _ = dst4.data_tensor
        _check(dst4.IsDataShared(), "const data_tensor must not trigger COW")

        # Three-way share
        a = Blob([4])
        a.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        b2 = Blob([4])
        c = Blob([4])
        b2.ShareData(a)
        c.ShareData(a)
        _check(b2.IsDataShared() and c.IsDataShared(), "three-way share")
        b2.UnshareData()
        _check(not b2.IsDataShared(), "b2 should break sharing")
        _check(c.IsDataShared(), "c should still share with a")

    except AssertionError as e:
        r.passed = False
        r.error = str(e)
    finally:
        r.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _results.append(r)
    return r


# ═══════════════════════════════════════════════════════════════════════
# Split COW behavior tests
# ═══════════════════════════════════════════════════════════════════════

def test_split_cow_N1():
    """N=1 Split: top shares data with bottom (zero-copy)."""
    r = TestResult("SplitCOW_N1")
    t0 = time.perf_counter()
    try:
        net, top_names = _build_split_net(1, feat_dim=4, batch_size=2)
        inp = np.random.RandomState(77).randn(2, 4).astype(np.float32)
        net.Forward({"data": inp})

        data_blob = net.blob_by_name("data")
        pass_blob = net.blob_by_name("split_0")
        _check(pass_blob.IsDataShared(), "N=1: passthrough must share data")
        _check(pass_blob.SharesDataWith(data_blob), "N=1: must share with bottom")
        _check(pass_blob.data_tensor.ctypes.data == data_blob.data_tensor.ctypes.data,
               "N=1: pointers must be equal (zero-copy)")
        r.details["shared"] = True
        r.details["pointers_equal"] = True
    except AssertionError as e:
        r.passed = False
        r.error = str(e)
    finally:
        r.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _results.append(r)
    return r


def test_split_cow_N2_data_shared():
    """N=2 Split: both tops share data before write."""
    r = TestResult("SplitCOW_N2_Shared")
    t0 = time.perf_counter()
    try:
        net, top_names = _build_split_net(2, feat_dim=4, batch_size=2)
        inp = np.random.RandomState(42).randn(2, 4).astype(np.float32)
        net.Forward({"data": inp})

        data_blob = net.blob_by_name("data")
        s0 = net.blob_by_name("split_0")
        s1 = net.blob_by_name("split_1")
        _check(s0.IsDataShared(), "N=2: split_0 must share data")
        _check(s1.IsDataShared(), "N=2: split_1 must share data")
        _check(s0.SharesDataWith(data_blob), "N=2: split_0 must share with bottom")
        _check(s1.SharesDataWith(data_blob), "N=2: split_1 must share with bottom")
        _check(s0.SharesDataWith(s1), "N=2: split_0 and split_1 must share")
        _check(s0.data_tensor.ctypes.data == data_blob.data_tensor.ctypes.data,
               "N=2: all three must share same physical memory")
        r.details["shared"] = True
        r.details["refcount"] = s0.DataRefCount()
    except AssertionError as e:
        r.passed = False
        r.error = str(e)
    finally:
        r.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _results.append(r)
    return r


def test_split_cow_N2_isolation():
    """N=2 Split: COW isolates sibling after write."""
    r = TestResult("SplitCOW_N2_Isolation")
    t0 = time.perf_counter()
    try:
        net, top_names = _build_split_net(2, feat_dim=4, batch_size=2)
        inp = np.random.RandomState(99).randn(2, 4).astype(np.float32)
        net.Forward({"data": inp})

        data_blob = net.blob_by_name("data")
        s0 = net.blob_by_name("split_0")
        s1 = net.blob_by_name("split_1")

        # COW trigger via mutable_data_tensor
        mt0 = s0.mutable_data_tensor()
        mt0[0, 0] = 999.0

        _check(not s0.IsDataShared(), "split_0 must break sharing after COW")
        _check(s0.DataRefCount() == 1, "split_0 refcount must be 1 after COW")
        _check(s1.IsDataShared(), "split_1 must remain shared")
        _check(s1.SharesDataWith(data_blob), "split_1 must still share with data")

        np.testing.assert_array_equal(s1.to_numpy(), inp,
            "COW on split_0 must not affect split_1 data")
        _check(s0.to_numpy()[0, 0] == 999.0, "split_0 must have modified value")
        r.details["cow_triggered"] = True
        r.details["isolation_verified"] = True
    except AssertionError as e:
        r.passed = False
        r.error = str(e)
    finally:
        r.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _results.append(r)
    return r


def test_split_cow_N4_isolation():
    """N=4 Split: COW isolates one branch, others unaffected."""
    r = TestResult("SplitCOW_N4_Isolation")
    t0 = time.perf_counter()
    try:
        net, top_names = _build_split_net(4, feat_dim=6, batch_size=2)
        inp = np.random.RandomState(55).randn(2, 6).astype(np.float32)
        net.Forward({"data": inp})

        data_blob = net.blob_by_name("data")
        splits = [net.blob_by_name(f"split_{i}") for i in range(4)]

        for i, s in enumerate(splits):
            _check(s.IsDataShared(), f"split_{i} must be shared before write")

        mt2 = splits[2].mutable_data_tensor()
        mt2[0, 0] = 777.0

        _check(not splits[2].IsDataShared(), "split_2 must break sharing")
        _check(splits[2].DataRefCount() == 1, "split_2 refcount must be 1")

        for i in [0, 1, 3]:
            _check(splits[i].IsDataShared(), f"split_{i} must remain shared")
            np.testing.assert_array_equal(splits[i].to_numpy(), inp,
                f"COW on split_2 must not affect split_{i}")

        _check(splits[2].to_numpy()[0, 0] == 777.0, "split_2 must have modified value")
        r.details["cow_triggered"] = True
        r.details["isolation_verified"] = True
    except AssertionError as e:
        r.passed = False
        r.error = str(e)
    finally:
        r.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _results.append(r)
    return r


def test_split_cow_inplace_relu():
    """N=2 Split with in-place ReLU: COW isolates raw branch."""
    r = TestResult("SplitCOW_InplaceReLU")
    t0 = time.perf_counter()
    try:
        feat_dim = 8
        batch_size = 2
        net = _build_split_inplace_net(feat_dim, batch_size)
        inp = np.random.RandomState(55).randn(batch_size, feat_dim).astype(np.float32)
        net.Forward({"data": inp})

        raw_branch = net.blob_by_name("raw_branch")
        relu_branch = net.blob_by_name("relu_branch")

        _check(not relu_branch.IsDataShared(),
               "In-place ReLU must trigger COW, breaking sharing")
        np.testing.assert_array_equal(raw_branch.to_numpy(), inp,
            "In-place ReLU on sibling must not corrupt raw_branch (COW isolation)")
        np.testing.assert_array_equal(relu_branch.to_numpy(), np.maximum(inp, 0))
        r.details["cow_triggered"] = True
        r.details["isolation_verified"] = True
    except AssertionError as e:
        r.passed = False
        r.error = str(e)
    finally:
        r.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _results.append(r)
    return r


def test_split_cow_const_access():
    """N=2 Split: const access (data_tensor) does NOT trigger COW."""
    r = TestResult("SplitCOW_ConstAccess")
    t0 = time.perf_counter()
    try:
        net, top_names = _build_split_net(2, feat_dim=4, batch_size=2)
        inp = np.random.RandomState(33).randn(2, 4).astype(np.float32)
        net.Forward({"data": inp})

        s0 = net.blob_by_name("split_0")
        s1 = net.blob_by_name("split_1")
        _ = s0.data_tensor
        _ = s1.data_tensor
        _check(s0.IsDataShared(), "const data_tensor must not trigger COW")
        _check(s1.IsDataShared(), "const data_tensor must not trigger COW")
        _check(s0.SharesDataWith(s1), "both must still share after const access")
        r.details["no_cow_triggered"] = True
    except AssertionError as e:
        r.passed = False
        r.error = str(e)
    finally:
        r.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _results.append(r)
    return r


# ═══════════════════════════════════════════════════════════════════════
# Memory savings measurement
# ═══════════════════════════════════════════════════════════════════════

def _measure_memcpy_equivalent(num_top: int, feat_dim: int, batch_size: int) -> int:
    """Calculate what memcpy would cost for the given Split topology."""
    count = batch_size * feat_dim
    return num_top * count * 4  # float32 = 4 bytes


def test_split_cow_memory_savings(num_top: int, feat_dim: int = 64, batch_size: int = 8):
    """Measure memory savings from COW vs hypothetical memcpy."""
    r = TestResult(f"SplitCOW_Mem{num_top}")
    t0 = time.perf_counter()
    try:
        count = batch_size * feat_dim
        memcpy_equivalent = _measure_memcpy_equivalent(num_top, feat_dim, batch_size)

        net, top_names = _build_split_net(num_top, feat_dim, batch_size)
        inp = np.random.RandomState(123).randn(batch_size, feat_dim).astype(np.float32)

        mem_before, blobs_before = _mem_snapshot()
        net.Forward({"data": inp})
        mem_after, blobs_after = _mem_snapshot()

        delta_mem = mem_after - mem_before
        # COW shares data, so actual allocation is far less than memcpy equivalent
        savings = memcpy_equivalent - max(0, delta_mem)

        data_blob = net.blob_by_name("data")
        splits = [net.blob_by_name(f"split_{i}") for i in range(num_top)]

        # Verify all tops share data
        all_shared = all(s.IsDataShared() for s in splits)
        min_refcount = min(s.DataRefCount() for s in splits)

        r.details = {
            "num_top": num_top,
            "feat_dim": feat_dim,
            "batch_size": batch_size,
            "count": count,
            "memcpy_bytes": memcpy_equivalent,
            "actual_delta_mem": delta_mem,
            "cow_saved_bytes": savings,
            "savings_pct": round(savings / memcpy_equivalent * 100, 1) if memcpy_equivalent > 0 else 0,
            "all_shared": all_shared,
            "min_refcount": min_refcount,
        }
        r.mem_saved_bytes = savings
    except Exception as e:
        r.passed = False
        r.error = str(e)
    finally:
        r.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _results.append(r)
    return r


# ═══════════════════════════════════════════════════════════════════════
# Stress test: N=100
# ═══════════════════════════════════════════════════════════════════════

def test_split_cow_stress_N100():
    """N=100 Split: stress test COW with large fan-out."""
    r = TestResult("SplitCOW_Stress_N100")
    t0 = time.perf_counter()
    try:
        feat_dim = 16
        batch_size = 4
        memcpy_equivalent = _measure_memcpy_equivalent(100, feat_dim, batch_size)

        net, top_names = _build_split_net(100, feat_dim, batch_size)
        inp = np.random.RandomState(7).randn(batch_size, feat_dim).astype(np.float32)

        mem_before, _ = _mem_snapshot()
        net.Forward({"data": inp})
        mem_after, _ = _mem_snapshot()

        delta_mem = mem_after - mem_before
        savings = memcpy_equivalent - max(0, delta_mem)

        data_blob = net.blob_by_name("data")
        # Verify all 100 tops share data
        shared_count = sum(1 for i in range(100)
                          if net.blob_by_name(f"split_{i}").IsDataShared())
        _check(shared_count == 100, f"All 100 tops must share data, got {shared_count}")

        # Trigger COW on one branch, verify isolation
        s50 = net.blob_by_name("split_50")
        mt50 = s50.mutable_data_tensor()
        mt50[0, 0] = 12345.0
        _check(not s50.IsDataShared(), "split_50 must break sharing after COW")
        _check(s50.to_numpy()[0, 0] == 12345.0, "split_50 must have modified value")

        # Other branches still shared
        s0 = net.blob_by_name("split_0")
        _check(s0.IsDataShared(), "split_0 must remain shared")
        np.testing.assert_array_equal(s0.to_numpy(), inp,
            "COW on split_50 must not affect split_0")

        r.details = {
            "num_top": 100,
            "memcpy_bytes": memcpy_equivalent,
            "actual_delta_mem": delta_mem,
            "cow_saved_bytes": savings,
            "savings_pct": round(savings / memcpy_equivalent * 100, 1) if memcpy_equivalent > 0 else 0,
            "all_shared": shared_count == 100,
            "isolation_verified": True,
        }
        r.mem_saved_bytes = savings
    except Exception as e:
        r.passed = False
        r.error = str(e)
    finally:
        r.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _results.append(r)
    return r


# ═══════════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════════

def _print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def _print_result(r: TestResult):
    status = "PASS" if r.passed else "FAIL"
    icon = "✓" if r.passed else "✗"
    print(f"  [{icon} {status}] {r.name:<30} {r.elapsed_ms:7.2f}ms", end="")
    if r.mem_saved_bytes > 0:
        print(f"  mem_saved={r.mem_saved_bytes}B", end="")
    if r.error:
        print(f"\n    Error: {r.error}", end="")
    if r.details:
        for k, v in r.details.items():
            if isinstance(v, float):
                print(f"  {k}={v:.1f}", end="")
            elif isinstance(v, bool):
                print(f"  {k}={'yes' if v else 'no'}", end="")
            elif isinstance(v, int) and abs(v) > 1000:
                print(f"  {k}={v}B", end="")
            elif k not in ("num_top", "feat_dim", "batch_size", "count"):
                print(f"  {k}={v}", end="")
    print()


def _export_csv(path: str):
    """Export results to CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "test_name", "passed", "elapsed_ms",
            "mem_saved_bytes", "error", "details"
        ])
        for r in _results:
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                r.name, r.passed, f"{r.elapsed_ms:.2f}",
                r.mem_saved_bytes, r.error,
                "; ".join(f"{k}={v}" for k, v in r.details.items())
            ])
    print(f"\nResults exported to: {path}")


def run_all(quick: bool = False, perf_only: bool = False, csv_path: Optional[str] = None):
    """Run all COW regression tests."""
    _print_header("Phase 2 COW Regression Tests")
    print(f"  Time: {datetime.now().isoformat(timespec='seconds')}")
    print(f"  Mode: {'quick' if quick else 'full'}{' (perf-only)' if perf_only else ''}")

    # ── Blob COW API tests ──
    if not perf_only:
        _print_header("1. Blob COW API Tests")
        test_blob_cow_api()
        _print_result(_results[-1])

        _print_header("2. Split COW Behavior Tests")
        for test_fn in [
            test_split_cow_N1,
            test_split_cow_N2_data_shared,
            test_split_cow_N2_isolation,
            test_split_cow_N4_isolation,
            test_split_cow_inplace_relu,
            test_split_cow_const_access,
        ]:
            test_fn()
            _print_result(_results[-1])

    # ── Memory savings tests ──
    _print_header("3. Memory Savings Measurement")
    N_values = [2, 4, 8, 16] if quick else [2, 4, 8, 16, 32, 64, 100]
    for n in N_values:
        test_split_cow_memory_savings(n)
        r = _results[-1]
        _print_result(r)
        if r.details:
            d = r.details
            print(f"       memcpy_equivalent={d.get('memcpy_bytes', 0)}B  "
                  f"cow_saved={d.get('cow_saved_bytes', 0)}B  "
                  f"savings={d.get('savings_pct', 0)}%")

    # ── Stress test ──
    if not quick:
        _print_header("4. Stress Test: N=100")
        test_split_cow_stress_N100()
        _print_result(_results[-1])

    # ── Summary ──
    _print_header("Summary")
    passed = sum(1 for r in _results if r.passed)
    failed = sum(1 for r in _results if not r.passed)
    total_mem_saved = sum(r.mem_saved_bytes for r in _results)

    print(f"  Total: {len(_results)} tests, {passed} passed, {failed} failed")
    print(f"  Total time: {sum(r.elapsed_ms for r in _results):.1f}ms")
    if total_mem_saved > 0:
        if total_mem_saved >= 1024 * 1024:
            print(f"  Total COW memory saved: {total_mem_saved / (1024*1024):.2f} MB")
        elif total_mem_saved >= 1024:
            print(f"  Total COW memory saved: {total_mem_saved / 1024:.2f} KB")
        else:
            print(f"  Total COW memory saved: {total_mem_saved} B")

    if csv_path:
        _export_csv(csv_path)

    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 COW Regression Test Suite")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick subset (N=1,2,4,8,16)")
    parser.add_argument("--perf-only", action="store_true",
                        help="Skip correctness tests, only measure memory savings")
    parser.add_argument("--csv", type=str, default=None,
                        help="Export results to CSV file")
    args = parser.parse_args()

    success = run_all(quick=args.quick, perf_only=args.perf_only, csv_path=args.csv)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()