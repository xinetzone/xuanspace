#!/usr/bin/env python3
"""Flaky test stress-runner for test_p2_other_ops.py.

Runs the P2 other-ops test module N times (default 50) in a tight loop and captures
diagnostics on the FIRST failure:

  * Full traceback
  * tracemalloc memory snapshot (top 20 allocating locations at time of failure)
  * RSS memory delta from start of run
  * Net-state blob dump (if the failure occurred during a Forward/net construction)
  * C++ callback registry state

Usage::

    # In Docker container (caffe-ffi-jupyter):
    python scripts/stress_test_p2.py                    # 50 runs, all P2 tests
    python scripts/stress_test_p2.py -n 100             # 100 runs
    python scripts/stress_test_p2.py -k test_forward_no # filter by test name
    python scripts/stress_test_p2.py --fail-fast        # stop on first failure (default)

The script exits with code 0 if all iterations pass, 1 if any failure is detected.
Failure artifacts are written to tests/python/.temp/stress_p2/ including:
  - failure_traceback_<run>.txt
  - memory_snapshot_<run>.txt
  - net_diag_<run>.txt (if net state is available)
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
import tracemalloc
from datetime import datetime
from pathlib import Path

# Ensure the project python/ dir is on sys.path so caffe_ffi imports work.
_project_root = Path(__file__).resolve().parent.parent
_python_dir = _project_root / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))

# Ensure KMP duplicate is handled before importing caffe_ffi
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_out_dir = _project_root / "tests" / "python" / ".temp" / "stress_p2"
_out_dir.mkdir(parents=True, exist_ok=True)


def _clear_callbacks() -> None:
    """Clear C++ static callback registries (mirrors conftest autouse fixture)."""
    try:
        from caffe_ffi import _ffi_api
        for name in ("caffe_ffi.data_io.clear", "caffe_ffi.python_layer.clear"):
            fn = _ffi_api.get_global_func(name)
            if fn is not None:
                try:
                    fn()
                except Exception:
                    pass
    except Exception:
        pass


def _get_rss_kb() -> int:
    """Return current process RSS in KB (Linux /proc-based; returns 0 on failure)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


def _take_memory_snapshot(tag: str, run_idx: int, snap_pre, snap_post) -> str:
    """Compare two tracemalloc snapshots and write a top-20 diff report."""
    stats = snap_post.compare_to(snap_pre, "lineno")
    lines = [f"=== Memory snapshot diff: {tag} (run {run_idx}) ==="]
    lines.append(f"Timestamp: {datetime.now().isoformat()}")
    lines.append(f"Top 20 allocating locations (size diff in bytes):")
    for stat in stats[:20]:
        lines.append(f"  {stat}")
    report = "\n".join(lines)
    out_path = _out_dir / f"memory_snapshot_{tag}_run{run_idx}.txt"
    out_path.write_text(report, encoding="utf-8")
    return str(out_path)


def _save_traceback(tag: str, run_idx: int, exc: BaseException) -> str:
    """Save a formatted traceback to disk."""
    tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    header = (
        f"=== FAILURE: {tag} (run {run_idx}) ===\n"
        f"Timestamp: {datetime.now().isoformat()}\n"
        f"Exception type: {type(exc).__name__}\n"
        f"Exception message: {exc}\n\n"
    )
    out_path = _out_dir / f"failure_traceback_{tag}_run{run_idx}.txt"
    out_path.write_text(header + tb_str, encoding="utf-8")
    return str(out_path)


def stress_run(num_runs: int, keyword: str | None, fail_fast: bool) -> int:
    """Run the P2 other-ops tests *num_runs* times, return 0 on all-pass, 1 on any fail."""
    import pytest
    import numpy as np
    from caffe_ffi import _ffi_api

    # Seed numpy for reproducibility (each run uses a deterministic seed derived from run idx)
    rng = np.random.RandomState(42)

    failures = 0
    total_start = time.perf_counter()
    rss_start = _get_rss_kb()
    print(f"[stress] Starting {num_runs} tight-loop iterations of test_p2_other_ops.py")
    print(f"[stress] RSS start: {rss_start} KB, output dir: {_out_dir}")
    if keyword:
        print(f"[stress] Test filter: -k {keyword!r}")

    tracemalloc.start(25)  # 25 frames deep for allocation trace
    snap_prev = tracemalloc.take_snapshot()

    args_base = [
        str(_project_root / "tests" / "python" / "test_p2_other_ops.py"),
        "-v",
        "--tb=long",
        "-p", "no:caffe_ffi.test.perf",  # disable perf-tracing plugin to reduce noise
    ]
    if keyword:
        args_base.extend(["-k", keyword])

    for run_idx in range(1, num_runs + 1):
        _clear_callbacks()
        gc.collect()

        run_t0 = time.perf_counter()
        rss_before = _get_rss_kb()
        np.random.seed(run_idx * 7 + 13)  # deterministic per-run seed

        # Capture tracemalloc snapshot before the test run
        snap_before = tracemalloc.take_snapshot()

        # Run pytest programmatically via pytest.main() in-process.
        # We redirect output to a buffer to keep the console clean.
        import io
        import contextlib
        buf = io.StringIO()
        exit_code = 1
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exit_code = pytest.main(args_base)
        except Exception as e:
            exit_code = 1
            buf.write(f"\n=== EXCEPTION IN PYTEST MAIN ===\n")
            buf.write(traceback.format_exc())

        snap_after = tracemalloc.take_snapshot()
        run_elapsed = time.perf_counter() - run_t0
        rss_after = _get_rss_kb()

        output = buf.getvalue()

        if exit_code == 0 and "FAILED" not in output and "ERROR" not in output:
            print(f"  [run {run_idx:3d}/{num_runs}] PASS  ({run_elapsed:.2f}s, "
                  f"RSS {rss_after} KB, Δ{rss_after - rss_before:+d} KB)")
            snap_prev = snap_after
            continue

        # ── FAILURE PATH ──
        failures += 1
        print(f"\n!!! [run {run_idx:3d}/{num_runs}] FAILURE DETECTED (exit_code={exit_code}) !!!")
        print(f"    Elapsed: {run_elapsed:.2f}s, RSS: {rss_before} KB -> {rss_after} KB "
              f"(Δ{rss_after - rss_before:+d} KB)")

        # Save traceback / output
        tag = f"run{run_idx}"
        # Save full pytest output
        out_log = _out_dir / f"pytest_output_{tag}.txt"
        out_log.write_text(output, encoding="utf-8")
        print(f"    Pytest output saved to: {out_log}")

        # Save memory snapshot
        snap_path = _take_memory_snapshot("on_failure", run_idx, snap_before, snap_after)
        print(f"    Memory snapshot saved to: {snap_path}")

        # Cumulative memory diff since start
        snap_cum_path = _take_memory_snapshot("cumulative", run_idx, snap_prev, snap_after)
        print(f"    Cumulative memory diff saved to: {snap_cum_path}")

        # Try to import and dump net state for a representative net
        try:
            sys.path.insert(0, str(_project_root / "tests" / "python"))
            from caffe_test_helpers import make_net, dump_net_state
            import textwrap
            diag_net = make_net(textwrap.dedent("""\
                name: "diag_stress_net"
                layer {
                  name: "data" type: "Input" top: "data"
                  input_param { shape { dim: 2 dim: 3 } }
                }
                layer {
                  name: "label" type: "Input" top: "label"
                  input_param { shape { dim: 2 } }
                }
                layer {
                  name: "h5_diag" type: "HDF5Output"
                  bottom: "data" bottom: "label"
                  hdf5_output_param { file_name: "diag.h5" }
                }
            """))
            diag = dump_net_state(diag_net, tag=f"diag_run{run_idx}")
            diag_path = _out_dir / f"net_diag_{tag}.txt"
            diag_path.write_text(diag, encoding="utf-8")
            print(f"    Net diagnostic dump saved to: {diag_path}")
            del diag_net
            gc.collect()
        except Exception as e:
            print(f"    (Could not produce net diag dump: {e})")

        # Save RSS info
        rss_path = _out_dir / f"rss_info_{tag}.txt"
        rss_path.write_text(
            f"run={run_idx}\nrss_before={rss_before} KB\nrss_after={rss_after} KB\n"
            f"delta={rss_after - rss_before} KB\nrss_start={rss_start} KB\n"
            f"total_elapsed={time.perf_counter() - total_start:.2f}s\n",
            encoding="utf-8",
        )

        if fail_fast:
            print(f"\n[stress] --fail-fast set, stopping after first failure.")
            break

    total_elapsed = time.perf_counter() - total_start
    tracemalloc.stop()
    _clear_callbacks()

    print(f"\n[stress] ===== SUMMARY =====")
    print(f"[stress] Total runs: {run_idx if failures else num_runs}/{num_runs}")
    print(f"[stress] Failures: {failures}")
    print(f"[stress] Total elapsed: {total_elapsed:.1f}s")
    print(f"[stress] RSS end: {_get_rss_kb()} KB (start was {rss_start} KB)")
    if failures:
        print(f"[stress] Artifacts saved under: {_out_dir}")
        return 1
    else:
        print("[stress] ALL RUNS PASSED ✓")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Stress-test test_p2_other_ops.py to reproduce flaky failures"
    )
    parser.add_argument("-n", "--num-runs", type=int, default=50,
                        help="Number of tight-loop iterations (default: 50)")
    parser.add_argument("-k", "--keyword", type=str, default=None,
                        help="Pytest -k keyword filter (e.g. 'test_forward_no')")
    parser.add_argument("--no-fail-fast", action="store_true",
                        help="Continue running even after a failure (default: stop on first)")
    args = parser.parse_args()

    sys.exit(stress_run(args.num_runs, args.keyword, fail_fast=not args.no_fail_fast))


if __name__ == "__main__":
    main()
