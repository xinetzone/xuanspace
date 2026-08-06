"""Phase 3.0: Log aggregation verification test for N=100 Split.

Verifies that when N=100 (>= kLogAggregateThreshold=32), the [SPLIT-PERF]
log output is controlled to a small number of lines, instead of O(N) per-top lines.

NOTE: caffe_ffi Logger sends WARN-level (including [SPLIT-PERF]) to stdout
(std::cout), NOT stderr. Use capfd (fd-level capture) and check .out.

These tests require CAFFE_FFI_ENABLE_PERF_LOG=ON at compile time. In Release
builds (PERF_LOG=OFF, the production default), PERF-dependent tests are
automatically skipped because no [SPLIT-PERF] lines are emitted.

Usage:
  pytest tests/python/test_phase3_log_aggregation.py -v -s
"""
import os
import re
import tempfile
import pytest
import numpy as np
import caffe_ffi
from .conftest import require_cpp_extension


# ── PERF_LOG runtime detection ────────────────────────────────────────

def _perf_log_enabled() -> bool:
    """Detect at runtime whether the C++ extension was compiled with PERF_LOG.

    Creates a tiny Split net, captures stdout at file-descriptor level
    (needed to catch C++ std::cout output), and checks for [SPLIT-PERF].
    Returns False if the build was compiled with CAFFE_FFI_ENABLE_PERF_LOG=OFF.
    """
    if not caffe_ffi.is_available():
        return False
    # Redirect fd 1 (stdout) to a temp file to capture C++ output
    stdout_fd = os.dup(1)
    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        os.dup2(tmp.fileno(), 1)
        prototxt = (
            'name: "perf_detect"\n'
            'layer { name: "data" type: "Input" top: "data" '
            'input_param { shape { dim: 1 dim: 16 } } }\n'
            'layer { name: "split" type: "Split" bottom: "data" '
            'top: "s0" top: "s1" top: "s2" }\n'
        )
        net = caffe_ffi.Net(prototxt)
        net.Forward({"data": np.random.randn(1, 16).astype(np.float32)})
        os.fsync(1)
    except Exception:
        return False
    finally:
        os.dup2(stdout_fd, 1)
        os.close(stdout_fd)
    try:
        tmp.seek(0)
        captured = tmp.read().decode("utf-8", errors="replace")
        return "[SPLIT-PERF]" in captured
    finally:
        tmp.close()


_perf_log = _perf_log_enabled()
skipif_no_perf_log = pytest.mark.skipif(
    not _perf_log,
    reason="CAFFE_FFI_ENABLE_PERF_LOG=OFF (Release build); PERF-dependent tests skipped"
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_split_prototxt(num_top: int, feat_dim: int, batch: int = 1, name: str = None) -> str:
    """Create a minimal prototxt with Input + Split(N=num_top)."""
    if name is None:
        name = f"split_log_test_N{num_top}"
    tops = "\n".join(f'  top: "split_{i}"' for i in range(num_top))
    return f"""name: "{name}"
layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape {{ dim: {batch} dim: {feat_dim} }} }}
}}
layer {{
  name: "split"
  type: "Split"
  bottom: "data"
{tops}
}}
"""


def _make_split_net(num_top: int, feat_dim: int, batch: int = 1):
    """Create a minimal Net with a single Split layer of N=num_top."""
    prototxt = _make_split_prototxt(num_top, feat_dim, batch)
    return caffe_ffi.Net(prototxt)


def _count_split_perf_lines(log_output: str) -> int:
    """Count lines containing [SPLIT-PERF] in log output."""
    return sum(1 for line in log_output.splitlines() if "[SPLIT-PERF]" in line)


# ── Test Cases ────────────────────────────────────────────────────────

@require_cpp_extension
class TestLogAggregationN100:
    """Verify Phase 3.0 log aggregation for N=100 Split."""

    @skipif_no_perf_log
    def test_n100_split_perf_lines_bounded(self, capfd):
        """N=100: [SPLIT-PERF] lines should be bounded (summary only, no per-top flood)."""
        num_top = 100
        C = 1024

        net = _make_split_net(num_top, C)
        inp = np.random.randn(1, C).astype(np.float32)

        # Capture stdout (C++ WARN logs go to stdout via std::cout)
        _ = capfd.readouterr()  # clear any prior output
        out = net.Forward({"data": inp})
        captured = capfd.readouterr()
        log_output = captured.out

        perf_lines = _count_split_perf_lines(log_output)
        # With log aggregation, we expect at most a few summary lines
        # (1 Reshape summary from Net init + 1 Forward summary, maybe one
        # extra Reshape if Forward triggers it = ~3 lines)
        assert perf_lines >= 1, (
            f"N=100: expected at least 1 [SPLIT-PERF] line, got {perf_lines}"
        )
        assert perf_lines <= 10, (
            f"N=100 produced {perf_lines} [SPLIT-PERF] lines, "
            f"expected <= 10. Phase 3.0 log aggregation may not be active.\n"
            f"Log output:\n{log_output}"
        )

    @skipif_no_perf_log
    def test_n100_forward_summary_present(self, capfd):
        """N=100: the Forward summary [SPLIT-PERF] log must be present."""
        num_top = 100
        C = 1024

        net = _make_split_net(num_top, C)
        inp = np.random.randn(1, C).astype(np.float32)

        _ = capfd.readouterr()
        out = net.Forward({"data": inp})
        captured = capfd.readouterr()
        log_output = captured.out

        # The summary must contain "Forward" and "SPLIT-PERF" with N=100
        has_forward_summary = any(
            "Forward" in line and "SPLIT-PERF" in line and f"N={num_top}" in line
            for line in log_output.splitlines()
        )
        assert has_forward_summary, (
            f"N={num_top} Forward summary [SPLIT-PERF] log is missing.\n"
            f"Log output:\n{log_output}"
        )

    @skipif_no_perf_log
    def test_n100_reshape_summary_present(self, capfd):
        """N=100: the Reshape summary [SPLIT-PERF] log must be present."""
        num_top = 100
        C = 1024

        _ = capfd.readouterr()
        net = _make_split_net(num_top, C)
        captured = capfd.readouterr()
        log_output = captured.out

        has_reshape_summary = any(
            "Reshape" in line and "SPLIT-PERF" in line and f"num_top={num_top}" in line
            for line in log_output.splitlines()
        )
        assert has_reshape_summary, (
            f"N={num_top} Reshape summary [SPLIT-PERF] log is missing.\n"
            f"Log output:\n{log_output}"
        )

    def test_n4_split_output_correct(self):
        """N=4: all top outputs should equal bottom input (functional correctness)."""
        num_top = 4
        C = 256

        net = _make_split_net(num_top, C)
        inp = np.random.randn(1, C).astype(np.float32)

        out = net.Forward({"data": inp})

        for i in range(num_top):
            key = f"split_{i}"
            assert key in out, f"Missing output '{key}'"
            np.testing.assert_array_almost_equal(
                net.blob_by_name(key).to_numpy(), inp,
                err_msg=f"split_{i} output differs from input"
            )

    def test_n100_split_output_correct(self):
        """N=100: all top outputs should equal bottom input."""
        num_top = 100
        C = 128

        net = _make_split_net(num_top, C)
        inp = np.random.randn(1, C).astype(np.float32)

        out = net.Forward({"data": inp})

        for i in range(num_top):
            key = f"split_{i}"
            assert key in out, f"Missing output '{key}'"
            np.testing.assert_array_almost_equal(
                net.blob_by_name(key).to_numpy(), inp,
                err_msg=f"split_{i} output differs from input"
            )

    def test_n100_forward_deterministic(self):
        """N=100: two Forward calls should produce identical results."""
        num_top = 100
        C = 64

        net = _make_split_net(num_top, C)
        inp = np.random.randn(1, C).astype(np.float32)

        net.Forward({"data": inp})
        out1 = {f"split_{i}": net.blob_by_name(f"split_{i}").to_numpy().copy()
                for i in range(num_top)}

        net.Forward({"data": inp})
        out2 = {f"split_{i}": net.blob_by_name(f"split_{i}").to_numpy().copy()
                for i in range(num_top)}

        for i in range(num_top):
            key = f"split_{i}"
            np.testing.assert_array_equal(out1[key], out2[key])


@require_cpp_extension
class TestLogAggregationBoundary:
    """Verify boundary behavior around kLogAggregateThreshold=32."""

    @skipif_no_perf_log
    @pytest.mark.parametrize("num_top,expect_aggregated", [
        (4, False),    # well below threshold
        (31, False),   # just below threshold
        (32, True),    # exactly at threshold
        (33, True),    # just above threshold
        (64, True),    # well above
    ])
    def test_threshold_boundary(self, capfd, num_top, expect_aggregated):
        """Verify log aggregation kicks in at exactly N=32."""
        C = 256
        _ = capfd.readouterr()
        net = _make_split_net(num_top, C)
        inp = np.random.randn(1, C).astype(np.float32)
        out = net.Forward({"data": inp})
        captured = capfd.readouterr()
        log_output = captured.out

        perf_lines = _count_split_perf_lines(log_output)

        # All N values should produce at least 1 [SPLIT-PERF] summary line
        assert perf_lines >= 1, (
            f"N={num_top}: expected at least 1 [SPLIT-PERF] line, got {perf_lines}\n"
            f"Log output:\n{log_output}"
        )

        if expect_aggregated:
            # Aggregated: bounded number of summary lines (summary only, no per-top flood)
            assert perf_lines <= 10, (
                f"N={num_top} should be aggregated but produced {perf_lines} lines\n"
                f"Log output:\n{log_output}"
            )
            # Verify log_aggregated=yes appears in summary
            has_agg_flag = any("log_aggregated=yes" in line for line in log_output.splitlines())
            assert has_agg_flag, (
                f"N={num_top}: expected log_aggregated=yes in [SPLIT-PERF] output"
            )
        else:
            # Not aggregated: summary lines present with log_aggregated=no
            has_no_agg_flag = any("log_aggregated=no" in line for line in log_output.splitlines())
            assert has_no_agg_flag, (
                f"N={num_top}: expected log_aggregated=no in [SPLIT-PERF] output"
            )

        # Functional correctness regardless of log mode
        for i in range(num_top):
            np.testing.assert_array_almost_equal(
                net.blob_by_name(f"split_{i}").to_numpy(), inp
            )


@require_cpp_extension
class TestLogAggregationCorrectness:
    """Verify log aggregation does not affect functional correctness."""

    def test_n100_split_forward_plus_relu(self):
        """N=100 Split + ReLU downstream: outputs correct."""
        num_top = 100
        tops = "\n".join(f'  top: "split_{i}"' for i in range(num_top))
        prototxt = f"""name: "split_relu_n100"
layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape {{ dim: 1 dim: 64 }} }}
}}
layer {{
  name: "split"
  type: "Split"
  bottom: "data"
{tops}
}}
layer {{
  name: "relu"
  type: "ReLU"
  bottom: "split_0"
  top: "relu_out"
}}
"""
        net = caffe_ffi.Net(prototxt)
        inp = np.random.randn(1, 64).astype(np.float32)

        out = net.Forward({"data": inp})

        assert "relu_out" in out
        expected = np.maximum(0, inp)
        np.testing.assert_array_almost_equal(
            net.blob_by_name("relu_out").to_numpy(), expected
        )


# ── Manual verification script ────────────────────────────────────────

if __name__ == "__main__":
    print("Phase 3.0 Log Aggregation Manual Verification")
    print("=" * 60)

    num_top = 100
    C = 1024

    net = _make_split_net(num_top, C)
    inp = np.random.randn(1, C).astype(np.float32)

    print(f"Running Split with N={num_top}, feat_dim={C}...")
    out = net.Forward({"data": inp})

    # Verify correctness
    all_ok = True
    for i in range(num_top):
        key = f"split_{i}"
        if not np.allclose(net.blob_by_name(key).to_numpy(), inp):
            print(f"  FAIL: {key} differs from input")
            all_ok = False
    if all_ok:
        print(f"  PASS: All {num_top} outputs match input")

    print(f"\nLog aggregation threshold: 32")
    print(f"N={num_top} >= 32 → aggregation ENABLED")
    print(f"Expected [SPLIT-PERF] lines: <= 10 (Reshape summary + Forward summary)")
