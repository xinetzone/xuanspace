"""Phase 3.0: Log aggregation verification test for N=100 Split.

Verifies that when N=100 (>= kLogAggregateThreshold=32), the [SPLIT-PERF]
log output is controlled to ≤ 10 lines, instead of the ~200 lines in Phase 2.

Test strategy:
  1. Capture CAFFE_FFI_LOG_WARN output during Split::Reshape() + Forward()
  2. Count lines containing [SPLIT-PERF]
  3. Assert line count ≤ 10

Usage:
  pytest tests/python/test_phase3_log_aggregation.py -v -s
"""
import os
import sys
import logging
import pytest
import numpy as np
import caffe_ffi


# ── Helpers ──────────────────────────────────────────────────────────

def _make_split_net(num_top: int, input_shape):
    """Create a minimal Net with a single Split layer of N=num_top."""
    from caffe_ffi import LayerParameter, NetParameter, Blob

    proto = NetParameter()
    proto.name = f"split_log_test_N{num_top}"

    # Input layer
    input_param = LayerParameter()
    input_param.type = "Input"
    input_param.name = "data"
    input_param.top.append("data")
    input_param.input_param.shape.add().dim[:] = list(input_shape)
    proto.layer.append(input_param)

    # Split layer
    split_param = LayerParameter()
    split_param.type = "Split"
    split_param.name = "split"
    split_param.bottom.append("data")
    for i in range(num_top):
        split_param.top.append(f"split_{i}")
    proto.layer.append(split_param)

    return proto


def _count_split_perf_lines(log_output: str) -> int:
    """Count lines containing [SPLIT-PERF] in log output."""
    return sum(1 for line in log_output.splitlines() if "[SPLIT-PERF]" in line)


def _count_reshape_lines(log_output: str) -> int:
    """Count lines containing 'Split Reshape' in log output."""
    return sum(1 for line in log_output.splitlines() if "Split Reshape" in line)


def _count_forward_lines(log_output: str) -> int:
    """Count lines containing 'Split Forward' in log output."""
    return sum(1 for line in log_output.splitlines() if "Split Forward" in line)


# ── Test Cases ────────────────────────────────────────────────────────

class TestLogAggregationN100:
    """Verify Phase 3.0 log aggregation for N=100 Split."""

    def test_n100_split_perf_lines_le_10(self):
        """N=100: [SPLIT-PERF] lines should be ≤ 10."""
        # This test requires the compiled C++ code with kLogAggregateThreshold.
        # When running in a build environment, it captures the actual log output.
        num_top = 100
        N = num_top
        C = 1024  # count per blob

        proto = _make_split_net(num_top, (1, C))
        net = caffe_ffi.Net(proto)
        inp = np.random.randn(1, C).astype(np.float32)

        # Capture WARN-level log output
        import io
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARN)
        logger = logging.getLogger("caffe_ffi")
        logger.addHandler(handler)

        try:
            out = net.Forward({"data": inp})
            log_output = log_capture.getvalue()
        finally:
            logger.removeHandler(handler)

        perf_lines = _count_split_perf_lines(log_output)
        assert perf_lines <= 10, (
            f"N=100 produced {perf_lines} [SPLIT-PERF] lines, "
            f"expected ≤ 10. Phase 3.0 log aggregation may not be active."
        )

    def test_n100_no_per_top_reshape_logs(self):
        """N=100: per-top 'Split Reshape: top[N]' should NOT appear."""
        num_top = 100

        proto = _make_split_net(num_top, (1, 512))
        net = caffe_ffi.Net(proto)
        inp = np.random.randn(1, 512).astype(np.float32)

        import io
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)  # Capture all levels
        logger = logging.getLogger("caffe_ffi")
        logger.addHandler(handler)

        try:
            out = net.Forward({"data": inp})
            log_output = log_capture.getvalue()
        finally:
            logger.removeHandler(handler)

        # Per-top reshape logs contain "Split Reshape: top["
        per_top_lines = _count_reshape_lines(log_output)
        assert per_top_lines == 0, (
            f"N=100 produced {per_top_lines} per-top reshape log lines, "
            f"expected 0 with log aggregation enabled."
        )

    def test_n4_per_top_logs_still_present(self):
        """N=4 (< threshold): per-top logs should still be present."""
        num_top = 4

        proto = _make_split_net(num_top, (1, 256))
        net = caffe_ffi.Net(proto)
        inp = np.random.randn(1, 256).astype(np.float32)

        import io
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger("caffe_ffi")
        logger.addHandler(handler)

        try:
            out = net.Forward({"data": inp})
            log_output = log_capture.getvalue()
        finally:
            logger.removeHandler(handler)

        # N=4 < 32, so per-top logs should still be emitted
        perf_lines = _count_split_perf_lines(log_output)
        # With N=4, we expect at least the summary line + some per-top info
        assert perf_lines >= 1, (
            f"N=4 produced only {perf_lines} [SPLIT-PERF] lines, "
            f"expected at least 1 (summary)."
        )

    def test_n100_forward_summary_present(self):
        """N=100: the summary [SPLIT-PERF] Forward log must be present."""
        num_top = 100

        proto = _make_split_net(num_top, (1, 1024))
        net = caffe_ffi.Net(proto)
        inp = np.random.randn(1, 1024).astype(np.float32)

        import io
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARN)
        logger = logging.getLogger("caffe_ffi")
        logger.addHandler(handler)

        try:
            out = net.Forward({"data": inp})
            log_output = log_capture.getvalue()
        finally:
            logger.removeHandler(handler)

        # The summary must contain "Forward(N=100" and "COW" or "COW-BATCH"
        has_forward_summary = any(
            "Forward(N=100" in line and "SPLIT-PERF" in line
            for line in log_output.splitlines()
        )
        assert has_forward_summary, (
            "N=100 Forward summary [SPLIT-PERF] log is missing. "
            "The summary log must always be emitted regardless of N."
        )

    def test_n100_reshape_summary_present(self):
        """N=100: the summary [SPLIT-PERF] Reshape log must be present."""
        num_top = 100

        proto = _make_split_net(num_top, (1, 1024))
        net = caffe_ffi.Net(proto)
        inp = np.random.randn(1, 1024).astype(np.float32)

        import io
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARN)
        logger = logging.getLogger("caffe_ffi")
        logger.addHandler(handler)

        try:
            out = net.Forward({"data": inp})
            log_output = log_capture.getvalue()
        finally:
            logger.removeHandler(handler)

        has_reshape_summary = any(
            "Reshape:" in line and "SPLIT-PERF" in line and "num_top=100" in line
            for line in log_output.splitlines()
        )
        assert has_reshape_summary, (
            "N=100 Reshape summary [SPLIT-PERF] log is missing."
        )


class TestLogAggregationBoundary:
    """Verify boundary behavior around kLogAggregateThreshold=32."""

    @pytest.mark.parametrize("num_top,expect_aggregated", [
        (31, False),   # just below threshold
        (32, True),    # exactly at threshold
        (33, True),    # just above threshold
    ])
    def test_threshold_boundary(self, num_top, expect_aggregated):
        """Verify log aggregation kicks in at exactly N=32."""
        proto = _make_split_net(num_top, (1, 256))
        net = caffe_ffi.Net(proto)
        inp = np.random.randn(1, 256).astype(np.float32)

        import io
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger("caffe_ffi")
        logger.addHandler(handler)

        try:
            out = net.Forward({"data": inp})
            log_output = log_capture.getvalue()
        finally:
            logger.removeHandler(handler)

        perf_lines = _count_split_perf_lines(log_output)

        if expect_aggregated:
            assert perf_lines <= 10, (
                f"N={num_top} should be aggregated but produced {perf_lines} lines"
            )
        else:
            # N=31 should still produce per-top logs
            assert perf_lines >= 1, (
                f"N={num_top} should have per-top logs but only produced {perf_lines}"
            )


class TestLogAggregationCorrectness:
    """Verify log aggregation does not affect functional correctness."""

    def test_n100_split_output_correct(self):
        """N=100: all top outputs should equal bottom input."""
        num_top = 100
        C = 128

        proto = _make_split_net(num_top, (1, C))
        net = caffe_ffi.Net(proto)
        inp = np.random.randn(1, C).astype(np.float32)

        out = net.Forward({"data": inp})

        for i in range(num_top):
            key = f"split_{i}"
            assert key in out, f"Missing output '{key}'"
            np.testing.assert_array_almost_equal(
                out[key], inp,
                err_msg=f"split_{i} output differs from input"
            )

    def test_n100_forward_deterministic(self):
        """N=100: two Forward calls should produce identical results."""
        num_top = 100
        C = 64

        proto = _make_split_net(num_top, (1, C))
        net = caffe_ffi.Net(proto)
        inp = np.random.randn(1, C).astype(np.float32)

        out1 = net.Forward({"data": inp})
        out2 = net.Forward({"data": inp})

        for i in range(num_top):
            key = f"split_{i}"
            np.testing.assert_array_equal(out1[key], out2[key])


# ── Manual verification script ────────────────────────────────────────

if __name__ == "__main__":
    """Manual verification: run with N=100 and count log lines.

    Usage:
      python tests/python/test_phase3_log_aggregation.py
    """
    print("Phase 3.0 Log Aggregation Manual Verification")
    print("=" * 60)

    num_top = 100
    C = 1024

    proto = _make_split_net(num_top, (1, C))
    net = caffe_ffi.Net(proto)
    inp = np.random.randn(1, C).astype(np.float32)

    print(f"Running Split with N={num_top}, count={C}...")
    out = net.Forward({"data": inp})

    # Verify correctness
    all_ok = True
    for i in range(num_top):
        key = f"split_{i}"
        if not np.allclose(out[key], inp):
            print(f"  FAIL: {key} differs from input")
            all_ok = False
    if all_ok:
        print(f"  PASS: All {num_top} outputs match input")

    print(f"\nLog aggregation threshold: 32")
    print(f"N={num_top} >= 32 → aggregation ENABLED")
    print(f"Expected [SPLIT-PERF] lines: ≤ 10 (Reshape summary + Forward summary)")
    print(f"Expected per-top logs: 0 (all skipped)")