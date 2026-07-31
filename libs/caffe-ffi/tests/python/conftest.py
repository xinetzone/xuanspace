from __future__ import annotations

import csv
import gc
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import pytest
import numpy as np

_project_root = Path(__file__).resolve().parent.parent.parent
_python_dir = _project_root / "python"
_temp_dir = _project_root / "tests" / "python" / ".temp"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))

from caffe_ffi import _ffi_api

# ─── Performance tracing infrastructure ────────────────────────────

_perf_logger = logging.getLogger("caffe_ffi.test.perf")
_csv_writer = None
_csv_file = None
_csv_path = None


def _ensure_csv():
    """Initialize CSV file for performance logging (lazy, on first write)."""
    global _csv_writer, _csv_file, _csv_path
    if _csv_writer is not None:
        return
    _temp_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _csv_path = _temp_dir / f"perf_log_{ts}.csv"
    _csv_file = open(_csv_path, "w", newline="", encoding="utf-8")
    _csv_writer = csv.writer(_csv_file)
    _csv_writer.writerow([
        "timestamp", "test_class", "test_name", "operation",
        "elapsed_ms", "delta_mem", "delta_blobs",
        "cow_events", "cow_bytes", "cow_saved_bytes",
        "extra_fields"
    ])
    _csv_file.flush()


def _write_csv_row(test_class: str, test_name: str, operation: str,
                   elapsed_ms: float, delta_mem: int, delta_blobs: int,
                   extra: str = "", *,
                   cow_events: int = 0, cow_bytes: int = 0,
                   cow_saved_bytes: int = 0):
    """Write one row to the performance CSV file."""
    _ensure_csv()
    _csv_writer.writerow([
        datetime.now().isoformat(timespec="milliseconds"),
        test_class, test_name, operation,
        f"{elapsed_ms:.4f}", delta_mem, delta_blobs,
        cow_events, cow_bytes, cow_saved_bytes,
        extra,
    ])
    _csv_file.flush()


def _write_cow_csv_row(test_class: str, test_name: str, operation: str,
                       refcount_before: int, copy_bytes: int, copy_us: float,
                       blob_id: int = -1, extra: str = ""):
    """Write a COW-specific event row to the performance CSV.
    
    Records a Copy-on-Write trigger event with refcount and copy details.
    Use operation='COW-Data' or 'COW-Diff' to distinguish data vs diff COW.
    """
    _ensure_csv()
    _csv_writer.writerow([
        datetime.now().isoformat(timespec="milliseconds"),
        test_class, test_name, operation,
        f"{copy_us / 1000.0:.4f}", 0, 0,  # elapsed_ms, delta_mem, delta_blobs
        1, copy_bytes, 0,  # cow_events=1, cow_bytes, cow_saved_bytes
        f"blob_id={blob_id} refcount_before={refcount_before} copy_bytes={copy_bytes} copy_us={copy_us:.1f} {extra}",
    ])
    _csv_file.flush()


def cow_snapshot(blob) -> dict:
    """Query COW state of a Blob's data and diff tensors.

    Returns a dict with:
      - data_shared: bool, whether data tensor is shared (refcount > 1)
      - diff_shared: bool, whether diff tensor is shared (refcount > 1)
      - data_refcount: int, data tensor refcount (0 if undefined)
      - diff_refcount: int, diff tensor refcount (0 if undefined)
    """
    return {
        "data_shared": blob.IsDataShared(),
        "diff_shared": blob.IsDiffShared(),
        "data_refcount": blob.DataRefCount(),
        "diff_refcount": blob.DiffRefCount(),
    }


if not _perf_logger.handlers:
    _perf_handler = logging.StreamHandler(sys.stderr)
    _perf_handler.setLevel(logging.INFO)
    _perf_handler.setFormatter(logging.Formatter(
        "%(asctime)s [PERF] %(message)s",
        datefmt="%H:%M:%S",
    ))
    _perf_logger.addHandler(_perf_handler)
    _perf_logger.propagate = False
_perf_logger.setLevel(logging.INFO)


def _mem_bytes_blobs():
    """Return (total_allocated_bytes, live_blob_count) after aggressive GC."""
    from caffe_ffi import total_allocated_bytes, live_blob_count
    for _ in range(3):
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
    return total_allocated_bytes(), live_blob_count()


_current_test_context = {"cls": "", "name": ""}


@contextmanager
def perf_trace(label: str, verbose: bool = True) -> Iterator[dict]:
    """Context manager that measures wall-clock time and memory delta for a block.

    Yields a dict that the caller may mutate to add extra fields (e.g. 'shape',
    'input_size'); these are appended to the exit log line and CSV.

    If an exception is raised inside the block, it is logged with exception type
    and message before being re-raised. The caller may set info['expected_error']=True
    to indicate the error is expected (e.g. boundary testing).

    Usage:
        with perf_trace("Net(prototxt)") as t:
            net = Net(prototxt)
            t['layers'] = len(net.layers_array())
        # Logs: [PERF] Net(prototxt) ... Δtime=12.3ms Δmem=+4096B Δblobs=+5 layers=5
    """
    mem_before, blobs_before = _mem_bytes_blobs()
    t0 = time.perf_counter()
    info: dict = {}
    exc_info = None
    try:
        yield info
    except BaseException as e:
        exc_info = (type(e).__name__, str(e)[:200])
        info["exception"] = f"{type(e).__name__}: {str(e)[:120]}"
        raise
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        mem_after, blobs_after = _mem_bytes_blobs()
        delta_mem = mem_after - mem_before
        delta_blobs = blobs_after - blobs_before
        extra_parts = [f"{k}={v}" for k, v in info.items()
                       if k not in ("elapsed_ms", "delta_mem", "delta_blobs")]
        extra_str = " ".join(extra_parts)
        if verbose:
            mem_str = f"+{delta_mem}B" if delta_mem >= 0 else f"{delta_mem}B"
            blob_str = f"+{delta_blobs}" if delta_blobs >= 0 else f"{delta_blobs}"
            if exc_info is not None:
                status = "EXC" if not info.get("expected_error") else "EXP"
                _perf_logger.info(
                    "%-40s Δtime=%7.2fms  Δmem=%8s  Δblobs=%4s  [%s] %s: %s  %s",
                    label, elapsed_ms, mem_str, blob_str, status,
                    exc_info[0], exc_info[1][:100], extra_str,
                )
            else:
                _perf_logger.info(
                    "%-40s Δtime=%7.2fms  Δmem=%8s  Δblobs=%4s  %s",
                    label, elapsed_ms, mem_str, blob_str, extra_str,
                )
        _write_csv_row(
            _current_test_context["cls"], _current_test_context["name"],
            label, elapsed_ms, delta_mem, delta_blobs, extra_str,
        )
        info["elapsed_ms"] = elapsed_ms
        info["delta_mem"] = delta_mem
        info["delta_blobs"] = delta_blobs

# Configure memory stress test logger to output INFO logs during test runs
_mem_stress_logger = logging.getLogger("caffe_ffi.test.memory_stress")
_mem_stress_logger.setLevel(logging.INFO)
if not _mem_stress_logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setLevel(logging.INFO)
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    ))
    _mem_stress_logger.addHandler(_handler)
    _mem_stress_logger.propagate = False

_previous_test_name = None
_test_baseline = None
_session_baseline = None
_previous_test_passed = True


def _current_mem_state():
    from caffe_ffi import total_allocated_bytes, live_blob_count
    # Aggressive GC: collect across all generations repeatedly until counts stabilize
    for _ in range(5):
        gc.collect(0)
        gc.collect(1)
        gc.collect(2)
    return (total_allocated_bytes(), live_blob_count())


def pytest_configure(config):
    global _session_baseline
    config.addinivalue_line(
        "markers", "require_cpp_extension: mark test as requiring C++ extension"
    )
    config.addinivalue_line(
        "markers", "leak_check: mark test to check for Blob memory leaks (default: autouse)"
    )
    if _ffi_api.is_available():
        _session_baseline = _current_mem_state()


def _check_cpp_extension_available() -> bool:
    return _ffi_api.is_available()


require_cpp_extension = pytest.mark.skipif(
    not _check_cpp_extension_available(),
    reason="C++ extension not available, skipping test"
)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Track whether the previous test passed (for leak checking)."""
    global _previous_test_passed
    outcome = yield
    _previous_test_passed = outcome.excinfo is None


def pytest_runtest_setup(item):
    """Check for leaks from the PREVIOUS test at the start of the current test.

    At this point, all function-scoped fixtures from the previous test have
    been torn down, so any leaked blobs indicate a real leak.
    Skips the check if the previous test failed (exception tracebacks hold locals).
    """
    global _previous_test_name, _test_baseline, _previous_test_passed
    if not _ffi_api.is_available():
        return

    current = _current_mem_state()

    leak_marker = item.get_closest_marker("leak_check")
    if leak_marker is not None and leak_marker.args and leak_marker.args[0] is False:
        _test_baseline = current
        _previous_test_name = item.name
        _previous_test_passed = True
        return

    if _test_baseline is not None and _previous_test_name is not None and _previous_test_passed:
        mem_before, blobs_before = _test_baseline
        mem_after, blobs_after = current
        leaked_bytes = mem_after - mem_before
        leaked_blobs = blobs_after - blobs_before
        if leaked_blobs > 0 or leaked_bytes > 0:
            # Positive leak: blobs/bytes INCREASED after a passing test → real leak
            pytest.fail(
                f"Memory leak detected from {_previous_test_name}: "
                f"+{leaked_blobs} Blob(s) still alive "
                f"(prev={blobs_before}, now={blobs_after}), "
                f"+{leaked_bytes} bytes leaked "
                f"(prev={mem_before}, now={mem_after})"
            )
        # Negative delta (blobs/bytes decreased) = delayed GC from a prior failed test's
        # exception traceback finally releasing locals. Not a leak — just reset baseline.

    _test_baseline = current
    _previous_test_name = item.name


def pytest_sessionfinish(session, exitstatus):
    """Final check: memory should return to session baseline after all tests."""
    global _csv_writer, _csv_file, _csv_path
    # Close CSV file and report path
    if _csv_file is not None:
        _csv_file.close()
        _csv_file = None
        _csv_writer = None
        _perf_logger.info("Performance CSV saved to: %s", _csv_path)
        print(f"\n[PERF-CSV] Performance log saved to: {_csv_path}", file=sys.stderr)

    if not _ffi_api.is_available() or _session_baseline is None:
        return
    import warnings
    current = _current_mem_state()
    mem_before, blobs_before = _session_baseline
    mem_after, blobs_after = current
    leaked_bytes = mem_after - mem_before
    leaked_blobs = blobs_after - blobs_before
    if leaked_blobs != 0 or leaked_bytes != 0:
        warnings.warn(
            f"Global memory leak after all tests: "
            f"{leaked_blobs} Blob(s) still alive, {leaked_bytes} bytes leaked. "
            f"(May be caused by failed tests holding traceback references.)",
            stacklevel=2,
        )


@pytest.fixture
def ptrace():
    """Provide perf_trace context manager as a fixture for P1 detail logging."""
    return perf_trace


# ─── Test-level timing autouse (logs per-test wall time + memory) ──

_P1_TEST_CLASSES = {
    "TestLayerStandalone", "TestLayerFromNet",
    "TestNetEmptyConstructor", "TestNetConstructorErrors",
    "TestNetForwardBoundaries", "TestNetConsistency",
}

_P2_TEST_CLASSES = {
    "TestNetTopologies", "TestNetReshapeDynamics", "TestLargeScaleForward",
}

_P2B_TEST_CLASSES = {
    "TestExtremeValues", "TestDTypeErrors", "TestNonContiguousArrays",
    "TestRecoveryAfterError", "TestSplitTopologies", "TestExtremeBoundaries",
    "TestBlobCOWApi", "TestSplitCOWBehavior",
}

_PERF_TEST_CLASSES = _P1_TEST_CLASSES | _P2_TEST_CLASSES | _P2B_TEST_CLASSES


@pytest.fixture(autouse=True)
def _test_timing_log(request):
    """Autouse fixture: log timing + memory delta for every P1/P2 test case."""
    test_name = request.node.name
    cls_name = request.cls.__name__ if request.cls else ""
    is_perf = cls_name in _PERF_TEST_CLASSES
    if not is_perf:
        yield
        return

    _current_test_context["cls"] = cls_name
    _current_test_context["name"] = test_name

    mem_before, blobs_before = _mem_bytes_blobs()
    t0 = time.perf_counter()
    _perf_logger.info("─── BEGIN %s.%s ───  mem=%dB blobs=%d", cls_name, test_name, mem_before, blobs_before)
    _write_csv_row(cls_name, test_name, "BEGIN", 0.0, 0, 0, f"mem={mem_before} blobs={blobs_before}")
    yield
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    mem_after, blobs_after = _mem_bytes_blobs()
    delta_mem = mem_after - mem_before
    delta_blobs = blobs_after - blobs_before
    mem_str = f"+{delta_mem}B" if delta_mem >= 0 else f"{delta_mem}B"
    blob_str = f"+{delta_blobs}" if delta_blobs >= 0 else f"{delta_blobs}"
    _perf_logger.info(
        "─── END   %s.%s ───  Δtime=%.2fms  Δmem=%s  Δblobs=%s  total=%dB/%d blobs",
        cls_name, test_name, elapsed_ms, mem_str, blob_str, mem_after, blobs_after,
    )
    _write_csv_row(cls_name, test_name, "END", elapsed_ms, delta_mem, delta_blobs,
                   f"total_mem={mem_after} total_blobs={blobs_after}")

    _current_test_context["cls"] = ""
    _current_test_context["name"] = ""


@pytest.fixture
def mlp_prototxt() -> str:
    """Simple MLP prototxt: Input -> InnerProduct -> ReLU -> InnerProduct -> Softmax."""
    return """name: "mlp_test"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 2 dim: 3 } }
}
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "data"
  top: "ip1"
  inner_product_param {
    num_output: 4
    bias_term: true
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "ip1"
  top: "ip1"
}
layer {
  name: "ip2"
  type: "InnerProduct"
  bottom: "ip1"
  top: "ip2"
  inner_product_param {
    num_output: 2
    bias_term: true
  }
}
layer {
  name: "prob"
  type: "Softmax"
  bottom: "ip2"
  top: "prob"
}
"""


@pytest.fixture
def mlp_weights():
    """Manual weights for MLP testing."""
    np.random.seed(42)
    W1 = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
    ], dtype=np.float32)
    b1 = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)

    W2 = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ], dtype=np.float32)
    b2 = np.array([0.001, 0.002], dtype=np.float32)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


@pytest.fixture
def mlp_net(mlp_prototxt):
    """Create MLP network (uses C++ extension when available)."""
    from caffe_ffi import net_from_param, net_param_from_string
    from caffe_ffi._core import Blob

    param = net_param_from_string(mlp_prototxt)

    if _ffi_api.is_available():
        net = net_from_param(param)
        W1 = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
        ], dtype=np.float32)
        b1 = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
        W2 = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ], dtype=np.float32)
        b2 = np.array([0.001, 0.002], dtype=np.float32)

        layers = net.layers_array()
        if len(layers) >= 1 and hasattr(layers[0], 'blobs') and len(layers[0].blobs) >= 2:
            layers[0].blobs[0].from_numpy(W1)
            layers[0].blobs[1].from_numpy(b1.reshape(-1))
        if len(layers) >= 3 and hasattr(layers[2], 'blobs') and len(layers[2].blobs) >= 2:
            layers[2].blobs[0].from_numpy(W2)
            layers[2].blobs[1].from_numpy(b2.reshape(-1))
    else:
        net = _build_mlp_python(param)

    return net


def _build_mlp_python(param):
    """Build a minimal MLP net in pure Python for testing (without C++ extension)."""
    from caffe_ffi._core import Net, Blob, Layer

    net = Net.__new__(Net)
    net._py_name = param.name
    net._py_blobs = {}
    net._py_layers = {}
    net._py_blob_list = []
    net._py_layer_list = []
    net._py_input_blobs = []
    net._py_output_blobs = []

    data_blob = Blob([2, 3])
    data_blob.name = "data"
    net._py_blobs["data"] = data_blob
    net._py_blob_list.append(data_blob)
    net._py_input_blobs.append(data_blob)

    ip1_blob = Blob([2, 4])
    ip1_blob.name = "ip1"
    net._py_blobs["ip1"] = ip1_blob
    net._py_blob_list.append(ip1_blob)

    ip2_blob = Blob([2, 2])
    ip2_blob.name = "ip2"
    net._py_blobs["ip2"] = ip2_blob
    net._py_blob_list.append(ip2_blob)

    prob_blob = Blob([2, 2])
    prob_blob.name = "prob"
    net._py_blobs["prob"] = prob_blob
    net._py_blob_list.append(prob_blob)
    net._py_output_blobs.append(prob_blob)

    W1 = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
    ], dtype=np.float32)
    b1 = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
    W2 = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ], dtype=np.float32)
    b2 = np.array([0.001, 0.002], dtype=np.float32)

    def _make_layer(name, type_str, weight_blobs=None):
        layer = Layer()
        layer._py_name = name
        layer._py_type_str = type_str
        layer._py_blobs = list(weight_blobs) if weight_blobs else []
        return layer

    ip1_w = Blob([4, 3])
    ip1_w.from_numpy(W1)
    ip1_b = Blob([4])
    ip1_b.from_numpy(b1)
    ip1_layer = _make_layer("ip1", "InnerProduct", [ip1_w, ip1_b])

    relu1_layer = _make_layer("relu1", "ReLU")

    ip2_w = Blob([2, 4])
    ip2_w.from_numpy(W2)
    ip2_b = Blob([2])
    ip2_b.from_numpy(b2)
    ip2_layer = _make_layer("ip2", "InnerProduct", [ip2_w, ip2_b])

    prob_layer = _make_layer("prob", "Softmax")

    net._py_layers["ip1"] = ip1_layer
    net._py_layers["relu1"] = relu1_layer
    net._py_layers["ip2"] = ip2_layer
    net._py_layers["prob"] = prob_layer
    net._py_layer_list = [ip1_layer, relu1_layer, ip2_layer, prob_layer]

    def _forward_pure_python(self, input_dict):
        data = input_dict.get("data", self._py_blobs["data"].data)
        self._py_blobs["data"].data = data

        W1 = self._py_layers["ip1"]._py_blobs[0].data
        b1 = self._py_layers["ip1"]._py_blobs[1].data
        ip1 = np.maximum(0, data @ W1.T + b1)
        self._py_blobs["ip1"].data = ip1

        W2 = self._py_layers["ip2"]._py_blobs[0].data
        b2 = self._py_layers["ip2"]._py_blobs[1].data
        ip2 = ip1 @ W2.T + b2
        self._py_blobs["ip2"].data = ip2

        exp_ip2 = np.exp(ip2 - np.max(ip2, axis=1, keepdims=True))
        prob = exp_ip2 / np.sum(exp_ip2, axis=1, keepdims=True)
        self._py_blobs["prob"].data = prob

        return {"prob": prob}

    import types
    net._forward_pure_python = types.MethodType(_forward_pure_python, net)

    return net
