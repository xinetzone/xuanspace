from __future__ import annotations

import gc
import sys
from pathlib import Path

import pytest
import numpy as np

_project_root = Path(__file__).resolve().parent.parent.parent
_python_dir = _project_root / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))

from caffe_ffi import _ffi_api

_previous_test_name = None
_test_baseline = None
_session_baseline = None
_previous_test_passed = True


def _current_mem_state():
    from caffe_ffi import total_allocated_bytes, live_blob_count
    gc.collect()
    gc.collect()
    gc.collect()
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
        if leaked_blobs != 0 or leaked_bytes > 0:
            pytest.fail(
                f"Memory leak detected from {_previous_test_name}: "
                f"{leaked_blobs} Blob(s) still alive "
                f"(prev={blobs_before}, now={blobs_after}), "
                f"{leaked_bytes} bytes leaked "
                f"(prev={mem_before}, now={mem_after})"
            )

    _test_baseline = current
    _previous_test_name = item.name


def pytest_sessionfinish(session, exitstatus):
    """Final check: memory should return to session baseline after all tests."""
    if not _ffi_api.is_available() or _session_baseline is None:
        return
    import warnings
    current = _current_mem_state()
    mem_before, blobs_before = _session_baseline
    mem_after, blobs_after = current
    leaked_bytes = mem_after - mem_before
    leaked_blobs = blobs_after - blobs_before
    if leaked_blobs != 0 or leaked_bytes > 0:
        warnings.warn(
            f"Global memory leak after all tests: "
            f"{leaked_blobs} Blob(s) still alive, {leaked_bytes} bytes leaked. "
            f"(May be caused by failed tests holding traceback references.)",
            stacklevel=2,
        )


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
