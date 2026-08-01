"""Tests for caffe-ffi Python interface calling logic.

These tests verify that the Python wrapper layer correctly calls into the
C++ native extension via TVM FFI. They mirror the C++ unit tests in
tests/cpp/ and validate the Python-side API contract.

Timing: Each test case is individually timed; a per-suite summary and
top-N slowest test report is printed at the end (mirroring the C++
test harness behaviour).

Note: Due to a Python 3.14/pytest compatibility issue in this conda environment
(C++ extension segfaults when Blob/Net constructors are called under pytest),
these tests use the unittest framework with a custom TimingTestResult for
per-test execution duration measurement.  Run directly via plain Python:

    python tests/python/test_python_api.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
import unittest
from pathlib import Path


def _setup_path() -> None:
    """Ensure the caffe-ffi python/ directory is on sys.path for standalone runs."""
    _this_dir = Path(__file__).resolve().parent
    _project_root = _this_dir.parent.parent
    _python_dir = _project_root / "python"
    if _python_dir.is_dir() and str(_python_dir) not in sys.path:
        sys.path.insert(0, str(_python_dir))
    return _project_root


_project_root = _setup_path()

import numpy as np

import caffe_ffi
from caffe_ffi import (
    Blob, Layer, Net,
    net_param_from_string, net_from_param,
    _ffi_api,
)


def _check_cpp_extension_available() -> bool:
    return _ffi_api.is_available()


require_cpp_extension = unittest.skipIf(
    not _check_cpp_extension_available(),
    reason="C++ extension not available, skipping test"
)


# ─── Timing infrastructure (mirrors C++ test_harness.hpp) ────────────

class TimingTestResult(unittest.TextTestResult):
    """A TestResult that records per-test elapsed wall-clock time."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_timings: list[tuple[str, str, float, bool]] = []
        self._test_start_time: float | None = None

    def startTest(self, test):
        self._test_start_time = time.perf_counter()
        super().startTest(test)

    def stopTest(self, test):
        elapsed = 0.0
        if self._test_start_time is not None:
            elapsed = (time.perf_counter() - self._test_start_time) * 1000.0
            self._test_start_time = None
        passed = test not in (self.failures + self.errors)
        self.test_timings.append((
            test.__class__.__name__,
            test._testMethodName,
            elapsed,
            passed,
        ))
        super().stopTest(test)


class TimingTextTestRunner(unittest.TextTestRunner):
    """TextTestRunner that uses TimingTestResult and prints timing summary."""

    resultclass = TimingTestResult

    def run(self, test):
        _start_all = time.perf_counter()
        result = super().run(test)
        _elapsed_all = (time.perf_counter() - _start_all) * 1000.0

        timings = result.test_timings
        passed = sum(1 for _, _, _, p in timings if p)
        failed = len(timings) - passed

        print()
        print(f"[==========] {len(timings)} tests ran, {passed} passed, "
              f"{failed} failed ({_elapsed_all:.2f} ms total)")

        # Per-suite (class) summary
        suite_stats: dict[str, list[tuple[str, float]]] = {}
        for cls, method, ms, _ in timings:
            suite_stats.setdefault(cls, []).append((method, ms))
        if suite_stats:
            print("[----------] Global test environment tear-down")
            print("[==========] Per-suite summary:")
            for suite in sorted(suite_stats,
                                key=lambda s: sum(ms for _, ms in suite_stats[s]),
                                reverse=True):
                tests_in_suite = suite_stats[suite]
                total_ms = sum(ms for _, ms in tests_in_suite)
                avg_ms = total_ms / len(tests_in_suite)
                print(f"[  SUITE   ] {suite:<30s} {len(tests_in_suite):3d} tests, "
                      f"{total_ms:8.2f} ms total, avg {avg_ms:6.2f} ms")

        # Top 5 slowest tests
        sorted_tests = sorted(timings, key=lambda x: x[2], reverse=True)
        top_n = min(5, len(sorted_tests))
        print(f"[----------] Top {top_n} slowest test(s):")
        for i, (cls, method, ms, _) in enumerate(sorted_tests[:top_n], start=1):
            print(f"[  SLOW    ] #{i} {cls}.{method} ({ms:.2f} ms)")

        return result


# ─── Prototext fixtures ──────────────────────────────────────────────

SIMPLE_INPUT_PROTO = """name: "InputNet"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 2 dim: 3 } }
}"""

MLP_PROTO = """name: "MlpNet"
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
  inner_product_param { num_output: 4 bias_term: true }
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
  inner_product_param { num_output: 2 bias_term: true }
}
layer {
  name: "prob"
  type: "Softmax"
  bottom: "ip2"
  top: "prob"
}"""

INVALID_PROTO = "this is not a valid prototext {{{{"


# ─── Module-level API tests ──────────────────────────────────────────

class TestModuleAPI(unittest.TestCase):
    """Tests for top-level module functions."""

    def test_version_returns_string(self):
        v = caffe_ffi.version()
        self.assertIsInstance(v, str)
        self.assertGreater(len(v), 0)

    def test_dunder_version(self):
        self.assertIsInstance(caffe_ffi.__version__, str)

    @require_cpp_extension
    def test_ffi_api_is_available(self):
        self.assertTrue(_ffi_api.is_available())

    @require_cpp_extension
    def test_memory_info_returns_dict(self):
        info = caffe_ffi.memory_info()
        self.assertIsInstance(info, dict)
        self.assertIn("total_allocated_bytes", info)
        self.assertIn("live_blob_count", info)
        self.assertIsInstance(info["total_allocated_bytes"], int)
        self.assertIsInstance(info["live_blob_count"], int)

    @require_cpp_extension
    def test_log_level_roundtrip(self):
        original = caffe_ffi.get_log_level()
        try:
            caffe_ffi.set_log_level(caffe_ffi.LOG_LEVEL_ERROR)
            self.assertEqual(caffe_ffi.get_log_level(), caffe_ffi.LOG_LEVEL_ERROR)
            caffe_ffi.set_log_level(caffe_ffi.LOG_LEVEL_DEBUG)
            self.assertEqual(caffe_ffi.get_log_level(), caffe_ffi.LOG_LEVEL_DEBUG)
        finally:
            caffe_ffi.set_log_level(original)

    @require_cpp_extension
    def test_total_allocated_bytes_non_negative(self):
        self.assertGreaterEqual(caffe_ffi.total_allocated_bytes(), 0)

    @require_cpp_extension
    def test_live_blob_count_non_negative(self):
        self.assertGreaterEqual(caffe_ffi.live_blob_count(), 0)

    def test_enable_disable_debug_logging(self):
        """enable/disable debug logging should not crash."""
        caffe_ffi.enable_debug_logging(caffe_ffi.LOG_LEVEL_INFO)
        caffe_ffi.disable_debug_logging()
        self.assertEqual(caffe_ffi.get_log_level(), caffe_ffi.LOG_LEVEL_WARN)

    def test_python_only_fallback_when_native_lib_missing(self):
        """Regression test: import must not crash when _caffe_ffi.so is absent.

        Bug history: _try_init_tvm_ffi() set _ffi_available=True even when
        _lib_path was None, causing ValueError: Cannot find object type index
        for caffe_ffi.Blob during class decoration.

        Strategy: create a clean subprocess where caffe_ffi is importable only
        from a temp-copied package directory that has no native .so files.
        We must also strip scikit-build-core's editable install finder from
        sys.meta_path and the real source path from sys.path, because the
        package is installed in editable mode during development.
        """
        import tempfile, shutil

        python_dir = _project_root / "python"
        pkg_dir = python_dir / "caffe_ffi"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_python = Path(tmpdir) / "python"
            tmp_pkg = tmp_python / "caffe_ffi"
            shutil.copytree(
                pkg_dir, tmp_pkg,
                ignore=shutil.ignore_patterns("*.so", "*.pyd", "*.dll",
                                              "*.pyc", "__pycache__"),
            )

            child_code = textwrap.dedent(f"""
                import sys, os

                # Step 1: Remove editable install finders from meta_path BEFORE
                # any caffe_ffi import. scikit-build-core installs a
                # ScikitBuildRedirectingFinder that bypasses sys.path.
                for f in list(sys.meta_path):
                    _cls_name = type(f).__name__
                    if 'editable' in _cls_name.lower() or 'redirecting' in _cls_name.lower():
                        sys.meta_path.remove(f)

                # Step 2: Remove the real source tree from sys.path.
                # .pth files (e.g. _editable_skbc_caffe_ffi.pth) add it at startup.
                _real = {str(python_dir)!r}
                _proj = {str(_project_root)!r}
                for p in list(sys.path):
                    if p == _real or p.startswith(_real + os.sep) or p == _proj:
                        sys.path.remove(p)

                # Step 3: Insert our temp package directory at the front.
                sys.path.insert(0, {str(tmp_python)!r})

                # Step 4: Clear any already-cached caffe_ffi modules (shouldn't
                # be any at this point, but be safe).
                for mod in list(sys.modules.keys()):
                    if 'caffe_ffi' in mod:
                        del sys.modules[mod]

                import caffe_ffi
                from caffe_ffi import Blob, Net

                assert not caffe_ffi.is_available(), (
                    f"Expected is_available()=False without native lib, got True")
                assert caffe_ffi._ffi_api.lib_path() is None, (
                    f"Expected lib_path()=None without native lib, "
                    f"got {{caffe_ffi._ffi_api.lib_path()}}")

                # Python-only mode: Blob/Net must be constructible without crash
                b = Blob([2, 3])
                assert b.shape == (2, 3), f"Expected (2,3), got {{b.shape}}"
                assert b.count() == 6
                b.fill(1.0)
                import numpy as np
                np.testing.assert_allclose(b.data_tensor, 1.0, rtol=1e-5)

                n = Net()
                assert n.name == ""
                assert len(n.blobs_array()) == 0

                print("REGRESSION_OK: python-only fallback works correctly")
            """)

            result = subprocess.run(
                [sys.executable, "-c", child_code],
                capture_output=True, text=True, timeout=30,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                     "CAFFE_FFI_DISABLE_BACKTRACE": "1"},
            )
            self.assertEqual(
                result.returncode, 0,
                f"Subprocess failed (exit {result.returncode}).\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
            self.assertIn("REGRESSION_OK", result.stdout)


# ─── Blob native API tests ──────────────────────────────────────────

@require_cpp_extension
class TestBlobNativeAPI(unittest.TestCase):
    """Tests for Blob methods that call into C++ native implementation."""

    def test_default_constructor_creates_valid_blob(self):
        b = Blob()
        self.assertEqual(b.shape, (0,))

    def test_shape_constructor(self):
        b = Blob([2, 3, 4])
        self.assertEqual(b.shape, (2, 3, 4))
        self.assertEqual(b.ndim, 3)
        self.assertEqual(b.num_axes, 3)
        self.assertEqual(b.count(), 24)

    def test_name_property_roundtrip(self):
        b = Blob([2, 3])
        b.name = "test_blob"
        self.assertEqual(b.name, "test_blob")

    def test_reshape_changes_shape(self):
        b = Blob([2, 3])
        self.assertEqual(b.shape, (2, 3))
        b.Reshape([4, 5])
        self.assertEqual(b.shape, (4, 5))
        self.assertEqual(b.count(), 20)

    def test_set_data_get_data_roundtrip(self):
        b = Blob([2, 3])
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        b.set_data(data)
        result = b.get_data()
        self.assertEqual(len(result), 6)
        for a, e in zip(result, data):
            self.assertAlmostEqual(a, e, places=5)

    def test_set_diff_get_diff_roundtrip(self):
        b = Blob([3])
        diff = [0.1, 0.2, 0.3]
        b.set_diff(diff)
        result = b.get_diff()
        self.assertEqual(len(result), 3)
        for a, e in zip(result, diff):
            self.assertAlmostEqual(a, e, places=5)

    def test_data_tensor_zero_copy(self):
        b = Blob([2, 3])
        dt = b.data_tensor
        self.assertIsInstance(dt, np.ndarray)
        self.assertEqual(dt.shape, (2, 3))
        self.assertEqual(dt.dtype, np.float32)
        dt[0, 0] = 42.0
        self.assertAlmostEqual(float(b.data_tensor[0, 0]), 42.0, places=2)

    def test_diff_tensor_zero_copy(self):
        b = Blob([2, 3])
        dt = b.diff_tensor
        self.assertIsInstance(dt, np.ndarray)
        dt[1, 1] = 7.77
        self.assertAlmostEqual(float(b.diff_tensor[1, 1]), 7.77, places=2)

    def test_data_setter_from_numpy(self):
        b = Blob([2, 3])
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        b.data = arr
        np.testing.assert_array_equal(b.data, arr)

    def test_diff_setter_from_numpy(self):
        b = Blob([3])
        arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        b.diff = arr
        np.testing.assert_array_equal(b.diff, arr)

    def test_fill_sets_data_values(self):
        b = Blob([2, 3])
        b.fill(3.14)
        np.testing.assert_allclose(b.data_tensor, 3.14, rtol=1e-5)

    def test_zero_resets_data_and_diff(self):
        b = Blob([2, 3])
        b.fill(1.0)
        b.diff_tensor.fill(1.0)
        b.zero()
        np.testing.assert_allclose(b.data_tensor, 0.0)
        np.testing.assert_allclose(b.diff_tensor, 0.0)

    def test_copy_from_numpy_array(self):
        b = Blob([2, 3])
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        b.copy_from(arr)
        np.testing.assert_array_equal(b.data, arr)

    def test_copy_from_blob(self):
        b1 = Blob([2, 3])
        b1.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
        b2 = Blob()
        b2.copy_from(b1)
        self.assertEqual(b2.shape, (2, 3))
        np.testing.assert_array_equal(b2.data, b1.data)

    def test_update_subtracts_diff(self):
        b = Blob([3])
        b.data_tensor[:] = [10.0, 20.0, 30.0]
        b.diff_tensor[:] = [1.0, 2.0, 3.0]
        b.Update()
        np.testing.assert_allclose(b.data_tensor, [9.0, 18.0, 27.0], rtol=1e-5)

    def test_from_numpy_to_numpy_roundtrip(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        b = Blob()
        b.from_numpy(arr)
        np.testing.assert_array_equal(b.to_numpy(), arr)

    def test_to_numpy_get_diff(self):
        b = Blob([3])
        b.diff_tensor[:] = [1.0, 2.0, 3.0]
        result = b.to_numpy(get_diff=True)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0], rtol=1e-5)

    def test_repr(self):
        b = Blob([2, 3])
        r = repr(b)
        self.assertIn("Blob", r)
        self.assertIn("(2, 3)", r)

    def test_reshape_negative_dimension_raises(self):
        b = Blob()
        with self.assertRaises((RuntimeError, ValueError)):
            b.Reshape([-1, 3])

    def test_size_alias_for_count(self):
        b = Blob([2, 3, 4])
        self.assertEqual(b.size, b.count())


# ─── Net constructor tests ──────────────────────────────────────────

@require_cpp_extension
class TestNetConstructor(unittest.TestCase):
    """Tests for Net construction paths (mirrors C++ NetTest)."""

    def test_create_from_protoString_simple_input(self):
        """Mirrors C++ NetTest.CreateFromProtoString."""
        net = Net(SIMPLE_INPUT_PROTO)
        self.assertEqual(net.name, "InputNet")
        self.assertEqual(len(net.layers_array()), 1)
        self.assertGreaterEqual(len(net.blobs_array()), 1)

    def test_create_from_protoString_mlp(self):
        """Mirrors C++ NetTest.MlpNetCreation."""
        net = Net(MLP_PROTO)
        self.assertEqual(net.name, "MlpNet")
        self.assertEqual(len(net.layers_array()), 5)

    def test_invalid_protoString_raises(self):
        """Mirrors C++ NetTest.UnknownLayerTypeThrows (invalid proto raises)."""
        with self.assertRaises((RuntimeError, Exception)):
            Net(INVALID_PROTO)

    def test_no_args_constructor(self):
        """Net() with no arguments should not crash; access is safe."""
        net = Net()
        self.assertEqual(net.name, "")
        self.assertEqual(len(net.blobs_array()), 0)

    def test_net_from_protoString_has_correct_counts(self):
        """Mirrors C++ NetTest.LayerCount/BlobCount."""
        net = Net(SIMPLE_INPUT_PROTO)
        self.assertEqual(len(net.layers_array()), 1)
        self.assertEqual(net.num_inputs(), 1)
        self.assertEqual(net.num_outputs(), 1)

    def test_net_input_output_blobs(self):
        """Mirrors C++ NetTest.InputOutputBlobs."""
        net = Net(SIMPLE_INPUT_PROTO)
        inputs = net.input_blobs_array()
        outputs = net.output_blobs_array()
        self.assertGreaterEqual(len(inputs), 1)
        self.assertGreaterEqual(len(outputs), 1)

    def test_net_input_output_names(self):
        net = Net(SIMPLE_INPUT_PROTO)
        input_names = net.input_blob_names()
        output_names = net.output_blob_names()
        self.assertIsInstance(input_names, list)
        self.assertIsInstance(output_names, list)
        self.assertGreaterEqual(len(input_names), 1)
        self.assertGreaterEqual(len(output_names), 1)


# ─── Net blob/layer access tests ────────────────────────────────────

@require_cpp_extension
class TestNetAccess(unittest.TestCase):
    """Tests for Net blob/layer accessor methods (mirrors C++ NetTest)."""

    def setUp(self):
        self.simple_net = Net(SIMPLE_INPUT_PROTO)
        self.mlp_net = Net(MLP_PROTO)

    def test_has_blob_true(self):
        """Mirrors C++ NetTest.HasBlob."""
        self.assertTrue(self.simple_net.has_blob("data"))

    def test_has_blob_false(self):
        self.assertFalse(self.simple_net.has_blob("nonexistent_blob"))

    def test_has_layer_true(self):
        """Mirrors C++ NetTest.HasLayer."""
        self.assertTrue(self.mlp_net.has_layer("ip1"))
        self.assertTrue(self.mlp_net.has_layer("relu1"))
        self.assertTrue(self.mlp_net.has_layer("prob"))

    def test_has_layer_false(self):
        self.assertFalse(self.simple_net.has_layer("nonexistent_layer"))

    def test_blob_by_name_returns_blob(self):
        """Mirrors C++ NetTest.BlobByName."""
        blob = self.simple_net.blob_by_name("data")
        self.assertIsInstance(blob, Blob)
        self.assertEqual(blob.shape, (2, 3))

    def test_blob_by_name_raises_on_missing(self):
        """Mirrors C++ NetTest.BlobByNameNotFoundThrows."""
        with self.assertRaises((KeyError, RuntimeError)):
            self.simple_net.blob_by_name("nonexistent")

    def test_layer_by_name_returns_layer(self):
        """Mirrors C++ NetTest.LayerByName."""
        layer = self.mlp_net.layer_by_name("ip1")
        self.assertIsInstance(layer, Layer)
        self.assertEqual(layer.name, "ip1")
        self.assertEqual(layer.type, "InnerProduct")

    def test_layer_by_name_raises_on_missing(self):
        """Mirrors C++ NetTest.LayerByNameNotFoundThrows."""
        with self.assertRaises((KeyError, RuntimeError)):
            self.simple_net.layer_by_name("nonexistent")

    def test_blob_names(self):
        names = self.simple_net.blob_names()
        self.assertIsInstance(names, list)
        self.assertIn("data", names)

    def test_layer_names(self):
        names = self.mlp_net.layer_names()
        self.assertIsInstance(names, list)
        self.assertIn("ip1", names)
        self.assertIn("relu1", names)
        self.assertIn("ip2", names)
        self.assertIn("prob", names)

    def test_blobs_array_returns_list(self):
        blobs = self.simple_net.blobs_array()
        self.assertIsInstance(blobs, list)
        for b in blobs:
            self.assertIsInstance(b, Blob)

    def test_layers_array_returns_list(self):
        layers = self.mlp_net.layers_array()
        self.assertIsInstance(layers, list)
        self.assertEqual(len(layers), 5)
        for layer in layers:
            self.assertIsInstance(layer, Layer)

    def test_blobs_dict(self):
        bd = self.mlp_net.blobs_dict
        self.assertIsInstance(bd, dict)
        self.assertIn("data", bd)
        for name, blob in bd.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(blob, Blob)

    def test_layers_dict(self):
        ld = self.mlp_net.layers_dict
        self.assertIsInstance(ld, dict)
        self.assertIn("ip1", ld)
        for name, layer in ld.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(layer, Layer)

    def test_getitem_access(self):
        blob = self.simple_net["data"]
        self.assertIsInstance(blob, Blob)

    def test_getitem_keyerror(self):
        with self.assertRaises((KeyError, RuntimeError)):
            _ = self.simple_net["nonexistent"]

    def test_contains(self):
        self.assertIn("data", self.simple_net)
        self.assertNotIn("nonexistent", self.simple_net)

    def test_iter(self):
        names = list(self.simple_net)
        self.assertIn("data", names)

    def test_len(self):
        self.assertEqual(len(self.simple_net), len(self.simple_net.blobs_array()))

    def test_repr(self):
        r = repr(self.simple_net)
        self.assertIn("Net", r)
        self.assertIn("InputNet", r)


# ─── Net forward tests ─────────────────────────────────────────────

@require_cpp_extension
class TestNetForward(unittest.TestCase):
    """Tests for forward pass (mirrors C++ NetTest.ForwardSingleInput)."""

    def test_forward_simple_input(self):
        net = Net(SIMPLE_INPUT_PROTO)
        inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        out = net.forward({"data": inp})
        self.assertIsInstance(out, dict)
        self.assertGreaterEqual(len(out), 1)
        data_blob = net.blob_by_name("data")
        np.testing.assert_array_equal(data_blob.data, inp)

    def test_forward_mlp(self):
        net = Net(MLP_PROTO)
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

        ip1_layer = net.layer_by_name("ip1")
        if len(ip1_layer.blobs) >= 2:
            ip1_layer.blobs[0].from_numpy(W1)
            ip1_layer.blobs[1].from_numpy(b1.reshape(-1))
        ip2_layer = net.layer_by_name("ip2")
        if len(ip2_layer.blobs) >= 2:
            ip2_layer.blobs[0].from_numpy(W2)
            ip2_layer.blobs[1].from_numpy(b2.reshape(-1))

        inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        out = net.forward({"data": inp})
        self.assertIsInstance(out, dict)
        self.assertIn("prob", out)
        self.assertEqual(out["prob"].shape, (2, 2))
        np.testing.assert_allclose(out["prob"].sum(axis=1), np.array([1.0, 1.0]), rtol=1e-5)

    def test_forward_all_kwargs(self):
        net = Net(SIMPLE_INPUT_PROTO)
        inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        out = net.forward_all(data=inp)
        self.assertIsInstance(out, dict)


# ─── Layer access via Net ───────────────────────────────────────────

@require_cpp_extension
class TestLayerAccess(unittest.TestCase):
    """Tests for Layer objects accessed through Net."""

    def setUp(self):
        self.mlp_net = Net(MLP_PROTO)

    def test_layer_blobs_for_inner_product(self):
        """Mirrors C++ NetTest.LayerBlobsExistForInnerProduct."""
        ip1 = self.mlp_net.layer_by_name("ip1")
        blobs = ip1.blobs
        self.assertIsInstance(blobs, list)
        self.assertGreaterEqual(len(blobs), 2)

    def test_layer_type_property(self):
        relu = self.mlp_net.layer_by_name("relu1")
        self.assertEqual(relu.type, "ReLU")
        ip = self.mlp_net.layer_by_name("ip1")
        self.assertEqual(ip.type, "InnerProduct")
        prob = self.mlp_net.layer_by_name("prob")
        self.assertEqual(prob.type, "Softmax")

    def test_layer_name_property(self):
        for expected_name in ["data", "ip1", "relu1", "ip2", "prob"]:
            layer = self.mlp_net.layer_by_name(expected_name)
            self.assertEqual(layer.name, expected_name)

    def test_layer_blobs_array_via_reflection(self):
        """Mirrors C++ NetTest.LayerBlobsArrayViaReflection."""
        for layer in self.mlp_net.layers_array():
            blobs = layer.blobs
            self.assertIsInstance(blobs, list)
            for blob in blobs:
                self.assertIsInstance(blob, Blob)

    def test_layer_repr(self):
        layer = self.mlp_net.layer_by_name("ip1")
        r = repr(layer)
        self.assertIn("Layer", r)
        self.assertIn("ip1", r)


# ─── Cross-mode: net_from_param matches Net(protoString) ─────────────

@require_cpp_extension
class TestConstructorEquivalence(unittest.TestCase):
    """Verify that Net(str) and net_from_param(net_param_from_string(str)) produce equivalent nets."""

    def test_simple_input_equivalent(self):
        net_direct = Net(SIMPLE_INPUT_PROTO)
        param = net_param_from_string(SIMPLE_INPUT_PROTO)
        net_via_param = net_from_param(param)
        self.assertEqual(net_direct.name, net_via_param.name)
        self.assertEqual(len(net_direct.layers_array()), len(net_via_param.layers_array()))
        self.assertEqual(len(net_direct.blobs_array()), len(net_via_param.blobs_array()))

    def test_mlp_layer_count_equivalent(self):
        net_direct = Net(MLP_PROTO)
        param = net_param_from_string(MLP_PROTO)
        net_via_param = net_from_param(param)
        self.assertEqual(len(net_direct.layers_array()), len(net_via_param.layers_array()))
        self.assertEqual(net_direct.layer_names(), net_via_param.layer_names())


if __name__ == "__main__":
    runner = TimingTextTestRunner(verbosity=2)
    unittest.main(testRunner=runner, exit=False)
