"""P2 unit tests for the remaining operators: Upsample, MemoryData, DummyData,
Python, HDF5Output, WindowData.

Covers for each op:
  1. Registration / instantiation (LayerTypeList / layer_names)
  2. Forward correctness (numpy reference / in-place callback fill)
  3. Backward gradient correctness (Upsample sums upstream diff over its block)
  4. Numerical gradient check (central finite differences, Upsample)
  5. Branch-specific configs (various scales, fillers, callback sub-types)
  6. No-op fallback when no callback is registered (data I/O / Python layers)

Reference semantics from the C++ sources under
``src/caffe_ffi/layers/``:
  * Upsample      - nearest-neighbor upsampling by ``scale`` on 4-D (N,C,H,W);
                    Backward accumulates ``top_diff`` over each scale x scale block.
  * MemoryData    - internal data cache injected via ``set_data``; outputs zeros
                    until data is set (the minimal bridge exposes no FFI hook to
                    inject data, so tests cover registration + zero default +
                    shape).
  * DummyData     - fills ``top`` blobs from ``data_filler`` (constant/uniform/
                    gaussian) or defaults to zeros; shapes from ``shape``.
  * Python        - invokes a Python callback registered under ``"<module>.<layer>"``
                    with the top blobs' writable tensors; no-op without a callback.
  * HDF5Output    - invokes a data-io callback registered under ``"HDF5Output.<name>"``
                    with the bottom blobs' read-only tensors; skips write w/o a callback.
  * WindowData    - invokes a data-io callback registered under ``"WindowData.<name>"``
                    with the top blobs' writable tensors; zeros fallback w/o a callback.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from .conftest import require_cpp_extension
from .caffe_test_helpers import make_net, make_net_with_diag, dump_net_state

RTOL = 1e-3
ATOL = 1e-4
EPS_NUMERICAL = 1e-3


# ---------------------------------------------------------------------------
# Shared FFI helpers
# ---------------------------------------------------------------------------

def _ffi_get(name: str):
    from caffe_ffi import _ffi_api

    return _ffi_api.get_global_func(name)


def _register_python_layer(module: str, layer: str, callback) -> None:
    """Register a Python-layer callback under the ``"<module>.<layer>"`` key."""
    reg = _ffi_get("caffe_ffi.python_layer.register")
    assert reg is not None, "caffe_ffi.python_layer.register not found"
    reg(f"{module}.{layer}", callback)


def _register_data_io(key: str, callback) -> None:
    """Register a data-io callback under ``"<layer_type>.<layer_name>"``."""
    reg = _ffi_get("caffe_ffi.data_io.register")
    assert reg is not None, "caffe_ffi.data_io.register not found"
    reg(key, callback)


def _fill_constant_cb(value: float, label: float = 0.0):
    """Return a callback that fills ``tensors[0]`` with *value* and ``tensors[1]``
    with *label* (in place via DLPack)."""

    def _cb(tensors):
        assert len(tensors) >= 1, "expected at least one tensor"
        data = np.from_dlpack(tensors[0])
        data[...] = value
        if len(tensors) >= 2:
            lab = np.from_dlpack(tensors[1])
            lab[...] = label

    return _cb


# ---------------------------------------------------------------------------
# Upsample
# ---------------------------------------------------------------------------

def _make_upsample_prototxt(N, C, H, W, scale):
    return textwrap.dedent(f"""\
        name: "test_upsample"
        layer {{
          name: "data"
          type: "Input"
          top: "data"
          input_param {{ shape {{ dim: {N} dim: {C} dim: {H} dim: {W} }} }}
        }}
        layer {{
          name: "up"
          type: "Upsample"
          bottom: "data"
          top: "out"
          upsample_param {{ scale: {scale} }}
        }}
    """)


def _upsample_numpy(x, scale):
    """Nearest-neighbor upsample reference (N,C,H,W) -> (N,C,H*scale,W*scale)."""
    x = np.asarray(x, dtype=np.float64)
    n, c, h, w = x.shape
    out = np.zeros((n, c, h * scale, w * scale), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            out[:, :, i * scale:(i + 1) * scale,
                     j * scale:(j + 1) * scale] = x[:, :, i:i + 1, j:j + 1]
    return out


@require_cpp_extension
class TestUpsample:
    def test_registration(self):
        net = make_net(_make_upsample_prototxt(1, 1, 2, 2, 2))
        types = [l.type for l in net.layers_array()]
        assert "Upsample" in types

    def test_forward_nearest_neighbor(self):
        x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)  # (1,1,2,2)
        net, _ = make_net_with_diag(_make_upsample_prototxt(1, 1, 2, 2, 2),
                                    tag="upsample_nearest")
        dump_net_state(net, tag="upsample_nearest_before_fwd")
        out = net.Forward({"data": x})["out"].to_numpy()
        dump_net_state(net, tag="upsample_nearest_after_fwd")
        expected = np.array([[[[1, 1, 2, 2],
                               [1, 1, 2, 2],
                               [3, 3, 4, 4],
                               [3, 3, 4, 4]]]], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_forward_scale3_shape(self):
        x = np.arange(1 * 1 * 2 * 3, dtype=np.float32).reshape(1, 1, 2, 3)
        net = make_net(_make_upsample_prototxt(1, 1, 2, 3, 3))
        out = net.Forward({"data": x})["out"].to_numpy()
        assert tuple(out.shape) == (1, 1, 6, 9)
        ref = _upsample_numpy(x, 3)
        np.testing.assert_allclose(out, ref, rtol=RTOL, atol=ATOL)

    def test_backward_gradient_sums_over_block(self):
        x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
        net, _ = make_net_with_diag(_make_upsample_prototxt(1, 1, 2, 2, 2),
                                    tag="upsample_bwd")
        dy = np.ones((1, 1, 4, 4), dtype=np.float32)
        dump_net_state(net, tag="upsample_bwd_before_fwd")
        net.Forward({"data": x})
        dump_net_state(net, tag="upsample_bwd_after_fwd")
        net.backward({"out": dy})
        dump_net_state(net, tag="upsample_bwd_after_bwd")
        dx = net.blob_by_name("data").diff
        # Each 2x2 block sums to 4.
        np.testing.assert_allclose(dx, np.full((1, 1, 2, 2), 4.0),
                                   rtol=RTOL, atol=ATOL)

    def test_numerical_gradient(self):
        np.random.seed(31)
        x = np.random.randn(1, 1, 2, 2).astype(np.float32) * 0.5
        net = make_net(_make_upsample_prototxt(1, 1, 2, 2, 2))
        dy = np.random.randn(1, 1, 4, 4).astype(np.float32) * 0.1

        # Analytical gradient.
        net.Forward({"data": x})
        net.backward({"out": dy})
        dx = net.blob_by_name("data").diff.copy()

        # Numerical gradient of L = sum(out * dy).
        def _loss(xx):
            out = net.Forward({"data": xx})["out"].to_numpy()
            return float(np.sum(out.astype(np.float64) * dy.astype(np.float64)))

        grad = np.zeros_like(x, dtype=np.float64)
        flat_x = x.astype(np.float64).ravel()
        flat_grad = grad.ravel()
        for i in range(flat_x.size):
            orig = flat_x[i]
            xp = x.copy()
            xp.ravel()[i] = orig + EPS_NUMERICAL
            lp = _loss(xp)
            xm = x.copy()
            xm.ravel()[i] = orig - EPS_NUMERICAL
            lm = _loss(xm)
            flat_grad[i] = (lp - lm) / (2.0 * EPS_NUMERICAL)
        np.testing.assert_allclose(dx, grad.astype(np.float32),
                                   rtol=1e-2, atol=1e-3)


# ---------------------------------------------------------------------------
# MemoryData
# ---------------------------------------------------------------------------

def _make_memorydata_prototxt(batch, channels, height, width):
    return textwrap.dedent(f"""\
        name: "test_memorydata"
        layer {{
          name: "mem"
          type: "MemoryData"
          top: "out"
          memory_data_param {{
            batch_size: {batch}
            channels: {channels}
            height: {height}
            width: {width}
          }}
        }}
    """)


@require_cpp_extension
class TestMemoryData:
    def test_registration(self):
        net = make_net(_make_memorydata_prototxt(2, 3, 4, 4))
        types = [l.type for l in net.layers_array()]
        assert "MemoryData" in types

    def test_forward_zeros_when_no_data(self):
        net, _ = make_net_with_diag(_make_memorydata_prototxt(2, 3, 4, 5),
                                    tag="memdata_zeros")
        dump_net_state(net, tag="memdata_zeros_before_fwd")
        out = net.Forward({})["out"].to_numpy()
        dump_net_state(net, tag="memdata_zeros_after_fwd")
        assert tuple(out.shape) == (2, 3, 4, 5)
        np.testing.assert_allclose(out, 0.0, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# DummyData
# ---------------------------------------------------------------------------

def _make_dummydata_prototxt(shape_dims, filler):
    dims = " ".join(f"dim: {d}" for d in shape_dims)
    return textwrap.dedent(f"""\
        name: "test_dummydata"
        layer {{
          name: "dum"
          type: "DummyData"
          top: "out"
          dummy_data_param {{
            shape {{ {dims} }}
            data_filler {{ {filler} }}
          }}
        }}
    """)


@require_cpp_extension
class TestDummyData:
    def test_registration(self):
        net = make_net(_make_dummydata_prototxt([2, 3], 'type: "constant" value: 1.0'))
        types = [l.type for l in net.layers_array()]
        assert "DummyData" in types

    def test_forward_constant(self):
        net = make_net(_make_dummydata_prototxt([2, 3], 'type: "constant" value: 7.0'))
        out = net.Forward({})["out"].to_numpy()
        assert tuple(out.shape) == (2, 3)
        np.testing.assert_allclose(out, 7.0, rtol=RTOL, atol=ATOL)

    def test_forward_uniform_range(self):
        net = make_net(_make_dummydata_prototxt([4, 4], 'type: "uniform" min: -2.0 max: 3.0'))
        out = net.Forward({})["out"].to_numpy()
        assert out.min() >= -2.0 - 1e-4 and out.max() <= 3.0 + 1e-4

    def test_forward_no_filler_zero(self):
        proto = textwrap.dedent("""\
            name: "test_dummydata_zero"
            layer {
              name: "dum"
              type: "DummyData"
              top: "out"
              dummy_data_param { shape { dim: 2 dim: 2 } }
            }
        """)
        net = make_net(proto)
        out = net.Forward({})["out"].to_numpy()
        np.testing.assert_allclose(out, 0.0, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

def _make_python_prototxt(module, layer):
    return textwrap.dedent(f"""\
        name: "test_python"
        layer {{
          name: "py"
          type: "Python"
          top: "out"
          python_param {{
            module: "{module}"
            layer: "{layer}"
          }}
        }}
    """)


@require_cpp_extension
class TestPythonLayer:
    def test_registration(self):
        net = make_net(_make_python_prototxt("mymod", "mylayer"))
        types = [l.type for l in net.layers_array()]
        assert "Python" in types

    def test_forward_no_op_without_callback(self):
        # No callback registered for "mymod.mylayer" -> layer degrades to no-op
        # and Forward must complete without error.
        net, _ = make_net_with_diag(_make_python_prototxt("mymod", "mylayer"),
                                    tag="python_nocb")
        dump_net_state(net, tag="python_nocb_before_fwd")
        net.Forward({})  # should not raise
        dump_net_state(net, tag="python_nocb_after_fwd")

    def test_forward_invokes_callback(self):
        # Register a callback keyed "<module>.<layer>". The minimal bridge calls
        # it with the top blobs' writable tensors; if the top has storage the
        # callback can fill it. We only assert the callback is invoked.
        calls = {"count": 0}

        def _cb(tensors):
            calls["count"] += 1

        _register_python_layer("mod", "layer", _cb)
        net, _ = make_net_with_diag(_make_python_prototxt("mod", "layer"),
                                    tag="python_cb")
        dump_net_state(net, tag="python_cb_before_fwd")
        net.Forward({})
        dump_net_state(net, tag="python_cb_after_fwd")
        assert calls["count"] == 1


# ---------------------------------------------------------------------------
# HDF5Output
# ---------------------------------------------------------------------------

def _make_hdf5output_prototxt(layer_name="h5out", file_name="out.h5", with_label=True):
    label_layer = ""
    label_bottom = ""
    if with_label:
        label_layer = ('  layer {\n'
                       '    name: "label"\n'
                       '    type: "Input"\n'
                       '    top: "label"\n'
                       '    input_param { shape { dim: 2 } }\n'
                       '  }\n')
        label_bottom = '  bottom: "label"\n'
    return textwrap.dedent(f"""\
        name: "test_hdf5output"
        layer {{
          name: "data"
          type: "Input"
          top: "data"
          input_param {{ shape {{ dim: 2 dim: 3 }} }}
        }}
{label_layer}        layer {{
          name: "{layer_name}"
          type: "HDF5Output"
          bottom: "data"
{label_bottom}          hdf5_output_param {{ file_name: "{file_name}" }}
        }}
    """)


@require_cpp_extension
class TestHDF5Output:
    def test_registration(self):
        net = make_net(_make_hdf5output_prototxt())
        types = [l.type for l in net.layers_array()]
        assert "HDF5Output" in types

    def test_forward_invokes_callback_with_bottom(self):
        seen = {"value": None, "shape": None, "label": None}

        def _cb(tensors):
            assert len(tensors) == 2
            data = np.from_dlpack(tensors[0])
            seen["value"] = data.copy()
            seen["shape"] = tuple(data.shape)
            seen["label"] = np.from_dlpack(tensors[1]).copy()

        _register_data_io("HDF5Output.h5out", _cb)
        net, _ = make_net_with_diag(_make_hdf5output_prototxt(),
                                    tag="hdf5_cb")
        dump_net_state(net, tag="hdf5_cb_before_fwd")
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        y = np.array([0.0, 1.0], dtype=np.float32)
        net.Forward({"data": x, "label": y})
        dump_net_state(net, tag="hdf5_cb_after_fwd")
        assert seen["shape"] == (2, 3)
        np.testing.assert_allclose(seen["value"], x, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(seen["label"], y, rtol=RTOL, atol=ATOL)

    def test_forward_no_callback_skips_write(self):
        net, _ = make_net_with_diag(_make_hdf5output_prototxt(layer_name="h5_nocb"),
                                    tag="hdf5_nocb")
        dump_net_state(net, tag="hdf5_nocb_before_fwd")
        x = np.random.randn(2, 3).astype(np.float32)
        y = np.random.randn(2).astype(np.float32)
        net.Forward({"data": x, "label": y})  # should not raise
        dump_net_state(net, tag="hdf5_nocb_after_fwd")


# ---------------------------------------------------------------------------
# WindowData
# ---------------------------------------------------------------------------

def _make_windowdata_prototxt(layer_name="win", batch_size=2):
    return textwrap.dedent(f"""\
        name: "test_windowdata"
        layer {{
          name: "{layer_name}"
          type: "WindowData"
          top: "data"
          top: "label"
          window_data_param {{
            source: "dummy.txt"
            batch_size: {batch_size}
          }}
        }}
    """)


@require_cpp_extension
class TestWindowData:
    def test_registration(self):
        net = make_net(_make_windowdata_prototxt())
        types = [l.type for l in net.layers_array()]
        assert "WindowData" in types

    def test_forward_fills_data_and_label(self):
        _register_data_io("WindowData.fill_win", _fill_constant_cb(3.5, label=2.0))
        net, _ = make_net_with_diag(_make_windowdata_prototxt(layer_name="fill_win", batch_size=2),
                                    tag="window_cb")
        dump_net_state(net, tag="window_cb_before_fwd")
        net.Forward({})
        dump_net_state(net, tag="window_cb_after_fwd")
        d = net.blob_by_name("data").to_numpy()
        l = net.blob_by_name("label").to_numpy()
        # WindowData Reshape: top[0] -> [batch,3,1,1], top[1] -> [batch].
        assert tuple(d.shape) == (2, 3, 1, 1)
        assert tuple(l.shape) == (2,)
        np.testing.assert_allclose(d, 3.5, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(l, 2.0, rtol=RTOL, atol=ATOL)

    def test_forward_no_callback_zeros(self):
        net, _ = make_net_with_diag(_make_windowdata_prototxt(layer_name="win_nocb", batch_size=2),
                                    tag="window_nocb")
        dump_net_state(net, tag="window_nocb_before_fwd")
        net.Forward({})
        dump_net_state(net, tag="window_nocb_after_fwd")
        d = net.blob_by_name("data").to_numpy()
        l = net.blob_by_name("label").to_numpy()
        np.testing.assert_allclose(d, 0.0, rtol=0, atol=0)
        np.testing.assert_allclose(l, 0.0, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Cross-layer registration sanity (LayerTypeList contains the P2 operators)
# ---------------------------------------------------------------------------

@require_cpp_extension
def test_p2_other_operators_registered_in_layer_factory():
    fn = _ffi_get("caffe_ffi.LayerTypeList")
    types = set(fn())
    for op in ("Upsample", "MemoryData", "DummyData", "Python",
               "HDF5Output", "WindowData"):
        assert op in types, f"{op} not registered in LayerTypeList"