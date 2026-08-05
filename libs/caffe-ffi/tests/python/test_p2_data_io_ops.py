"""P2 unit tests for the data I/O layers (Data / ImageData / HDF5Data).

These layers read their payload through a **Python-side** callback registered via
the FFI function ``caffe_ffi.data_io.register`` under the key ``"<layer_type>.<name>"``.
On ``Forward`` the C++ layer invokes the callback with the mutable data tensors of its
top blobs (DLPack interop), allowing the Python side to fill the outputs in place.

Test coverage (Task 7 / TR-4.1 / TR-4.2):
  * Data:       callback fills ``data`` + ``label``; output matches; zeros fallback
  * ImageData:  callback fills image + label; output shape ``[N,C,H,W]``; zeros fallback
  * HDF5Data:   callback fills ``data`` + ``label``; output matches; zeros fallback
  * Layer registration (LayerTypeList / layer_names) for all three layers

Reference semantics derived from ``src/caffe_ffi/layers/{data,image_data,hdf5_data}_layer.cpp``
(placeholder Reshape shapes are filled in place by the registered callback).
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from .conftest import require_cpp_extension
from .caffe_test_helpers import make_net


def _register_data_io(cb_type: str, layer_name: str, callback) -> None:
    """Register a data-source callback under ``<cb_type>.<layer_name>``.

    The callback is a plain Python callable receiving an ``Array<Tensor>`` (the
    top blobs' mutable data tensors). It must fill them in place via a DLPack view.
    """
    from caffe_ffi import _ffi_api

    reg = _ffi_api.get_global_func("caffe_ffi.data_io.register")
    assert reg is not None, "caffe_ffi.data_io.register global func not found"
    reg(f"{cb_type}.{layer_name}", callback)


def _fill_constant_cb(value: float, label: float = 0.0):
    """Return a callback that fills ``top[0]`` with *value* and ``top[1]`` with *label*."""

    def _cb(tensors):
        assert len(tensors) >= 1, "callback expected at least one top tensor"
        data = np.from_dlpack(tensors[0])
        data[...] = value
        if len(tensors) >= 2:
            lab = np.from_dlpack(tensors[1])
            lab[...] = label

    return _cb


def _fill_seq_cb(start: float = 0.0):
    """Return a callback that fills ``top[0]`` with a running counter, ``top[1]`` with labels."""

    def _cb(tensors):
        data = np.from_dlpack(tensors[0])
        flat = np.arange(data.size, dtype=np.float32).reshape(data.shape) + start
        data[...] = flat
        if len(tensors) >= 2:
            lab = np.from_dlpack(tensors[1])
            lab[...] = np.arange(lab.size, dtype=np.float32).reshape(lab.shape) + 100.0

    return _cb


# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────

@require_cpp_extension
class TestDataLayer:
    def _net(self, layer_name="mydata", batch_size=2, source="dummy.txt"):
        return make_net(textwrap.dedent(f"""\
            name: "data_test"
            layer {{
              name: "{layer_name}"
              type: "Data"
              top: "data"
              top: "label"
              data_param {{ batch_size: {batch_size} source: "{source}" }}
            }}
        """))

    def test_forward_fills_data_and_label(self):
        _register_data_io("Data", "fill_data", _fill_constant_cb(7.0, label=3.0))
        net = self._net(layer_name="fill_data", batch_size=2)
        out = net.Forward({})
        d = net.blob_by_name("data").to_numpy()
        l = net.blob_by_name("label").to_numpy()
        # Data Reshape: top[0] -> [batch,3,1,1], top[1] -> [batch]
        assert tuple(d.shape) == (2, 3, 1, 1)
        assert tuple(l.shape) == (2,)
        np.testing.assert_allclose(d, 7.0, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(l, 3.0, rtol=1e-5, atol=1e-6)

    def test_forward_sequence_values(self):
        cb = _fill_seq_cb(start=0.0)
        _register_data_io("Data", "fill_seq", cb)
        net = self._net(layer_name="fill_seq", batch_size=3)
        net.Forward({})
        d = net.blob_by_name("data").to_numpy()
        expected = np.arange(3 * 3 * 1 * 1, dtype=np.float32).reshape(3, 3, 1, 1)
        np.testing.assert_array_equal(d, expected)

    def test_no_callback_zeros_fallback(self):
        # No callback registered under this key -> C++ fills zeros.
        net = self._net(layer_name="no_cb", batch_size=2)
        net.Forward({})
        d = net.blob_by_name("data").to_numpy()
        l = net.blob_by_name("label").to_numpy()
        np.testing.assert_allclose(d, 0.0, rtol=0, atol=0)
        np.testing.assert_allclose(l, 0.0, rtol=0, atol=0)

    def test_layer_registered(self):
        net = self._net()
        assert "mydata" in net.layer_names()
        assert net.blob_by_name("data").shape is not None


# ──────────────────────────────────────────────────────────────────────
# ImageData
# ──────────────────────────────────────────────────────────────────────

@require_cpp_extension
class TestImageDataLayer:
    def _net(self, layer_name="myimg", batch_size=1, is_color=True,
             new_height=4, new_width=5, source="imgs.txt", scale=0.0):
        return make_net(textwrap.dedent(f"""\
            name: "imagedata_test"
            layer {{
              name: "{layer_name}"
              type: "ImageData"
              top: "image"
              top: "label"
              image_data_param {{
                batch_size: {batch_size}
                is_color: {str(is_color).lower()}
                new_height: {new_height}
                new_width: {new_width}
                source: "{source}"
                scale: {scale}
              }}
            }}
        """))

    def test_forward_shape_and_values(self):
        _register_data_io("ImageData", "fill_img", _fill_constant_cb(2.5, label=1.0))
        net = self._net(layer_name="fill_img", batch_size=2, is_color=True,
                        new_height=4, new_width=5)
        net.Forward({})
        img = net.blob_by_name("image").to_numpy()
        lab = net.blob_by_name("label").to_numpy()
        # ImageData Reshape: top[0] -> [batch, channels, h, w]; color -> 3 channels.
        assert tuple(img.shape) == (2, 3, 4, 5)
        assert tuple(lab.shape) == (2,)
        np.testing.assert_allclose(img, 2.5, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(lab, 1.0, rtol=1e-5, atol=1e-6)

    def test_gray_single_channel(self):
        _register_data_io("ImageData", "fill_gray", _fill_constant_cb(9.0))
        net = self._net(layer_name="fill_gray", batch_size=1, is_color=False,
                        new_height=3, new_width=3)
        net.Forward({})
        img = net.blob_by_name("image").to_numpy()
        assert tuple(img.shape) == (1, 1, 3, 3)  # is_color=False -> 1 channel
        np.testing.assert_allclose(img, 9.0, rtol=1e-5, atol=1e-6)

    def test_no_callback_zeros_fallback(self):
        net = self._net(layer_name="img_no_cb", batch_size=1, is_color=True,
                        new_height=2, new_width=2)
        net.Forward({})
        img = net.blob_by_name("image").to_numpy()
        np.testing.assert_allclose(img, 0.0, rtol=0, atol=0)

    def test_layer_registered(self):
        net = self._net()
        assert "myimg" in net.layer_names()


# ──────────────────────────────────────────────────────────────────────
# HDF5Data
# ──────────────────────────────────────────────────────────────────────

@require_cpp_extension
class TestHDF5DataLayer:
    def _net(self, layer_name="myh5", batch_size=2, source="data.h5", shuffle=True):
        return make_net(textwrap.dedent(f"""\
            name: "hdf5data_test"
            layer {{
              name: "{layer_name}"
              type: "HDF5Data"
              top: "hdata"
              top: "hlabel"
              hdf5_data_param {{
                batch_size: {batch_size}
                source: "{source}"
                shuffle: {str(shuffle).lower()}
              }}
            }}
        """))

    def test_forward_fills_data_and_label(self):
        _register_data_io("HDF5Data", "fill_h5", _fill_constant_cb(4.25, label=8.0))
        net = self._net(layer_name="fill_h5", batch_size=2)
        net.Forward({})
        d = net.blob_by_name("hdata").to_numpy()
        l = net.blob_by_name("hlabel").to_numpy()
        # HDF5Data Reshape: top[0] -> [batch,1,1,1], top[1] -> [batch].
        assert tuple(d.shape) == (2, 1, 1, 1)
        assert tuple(l.shape) == (2,)
        np.testing.assert_allclose(d, 4.25, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(l, 8.0, rtol=1e-5, atol=1e-6)

    def test_no_callback_zeros_fallback(self):
        net = self._net(layer_name="h5_no_cb", batch_size=2)
        net.Forward({})
        d = net.blob_by_name("hdata").to_numpy()
        l = net.blob_by_name("hlabel").to_numpy()
        np.testing.assert_allclose(d, 0.0, rtol=0, atol=0)
        np.testing.assert_allclose(l, 0.0, rtol=0, atol=0)

    def test_layer_registered(self):
        net = self._net()
        assert "myh5" in net.layer_names()


# ──────────────────────────────────────────────────────────────────────
# Cross-layer registration sanity (LayerTypeList contains the P2 operators)
# ──────────────────────────────────────────────────────────────────────

@require_cpp_extension
def test_data_io_operators_registered_in_layer_factory():
    from caffe_ffi import _ffi_api

    fn = _ffi_api.get_global_func("caffe_ffi.LayerTypeList")
    types = set(fn())
    for op in ("Data", "ImageData", "HDF5Data"):
        assert op in types, f"{op} not registered in LayerTypeList"