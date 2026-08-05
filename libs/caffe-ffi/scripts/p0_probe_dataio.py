#!/usr/bin/env python3
"""Quick probe: verify data_io.register callback mechanism for Data layers."""
from __future__ import annotations

import numpy as np

import caffe_ffi
from caffe_ffi import _ffi_api

# Register a data source callback under key "Data.mydata"
_key = "Data.mydata"


def _ds(tensors):
    # tensors: Array<Tensor> of the top blobs (mutable). Fill top[0] via numpy.
    t = tensors[0]
    arr = np.from_dlpack(t)  # zero-copy DLPack view
    arr[...] = 7.0
    print("callback: filled", arr.shape, "->", arr.ravel()[:4])


# Register via FFI global func. Function type auto-converts a Python callable.
reg = _ffi_api.get_global_func("caffe_ffi.data_io.register")
print("reg func:", reg)
reg(_key, _ds)
print("registered", _key)

prototxt = """
name: "data_probe"
layer {
  name: "mydata"
  type: "Data"
  top: "data"
  top: "label"
  data_param { batch_size: 2 source: "dummy.txt" }
}
"""
net = caffe_ffi.net_from_param(caffe_ffi.net_param_from_string(prototxt))
net.Forward({})
d = net.blob_by_name("data").to_numpy()
l = net.blob_by_name("label").to_numpy()
print("data shape", d.shape)
print("label shape", l.shape)
assert np.all(d == 7.0), "data should be filled by callback"
print("PROBE OK")