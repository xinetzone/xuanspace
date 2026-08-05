"""Protobuf protocol-compatibility tests for caffe-ffi.

Verifies that the generated ``caffe_pb2`` (from ``proto/caffe/proto/caffe.proto``)
can parse standard BVLC/Caffe-SSD ``norm_param { ... }`` blocks (previously
unparseable because caffe-ffi only exposed ``l2_norm_param``), while remaining
backward compatible with the existing ``l2_norm_param`` field.

These tests operate purely on the protobuf protocol layer and do **not** require
the C++ extension, so they can run even when the native extension is unavailable.
"""
from __future__ import annotations

import pytest
from google.protobuf import text_format

# The caffe_ffi package re-exports the generated caffe_pb2 (see __init__.py).
# Importing through the package keeps a single registration in the descriptor
# pool, avoiding a "duplicate file name" error. These tests only touch the
# protobuf protocol layer and do not require the C++ extension.
from caffe_ffi import caffe_pb2


NORM_PARAM_PROTO = """
name: "norm_net"
layer {
  name: "norm1"
  type: "Normalize"
  bottom: "data"
  top: "norm1"
  norm_param {
    across_spatial: true
    scale_filler { type: "constant" value: 1 }
    channel_shared: false
    eps: 1e-5
  }
}
"""

L2_NORM_PROTO = """
name: "l2_net"
layer {
  name: "l2"
  type: "L2Norm"
  bottom: "data"
  top: "l2"
  l2_norm_param {
    axis: 1
    eps: 1e-5
  }
}
"""


def test_caffe_pb2_imports_and_has_norm_param_field():
    """The generated caffe_pb2 imports and exposes ``norm_param`` /
    ``NormalizeParameter``."""
    lp = caffe_pb2.LayerParameter()
    assert lp.norm_param.across_spatial is True  # default true (BVLC)
    assert lp.norm_param.channel_shared is True  # default true (BVLC)
    assert lp.norm_param.eps == pytest.approx(1e-10)
    assert caffe_pb2.NormalizeParameter is not None

    fields = {f.name: f.number for f in caffe_pb2.LayerParameter.DESCRIPTOR.fields}
    assert fields["norm_param"] == 190
    # Backward compat field intact
    assert fields["l2_norm_param"] == 158


def test_parse_norm_param_prototxt():
    """A standard BVLC/Caffe-SSD `norm_param { ... }` block parses without error."""
    net = caffe_pb2.NetParameter()
    text_format.Parse(NORM_PARAM_PROTO, net)
    layer = net.layer[0]
    assert layer.type == "Normalize"
    assert layer.norm_param.across_spatial is True
    assert layer.norm_param.channel_shared is False
    assert layer.norm_param.eps == pytest.approx(1e-5)
    assert layer.norm_param.scale_filler.type == "constant"
    assert layer.norm_param.scale_filler.value == pytest.approx(1.0)


def test_parse_l2_norm_param_still_works():
    """Existing `l2_norm_param` remains parseable (backward compatible)."""
    net = caffe_pb2.NetParameter()
    text_format.Parse(L2_NORM_PROTO, net)
    layer = net.layer[0]
    assert layer.type == "L2Norm"
    assert layer.l2_norm_param.axis == 1
    assert layer.l2_norm_param.eps == pytest.approx(1e-5)
    # norm_param is not set on this layer
    assert not layer.HasField("norm_param")


def test_serialization_roundtrip_norm_param():
    """Serialize/parse round-trip preserves the norm_param fields."""
    net = caffe_pb2.NetParameter()
    text_format.Parse(NORM_PARAM_PROTO, net)
    raw = net.SerializeToString()

    net2 = caffe_pb2.NetParameter()
    net2.ParseFromString(raw)
    layer = net2.layer[0]
    assert layer.norm_param.across_spatial is True
    assert layer.norm_param.channel_shared is False
    assert layer.norm_param.eps == pytest.approx(1e-5)