"""Tests for caffe-ffi model serialization (``caffe_ffi.serialization``).

Covers Task 33 (P4-训练工程化) model persistence:

1. **Weight dict round-trip** — ``weights_to_dict`` / ``dict_to_weights``
   export and restore all learnable blobs; unknown keys are ignored.
2. **caffemodel save/load round-trip** — ``save_net`` / ``load_net`` /
   ``net_parameter_to_file`` write a valid ``NetParameter`` protobuf that a
   fresh ``Net`` can load back, preserving weights and shapes.
3. **Protobuf validity** — the produced file parses via ``read_net_from_binary``
   and contains the expected layer names / blob shapes.

These tests require a real network with learnable blobs, so they are gated with
``@require_cpp_extension``.
"""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

from caffe_ffi import (
    net_from_param,
    net_param_from_string,
    read_net_from_binary,
    save_net,
    load_net,
    weights_to_dict,
    dict_to_weights,
    net_parameter_to_file,
)

from .conftest import require_cpp_extension


SIMPLE_MLP_PROTO = '''
name: "simple_mlp"
layer { name: "data" type: "Input" top: "data"
  input_param { shape { dim: 4 dim: 8 } } }
layer { name: "fc1" type: "InnerProduct" bottom: "data" top: "fc1"
  inner_product_param { num_output: 6 weight_filler { type: "msra" } } }
layer { name: "relu1" type: "ReLU" bottom: "fc1" top: "fc1" }
layer { name: "fc2" type: "InnerProduct" bottom: "fc1" top: "fc2"
  inner_product_param { num_output: 3 weight_filler { type: "msra" } } }
'''


@require_cpp_extension
def make_net():
    return net_from_param(net_param_from_string(SIMPLE_MLP_PROTO))


class TestWeightsDict:
    @require_cpp_extension
    def test_roundtrip_preserves_weights(self):
        net = make_net()
        # Overwrite fc1 weights with a known pattern
        fc1 = net.layer_by_name("fc1").blobs[0]
        fc1.data_tensor[:] = np.arange(fc1.data_tensor.size, dtype=np.float32).reshape(fc1.shape)

        weights = weights_to_dict(net)
        assert "fc1:0" in weights and "fc1:1" in weights
        assert "fc2:0" in weights and "fc2:1" in weights
        np.testing.assert_array_equal(weights["fc1:0"], fc1.data_tensor)

        # Corrupt the net, then restore
        fc1.data_tensor[:] = 0.0
        dict_to_weights(net, weights)
        np.testing.assert_array_equal(fc1.data_tensor, weights["fc1:0"])

    @require_cpp_extension
    def test_returns_copies_not_views(self):
        net = make_net()
        weights = weights_to_dict(net)
        fc1 = net.layer_by_name("fc1").blobs[0]
        weights["fc1:0"][:] = 123.0
        # Mutating the exported dict must not touch the net's weights
        assert not np.allclose(fc1.data_tensor, 123.0)

    @require_cpp_extension
    def test_dict_to_weights_ignores_unknown_keys(self):
        net = make_net()
        before = weights_to_dict(net)
        # Unknown key should be silently ignored
        dict_to_weights(net, {**before, "no_such_layer:0": np.zeros(3, dtype=np.float32)})
        after = weights_to_dict(net)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])


class TestCaffemodel:
    @require_cpp_extension
    def test_save_load_roundtrip(self, tmp_path):
        net = make_net()
        fc1 = net.layer_by_name("fc1").blobs[0]
        fc1.data_tensor[:] = np.arange(fc1.data_tensor.size, dtype=np.float32).reshape(fc1.shape)

        path = tmp_path / "model.caffemodel"
        save_net(net, path)
        assert path.exists()

        net2 = make_net()
        load_net(net2, path)
        np.testing.assert_array_equal(net2.layer_by_name("fc1").blobs[0].data_tensor, fc1.data_tensor)

    @require_cpp_extension
    def test_net_parameter_to_file_produces_valid_protobuf(self, tmp_path):
        net = make_net()
        path = tmp_path / "weights.caffemodel"
        net_parameter_to_file(net, path)

        param = read_net_from_binary(path)
        assert param.name in ("simple_mlp", "caffe_ffi")
        layer_names = {l.name for l in param.layer}
        assert {"fc1", "fc2"} <= layer_names

        fc1_proto = next(l for l in param.layer if l.name == "fc1")
        assert len(fc1_proto.blobs) == 2  # weight + bias
        assert list(fc1_proto.blobs[0].shape.dim) == list(net.layer_by_name("fc1").blobs[0].shape)

    @require_cpp_extension
    def test_load_into_net_with_extra_layer_ignores_missing(self, tmp_path):
        net = make_net()
        path = tmp_path / "model.caffemodel"
        save_net(net, path)

        # A net with an extra layer still loads the shared weights fine
        proto = SIMPLE_MLP_PROTO + '''
layer { name: "fc3" type: "InnerProduct" bottom: "fc2" top: "fc3"
  inner_product_param { num_output: 2 weight_filler { type: "msra" } } }
'''
        net2 = net_from_param(net_param_from_string(proto))
        load_net(net2, path)
        np.testing.assert_array_equal(
            net2.layer_by_name("fc1").blobs[0].data_tensor,
            net.layer_by_name("fc1").blobs[0].data_tensor,
        )