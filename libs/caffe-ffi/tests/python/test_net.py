from __future__ import annotations

import numpy as np
import pytest

import caffe_ffi
from caffe_ffi import (
    Net, Blob, Layer, caffe_pb2,
    net_param_from_string, net_from_param
)
from .conftest import require_cpp_extension


class TestNetParse:
    def test_parse_prototxt_string(self):
        prototxt = """name: "test"
input: "data"
input_dim: 1
input_dim: 3
layer {
  name: "ip1"
  type: "InnerProduct"
  bottom: "data"
  top: "ip1"
  inner_product_param { num_output: 2 }
}
"""
        param = net_param_from_string(prototxt)
        assert param.name == "test"
        assert len(param.input) == 1
        assert param.input[0] == "data"
        assert len(param.layer) == 1
        assert param.layer[0].name == "ip1"
        assert param.layer[0].type == "InnerProduct"

    def test_parse_multilayer_prototxt(self):
        prototxt = """name: "mlp"
input: "data"
input_shape { dim: 1 dim: 10 }
layer { name: "ip1" type: "InnerProduct" bottom: "data" top: "ip1" inner_product_param { num_output: 5 } }
layer { name: "relu1" type: "ReLU" bottom: "ip1" top: "ip1" }
layer { name: "prob" type: "Softmax" bottom: "ip1" top: "prob" }
"""
        param = net_param_from_string(prototxt)
        assert param.name == "mlp"
        assert len(param.layer) == 3
        layer_types = [l.type for l in param.layer]
        assert layer_types == ["InnerProduct", "ReLU", "Softmax"]


class TestNetBuild:
    def test_build_from_param(self):
        param = caffe_pb2.NetParameter()
        param.name = "test_net"
        param.input.append("data")
        param.input_dim.extend([1, 3])
        
        layer = param.layer.add()
        layer.name = "ip1"
        layer.type = "InnerProduct"
        layer.bottom.append("data")
        layer.top.append("ip1")
        layer.inner_product_param.num_output = 2
        
        net = net_from_param(param)
        assert net.name == "test_net"
        assert len(net.blobs_array()) >= 2
        assert len(net.layers_array()) >= 1

    def test_net_name_property(self, mlp_net):
        assert mlp_net.name == "mlp_test"


class TestNetBlobAccess:
    def test_has_blob(self, mlp_net):
        assert mlp_net.has_blob("data")
        assert mlp_net.has_blob("ip1")
        assert not mlp_net.has_blob("nonexistent")

    def test_has_layer(self, mlp_net):
        assert mlp_net.has_layer("ip1")
        assert mlp_net.has_layer("relu1")
        assert not mlp_net.has_layer("nonexistent")

    def test_blob_by_name(self, mlp_net):
        blob = mlp_net.blob_by_name("data")
        assert isinstance(blob, Blob)
        assert blob.shape == (2, 3)

    def test_layer_by_name(self, mlp_net):
        layer = mlp_net.layer_by_name("ip1")
        assert isinstance(layer, Layer)

    def test_blob_by_name_keyerror(self, mlp_net):
        with pytest.raises(KeyError):
            mlp_net.blob_by_name("nonexistent")

    def test_layer_by_name_keyerror(self, mlp_net):
        with pytest.raises(KeyError):
            mlp_net.layer_by_name("nonexistent")

    def test_getitem(self, mlp_net):
        blob = mlp_net["data"]
        assert isinstance(blob, Blob)

    def test_getitem_keyerror(self, mlp_net):
        with pytest.raises(KeyError):
            _ = mlp_net["nonexistent"]

    def test_contains(self, mlp_net):
        assert "data" in mlp_net
        assert "ip1" in mlp_net
        assert "nonexistent" not in mlp_net

    def test_blobs_dict(self, mlp_net):
        bd = mlp_net.blobs_dict
        assert isinstance(bd, dict)
        assert "data" in bd
        assert "ip1" in bd
        for name, blob in bd.items():
            assert isinstance(name, str)
            assert isinstance(blob, Blob)

    def test_layers_dict(self, mlp_net):
        ld = mlp_net.layers_dict
        assert isinstance(ld, dict)
        assert "ip1" in ld
        assert "relu1" in ld
        for name, layer in ld.items():
            assert isinstance(name, str)
            assert isinstance(layer, Layer)

    def test_iter(self, mlp_net):
        names = list(mlp_net)
        assert "data" in names
        assert "ip1" in names

    def test_len(self, mlp_net):
        assert len(mlp_net) == len(mlp_net.blobs_array())


class TestNetForward:
    @require_cpp_extension
    def test_forward_executes(self, mlp_net):
        inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        out = mlp_net.forward({"data": inp})
        assert isinstance(out, dict)
        assert len(out) > 0

    @require_cpp_extension
    def test_forward_output_shape(self, mlp_net):
        inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        out = mlp_net.forward({"data": inp})
        if "prob" in out:
            assert out["prob"].shape == (2, 2)

    @require_cpp_extension
    def test_forward_all(self, mlp_net):
        inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        out = mlp_net.forward_all(data=inp)
        assert isinstance(out, dict)

    def test_forward_pure_python_reference(self, mlp_net):
        if mlp_net._is_native:
            pytest.skip("Requires pure Python net for reference test")
        inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        out = mlp_net.forward({"data": inp})
        assert "prob" in out
        assert out["prob"].shape == (2, 2)
        np.testing.assert_allclose(out["prob"].sum(axis=1), np.array([1.0, 1.0]), rtol=1e-5)


class TestNetRepr:
    def test_repr(self, mlp_net):
        r = repr(mlp_net)
        assert "Net" in r
        assert "mlp_test" in r
