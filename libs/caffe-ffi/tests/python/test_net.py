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


# ─── P1: Net boundary tests ──────────────────────────────────────

class TestNetEmptyConstructor:
    """Boundary tests for Net() with no arguments (empty net)."""

    def test_empty_net_name(self):
        """Net() with no args has empty name."""
        net = Net()
        assert net.name == ""

    def test_empty_net_no_blobs(self):
        """Net() with no args has 0 blobs."""
        net = Net()
        assert len(net.blobs_array()) == 0

    def test_empty_net_no_layers(self):
        """Net() with no args has 0 layers."""
        net = Net()
        assert len(net.layers_array()) == 0

    def test_empty_net_zero_inputs(self):
        """Net() with no args reports 0 inputs."""
        net = Net()
        assert net.num_inputs() == 0

    def test_empty_net_zero_outputs(self):
        """Net() with no args reports 0 outputs."""
        net = Net()
        assert net.num_outputs() == 0

    def test_empty_net_input_names_empty(self):
        """Net() with no args returns empty input name list."""
        net = Net()
        assert net.input_blob_names() == []

    def test_empty_net_output_names_empty(self):
        """Net() with no args returns empty output name list."""
        net = Net()
        assert net.output_blob_names() == []

    def test_empty_net_blob_names_empty(self):
        """Net() with no args returns empty blob name list."""
        net = Net()
        assert net.blob_names() == []

    def test_empty_net_layer_names_empty(self):
        """Net() with no args returns empty layer name list."""
        net = Net()
        assert net.layer_names() == []

    def test_empty_net_blobs_dict_empty(self):
        """Net() with no args returns empty blobs_dict."""
        net = Net()
        assert net.blobs_dict == {}

    def test_empty_net_layers_dict_empty(self):
        """Net() with no args returns empty layers_dict."""
        net = Net()
        assert net.layers_dict == {}

    def test_empty_net_len_zero(self):
        """len(Net()) == 0."""
        net = Net()
        assert len(net) == 0

    def test_empty_net_iter_empty(self):
        """iter(Net()) yields no names."""
        net = Net()
        assert list(net) == []

    def test_empty_net_contains_false(self):
        """Any name not in empty net."""
        net = Net()
        assert "data" not in net
        assert "" not in net

    def test_empty_net_repr(self):
        """Empty net repr shows 0 blobs and 0 layers."""
        net = Net()
        r = repr(net)
        assert "0 blobs" in r
        assert "0 layers" in r

    def test_empty_net_has_blob_false(self):
        """has_blob returns False for any name on empty net."""
        net = Net()
        assert not net.has_blob("data")
        assert not net.has_blob("")

    def test_empty_net_has_layer_false(self):
        """has_layer returns False for any name on empty net."""
        net = Net()
        assert not net.has_layer("data")
        assert not net.has_layer("")


@require_cpp_extension
class TestNetConstructorErrors:
    """Boundary tests for Net constructor error cases."""

    def test_empty_string_prototxt_raises(self):
        """Net('') raises ValueError (empty proto text)."""
        with pytest.raises((ValueError, RuntimeError)):
            Net("")

    def test_invalid_prototxt_raises(self):
        """Net with garbage text raises RuntimeError (parse failure)."""
        with pytest.raises((ValueError, RuntimeError)):
            Net("this is not a valid prototxt {{{{")

    def test_whitespace_only_prototxt(self):
        """Net with whitespace-only string may raise or create empty net."""
        try:
            net = Net("   \n\t  ")
            # If it succeeds, net should be empty or near-empty
            assert len(net.layers_array()) == 0
        except (ValueError, RuntimeError):
            pass  # Also acceptable


@require_cpp_extension
class TestNetForwardBoundaries:
    """Boundary tests for Net forward pass edge cases."""

    @pytest.fixture
    def simple_input_net(self):
        prototxt = """name: "input_only"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 } } }"""
        return Net(prototxt)

    @pytest.fixture
    def mlp_net_loaded(self):
        """MLP net with weights loaded correctly by layer name (not by fragile index)."""
        prototxt = """name: "mlp_loaded"
layer { name: "data" type: "Input" top: "data" input_param { shape { dim: 2 dim: 3 } } }
layer { name: "ip1" type: "InnerProduct" bottom: "data" top: "ip1" inner_product_param { num_output: 4 bias_term: true } }
layer { name: "relu1" type: "ReLU" bottom: "ip1" top: "ip1" }
layer { name: "ip2" type: "InnerProduct" bottom: "ip1" top: "ip2" inner_product_param { num_output: 2 bias_term: true } }
layer { name: "prob" type: "Softmax" bottom: "ip2" top: "prob" }"""
        param = net_param_from_string(prototxt)
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
        # Load weights by layer name, not by fragile array index
        ip1 = net.layer_by_name("ip1")
        if len(ip1.blobs) >= 2:
            ip1.blobs[0].from_numpy(W1)
            ip1.blobs[1].from_numpy(b1.reshape(-1))
        ip2 = net.layer_by_name("ip2")
        if len(ip2.blobs) >= 2:
            ip2.blobs[0].from_numpy(W2)
            ip2.blobs[1].from_numpy(b2.reshape(-1))
        return net

    def test_forward_no_args(self, simple_input_net):
        """forward() with no arguments does not crash."""
        out = simple_input_net.forward()
        assert isinstance(out, dict)

    def test_forward_empty_dict(self, simple_input_net):
        """forward({}) with empty dict does not crash."""
        out = simple_input_net.forward({})
        assert isinstance(out, dict)

    def test_forward_none_explicit(self, simple_input_net):
        """forward(None) should not crash (treated as no inputs)."""
        out = simple_input_net.forward(None)
        assert isinstance(out, dict)

    def test_forward_wrong_input_name_silently_ignored(self, simple_input_net):
        """forward with non-existent input name should not crash."""
        inp = np.zeros((2, 3), dtype=np.float32)
        out = simple_input_net.forward({"nonexistent": inp})
        assert isinstance(out, dict)

    def test_forward_list_input_auto_converts(self, simple_input_net):
        """forward accepts Python list input and converts to float32."""
        out = simple_input_net.forward({"data": [[1, 2, 3], [4, 5, 6]]})
        assert isinstance(out, dict)
        assert "data" in out
        assert out["data"].dtype == np.float32

    def test_forward_float64_converts_to_float32(self, simple_input_net):
        """forward auto-converts float64 input to float32."""
        inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        out = simple_input_net.forward({"data": inp})
        assert out["data"].dtype == np.float32

    def test_forward_int_input_converts(self, simple_input_net):
        """forward accepts integer input and converts to float32."""
        inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        out = simple_input_net.forward({"data": inp})
        assert out["data"].dtype == np.float32

    def test_forward_returns_dict_of_numpy_arrays(self, mlp_net_loaded):
        """forward returns dict mapping str to np.ndarray."""
        inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        out = mlp_net_loaded.forward({"data": inp})
        assert isinstance(out, dict)
        for name, arr in out.items():
            assert isinstance(name, str)
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == np.float32

    def test_forward_all_no_kwargs(self, simple_input_net):
        """forward_all() with no kwargs does not crash."""
        out = simple_input_net.forward_all()
        assert isinstance(out, dict)

    def test_forward_deterministic(self, mlp_net_loaded):
        """Same input should produce same output on multiple forwards."""
        inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        out1 = mlp_net_loaded.forward({"data": inp})
        out2 = mlp_net_loaded.forward({"data": inp})
        for key in out1:
            np.testing.assert_array_equal(out1[key], out2[key])

    def test_forward_output_not_input_reference(self, mlp_net_loaded):
        """Forward output arrays are copies, not views into internal buffers."""
        inp = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        out = mlp_net_loaded.forward({"data": inp})
        if "prob" in out:
            out["prob"][0, 0] = 999.0
            out2 = mlp_net_loaded.forward({"data": inp})
            assert out2["prob"][0, 0] != 999.0

    def test_forward_nan_input_no_crash(self, mlp_net_loaded):
        """Forward with NaN input should not crash (may produce NaN output)."""
        inp = np.full((2, 3), np.nan, dtype=np.float32)
        out = mlp_net_loaded.forward({"data": inp})
        assert isinstance(out, dict)

    def test_forward_inf_input_no_crash(self, mlp_net_loaded):
        """Forward with Inf input should not crash."""
        inp = np.full((2, 3), np.inf, dtype=np.float32)
        out = mlp_net_loaded.forward({"data": inp})
        assert isinstance(out, dict)

    def test_forward_zero_input(self, mlp_net_loaded):
        """Forward with all-zero input produces valid output."""
        inp = np.zeros((2, 3), dtype=np.float32)
        out = mlp_net_loaded.forward({"data": inp})
        if "prob" in out:
            prob = out["prob"]
            np.testing.assert_allclose(prob.sum(axis=1), np.array([1.0, 1.0]), rtol=1e-5)


@require_cpp_extension
class TestNetConsistency:
    """Consistency checks between different Net accessor methods."""

    def test_blob_names_count_matches_blobs_array(self, mlp_net):
        """blob_names() length matches blobs_array() length."""
        assert len(mlp_net.blob_names()) == len(mlp_net.blobs_array())

    def test_layer_names_count_matches_layers_array(self, mlp_net):
        """layer_names() length matches layers_array() length."""
        assert len(mlp_net.layer_names()) == len(mlp_net.layers_array())

    def test_input_blobs_count_matches_num_inputs(self, mlp_net):
        """input_blobs_array() length matches num_inputs()."""
        assert len(mlp_net.input_blobs_array()) == mlp_net.num_inputs()

    def test_output_blobs_count_matches_num_outputs(self, mlp_net):
        """output_blobs_array() length matches num_outputs()."""
        assert len(mlp_net.output_blobs_array()) == mlp_net.num_outputs()

    def test_blobs_dict_keys_match_blob_names(self, mlp_net):
        """blobs_dict keys are consistent with blob_names()."""
        bd = mlp_net.blobs_dict
        names = mlp_net.blob_names()
        for name in names:
            assert name in bd

    def test_layers_dict_keys_match_layer_names(self, mlp_net):
        """layers_dict keys are consistent with layer_names()."""
        ld = mlp_net.layers_dict
        names = mlp_net.layer_names()
        for name in names:
            assert name in ld

    def test_layer_by_name_consistent_with_layers_array(self, mlp_net):
        """layer_by_name returns the same object as layers_array for same name."""
        for layer in mlp_net.layers_array():
            name = layer.name
            fetched = mlp_net.layer_by_name(name)
            assert fetched.name == name
            assert fetched.type == layer.type

    def test_blob_by_name_consistent_with_blobs_array(self, mlp_net):
        """blob_by_name returns blob with correct shape for known names."""
        data_blob = mlp_net.blob_by_name("data")
        assert data_blob.shape == (2, 3)

    def test_iter_yields_blob_names(self, mlp_net):
        """list(net) contains all blob names from blob_names()."""
        names = list(mlp_net)
        assert set(names) == set(mlp_net.blob_names())

    def test_len_matches_blobs_array(self, mlp_net):
        """len(net) == len(net.blobs_array())."""
        assert len(mlp_net) == len(mlp_net.blobs_array())

    def test_input_names_consistent_with_input_blobs(self, mlp_net):
        """input_blob_names length matches input_blobs_array length."""
        names = mlp_net.input_blob_names()
        blobs = mlp_net.input_blobs_array()
        assert len(names) == len(blobs)

    def test_output_names_consistent_with_output_blobs(self, mlp_net):
        """output_blob_names length matches output_blobs_array length."""
        names = mlp_net.output_blob_names()
        blobs = mlp_net.output_blobs_array()
        assert len(names) == len(blobs)

