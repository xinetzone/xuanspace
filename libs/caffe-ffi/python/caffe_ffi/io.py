from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from google.protobuf import text_format

from . import caffe_pb2
from . import _ffi_api
from ._core import Net


def read_net_from_prototxt(prototxt_path: Union[str, Path]) -> caffe_pb2.NetParameter:
    prototxt_path = Path(prototxt_path)
    param = caffe_pb2.NetParameter()
    with open(prototxt_path, 'r', encoding='utf-8') as f:
        text_format.Parse(f.read(), param)
    return param


def read_net_from_binary(caffemodel_path: Union[str, Path]) -> caffe_pb2.NetParameter:
    caffemodel_path = Path(caffemodel_path)
    param = caffe_pb2.NetParameter()
    with open(caffemodel_path, 'rb') as f:
        param.ParseFromString(f.read())
    return param


def net_param_from_string(prototxt_str: str) -> caffe_pb2.NetParameter:
    param = caffe_pb2.NetParameter()
    text_format.Parse(prototxt_str, param)
    return param


def read_net(
    prototxt_path: Union[str, Path],
    caffemodel_path: Optional[Union[str, Path]] = None,
) -> Net:
    from . import _ffi_api as ffi
    
    prototxt_path = Path(prototxt_path)
    
    if ffi.is_available() and caffemodel_path is None:
        new_net_from_file = ffi.get_global_func("caffe_ffi.NewNetFromFile")
        if new_net_from_file is not None:
            return new_net_from_file(str(prototxt_path))
    
    param = read_net_from_prototxt(prototxt_path)
    
    if caffemodel_path is not None:
        weights = read_net_from_binary(caffemodel_path)
        _merge_weights(param, weights)
    
    if ffi.is_available():
        new_net_from_str = ffi.get_global_func("caffe_ffi.NewNetFromProtoString")
        if new_net_from_str is not None:
            proto_text = text_format.MessageToString(param)
            return new_net_from_str(proto_text)
    
    net = Net(param=param)
    _build_net_from_param(net, param)
    return net


def net_from_param(param: caffe_pb2.NetParameter) -> Net:
    from . import _ffi_api as ffi
    
    if ffi.is_available():
        new_net_from_str = ffi.get_global_func("caffe_ffi.NewNetFromProtoString")
        if new_net_from_str is not None:
            proto_text = text_format.MessageToString(param)
            return new_net_from_str(proto_text)
    
    net = Net(param=param)
    _build_net_from_param(net, param)
    return net


def _merge_weights(param: caffe_pb2.NetParameter, weights: caffe_pb2.NetParameter) -> None:
    layer_map = {layer.name: layer for layer in param.layer}
    for w_layer in weights.layer:
        if w_layer.name in layer_map and w_layer.blobs:
            layer_map[w_layer.name].blobs.extend(w_layer.blobs)


def _build_net_from_param(net: Net, param: caffe_pb2.NetParameter) -> None:
    from ._core import Blob, Layer as _LayerCls

    net._py_name = param.name if param.name else ""
    net._py_blobs = {}
    net._py_layers = {}
    net._py_blob_list = []
    net._py_layer_list = []
    net._py_input_blobs = []
    net._py_output_blobs = []

    for input_name in param.input:
        blob = Blob()
        blob.name = input_name
        if param.input_shape:
            for i, shape in enumerate(param.input_shape):
                if i == len(net._py_input_blobs):
                    dims = list(shape.dim)
                    blob.Reshape(dims)
                    break
        elif param.input_dim:
            blob.Reshape(list(param.input_dim))
        net._py_blobs[input_name] = blob
        net._py_blob_list.append(blob)
        net._py_input_blobs.append(blob)

    for layer_param in param.layer:
        layer = _LayerCls()
        layer._py_name = layer_param.name
        layer._py_type_str = layer_param.type
        layer._py_blobs = []

        for blob_proto in layer_param.blobs:
            blob = Blob()
            if blob_proto.HasField('shape') and blob_proto.shape.dim:
                dims = list(blob_proto.shape.dim)
            else:
                dims = [blob_proto.num, blob_proto.channels, blob_proto.height, blob_proto.width]
                dims = [d for d in dims if d != 0]

            data_list = None
            if blob_proto.data:
                data_list = list(blob_proto.data)
            elif blob_proto.double_data:
                data_list = [float(v) for v in blob_proto.double_data]

            if not dims and data_list:
                dims = [len(data_list)]
            if not dims:
                dims = [0]

            blob.Reshape(dims)
            if data_list:
                blob.data = np.array(data_list, dtype=np.float32).reshape(dims)
            layer._py_blobs.append(blob)

        net._py_layers[layer_param.name] = layer
        net._py_layer_list.append(layer)

        for top_name in layer_param.top:
            if top_name not in net._py_blobs:
                blob = Blob()
                blob.name = top_name
                net._py_blobs[top_name] = blob
                net._py_blob_list.append(blob)
                if layer_param.type in ["Softmax", "SoftmaxWithLoss"]:
                    net._py_output_blobs.append(blob)

    if not net._py_output_blobs:
        for layer_param in reversed(param.layer):
            for top_name in layer_param.top:
                if top_name in net._py_blobs and top_name not in [b.name for b in net._py_input_blobs]:
                    net._py_output_blobs.append(net._py_blobs[top_name])
                    break
            if net._py_output_blobs:
                break
