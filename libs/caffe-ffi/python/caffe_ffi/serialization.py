"""Model serialization for caffe-ffi: save/load weights in caffemodel format.

The caffemodel format is a serialized ``caffe.NetParameter`` protobuf. Because
caffe-ffi loads weights by matching layer *names* (see :meth:`Net.CopyTrainedLayersFrom`),
a weight-only caffemodel only needs each layer's ``name`` and its ``blobs``.

This module provides:

* :func:`save_net` — write a ``Net``'s current weights to a ``.caffemodel`` file.
* :func:`load_net` — load weights from a ``.caffemodel`` file into a ``Net``.
* :func:`weights_to_dict` / :func:`dict_to_weights` — round-trip weights to a
  plain ``dict`` keyed by ``"layer_name:blob_index"`` (useful for pickling,
  checkpointing, and inspecting weights).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np

from . import caffe_pb2
from ._core import Net

__all__ = [
    "save_net",
    "load_net",
    "weights_to_dict",
    "dict_to_weights",
    "net_parameter_to_file",
]


def _blob_to_proto(blob) -> caffe_pb2.BlobProto:
    """Convert a Blob to a BlobProto (shape + data)."""
    proto = caffe_pb2.BlobProto()
    data = blob.data_tensor
    proto.shape.dim.extend(int(d) for d in data.shape)
    proto.data.extend(data.flatten().tolist())
    return proto


def _iter_weight_blobs(net: Net):
    """Yield ``(layer_name, layer_type, blob_index, blob)`` for every layer blob."""
    for layer in net.layers_array():
        blobs = layer.blobs
        for i, blob in enumerate(blobs):
            yield layer.name, layer.type, i, blob


def net_parameter_to_file(net: Net, path: Union[str, Path]) -> None:
    """Serialize a Net's weights to a ``.caffemodel`` file (NetParameter protobuf)."""
    path = Path(path)
    param = caffe_pb2.NetParameter()
    param.name = net.name or "caffe_ffi"
    for layer_name, layer_type, _i, blob in _iter_weight_blobs(net):
        lp = param.layer.add()
        lp.name = layer_name
        lp.type = layer_type
        lp.blobs.append(_blob_to_proto(blob))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(param.SerializeToString())


def save_net(net: Net, path: Union[str, Path]) -> None:
    """Save a Net's weights to a ``.caffemodel`` file.

    Parameters
    ----------
    net : Net
        The network whose weights are to be saved.
    path : str or Path
        Destination file path (`.caffemodel`).
    """
    net_parameter_to_file(net, path)


def load_net(net: Net, path: Union[str, Path]) -> None:
    """Load weights from a ``.caffemodel`` file into ``net``.

    Weights are matched to layers by name (see :meth:`Net.CopyTrainedLayersFrom`).
    """
    net.copy_from(path)


def weights_to_dict(net: Net) -> Dict[str, np.ndarray]:
    """Export all learnable weights as a dict keyed by ``"layer_name:blob_index"``.

    Returns shallow copies of the current weight tensors. Use with
    :func:`dict_to_weights` to restore them later.
    """
    return {f"{name}:{i}": blob.data_tensor.copy() for name, _t, i, blob in _iter_weight_blobs(net)}


def dict_to_weights(net: Net, weights: Dict[str, np.ndarray]) -> None:
    """Restore weights from the dict produced by :func:`weights_to_dict`.

    Keys that do not match any layer:blob_index in ``net`` are ignored.
    """
    by_key = {}
    for layer in net.layers_array():
        for i, blob in enumerate(layer.blobs):
            by_key[f"{layer.name}:{i}"] = blob
    for key, arr in weights.items():
        blob = by_key.get(key)
        if blob is None:
            continue
        arr = np.asarray(arr, dtype=np.float32)
        if tuple(arr.shape) != blob.shape:
            blob.Reshape(list(arr.shape))
        blob.data_tensor[:] = arr