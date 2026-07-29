from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np

from . import _ffi_api

_logger = logging.getLogger("caffe_ffi.core")


def _import_tvm_ffi():
    try:
        import tvm_ffi
        return tvm_ffi
    except ImportError:
        return None


_tvm_ffi = _import_tvm_ffi()
_NATIVE_MODE = _ffi_api.is_available() and _tvm_ffi is not None

if _NATIVE_MODE:
    _Object = _tvm_ffi.Object
    _register_object = _tvm_ffi.register_object
    _get_global_func = _tvm_ffi.get_global_func
else:
    _Object = object
    _register_object = None
    _get_global_func = None


def _noop_decorator(type_key):
    def decorator(cls):
        return cls
    return decorator


_reg = _register_object if _NATIVE_MODE else _noop_decorator


def _get_fn(name: str):
    if not _NATIVE_MODE:
        return None
    return _get_global_func(f"caffe_ffi.{name}", allow_missing=True)


def _native_method(obj, name: str):
    """Call a C++ registered method on an object, bypassing Python overrides."""
    if not _NATIVE_MODE:
        raise RuntimeError(f"Native method '{name}' not available in Python-only mode")
    # Search through MRO for the method in C++ type info
    for cls in type(obj).__mro__:
        info = getattr(cls, '__tvm_ffi_type_info__', None)
        if info is not None:
            for m in info.methods:
                if m.name == name:
                    bound = m.func
                    if not m.is_static:
                        import types
                        bound = types.MethodType(m.func, obj)
                    return bound
    raise AttributeError(f"Native method '{name}' not found on {type(obj).__name__}")


@_reg("caffe_ffi.Blob")
class Blob(_Object):
    """N-dimensional tensor blob for storing network data and gradients.

    A Blob wraps a C++ tensor via TVM FFI when the native extension is available,
    providing zero-copy numpy interop through DLPack.

    Parameters
    ----------
    shape : list of int, optional
        Initial shape of the blob. If None, creates an empty blob (shape ()).
    """

    __slots__ = ('__dict__',)

    def __init__(self, shape: Optional[List[int]] = None, handle=None):
        if _NATIVE_MODE and handle is None:
            ctor = _get_fn("NewBlob")
            self.__init_handle_by_constructor__(ctor)
            if shape is not None:
                _native_method(self, 'Reshape')(list(shape))
        elif not _NATIVE_MODE:
            self._py_init(shape)

    def _py_init(self, shape=None):
        if shape is None:
            shape = ()
        self._py_data = np.zeros(shape, dtype=np.float32)
        self._py_diff = np.zeros_like(self._py_data)
        self._py_shape = list(shape)
        self._py_name = ""

    @property
    def _is_native(self) -> bool:
        return _NATIVE_MODE and self.__chandle__() != 0

    @property
    def shape(self) -> tuple:
        if self._is_native:
            return tuple(_native_method(self, 'shape')())
        return tuple(self._py_shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def num_axes(self) -> int:
        return self.ndim

    @property
    def size(self) -> int:
        return self.count()

    def count(self, start_axis: int = 0, end_axis: Optional[int] = None) -> int:
        if self._is_native:
            if end_axis is None:
                return int(_native_method(self, 'count')())
            s = self.shape
            if end_axis < 0:
                end_axis += len(s)
            result = 1
            for i in range(start_axis, end_axis + 1):
                result *= s[i]
            return result
        s = self._py_shape
        if end_axis is None:
            end_axis = len(s)
        elif end_axis < 0:
            end_axis += len(s)
        result = 1
        for i in range(start_axis, end_axis):
            result *= s[i]
        return result

    def Reshape(self, shape: List[int]) -> None:
        if self._is_native:
            _native_method(self, 'Reshape')(list(shape))
        else:
            shape_list = list(shape)
            self._py_data = np.zeros(shape_list, dtype=np.float32)
            self._py_diff = np.zeros_like(self._py_data)
            self._py_shape = shape_list

    @property
    def name(self) -> str:
        if self._is_native:
            return str(_native_method(self, 'name')())
        return getattr(self, '_py_name', '')

    @name.setter
    def name(self, value: str) -> None:
        if self._is_native:
            _native_method(self, 'set_name')(str(value))
        else:
            self._py_name = value

    @property
    def data_tensor(self) -> np.ndarray:
        """Zero-copy numpy view of the data tensor (modifications affect C++ memory)."""
        if self._is_native:
            fn = _get_fn("BlobDataTensor")
            return np.from_dlpack(fn(self))
        return self._py_data

    @property
    def diff_tensor(self) -> np.ndarray:
        """Zero-copy numpy view of the diff tensor (modifications affect C++ memory)."""
        if self._is_native:
            fn = _get_fn("BlobDiffTensor")
            return np.from_dlpack(fn(self))
        return self._py_diff

    @property
    def data(self) -> np.ndarray:
        """Get data as numpy array (returns a copy for safety)."""
        return self.data_tensor.copy()

    @data.setter
    def data(self, value: np.ndarray) -> None:
        """Set data from numpy array."""
        arr = np.asarray(value, dtype=np.float32)
        if tuple(arr.shape) != self.shape:
            self.Reshape(list(arr.shape))
        self.data_tensor[:] = arr

    @property
    def diff(self) -> np.ndarray:
        """Get diff as numpy array (returns a copy for safety)."""
        return self.diff_tensor.copy()

    @diff.setter
    def diff(self, value: np.ndarray) -> None:
        """Set diff from numpy array."""
        arr = np.asarray(value, dtype=np.float32)
        if tuple(arr.shape) != self.shape:
            self.Reshape(list(arr.shape))
        self.diff_tensor[:] = arr

    def Update(self) -> None:
        """Update data by subtracting diff (data -= diff)."""
        if self._is_native:
            fn = _get_fn("BlobUpdate")
            fn(self)
        elif self._py_data is not None and self._py_diff is not None:
            self._py_data -= self._py_diff

    def zero(self) -> Blob:
        """Set data and diff to all zeros."""
        self.data_tensor.fill(0)
        self.diff_tensor.fill(0)
        return self

    def fill(self, value: float) -> Blob:
        """Fill data with a constant value."""
        self.data_tensor.fill(np.float32(value))
        self.diff_tensor.fill(0)
        return self

    def copy_from(self, other) -> Blob:
        """Copy data from another blob or numpy array."""
        if isinstance(other, Blob):
            other_data = other.data_tensor
        else:
            other_data = np.asarray(other, dtype=np.float32)
        if tuple(other_data.shape) != self.shape:
            self.Reshape(list(other_data.shape))
        self.data_tensor[:] = other_data
        return self

    def from_numpy(self, arr: np.ndarray, set_diff: bool = False) -> Blob:
        """Reshape blob and set data from numpy array."""
        arr = np.asarray(arr, dtype=np.float32)
        self.Reshape(list(arr.shape))
        if set_diff:
            self.diff = arr
        else:
            self.data = arr
        return self

    def to_numpy(self, get_diff: bool = False) -> np.ndarray:
        """Convert blob to numpy array (returns a copy)."""
        return self.diff_tensor.copy() if get_diff else self.data_tensor.copy()

    def get_data(self) -> List[float]:
        if self._is_native:
            return list(_native_method(self, 'get_data')())
        return self._py_data.flatten().tolist()

    def set_data(self, data) -> None:
        if self._is_native:
            # Convert input to numpy float32 array for zero-copy DLPack interop
            if not isinstance(data, np.ndarray):
                data = np.array(data, dtype=np.float32)
            if data.dtype != np.float32:
                data = data.astype(np.float32)
            expected_shape = tuple(self.shape)
            if data.shape != expected_shape:
                data = data.reshape(expected_shape)
            _native_method(self, 'set_data')(data)
        else:
            self._py_data = np.array(data, dtype=np.float32).reshape(self._py_shape)

    def get_diff(self) -> List[float]:
        if self._is_native:
            return list(_native_method(self, 'get_diff')())
        return self._py_diff.flatten().tolist()

    def set_diff(self, diff) -> None:
        if self._is_native:
            if not isinstance(diff, np.ndarray):
                diff = np.array(diff, dtype=np.float32)
            if diff.dtype != np.float32:
                diff = diff.astype(np.float32)
            expected_shape = tuple(self.shape)
            if diff.shape != expected_shape:
                diff = diff.reshape(expected_shape)
            _native_method(self, 'set_diff')(diff)
        else:
            self._py_diff = np.array(diff, dtype=np.float32).reshape(self._py_shape)

    @property
    def construction_backtrace(self) -> str:
        if self._is_native:
            return str(_native_method(self, 'construction_backtrace')())
        return "(backtrace not available: Python-only mode)"

    def __repr__(self) -> str:
        return f"Blob(shape={self.shape}, dtype=float32)"


@_reg("caffe_ffi.Layer")
class Layer(_Object):
    """Neural network layer.

    Layers are typically created by Net during network construction.
    Each layer has a type, name, and parameter blobs (weights/biases).
    """

    __slots__ = ('__dict__',)

    def __init__(self, handle=None):
        if not _NATIVE_MODE:
            self._py_name = ""
            self._py_type_str = ""
            self._py_blobs = []

    @property
    def _is_native(self) -> bool:
        return _NATIVE_MODE and self.__chandle__() != 0

    @property
    def type(self) -> str:
        if self._is_native:
            return str(_native_method(self, 'type')())
        return getattr(self, '_py_type_str', '')

    @property
    def name(self) -> str:
        if self._is_native:
            return str(_native_method(self, 'name')())
        return getattr(self, '_py_name', '')

    @property
    def blobs(self) -> List[Blob]:
        if self._is_native:
            return list(_native_method(self, 'blobs_array')())
        return list(getattr(self, '_py_blobs', []))

    def __repr__(self) -> str:
        name = self.name
        if name:
            return f"Layer(name='{name}', type='{self.type}')"
        return f"Layer(type='{self.type}')"


@_reg("caffe_ffi.Net")
class Net(_Object):
    """Neural network container.

    A Net represents a complete neural network with layers and blobs
    connected in a directed acyclic graph.

    Parameters
    ----------
    param : str or Path, optional
        Path to a .prototxt file, or a prototxt string defining the network.
    """

    __slots__ = ('__dict__',)

    def __init__(self, param: Optional[Union[str, os.PathLike]] = None, handle=None):
        if _NATIVE_MODE and handle is None and param is not None:
            param_str = str(param)
            if os.path.isfile(param_str):
                ctor = _get_fn("NewNetFromFile")
                if ctor is None:
                    raise RuntimeError("caffe_ffi.NewNetFromFile not available")
                self.__init_handle_by_constructor__(ctor, param_str)
            else:
                ctor = _get_fn("NewNetFromProtoString")
                if ctor is None:
                    raise RuntimeError("caffe_ffi.NewNetFromProtoString not available")
                self.__init_handle_by_constructor__(ctor, param_str)
        elif not _NATIVE_MODE:
            self._py_init(param)

    def _py_init(self, param=None):
        self._py_name = getattr(param, 'name', '') if param else ''
        self._py_blobs = {}
        self._py_layers = {}
        self._py_blob_list = []
        self._py_layer_list = []
        self._py_input_blobs = []
        self._py_output_blobs = []
        self._py_input_blob_names = []
        self._py_output_blob_names = []

    @property
    def _is_native(self) -> bool:
        return _NATIVE_MODE and self.__chandle__() != 0

    @property
    def name(self) -> str:
        if self._is_native:
            return str(_native_method(self, 'name')())
        return getattr(self, '_py_name', '')

    def Forward(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Blob]:
        if inputs is None:
            inputs = {}
        if self._is_native:
            input_map = {}
            for k, v in inputs.items():
                if isinstance(v, np.ndarray):
                    # Pass numpy array directly for zero-copy DLPack Tensor interop
                    if v.dtype != np.float32:
                        v = v.astype(np.float32)
                    input_map[k] = v
                elif isinstance(v, Blob):
                    # Use the blob's data tensor directly
                    input_map[k] = v.to_numpy()
                elif isinstance(v, (list, tuple)):
                    # Convert list/tuple to numpy float32 array
                    input_map[k] = np.array(v, dtype=np.float32)
                else:
                    input_map[k] = np.array(v, dtype=np.float32)
            result = _native_method(self, 'Forward')(input_map)
            return {str(k): v for k, v in result.items()}
        return self._py_forward(inputs)

    def _py_forward(self, inputs):
        return {}

    def forward(self, input_dict: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """Run forward pass and return output blobs as numpy arrays.

        Parameters
        ----------
        input_dict : dict of str to ndarray, optional
            Mapping from input blob names to numpy arrays.

        Returns
        -------
        dict of str to ndarray
            Mapping from output blob names to numpy arrays (copies).
        """
        if input_dict is None:
            input_dict = {}

        if self._is_native:
            for name, arr in input_dict.items():
                try:
                    blob = self.blob_by_name(name)
                    blob.data = np.asarray(arr, dtype=np.float32)
                except (KeyError, RuntimeError):
                    pass

            result_map = self.Forward()
            result = {}
            if result_map:
                for name, blob in result_map.items():
                    result[str(name)] = blob.data
            else:
                output_blobs = self.output_blobs_array()
                output_names = self.output_blob_names()
                for i, blob in enumerate(output_blobs):
                    name = str(output_names[i]) if i < len(output_names) else f"output_{i}"
                    result[name] = blob.data
            return result
        return self._forward_pure_python(input_dict)

    def forward_all(self, **kwargs: np.ndarray) -> Dict[str, np.ndarray]:
        """Convenience wrapper for forward with keyword arguments."""
        return self.forward(kwargs)

    def blobs_array(self) -> List[Blob]:
        if self._is_native:
            return list(_native_method(self, 'blobs_array')())
        return list(getattr(self, '_py_blob_list', []))

    def layers_array(self) -> List[Layer]:
        if self._is_native:
            return list(_native_method(self, 'layers_array')())
        return list(getattr(self, '_py_layer_list', []))

    def input_blobs_array(self) -> List[Blob]:
        if self._is_native:
            return list(_native_method(self, 'input_blobs_array')())
        return list(getattr(self, '_py_input_blobs', []))

    def output_blobs_array(self) -> List[Blob]:
        if self._is_native:
            return list(_native_method(self, 'output_blobs_array')())
        return list(getattr(self, '_py_output_blobs', []))

    def blob_names(self) -> List[str]:
        if self._is_native:
            return [str(n) for n in _native_method(self, 'blob_names')()]
        return list(getattr(self, '_py_blobs', {}).keys())

    def layer_names(self) -> List[str]:
        if self._is_native:
            return [str(n) for n in _native_method(self, 'layer_names')()]
        return list(getattr(self, '_py_layers', {}).keys())

    def input_blob_names(self) -> List[str]:
        if self._is_native:
            return [str(n) for n in _native_method(self, 'input_blob_names')()]
        return list(getattr(self, '_py_input_blob_names', []))

    def output_blob_names(self) -> List[str]:
        if self._is_native:
            return [str(n) for n in _native_method(self, 'output_blob_names')()]
        return list(getattr(self, '_py_output_blob_names', []))

    def blob_by_name(self, name: str) -> Blob:
        if self._is_native:
            return _native_method(self, 'blob_by_name')(str(name))
        if name in getattr(self, '_py_blobs', {}):
            return self._py_blobs[name]
        raise KeyError(f"Blob '{name}' not found")

    def layer_by_name(self, name: str) -> Layer:
        if self._is_native:
            return _native_method(self, 'layer_by_name')(str(name))
        if name in getattr(self, '_py_layers', {}):
            return self._py_layers[name]
        raise KeyError(f"Layer '{name}' not found")

    def has_blob(self, name: str) -> bool:
        if self._is_native:
            return bool(_native_method(self, 'has_blob')(str(name)))
        return name in getattr(self, '_py_blobs', {})

    def has_layer(self, name: str) -> bool:
        if self._is_native:
            return bool(_native_method(self, 'has_layer')(str(name)))
        return name in getattr(self, '_py_layers', {})

    def num_inputs(self) -> int:
        if self._is_native:
            return int(_native_method(self, 'num_inputs')())
        return len(getattr(self, '_py_input_blobs', []))

    def num_outputs(self) -> int:
        if self._is_native:
            return int(_native_method(self, 'num_outputs')())
        return len(getattr(self, '_py_output_blobs', []))

    @property
    def blobs_dict(self) -> Dict[str, Blob]:
        result = {}
        names = self.blob_names()
        blobs = self.blobs_array()
        for i, blob in enumerate(blobs):
            if i < len(names):
                result[names[i]] = blob
            else:
                result[f"blob_{i}"] = blob
        return result

    @property
    def layers_dict(self) -> Dict[str, Layer]:
        result = {}
        names = self.layer_names()
        layers = self.layers_array()
        for i, layer in enumerate(layers):
            if i < len(names):
                result[names[i]] = layer
            else:
                result[f"layer_{i}"] = layer
        return result

    def CopyTrainedLayersFrom(self, trained_filename: Union[str, Path]) -> None:
        if self._is_native:
            _native_method(self, 'CopyTrainedLayersFrom')(str(trained_filename))
        else:
            self._copy_from_pure_python(trained_filename)

    def copy_from(self, trained_filename: Union[str, Path]) -> None:
        """Copy trained layers from a caffemodel file."""
        self.CopyTrainedLayersFrom(trained_filename)

    def _copy_from_pure_python(self, trained_filename):
        from .io import read_net_from_binary
        trained_net_param = read_net_from_binary(trained_filename)
        trained_layer_map = {layer.name: layer for layer in trained_net_param.layer}
        for layer in self.layers_array():
            layer_name = layer.name
            if not layer_name or layer_name not in trained_layer_map:
                continue
            source_layer = trained_layer_map[layer_name]
            target_blobs = layer.blobs
            num_blobs_to_copy = min(len(target_blobs), len(source_layer.blobs))
            for j in range(num_blobs_to_copy):
                source_blob_proto = source_layer.blobs[j]
                target_blob = target_blobs[j]
                if source_blob_proto.HasField('shape') and source_blob_proto.shape.dim:
                    dims = list(source_blob_proto.shape.dim)
                else:
                    dims = [source_blob_proto.num, source_blob_proto.channels,
                           source_blob_proto.height, source_blob_proto.width]
                    dims = [d for d in dims if d != 0]
                data_list = None
                if source_blob_proto.data:
                    data_list = list(source_blob_proto.data)
                elif source_blob_proto.double_data:
                    data_list = [float(v) for v in source_blob_proto.double_data]
                if not dims and data_list:
                    dims = [len(data_list)]
                if dims:
                    target_blob.Reshape(dims)
                if data_list:
                    target_blob.data = np.array(data_list, dtype=np.float32).reshape(target_blob.shape)

    def _forward_pure_python(self, input_dict):
        for name, arr in input_dict.items():
            if name in getattr(self, '_py_blobs', {}):
                self._py_blobs[name].from_numpy(arr)
        result = {}
        for blob in getattr(self, '_py_output_blobs', []):
            result[blob.name] = blob.data
        return result

    def __getitem__(self, name: str) -> Blob:
        return self.blob_by_name(name)

    def __contains__(self, name: str) -> bool:
        return self.has_blob(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self.blobs_dict.keys())

    def __len__(self) -> int:
        return len(self.blobs_array())

    def __repr__(self) -> str:
        return f"Net(name='{self.name}', {len(self.blobs_array())} blobs, {len(self.layers_array())} layers)"
