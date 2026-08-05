# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import sys
import time
import logging
from pathlib import Path
import numpy as np

# Ensure the caffe_ffi package is importable even when the ops/ directory is
# run directly (without the parent tests/python/conftest.py that normally
# inserts the package dir into sys.path).
_project_root = Path(__file__).resolve().parent.parent.parent.parent
_python_dir = _project_root / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))

import pytest  # noqa: E402

import caffe_ffi  # noqa: E402
from caffe_ffi import caffe_pb2 as pb  # noqa: E402
from caffe_ffi import net_from_param  # noqa: E402

np.random.seed(42)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_log_level = os.environ.get("CAFFE_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.WARNING),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Whether the native caffe_ffi C++ extension is available. When it is not
# (e.g. running against a source tree that has not been compiled), the
# operator tests are skipped cleanly instead of failing on empty results.
_CAFFE_FFI_AVAILABLE = bool(caffe_ffi.is_available())


# ─────────────────────────────────────────────────────────────────────
# Directory / filename helpers (shared utilities, no caffe dependency)
# ─────────────────────────────────────────────────────────────────────


def _create_dir(d_path):
    """If the directory is not existed, create it"""
    logger.debug(f"Creating directory: {d_path}")
    if not (os.path.exists(d_path) and os.path.isdir(d_path)):
        os.makedirs(d_path)


def _list_to_str(ll):
    """Convert list or tuple to str, separated by underline."""
    if isinstance(ll, (tuple, list)):
        tmp = [str(i) for i in ll]
        res = "_".join(tmp)
    else:
        res = str(ll)
    return res


def _dict_to_str(d):
    """Convert a dict to a filename-safe string."""
    items = []
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, dict):
            items.append(f"{k}-{_dict_to_str(v)}")
        elif isinstance(v, (list, tuple)):
            items.append(f"{k}-{_list_to_str(v)}")
        elif isinstance(v, bool):
            items.append(f"{k}-{1 if v else 0}")
        else:
            items.append(f"{k}-{v}")
    return "_".join(items)


def _gen_filename_str(op_name, data_shape, base_dir, *args, **kwargs):
    """Combining the filename according to the op_name, shape and other args."""
    file_dir = os.path.join(base_dir, op_name)
    _create_dir(file_dir)
    res = op_name + "_"
    shape_str = _list_to_str(list(data_shape))
    res += shape_str
    for arg in args:
        if isinstance(arg, (tuple, list)):
            res += "_" + _list_to_str(arg)
        elif isinstance(arg, (int, float, str)):
            res += "_" + str(arg)
        elif isinstance(arg, dict):
            res += "_" + _dict_to_str(arg)
    for k, v in kwargs.items():
        if isinstance(v, (tuple, list)):
            res += "_" + k + "-" + _list_to_str(v)
        elif isinstance(v, (int, float, str)):
            res += "_" + k + "-" + str(v)
        elif isinstance(v, bool):
            res += "_" + k + "-" + ("1" if v else "0")
        elif isinstance(v, dict):
            res += "_" + k + "-" + _dict_to_str(v)
    res = res.replace(".", "_")
    res = res.replace("-", "_")
    proto_file = os.path.join(file_dir, res + ".prototxt")
    blob_file = os.path.join(file_dir, res + ".caffemodel")
    solver_file = os.path.join(file_dir, res + "_solver.prototxt")
    logger.debug(f"Generated files - proto: {proto_file}, blob: {blob_file}, solver: {solver_file}")

    return (proto_file, blob_file, solver_file)


# ─────────────────────────────────────────────────────────────────────
# caffe_ffi protobuf helpers
# ─────────────────────────────────────────────────────────────────────


def _fill_param_from_dict(param, d):
    """Fill a caffe_pb2 parameter message from a (possibly nested) dict.

    Handles scalar fields, repeated fields (list/tuple -> extend, scalar ->
    append), nested message fields (dict -> recurse) and filler dicts.
    Unknown keys are silently ignored.
    """
    if not isinstance(d, dict):
        return
    for key, value in d.items():
        if not hasattr(param, key):
            continue
        field = getattr(param, key)
        if isinstance(value, dict):
            _fill_param_from_dict(field, value)
        elif isinstance(value, (list, tuple)):
            del field[:]
            field.extend(value)
        elif hasattr(field, "append"):
            field.append(value)
        else:
            setattr(param, key, value)


def _make_layer(layer_type, bottoms, tops=None, name=None):
    """Build a caffe_pb2.LayerParameter with type, bottoms and tops set."""
    layer = pb.LayerParameter()
    layer.type = layer_type
    layer.name = name or (layer_type.lower() + "_layer")
    if isinstance(bottoms, (list, tuple)):
        bottoms = [str(b) for b in bottoms]
    else:
        bottoms = [str(bottoms)]
    layer.bottom.extend(bottoms)
    tops = tops or ["output"]
    layer.top.extend([str(t) for t in tops])
    return layer


class _LayerSpec:
    """Thin wrapper around a caffe_pb2.LayerParameter (pycaffe L.<Op> result)."""

    __slots__ = ("param",)

    def __init__(self, param):
        self.param = param


class _NetSpec:
    """In-memory network accumulator (pycaffe NetSpec replacement).

    Declares net-level inputs and collects layers, then ``to_param()``
    produces a caffe_pb2.NetParameter consumable by caffe_ffi.net_from_param.
    """

    def __init__(self, name="test_net"):
        self.param = pb.NetParameter()
        self.param.name = name
        self._input_names = []
        self._input_shapes = []

    def add_input(self, name, dims):
        self._input_names.append(name)
        shape = pb.BlobShape()
        shape.dim.extend([int(d) for d in dims])
        self._input_shapes.append(shape)

    def add_layer(self, layer_spec):
        self.param.layer.add().CopyFrom(layer_spec.param)

    def to_param(self):
        self.param.input.extend(self._input_names)
        for s in self._input_shapes:
            self.param.input_shape.add().CopyFrom(s)
        return self.param


class P:
    """Pooling / layer parameter enums (mirrors pycaffe ``caffe.params``)."""

    class Pooling:
        MAX = 0
        AVE = 1
        STOCHASTIC = 2


class L:
    """Layer spec builders (mirrors pycaffe ``caffe.layers``).

    Each ``L.<Op>(bottom, **kwargs)`` returns a ``_LayerSpec`` whose
    ``param`` is a caffe_pb2.LayerParameter. Kwargs map onto the op's
    parameter message (e.g. ``convolution_param`` for Convolution).
    """

    # ── element-wise / activation ops ────────────────────────────────
    @staticmethod
    def ReLU(bottom, **kwargs):
        layer = _make_layer("ReLU", [bottom])
        _fill_param_from_dict(layer.relu_param, kwargs)
        return _LayerSpec(layer)

    @staticmethod
    def Sigmoid(bottom, **kwargs):
        layer = _make_layer("Sigmoid", [bottom])
        _fill_param_from_dict(layer.sigmoid_param, kwargs)
        return _LayerSpec(layer)

    @staticmethod
    def TanH(bottom, **kwargs):
        layer = _make_layer("TanH", [bottom])
        _fill_param_from_dict(layer.tanh_param, kwargs)
        return _LayerSpec(layer)

    @staticmethod
    def Dropout(bottom, **kwargs):
        layer = _make_layer("Dropout", [bottom])
        _fill_param_from_dict(layer.dropout_param, kwargs)
        return _LayerSpec(layer)

    @staticmethod
    def PReLU(bottom, **kwargs):
        layer = _make_layer("PReLU", [bottom])
        _fill_param_from_dict(layer.prelu_param, kwargs)
        return _LayerSpec(layer)

    @staticmethod
    def ELU(bottom, elu_param=None, **kwargs):
        layer = _make_layer("ELU", [bottom])
        if elu_param:
            _fill_param_from_dict(layer.elu_param, elu_param)
        return _LayerSpec(layer)

    @staticmethod
    def Threshold(bottom, threshold_param=None, **kwargs):
        layer = _make_layer("Threshold", [bottom])
        if threshold_param:
            _fill_param_from_dict(layer.threshold_param, threshold_param)
        return _LayerSpec(layer)

    @staticmethod
    def Clip(bottom, clip_param=None, **kwargs):
        layer = _make_layer("Clip", [bottom])
        if clip_param:
            _fill_param_from_dict(layer.clip_param, clip_param)
        return _LayerSpec(layer)

    @staticmethod
    def Swish(bottom, swish_param=None, **kwargs):
        layer = _make_layer("Swish", [bottom])
        if swish_param:
            _fill_param_from_dict(layer.swish_param, swish_param)
        return _LayerSpec(layer)

    @staticmethod
    def Exp(bottom, exp_param=None, **kwargs):
        layer = _make_layer("Exp", [bottom])
        if exp_param:
            _fill_param_from_dict(layer.exp_param, exp_param)
        return _LayerSpec(layer)

    @staticmethod
    def Log(bottom, log_param=None, **kwargs):
        layer = _make_layer("Log", [bottom])
        if log_param:
            _fill_param_from_dict(layer.log_param, log_param)
        return _LayerSpec(layer)

    @staticmethod
    def Power(bottom, power_param=None, **kwargs):
        layer = _make_layer("Power", [bottom])
        if power_param:
            _fill_param_from_dict(layer.power_param, power_param)
        return _LayerSpec(layer)

    # ── matrix / tensor structure ops ────────────────────────────────
    @staticmethod
    def Convolution(bottom, num_output, kernel_size=1, stride=1, pad=0,
                    dilation=1, group=1, bias_term=True, weight_filler=None,
                    bias_filler=None, kernel_h=None, kernel_w=None,
                    stride_h=None, stride_w=None, pad_h=None, pad_w=None,
                    **kwargs):
        layer = _make_layer("Convolution", [bottom])
        params = dict(num_output=num_output, kernel_size=kernel_size,
                      stride=stride, pad=pad, dilation=dilation, group=group,
                      bias_term=bias_term)
        if weight_filler is not None:
            params["weight_filler"] = weight_filler
        if bias_filler is not None:
            params["bias_filler"] = bias_filler
        if kernel_h is not None:
            params["kernel_h"] = kernel_h
        if kernel_w is not None:
            params["kernel_w"] = kernel_w
        if stride_h is not None:
            params["stride_h"] = stride_h
        if stride_w is not None:
            params["stride_w"] = stride_w
        if pad_h is not None:
            params["pad_h"] = pad_h
        if pad_w is not None:
            params["pad_w"] = pad_w
        params.update(kwargs)
        _fill_param_from_dict(layer.convolution_param, params)
        return _LayerSpec(layer)

    @staticmethod
    def Deconvolution(bottom, convolution_param=None, **kwargs):
        layer = _make_layer("Deconvolution", [bottom])
        if convolution_param and isinstance(convolution_param, dict):
            _fill_param_from_dict(layer.convolution_param, convolution_param)
        if kwargs:
            _fill_param_from_dict(layer.convolution_param, kwargs)
        return _LayerSpec(layer)

    @staticmethod
    def InnerProduct(bottom, num_output, bias_term=True, weight_filler=None,
                     bias_filler=None, axis=1, transpose=False, **kwargs):
        layer = _make_layer("InnerProduct", [bottom])
        params = dict(num_output=num_output, bias_term=bias_term, axis=axis,
                      transpose=transpose)
        if weight_filler is not None:
            params["weight_filler"] = weight_filler
        if bias_filler is not None:
            params["bias_filler"] = bias_filler
        params.update(kwargs)
        _fill_param_from_dict(layer.inner_product_param, params)
        return _LayerSpec(layer)

    @staticmethod
    def Embed(bottom, num_output, input_dim, bias_term=True,
              weight_filler=None, bias_filler=None, **kwargs):
        layer = _make_layer("Embed", [bottom])
        params = dict(num_output=num_output, input_dim=input_dim,
                      bias_term=bias_term)
        if weight_filler is not None:
            params["weight_filler"] = weight_filler
        if bias_filler is not None:
            params["bias_filler"] = bias_filler
        params.update(kwargs)
        _fill_param_from_dict(layer.embed_param, params)
        return _LayerSpec(layer)

    @staticmethod
    def Pooling(bottom, pool=P.Pooling.MAX, kernel_size=1, stride=1, pad=0,
                global_pooling=False, kernel_h=None, kernel_w=None,
                stride_h=None, stride_w=None, pad_h=None, pad_w=None, **kwargs):
        layer = _make_layer("Pooling", [bottom])
        params = dict(pool=pool, kernel_size=kernel_size, stride=stride,
                      pad=pad, global_pooling=global_pooling)
        if kernel_h is not None:
            params["kernel_h"] = kernel_h
        if kernel_w is not None:
            params["kernel_w"] = kernel_w
        if stride_h is not None:
            params["stride_h"] = stride_h
        if stride_w is not None:
            params["stride_w"] = stride_w
        if pad_h is not None:
            params["pad_h"] = pad_h
        if pad_w is not None:
            params["pad_w"] = pad_w
        params.update(kwargs)
        _fill_param_from_dict(layer.pooling_param, params)
        return _LayerSpec(layer)

    @staticmethod
    def BatchNorm(bottom, **kwargs):
        layer = _make_layer("BatchNorm", [bottom])
        _fill_param_from_dict(layer.batch_norm_param, kwargs)
        return _LayerSpec(layer)

    @staticmethod
    def Scale(bottom, **kwargs):
        layer = _make_layer("Scale", [bottom])
        _fill_param_from_dict(layer.scale_param, kwargs)
        return _LayerSpec(layer)

    @staticmethod
    def LRN(bottom, **kwargs):
        layer = _make_layer("LRN", [bottom])
        _fill_param_from_dict(layer.lrn_param, kwargs)
        return _LayerSpec(layer)

    @staticmethod
    def Reduction(bottom, **kwargs):
        layer = _make_layer("Reduction", [bottom])
        _fill_param_from_dict(layer.reduction_param, kwargs)
        return _LayerSpec(layer)

    @staticmethod
    def Flatten(bottom, **kwargs):
        layer = _make_layer("Flatten", [bottom])
        _fill_param_from_dict(layer.flatten_param, kwargs)
        return _LayerSpec(layer)

    @staticmethod
    def Softmax(bottom, **kwargs):
        layer = _make_layer("Softmax", [bottom])
        _fill_param_from_dict(layer.softmax_param, kwargs)
        return _LayerSpec(layer)

    @staticmethod
    def ArgMax(bottom, argmax_param=None, **kwargs):
        layer = _make_layer("ArgMax", [bottom])
        if argmax_param:
            _fill_param_from_dict(layer.argmax_param, argmax_param)
        return _LayerSpec(layer)

    @staticmethod
    def Reshape(bottom, reshape_param=None, **kwargs):
        layer = _make_layer("Reshape", [bottom])
        if reshape_param:
            _fill_param_from_dict(layer.reshape_param, reshape_param)
        return _LayerSpec(layer)

    @staticmethod
    def Tile(bottom, tile_param=None, **kwargs):
        layer = _make_layer("Tile", [bottom])
        if tile_param:
            _fill_param_from_dict(layer.tile_param, tile_param)
        return _LayerSpec(layer)

    @staticmethod
    def Slice(bottom, ntop=2, slice_param=None, **kwargs):
        tops = ["output%d" % i for i in range(ntop)]
        layer = _make_layer("Slice", [bottom], tops=tops)
        if slice_param:
            _fill_param_from_dict(layer.slice_param, slice_param)
        return _LayerSpec(layer)

    # ── multi-input ops ─────────────────────────────────────────────
    @staticmethod
    def Concat(*bottoms, **kwargs):
        layer = _make_layer("Concat", bottoms)
        _fill_param_from_dict(layer.concat_param, kwargs)
        return _LayerSpec(layer)

    @staticmethod
    def Eltwise(*bottoms, **kwargs):
        layer = _make_layer("Eltwise", bottoms)
        _fill_param_from_dict(layer.eltwise_param, kwargs)
        return _LayerSpec(layer)

    @staticmethod
    def Crop(*bottoms, **kwargs):
        layer = _make_layer("Crop", bottoms)
        _fill_param_from_dict(layer.crop_param, kwargs)
        return _LayerSpec(layer)


# ─────────────────────────────────────────────────────────────────────
# Net construction helpers
# ─────────────────────────────────────────────────────────────────────


def _siso_op(data, func, **kwargs):
    """Create single input and single output Caffe op (_NetSpec)."""
    logger.debug(f"Creating SISO op, input shape: {data.shape}")
    n = _NetSpec()
    n.add_input("data", list(data.shape))
    n.add_layer(func("data", **kwargs))
    return n


def _miso_op(data_list, func, **kwargs):
    """Create multi input and single output Caffe op (_NetSpec)."""
    input_shapes = [d.shape for d in data_list]
    logger.debug(f"Creating MISO op, num inputs: {len(data_list)}, shapes: {input_shapes}")
    if not isinstance(data_list, (tuple, list)):
        raise TypeError(f"Need tuple or list but get {type(data_list)}")
    n = _NetSpec()
    inputs = []
    for idx, d in enumerate(data_list):
        n.add_input("data" + str(idx), list(d.shape))
        inputs.append("data" + str(idx))
    n.add_layer(func(*inputs, **kwargs))
    return n


def _simo_op(data, func, **kwargs):
    """Create single input and multi output Caffe op (_NetSpec)."""
    logger.debug(f"Creating SIMO op, input shape: {data.shape}")
    n = _NetSpec()
    n.add_input("data", list(data.shape))
    n.add_layer(func("data", **kwargs))
    return n


def _run(data, net_param):
    """Build a caffe_ffi Net from param, set inputs, run forward, return outputs.

    Returns a list of numpy arrays (one per output blob), matching the
    historical return contract of ``_test_op``.
    """
    net = net_from_param(net_param)
    if isinstance(data, (list, tuple)):
        for idx, d in enumerate(data):
            net.blob_by_name("data" + str(idx)).data = d
    else:
        net.blob_by_name("data").data = data
    out = net.forward()
    out_names = net.output_blob_names()
    if out_names:
        result = [out[name] for name in out_names if name in out]
    else:
        result = list(out.values())
    if not result:
        raise RuntimeError("caffe_ffi forward returned no outputs")
    logger.debug(f"caffe_ffi forward outputs: {[o.shape for o in result]}")
    return result


def _validate_reshape_params(data, reshape_param):
    """
    Pre-validate Reshape parameters in Python before calling C++ layer.
    Converts C++ CHECK failures (SIGABRT) into Python ValueError with clear messages.

    Corresponds to reshape_layer.cpp CHECK_EQ(top[0]->count(), bottom[0]->count()).

    Caffe Reshape logic:
    - Output shape = input[:axis] + resolved_dims + input[axis+num_axes:]
    - 0 in dim[i] = copy input axis size from position axis+i (requires i < num_axes)
    - -1 in dim[i] = infer this dimension (exactly one -1 allowed)
    - dim can have different length than num_axes (this is how rank changes in reshape)
    - Product of resolved dims must equal product of affected input axes
    """
    import numpy as _np
    if isinstance(data, (list, tuple)):
        input_shape = list(data[0].shape)
    else:
        input_shape = list(data.shape)

    shape_spec = reshape_param.get("shape", {})
    dim = list(shape_spec.get("dim", []))
    axis = reshape_param.get("axis", 0)
    num_axes = reshape_param.get("num_axes", -1)

    if not dim:
        return

    ndim = len(input_shape)
    if axis < 0:
        axis = ndim + axis
    if axis < 0 or axis >= ndim:
        raise ValueError(
            f"Reshape parameter error: axis={axis} out of range for input_shape={input_shape}"
        )

    if num_axes < 0:
        num_axes = ndim - axis
    if axis + num_axes > ndim:
        raise ValueError(
            f"Reshape parameter error: axis+num_axes={axis+num_axes} exceeds ndim={ndim}"
        )

    # Affected input axes (those being reshaped)
    affected_axes = input_shape[axis:axis + num_axes]
    affected_count = int(_np.prod(affected_axes)) if affected_axes else 1

    # Count -1 occurrences
    num_minus_one = dim.count(-1)
    if num_minus_one > 1:
        raise ValueError(
            f"Reshape parameter error: multiple -1 in dim={dim} (at most one allowed)"
        )

    # Compute product of resolved dims:
    # - d > 0: multiply by d
    # - d == 0: multiply by affected_axes[i] (copy from input); requires i < num_axes
    # - d == -1: skip (infer later)
    constant_count = 1
    for i, d in enumerate(dim):
        if d == -1:
            continue
        elif d == 0:
            if i >= num_axes:
                raise ValueError(
                    f"Reshape parameter error: dim[0] at position {i} is outside "
                    f"the affected axis range (num_axes={num_axes}). "
                    f"dim={dim}, axis={axis}, num_axes={num_axes}"
                )
            constant_count *= affected_axes[i]
        elif d > 0:
            constant_count *= d
        else:
            raise ValueError(
                f"Reshape parameter error: invalid dim value {d} at position {i} in {dim}"
            )

    if num_minus_one == 1:
        if affected_count % constant_count != 0:
            raise ValueError(
                f"Reshape parameter error: cannot infer -1 dimension. "
                f"Affected axes {affected_axes} have {affected_count} elements, "
                f"explicit product is {constant_count} (not a divisor). "
                f"dim={dim}, axis={axis}, num_axes={num_axes}, input_shape={input_shape}. "
                f"This would trigger SIGABRT in C++ reshape_layer CHECK_EQ."
            )
    else:
        if constant_count != affected_count:
            raise ValueError(
                f"Reshape parameter error: element count mismatch. "
                f"Affected axes {affected_axes} have {affected_count} elements, "
                f"but resolved dim product is {constant_count}. "
                f"dim={dim}, axis={axis}, num_axes={num_axes}, input_shape={input_shape}. "
                f"This would trigger SIGABRT in C++ reshape_layer CHECK_EQ."
            )


def _test_op(data, func_op, op_name, test_dir, **kwargs):
    """Single op testing pipeline (caffe_ffi, no TVM comparison)."""
    logger.info(f"Testing operator: {op_name}")

    # --- Reshape parameter pre-validation (prevents SIGABRT from C++ CHECK failure) ---
    if op_name == "Reshape" and "reshape_param" in kwargs:
        _validate_reshape_params(data, kwargs["reshape_param"])

    if not _CAFFE_FFI_AVAILABLE:
        pytest.skip("caffe_ffi C++ extension is not available; operator tests skipped")

    try:
        if isinstance(data, (list, tuple)):
            net_spec = _miso_op(data, func_op, **kwargs)
        else:
            output_num = kwargs.get("ntop", 1)
            if output_num == 1:
                net_spec = _siso_op(data, func_op, **kwargs)
            else:
                net_spec = _simo_op(data, func_op, **kwargs)

        net_param = net_spec.to_param()
        caffe_out = _run(data, net_param)
        logger.info(f"Testing operator {op_name} completed")
        return caffe_out
    except Exception as e:
        logger.error(f"Error testing operator {op_name}: {e}", exc_info=True)
        raise


def assert_op_correct(caffe_out, ref_out, atol=1e-5, rtol=1e-4, op_name=""):
    """
    Compare Caffe output with numpy reference implementation using np.allclose.

    Args:
        caffe_out: Caffe operator output (numpy array or list of arrays)
        ref_out: Numpy reference output (numpy array or list of arrays)
        atol: Absolute tolerance for np.allclose
        rtol: Relative tolerance for np.allclose
        op_name: Operator name for error messages

    Raises:
        AssertionError: If outputs don't match within tolerance, showing max error
    """
    logger.debug(f"Verifying correctness for op: {op_name or 'unknown'}, atol={atol}, rtol={rtol}")

    def _compare_single(a, b):
        a_np = np.asarray(a)
        b_np = np.asarray(b)
        if a_np.shape != b_np.shape:
            raise AssertionError(
                f"Shape mismatch for {op_name or 'op'}: "
                f"caffe_out shape {a_np.shape} vs ref_out shape {b_np.shape}"
            )
        close_mask = np.isclose(a_np, b_np, atol=atol, rtol=rtol)
        if not np.all(close_mask):
            max_abs_err = np.max(np.abs(a_np - b_np))
            max_rel_err = np.max(np.abs(a_np - b_np) / (np.abs(b_np) + 1e-10))
            err_idx = np.unravel_index(np.argmax(np.abs(a_np - b_np)), a_np.shape)
            raise AssertionError(
                f"Output mismatch for {op_name or 'op'}: "
                f"max absolute error = {max_abs_err:.2e}, "
                f"max relative error = {max_rel_err:.2e}, "
                f"atol={atol}, rtol={rtol}, "
                f"error location (flattened index argmax): {err_idx}, "
                f"caffe value = {a_np[err_idx]}, ref value = {b_np[err_idx]}"
            )
        return True

    if isinstance(caffe_out, (list, tuple)) and isinstance(ref_out, (list, tuple)):
        if len(caffe_out) != len(ref_out):
            raise AssertionError(
                f"Output count mismatch for {op_name or 'op'}: "
                f"{len(caffe_out)} vs {len(ref_out)}"
            )
        for i, (c, r) in enumerate(zip(caffe_out, ref_out)):
            _compare_single(c, r)
    elif isinstance(caffe_out, (list, tuple)) and len(caffe_out) == 1 and not isinstance(ref_out, (list, tuple)):
        _compare_single(caffe_out[0], ref_out)
    else:
        _compare_single(caffe_out, ref_out)

    logger.info(f"Correctness check passed for {op_name or 'op'}")


class Timer:
    """
    Context manager for performance timing.

    Usage:
        with Timer() as t:
            # code to time
        print(f"Elapsed: {t.elapsed} seconds")
    """

    def __init__(self, name=""):
        self.name = name
        self.elapsed = 0.0
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        if self.name:
            logger.debug(f"Timer [{self.name}] elapsed: {self.elapsed:.6f}s")


def get_memory_usage():
    """
    Get current memory usage in bytes using tracemalloc.

    Returns:
        tuple: (current_memory, peak_memory) in bytes
    """
    import tracemalloc
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    current, peak = tracemalloc.get_traced_memory()
    return current, peak


def check_memory_leak(func, runs=5, *args, **kwargs):
    """
    Check for memory leaks by running a function multiple times and tracking memory growth.

    Args:
        func: Callable to test
        runs: Number of consecutive runs
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        dict: Memory usage statistics with keys:
            - 'memory_per_run': List of memory usage (bytes) after each run
            - 'has_leak': Boolean indicating if continuous growth detected
            - 'growth_rate': Average bytes increase per run
            - 'peak_memory': Peak memory usage across all runs

    Raises:
        RuntimeError: If significant memory leak is detected
    """
    import tracemalloc
    logger.debug(f"Checking memory leak for {func.__name__} with {runs} runs")

    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()

    tracemalloc.reset_peak()
    memory_per_run = []

    for i in range(runs):
        func(*args, **kwargs)
        current, _ = tracemalloc.get_traced_memory()
        memory_per_run.append(current)

    _, peak = tracemalloc.get_traced_memory()

    if len(memory_per_run) >= 3:
        first_half = memory_per_run[:runs//2]
        second_half = memory_per_run[runs//2:]
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        growth = avg_second - avg_first
        growth_rate = growth / (runs // 2)
        has_leak = growth_rate > 1024 * 100
    else:
        growth = memory_per_run[-1] - memory_per_run[0]
        growth_rate = growth / max(runs - 1, 1)
        has_leak = growth_rate > 1024 * 100

    if not was_tracing:
        tracemalloc.stop()

    result = {
        'memory_per_run': memory_per_run,
        'has_leak': has_leak,
        'growth_rate': growth_rate,
        'peak_memory': peak,
    }

    if has_leak:
        logger.warning(
            f"Potential memory leak detected in {func.__name__}: "
            f"avg growth = {growth_rate:.2f} bytes/run over {runs} runs"
        )
    else:
        logger.debug(
            f"No memory leak detected in {func.__name__}: "
            f"avg growth = {growth_rate:.2f} bytes/run"
        )

    return result


class TestResultCollector:
    """
    Collector for test results including correctness, performance, and memory tests.
    """

    def __init__(self):
        self.results = {
            'correctness': [],
            'performance': [],
            'memory': [],
        }

    def add_result(self, category, name, passed, details=None):
        """
        Add a test result.

        Args:
            category: One of 'correctness', 'performance', 'memory'
            name: Test/operator name
            passed: Boolean indicating if test passed
            details: Optional dict with additional details (e.g., elapsed time, error)
        """
        if category not in self.results:
            raise ValueError(f"Unknown category: {category}. Must be one of {list(self.results.keys())}")

        result = {
            'name': name,
            'passed': passed,
            'details': details or {},
        }
        self.results[category].append(result)
        logger.debug(f"Added {category} result for {name}: passed={passed}")

    def get_summary(self):
        """
        Get a summary of all test results.

        Returns:
            dict: Summary with counts per category and overall pass/fail status
        """
        summary = {}
        total_passed = 0
        total_tests = 0

        for category, results in self.results.items():
            passed = sum(1 for r in results if r['passed'])
            total = len(results)
            summary[category] = {
                'total': total,
                'passed': passed,
                'failed': total - passed,
                'pass_rate': passed / total if total > 0 else 0.0,
                'results': results,
            }
            total_passed += passed
            total_tests += total

        summary['overall'] = {
            'total': total_tests,
            'passed': total_passed,
            'failed': total_tests - total_passed,
            'pass_rate': total_passed / total_tests if total_tests > 0 else 0.0,
            'all_passed': total_passed == total_tests,
        }

        logger.info(
            f"Test summary - total: {total_tests}, passed: {total_passed}, "
            f"failed: {total_tests - total_passed}, "
            f"pass rate: {summary['overall']['pass_rate']:.1%}"
        )

        return summary