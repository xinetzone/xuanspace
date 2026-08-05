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

import logging
import numpy as np
import pytest
from utils import L, _test_op, assert_op_correct

logger = logging.getLogger(__name__)


def _test_reshape(data, test_dir, **kwargs):
    """One iteration of Reshape."""
    logger.info(f"Testing Reshape, input shape: {data.shape}")
    logger.debug(f"Reshape params: {kwargs}")
    return _test_op(data, L.Reshape, "Reshape", test_dir, **kwargs)


@pytest.mark.correctness
@pytest.mark.edge
def test_forward_Reshape(caffe_test_dir):
    """Reshape"""
    logger.info("Running test_forward_Reshape")
    data = np.random.rand(1, 8, 6).astype(np.float32)
    logger.debug(f"Testing Reshape to [4,3,4], shape: {data.shape}")
    _test_reshape(data, caffe_test_dir, reshape_param={"shape": {"dim": [4, 3, 4]}})
    logger.debug(f"Testing Reshape to [2,0,3] (infer), shape: {data.shape}")
    _test_reshape(data, caffe_test_dir, reshape_param={"shape": {"dim": [2, 0, 3]}})
    logger.debug(f"Testing Reshape to [2,0,-1] (flatten), shape: {data.shape}")
    _test_reshape(data, caffe_test_dir, reshape_param={"shape": {"dim": [2, 0, -1]}})
    logger.debug(f"Testing Reshape to [0,-1], shape: {data.shape}")
    _test_reshape(data, caffe_test_dir, reshape_param={"shape": {"dim": [0, -1]}})

    logger.debug(f"Testing Reshape with axis=2, shape: {data.shape}")
    _test_reshape(data, caffe_test_dir, reshape_param={"shape": {"dim": [2, 3]}, "axis": 2})
    logger.debug(f"Testing Reshape with axis=1, shape: {data.shape}")
    _test_reshape(data, caffe_test_dir, reshape_param={"shape": {"dim": [4, 3, 4]}, "axis": 1})
    logger.debug(f"Testing Reshape with axis=-3, shape: {data.shape}")
    _test_reshape(data, caffe_test_dir, reshape_param={"shape": {"dim": [4, 3, 4]}, "axis": -3})

    logger.debug(f"Testing Reshape with axis=1, num_axes=1, shape: {data.shape}")
    _test_reshape(data, caffe_test_dir, reshape_param={"shape": {"dim": [2, 4]}, "axis": 1, "num_axes": 1})
    logger.debug(f"Testing Reshape with axis=1, num_axes=2, shape: {data.shape}")
    _test_reshape(data, caffe_test_dir, reshape_param={"shape": {"dim": [3, 16]}, "axis": 1, "num_axes": 2})


@pytest.mark.correctness
def test_reshape_correctness(caffe_test_dir):
    """Reshape correctness test for simple explicit dimensions."""
    logger.info("Running test_reshape_correctness")
    np.random.seed(42)

    logger.debug("Testing simple 4D->2D reshape")
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    target_shape = [2, 60]
    ref = x.reshape(target_shape)
    caffe_out = _test_reshape(x, caffe_test_dir, reshape_param={"shape": {"dim": target_shape}})
    assert_op_correct(caffe_out, ref, op_name="Reshape(4D->2D)")

    logger.debug("Testing simple 2D->4D reshape")
    x = np.random.randn(6, 20).astype(np.float32)
    target_shape = [2, 3, 4, 5]
    ref = x.reshape(target_shape)
    caffe_out = _test_reshape(x, caffe_test_dir, reshape_param={"shape": {"dim": target_shape}})
    assert_op_correct(caffe_out, ref, op_name="Reshape(2D->4D)")

    logger.debug("Testing 3D reshape with -1 inference")
    x = np.random.randn(2, 3, 12).astype(np.float32)
    target_shape = [2, -1, 6]
    ref = x.reshape([2, 6, 6])
    caffe_out = _test_reshape(x, caffe_test_dir, reshape_param={"shape": {"dim": target_shape}})
    assert_op_correct(caffe_out, ref, op_name="Reshape(3D-infer)")

    logger.debug("Testing flatten to 2D [0,-1]")
    x = np.random.randn(4, 5, 6).astype(np.float32)
    ref = x.reshape([x.shape[0], -1])
    caffe_out = _test_reshape(x, caffe_test_dir, reshape_param={"shape": {"dim": [0, -1]}})
    assert_op_correct(caffe_out, ref, op_name="Reshape(flatten-0,-1)")

    logger.debug("Testing identity reshape (same dimensions)")
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    target_shape = list(x.shape)
    ref = x.reshape(target_shape)
    caffe_out = _test_reshape(x, caffe_test_dir, reshape_param={"shape": {"dim": target_shape}})
    assert_op_correct(caffe_out, ref, op_name="Reshape(identity)")


@pytest.mark.edge
def test_reshape_edge_cases(caffe_test_dir):
    """Reshape edge cases - verify runs without crash and output shape."""
    logger.info("Running test_reshape_edge_cases")

    logger.debug("Testing all zeros input")
    x = np.zeros((2, 3, 4, 5), dtype=np.float32)
    caffe_out = _test_reshape(x, caffe_test_dir, reshape_param={"shape": {"dim": [2, 60]}})
    assert caffe_out[0].shape == (2, 60), f"Expected (2,60) got {caffe_out[0].shape}"
    assert np.all(caffe_out[0] == 0)

    logger.debug("Testing all ones input")
    x = np.ones((3, 4, 5), dtype=np.float32)
    target_shape = [3, 20]
    caffe_out = _test_reshape(x, caffe_test_dir, reshape_param={"shape": {"dim": target_shape}})
    assert caffe_out[0].shape == tuple(target_shape)
    assert np.all(caffe_out[0] == 1.0)

    logger.debug("Testing single element reshape")
    x = np.array([[[[42.0]]]], dtype=np.float32)
    caffe_out = _test_reshape(x, caffe_test_dir, reshape_param={"shape": {"dim": [1]}})
    assert caffe_out[0].shape == (1,)
    assert np.allclose(caffe_out[0], [42.0])

    logger.debug("Testing 1D input reshape")
    x = np.arange(24, dtype=np.float32)
    caffe_out = _test_reshape(x, caffe_test_dir, reshape_param={"shape": {"dim": [2, 3, 4]}})
    assert caffe_out[0].shape == (2, 3, 4)
    assert_op_correct(caffe_out, x.reshape(2, 3, 4), op_name="Reshape(1D->3D)")
