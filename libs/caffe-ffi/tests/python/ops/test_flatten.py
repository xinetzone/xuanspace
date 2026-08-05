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


def _test_flatten(data, test_dir, axis=1):
    """One iteration of Flatten"""
    logger.info(f"Testing Flatten, input shape: {data.shape}")
    logger.debug(f"Flatten params - axis: {axis}")
    return _test_op(data, L.Flatten, "Flatten", test_dir, axis=axis)


def test_forward_Flatten(caffe_test_dir):
    """Flatten"""
    logger.info("Running test_forward_Flatten")
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    logger.debug(f"Calling _test_flatten with data shape {data.shape}, default axis=1")
    _test_flatten(data, caffe_test_dir)
    logger.debug(f"Calling _test_flatten with data shape {data.shape}, axis=1")
    _test_flatten(data, caffe_test_dir, axis=1)


def _flatten_ref(x, axis=1):
    if axis == 0:
        return x.reshape(-1)
    shape = list(x.shape[:axis]) + [-1]
    return x.reshape(shape).astype(np.float32)


@pytest.mark.correctness
def test_flatten_correctness(caffe_test_dir):
    """Flatten correctness test with numpy reference."""
    logger.info("Running test_flatten_correctness")
    np.random.seed(42)

    logger.debug("Testing Flatten 4D default axis=1")
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    ref = _flatten_ref(x, axis=1)
    caffe_out = _test_flatten(x, caffe_test_dir, axis=1)
    assert_op_correct(caffe_out, ref, op_name="Flatten(4D-axis1)")
    assert caffe_out[0].shape == (2, 60)

    logger.debug("Testing Flatten 3D axis=1")
    x = np.random.randn(4, 5, 6).astype(np.float32)
    ref = _flatten_ref(x, axis=1)
    caffe_out = _test_flatten(x, caffe_test_dir, axis=1)
    assert_op_correct(caffe_out, ref, op_name="Flatten(3D-axis1)")
    assert caffe_out[0].shape == (4, 30)

    logger.debug("Testing Flatten 4D axis=2")
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    ref = _flatten_ref(x, axis=2)
    caffe_out = _test_flatten(x, caffe_test_dir, axis=2)
    assert_op_correct(caffe_out, ref, op_name="Flatten(4D-axis2)")
    assert caffe_out[0].shape == (2, 3, 20)

    logger.debug("Testing Flatten 2D identity")
    x = np.random.randn(5, 10).astype(np.float32)
    ref = _flatten_ref(x, axis=1)
    caffe_out = _test_flatten(x, caffe_test_dir, axis=1)
    assert_op_correct(caffe_out, ref, op_name="Flatten(2D-axis1)")
    assert caffe_out[0].shape == (5, 10)


@pytest.mark.edge
def test_flatten_edge_cases(caffe_test_dir):
    """Flatten edge cases."""
    logger.info("Running test_flatten_edge_cases")

    logger.debug("Testing all zeros input")
    x = np.zeros((2, 3, 4, 5), dtype=np.float32)
    caffe_out = _test_flatten(x, caffe_test_dir, axis=1)
    assert caffe_out[0].shape == (2, 60)
    assert np.all(caffe_out[0] == 0)

    logger.debug("Testing all ones input")
    x = np.ones((1, 5, 2, 2), dtype=np.float32)
    caffe_out = _test_flatten(x, caffe_test_dir, axis=1)
    assert caffe_out[0].shape == (1, 20)
    assert np.all(caffe_out[0] == 1.0)

    logger.debug("Testing batch=1, single feature map")
    x = np.random.randn(1, 1, 1, 10).astype(np.float32)
    caffe_out = _test_flatten(x, caffe_test_dir, axis=1)
    assert caffe_out[0].shape == (1, 10)

    logger.debug("Testing large batch")
    x = np.random.randn(32, 3, 8, 8).astype(np.float32)
    caffe_out = _test_flatten(x, caffe_test_dir, axis=1)
    assert caffe_out[0].shape == (32, 192)
