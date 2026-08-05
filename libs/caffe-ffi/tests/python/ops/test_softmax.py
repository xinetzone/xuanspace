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


def _test_softmax(data, test_dir, **kwargs):
    """One iteration of Softmax"""
    logger.info(f"Testing Softmax, input shape: {data.shape}")
    logger.debug(f"Softmax params: {kwargs}")
    return _test_op(data, L.Softmax, "Softmax", test_dir, **kwargs)


def test_forward_Softmax(caffe_test_dir):
    """Softmax"""
    logger.info("Running test_forward_Softmax")
    logger.debug("Testing Softmax 4D, default axis=1")
    _test_softmax(np.random.rand(1, 3, 10, 10).astype(np.float32), caffe_test_dir)
    logger.debug("Testing Softmax 4D, axis=2")
    _test_softmax(np.random.rand(1, 3, 10, 10).astype(np.float32), caffe_test_dir, axis=2)
    logger.debug("Testing Softmax 2D, axis=0")
    _test_softmax(np.random.rand(10, 10).astype(np.float32), caffe_test_dir, axis=0)
    logger.debug("Testing Softmax 3D, axis=1")
    _test_softmax(np.random.rand(2, 10, 10).astype(np.float32), caffe_test_dir, axis=1)


def _softmax_ref(x, axis=1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return (exp_x / np.sum(exp_x, axis=axis, keepdims=True)).astype(np.float32)


@pytest.mark.correctness
def test_softmax_correctness(caffe_test_dir):
    """Softmax correctness test with numpy reference."""
    logger.info("Running test_softmax_correctness")
    np.random.seed(42)

    logger.debug("Testing Softmax 4D, axis=1 (default)")
    x = np.random.randn(2, 5, 6, 6).astype(np.float32)
    ref = _softmax_ref(x, axis=1)
    caffe_out = _test_softmax(x, caffe_test_dir, axis=1)
    assert_op_correct(caffe_out, ref, op_name="Softmax(4D-axis1)")

    logger.debug("Testing Softmax 4D, axis=2")
    x = np.random.randn(1, 3, 8, 8).astype(np.float32)
    ref = _softmax_ref(x, axis=2)
    caffe_out = _test_softmax(x, caffe_test_dir, axis=2)
    assert_op_correct(caffe_out, ref, op_name="Softmax(4D-axis2)")

    logger.debug("Testing Softmax 4D, axis=3")
    x = np.random.randn(1, 3, 4, 6).astype(np.float32)
    ref = _softmax_ref(x, axis=3)
    caffe_out = _test_softmax(x, caffe_test_dir, axis=3)
    assert_op_correct(caffe_out, ref, op_name="Softmax(4D-axis3)")

    logger.debug("Testing Softmax 2D, axis=0")
    x = np.random.randn(5, 4).astype(np.float32)
    ref = _softmax_ref(x, axis=0)
    caffe_out = _test_softmax(x, caffe_test_dir, axis=0)
    assert_op_correct(caffe_out, ref, op_name="Softmax(2D-axis0)")

    logger.debug("Testing Softmax 2D, axis=1")
    x = np.random.randn(5, 4).astype(np.float32)
    ref = _softmax_ref(x, axis=1)
    caffe_out = _test_softmax(x, caffe_test_dir, axis=1)
    assert_op_correct(caffe_out, ref, op_name="Softmax(2D-axis1)")

    logger.debug("Testing Softmax 3D, axis=1")
    x = np.random.randn(2, 4, 3).astype(np.float32)
    ref = _softmax_ref(x, axis=1)
    caffe_out = _test_softmax(x, caffe_test_dir, axis=1)
    assert_op_correct(caffe_out, ref, op_name="Softmax(3D-axis1)")


@pytest.mark.edge
def test_softmax_edge_cases(caffe_test_dir):
    """Softmax edge cases."""
    logger.info("Running test_softmax_edge_cases")

    logger.debug("Testing uniform input (all equal values)")
    x = np.full((1, 5, 4, 4), 3.0, dtype=np.float32)
    ref = _softmax_ref(x, axis=1)
    caffe_out = _test_softmax(x, caffe_test_dir, axis=1)
    assert_op_correct(caffe_out, ref, op_name="Softmax(uniform)")
    np.testing.assert_allclose(caffe_out[0], ref, atol=1e-5)
    expected_val = 1.0 / x.shape[1]
    np.testing.assert_allclose(caffe_out[0], np.full_like(x, expected_val, dtype=np.float32), atol=1e-5)

    logger.debug("Testing one-hot-like large differences")
    x = np.zeros((1, 5, 3, 3), dtype=np.float32)
    x[:, 0, :, :] = 100.0
    ref = _softmax_ref(x, axis=1)
    caffe_out = _test_softmax(x, caffe_test_dir, axis=1)
    assert_op_correct(caffe_out, ref, op_name="Softmax(one-hot-large)")
    np.testing.assert_allclose(caffe_out[0][:, 0, :, :], np.ones((1, 3, 3), dtype=np.float32), atol=1e-5)

    logger.debug("Testing numerical stability with large values")
    x = np.full((1, 3, 4, 4), 1000.0, dtype=np.float32)
    x[:, 0, :, :] = 1001.0
    ref = _softmax_ref(x, axis=1)
    caffe_out = _test_softmax(x, caffe_test_dir, axis=1)
    assert_op_correct(caffe_out, ref, op_name="Softmax(large-values)")

    logger.debug("Testing all zeros input")
    x = np.zeros((1, 3, 4, 4), dtype=np.float32)
    ref = _softmax_ref(x, axis=1)
    caffe_out = _test_softmax(x, caffe_test_dir, axis=1)
    assert_op_correct(caffe_out, ref, op_name="Softmax(zeros)")

    logger.debug("Testing negative large values")
    x = np.full((1, 3, 4, 4), -1000.0, dtype=np.float32)
    x[:, 1, :, :] = -999.0
    ref = _softmax_ref(x, axis=1)
    caffe_out = _test_softmax(x, caffe_test_dir, axis=1)
    assert_op_correct(caffe_out, ref, op_name="Softmax(negative-large)")
