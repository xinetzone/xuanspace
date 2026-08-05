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


def _test_concat(data_list, test_dir, axis=1):
    """One iteration of Concat"""
    input_shapes = [d.shape for d in data_list]
    logger.info(f"Testing Concat, num inputs: {len(data_list)}, shapes: {input_shapes}")
    logger.debug(f"Concat params - axis: {axis}")
    return _test_op(data_list, L.Concat, "Concat", test_dir, axis=axis)


def test_forward_Concat(caffe_test_dir):
    """Concat"""
    logger.info("Running test_forward_Concat")
    logger.debug("Testing Concat 4D, axis=1")
    _test_concat([np.random.rand(1, 3, 10, 10), np.random.rand(1, 2, 10, 10)], caffe_test_dir, axis=1)
    logger.debug("Testing Concat 3D, axis=0")
    _test_concat([np.random.rand(3, 10, 10), np.random.rand(2, 10, 10)], caffe_test_dir, axis=0)
    logger.debug("Testing Concat 2D, axis=0")
    _test_concat([np.random.rand(3, 10), np.random.rand(2, 10)], caffe_test_dir, axis=0)


@pytest.mark.correctness
def test_concat_correctness(caffe_test_dir):
    """Concat correctness test with numpy concatenate reference."""
    logger.info("Running test_concat_correctness")
    np.random.seed(42)

    logger.debug("Testing Concat 4D axis=1 (channel)")
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    b = np.random.randn(2, 4, 4, 5).astype(np.float32)
    ref = np.concatenate([a, b], axis=1).astype(np.float32)
    caffe_out = _test_concat([a, b], caffe_test_dir, axis=1)
    assert_op_correct(caffe_out, ref, op_name="Concat(4D-axis1)")
    assert caffe_out[0].shape == (2, 7, 4, 5)

    logger.debug("Testing Concat 4D axis=0 (batch)")
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    b = np.random.randn(3, 3, 4, 5).astype(np.float32)
    ref = np.concatenate([a, b], axis=0).astype(np.float32)
    caffe_out = _test_concat([a, b], caffe_test_dir, axis=0)
    assert_op_correct(caffe_out, ref, op_name="Concat(4D-axis0)")
    assert caffe_out[0].shape == (5, 3, 4, 5)

    logger.debug("Testing Concat 4D axis=3 (width)")
    a = np.random.randn(1, 2, 3, 4).astype(np.float32)
    b = np.random.randn(1, 2, 3, 5).astype(np.float32)
    ref = np.concatenate([a, b], axis=3).astype(np.float32)
    caffe_out = _test_concat([a, b], caffe_test_dir, axis=3)
    assert_op_correct(caffe_out, ref, op_name="Concat(4D-axis3)")
    assert caffe_out[0].shape == (1, 2, 3, 9)

    logger.debug("Testing Concat 3 inputs axis=1")
    a = np.random.randn(1, 2, 3, 3).astype(np.float32)
    b = np.random.randn(1, 3, 3, 3).astype(np.float32)
    c = np.random.randn(1, 4, 3, 3).astype(np.float32)
    ref = np.concatenate([a, b, c], axis=1).astype(np.float32)
    caffe_out = _test_concat([a, b, c], caffe_test_dir, axis=1)
    assert_op_correct(caffe_out, ref, op_name="Concat(3in-axis1)")
    assert caffe_out[0].shape == (1, 9, 3, 3)

    logger.debug("Testing Concat 2D axis=0")
    a = np.random.randn(3, 5).astype(np.float32)
    b = np.random.randn(4, 5).astype(np.float32)
    ref = np.concatenate([a, b], axis=0).astype(np.float32)
    caffe_out = _test_concat([a, b], caffe_test_dir, axis=0)
    assert_op_correct(caffe_out, ref, op_name="Concat(2D-axis0)")

    logger.debug("Testing Concat 2D axis=1")
    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(3, 6).astype(np.float32)
    ref = np.concatenate([a, b], axis=1).astype(np.float32)
    caffe_out = _test_concat([a, b], caffe_test_dir, axis=1)
    assert_op_correct(caffe_out, ref, op_name="Concat(2D-axis1)")


@pytest.mark.edge
def test_concat_edge_cases(caffe_test_dir):
    """Concat edge cases."""
    logger.info("Running test_concat_edge_cases")

    logger.debug("Testing all zeros concat")
    a = np.zeros((2, 3, 4, 4), dtype=np.float32)
    b = np.zeros((2, 2, 4, 4), dtype=np.float32)
    caffe_out = _test_concat([a, b], caffe_test_dir, axis=1)
    assert caffe_out[0].shape == (2, 5, 4, 4)
    assert np.all(caffe_out[0] == 0)

    logger.debug("Testing all ones concat")
    a = np.ones((1, 2, 3, 3), dtype=np.float32)
    b = np.ones((1, 3, 3, 3), dtype=np.float32)
    caffe_out = _test_concat([a, b], caffe_test_dir, axis=1)
    assert caffe_out[0].shape == (1, 5, 3, 3)
    assert np.all(caffe_out[0] == 1.0)

    logger.debug("Testing concat single size dimension (1 element)")
    a = np.array([[[[1.0]]]], dtype=np.float32)
    b = np.array([[[[2.0]]]], dtype=np.float32)
    caffe_out = _test_concat([a, b], caffe_test_dir, axis=1)
    assert caffe_out[0].shape == (1, 2, 1, 1)

    logger.debug("Testing negative values concat")
    a = np.full((1, 2, 2, 2), -1.0, dtype=np.float32)
    b = np.full((1, 2, 2, 2), 1.0, dtype=np.float32)
    caffe_out = _test_concat([a, b], caffe_test_dir, axis=1)
    ref = np.concatenate([a, b], axis=1)
    assert_op_correct(caffe_out, ref, op_name="Concat(neg-pos)")
