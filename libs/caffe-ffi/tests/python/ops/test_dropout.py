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


def _test_dropout(data, test_dir, **kwargs):
    """One iteration of Dropout"""
    logger.info(f"Testing Dropout, input shape: {data.shape}")
    logger.debug(f"Dropout params: {kwargs}")
    return _test_op(data, L.Dropout, "Dropout", test_dir, **kwargs)


def test_forward_Dropout(caffe_test_dir):
    """Dropout"""
    logger.info("Running test_forward_Dropout")
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    logger.debug(f"Calling _test_dropout with data shape {data.shape}, default params")
    _test_dropout(data, caffe_test_dir)
    logger.debug(f"Calling _test_dropout with data shape {data.shape}, dropout_ratio=0.7")
    _test_dropout(data, caffe_test_dir, dropout_ratio=0.7)


@pytest.mark.correctness
def test_dropout_correctness(caffe_test_dir):
    """Dropout correctness test - in TEST mode Dropout is identity (pass-through)."""
    logger.info("Running test_dropout_correctness")
    np.random.seed(42)

    logger.debug("Testing Dropout identity on random data")
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)
    caffe_out = _test_dropout(x, caffe_test_dir)
    assert_op_correct(caffe_out, x, op_name="Dropout(identity-default)", atol=1e-6, rtol=0)

    logger.debug("Testing Dropout identity with dropout_ratio=0.5")
    x = np.random.randn(1, 5, 4, 4).astype(np.float32)
    caffe_out = _test_dropout(x, caffe_test_dir, dropout_ratio=0.5)
    assert_op_correct(caffe_out, x, op_name="Dropout(identity-ratio0.5)", atol=1e-6, rtol=0)

    logger.debug("Testing Dropout identity with dropout_ratio=0.9")
    x = np.random.randn(3, 2, 5, 5).astype(np.float32)
    caffe_out = _test_dropout(x, caffe_test_dir, dropout_ratio=0.9)
    assert_op_correct(caffe_out, x, op_name="Dropout(identity-ratio0.9)", atol=1e-6, rtol=0)

    logger.debug("Testing Dropout on 2D data")
    x = np.random.randn(10, 20).astype(np.float32)
    caffe_out = _test_dropout(x, caffe_test_dir)
    assert_op_correct(caffe_out, x, op_name="Dropout(2D-identity)", atol=1e-6, rtol=0)


@pytest.mark.edge
def test_dropout_edge_cases(caffe_test_dir):
    """Dropout edge cases."""
    logger.info("Running test_dropout_edge_cases")

    logger.debug("Testing all zeros input")
    x = np.zeros((2, 3, 4, 4), dtype=np.float32)
    caffe_out = _test_dropout(x, caffe_test_dir)
    assert caffe_out[0].shape == x.shape
    assert np.all(caffe_out[0] == 0)

    logger.debug("Testing all ones input")
    x = np.ones((1, 2, 3, 3), dtype=np.float32)
    caffe_out = _test_dropout(x, caffe_test_dir, dropout_ratio=0.7)
    assert caffe_out[0].shape == x.shape
    assert np.all(caffe_out[0] == 1.0)

    logger.debug("Testing all negative values")
    x = np.full((2, 3, 2, 2), -1.0, dtype=np.float32)
    caffe_out = _test_dropout(x, caffe_test_dir)
    assert caffe_out[0].shape == x.shape
    assert np.all(caffe_out[0] == -1.0)

    logger.debug("Testing large values")
    x = np.full((1, 1, 2, 2), 1e4, dtype=np.float32)
    caffe_out = _test_dropout(x, caffe_test_dir)
    assert caffe_out[0].shape == x.shape
    assert np.allclose(caffe_out[0], x, atol=1e-3)
