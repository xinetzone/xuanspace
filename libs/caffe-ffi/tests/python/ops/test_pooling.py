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
from utils import L, P, _test_op, assert_op_correct

logger = logging.getLogger(__name__)


def _test_pooling(data, test_dir, **kwargs):
    """One iteration of Pooling."""
    logger.info(f"Testing Pooling, input shape: {data.shape}")
    logger.debug(f"Pooling params: {kwargs}")
    return _test_op(data, L.Pooling, "Pooling", test_dir, **kwargs)


def test_forward_Pooling(caffe_test_dir):
    """Pooling"""
    logger.info("Running test_forward_Pooling")
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    logger.debug(f"Testing MAX pooling with kernel_size=2, stride=2, shape: {data.shape}")
    _test_pooling(data, caffe_test_dir, kernel_size=2, stride=2, pad=0, pool=P.Pooling.MAX)
    logger.debug(f"Testing MAX pooling with explicit h/w params, shape: {data.shape}")
    _test_pooling(
        data, caffe_test_dir, kernel_h=2, kernel_w=3, stride_h=2, stride_w=1, pad_h=1, pad_w=2, pool=P.Pooling.MAX
    )
    logger.debug(f"Testing MAX global pooling, shape: {data.shape}")
    _test_pooling(data, caffe_test_dir, pool=P.Pooling.MAX, global_pooling=True)

    logger.debug(f"Testing AVE pooling with kernel_size=2, stride=2, shape: {data.shape}")
    _test_pooling(data, caffe_test_dir, kernel_size=2, stride=2, pad=0, pool=P.Pooling.AVE)
    logger.debug(f"Testing AVE pooling with explicit h/w params, shape: {data.shape}")
    _test_pooling(
        data, caffe_test_dir, kernel_h=2, kernel_w=3, stride_h=2, stride_w=1, pad_h=1, pad_w=2, pool=P.Pooling.AVE
    )
    logger.debug(f"Testing AVE global pooling, shape: {data.shape}")
    _test_pooling(data, caffe_test_dir, pool=P.Pooling.AVE, global_pooling=True)


@pytest.mark.correctness
def test_pooling_correctness(caffe_test_dir):
    """Pooling runs without crash and produces correct output shape."""
    logger.info("Running test_pooling_correctness")
    np.random.seed(42)

    logger.debug("Testing MAX pooling shape: (1,3,10,10) -> (1,3,5,5) with kernel=2,stride=2,pad=0")
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    caffe_out = _test_pooling(x, caffe_test_dir, kernel_size=2, stride=2, pad=0, pool=P.Pooling.MAX)
    assert caffe_out[0].shape == (1, 3, 5, 5), f"Expected (1,3,5,5) got {caffe_out[0].shape}"

    logger.debug("Testing AVE pooling shape: (1,3,8,8) -> (1,3,4,4) with kernel=2,stride=2,pad=0")
    x = np.random.randn(1, 3, 8, 8).astype(np.float32)
    caffe_out = _test_pooling(x, caffe_test_dir, kernel_size=2, stride=2, pad=0, pool=P.Pooling.AVE)
    assert caffe_out[0].shape == (1, 3, 4, 4), f"Expected (1,3,4,4) got {caffe_out[0].shape}"

    logger.debug("Testing global pooling shape: (1,3,10,10) -> (1,3,1,1)")
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    caffe_out = _test_pooling(x, caffe_test_dir, pool=P.Pooling.MAX, global_pooling=True)
    assert caffe_out[0].shape == (1, 3, 1, 1), f"Expected (1,3,1,1) got {caffe_out[0].shape}"

    logger.debug("Testing AVE global pooling shape: (2,5,6,6) -> (2,5,1,1)")
    x = np.random.randn(2, 5, 6, 6).astype(np.float32)
    caffe_out = _test_pooling(x, caffe_test_dir, pool=P.Pooling.AVE, global_pooling=True)
    assert caffe_out[0].shape == (2, 5, 1, 1), f"Expected (2,5,1,1) got {caffe_out[0].shape}"


@pytest.mark.edge
def test_pooling_edge_cases(caffe_test_dir):
    """Pooling edge cases."""
    logger.info("Running test_pooling_edge_cases")

    logger.debug("Testing MAX pooling with all ones input (output should be all ones)")
    x = np.ones((1, 3, 8, 8), dtype=np.float32)
    caffe_out = _test_pooling(x, caffe_test_dir, kernel_size=2, stride=2, pad=0, pool=P.Pooling.MAX)
    assert caffe_out[0].shape == (1, 3, 4, 4), f"Expected (1,3,4,4) got {caffe_out[0].shape}"
    np.testing.assert_allclose(caffe_out[0], np.ones((1, 3, 4, 4), dtype=np.float32), atol=1e-6)

    logger.debug("Testing AVE pooling with all ones input (output should be all ones, no padding)")
    x = np.ones((1, 2, 4, 4), dtype=np.float32)
    caffe_out = _test_pooling(x, caffe_test_dir, kernel_size=2, stride=2, pad=0, pool=P.Pooling.AVE)
    assert caffe_out[0].shape == (1, 2, 2, 2), f"Expected (1,2,2,2) got {caffe_out[0].shape}"
    np.testing.assert_allclose(caffe_out[0], np.ones((1, 2, 2, 2), dtype=np.float32), atol=1e-6)

    logger.debug("Testing MAX global pooling with all zeros input (output should be all zeros)")
    x = np.zeros((2, 3, 5, 5), dtype=np.float32)
    caffe_out = _test_pooling(x, caffe_test_dir, pool=P.Pooling.MAX, global_pooling=True)
    assert caffe_out[0].shape == (2, 3, 1, 1), f"Expected (2,3,1,1) got {caffe_out[0].shape}"
    np.testing.assert_allclose(caffe_out[0], np.zeros((2, 3, 1, 1), dtype=np.float32), atol=1e-6)

    logger.debug("Testing AVE global pooling with all same values (output should equal that value)")
    val = 3.5
    x = np.full((1, 4, 6, 6), val, dtype=np.float32)
    caffe_out = _test_pooling(x, caffe_test_dir, pool=P.Pooling.AVE, global_pooling=True)
    assert caffe_out[0].shape == (1, 4, 1, 1), f"Expected (1,4,1,1) got {caffe_out[0].shape}"
    np.testing.assert_allclose(caffe_out[0], np.full((1, 4, 1, 1), val, dtype=np.float32), rtol=1e-5)

    logger.debug("Testing pooling with 1x1 kernel stride=1 (identity)")
    x = np.random.randn(1, 3, 6, 6).astype(np.float32)
    caffe_out = _test_pooling(x, caffe_test_dir, kernel_size=1, stride=1, pad=0, pool=P.Pooling.MAX)
    assert caffe_out[0].shape == x.shape, f"Expected {x.shape} got {caffe_out[0].shape}"
