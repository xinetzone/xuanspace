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
from utils import L, _test_op

logger = logging.getLogger(__name__)


def _test_scale(data, test_dir, **kwargs):
    """One iteration of Scale."""
    logger.info(f"Testing Scale, input shape: {data.shape}")
    logger.debug(f"Scale params: {kwargs}")
    return _test_op(data, L.Scale, "Scale", test_dir, **kwargs)


def test_forward_Scale(caffe_test_dir):
    """Scale"""
    logger.info("Running test_forward_Scale")
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    logger.debug(f"Testing Scale with xavier filler, no bias, shape: {data.shape}")
    _test_scale(data, caffe_test_dir, filler=dict(type="xavier"))
    logger.debug(f"Testing Scale with xavier filler and bias_term=True, shape: {data.shape}")
    _test_scale(data, caffe_test_dir, filler=dict(type="xavier"), bias_term=True, bias_filler=dict(type="xavier"))


@pytest.mark.correctness
def test_scale_correctness(caffe_test_dir):
    """Scale runs without crash and preserves input shape."""
    logger.info("Running test_scale_correctness")
    np.random.seed(42)

    logger.debug("Testing Scale on 4D data without bias")
    x = np.random.randn(2, 5, 8, 8).astype(np.float32)
    caffe_out = _test_scale(x, caffe_test_dir, filler=dict(type="xavier"))
    assert caffe_out[0].shape == x.shape, f"Expected shape {x.shape} got {caffe_out[0].shape}"

    logger.debug("Testing Scale on 4D data with bias")
    x = np.random.randn(1, 3, 6, 6).astype(np.float32)
    caffe_out = _test_scale(
        x,
        caffe_test_dir,
        filler=dict(type="xavier"),
        bias_term=True,
        bias_filler=dict(type="xavier"),
    )
    assert caffe_out[0].shape == x.shape, f"Expected shape {x.shape} got {caffe_out[0].shape}"

    logger.debug("Testing Scale on 2D data")
    x = np.random.randn(4, 10).astype(np.float32)
    caffe_out = _test_scale(x, caffe_test_dir, filler=dict(type="xavier"))
    assert caffe_out[0].shape == x.shape, f"Expected shape {x.shape} got {caffe_out[0].shape}"


@pytest.mark.edge
def test_scale_edge_cases(caffe_test_dir):
    """Scale edge cases."""
    logger.info("Running test_scale_edge_cases")

    logger.debug("Testing Scale with all zeros input (no bias)")
    x = np.zeros((2, 3, 4, 4), dtype=np.float32)
    caffe_out = _test_scale(x, caffe_test_dir, filler=dict(type="xavier"))
    assert caffe_out[0].shape == x.shape, f"Expected shape {x.shape} got {caffe_out[0].shape}"

    logger.debug("Testing Scale with all zeros input (with bias)")
    x = np.zeros((1, 4, 5, 5), dtype=np.float32)
    caffe_out = _test_scale(
        x,
        caffe_test_dir,
        filler=dict(type="xavier"),
        bias_term=True,
        bias_filler=dict(type="constant", value=0.0),
    )
    assert caffe_out[0].shape == x.shape, f"Expected shape {x.shape} got {caffe_out[0].shape}"

    logger.debug("Testing Scale with all ones input")
    x = np.ones((1, 2, 3, 3), dtype=np.float32)
    caffe_out = _test_scale(x, caffe_test_dir, filler=dict(type="xavier"))
    assert caffe_out[0].shape == x.shape, f"Expected shape {x.shape} got {caffe_out[0].shape}"

    logger.debug("Testing Scale with constant filler (scale=1, bias=0 should be identity)")
    x = np.random.randn(2, 3, 4, 4).astype(np.float32)
    caffe_out = _test_scale(
        x,
        caffe_test_dir,
        filler=dict(type="constant", value=1.0),
        bias_term=True,
        bias_filler=dict(type="constant", value=0.0),
    )
    assert caffe_out[0].shape == x.shape, f"Expected shape {x.shape} got {caffe_out[0].shape}"

    logger.debug("Testing Scale with large values")
    x = np.random.randn(1, 3, 4, 4).astype(np.float32) * 100
    caffe_out = _test_scale(x, caffe_test_dir, filler=dict(type="xavier"))
    assert caffe_out[0].shape == x.shape, f"Expected shape {x.shape} got {caffe_out[0].shape}"
