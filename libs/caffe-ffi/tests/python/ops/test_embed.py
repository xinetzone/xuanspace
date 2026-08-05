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


def _test_embed(data, test_dir, **kwargs):
    """One iteration of Embed"""
    logger.info(f"Testing Embed, input shape: {data.shape}")
    logger.debug(f"Embed params: {kwargs}")
    return _test_op(data, L.Embed, "Embed", test_dir, **kwargs)


def test_forward_Embed(caffe_test_dir):
    """Embed"""
    logger.info("Running test_forward_Embed")
    k = 20
    data = list(i for i in range(k))
    np.random.shuffle(data)
    data = np.asarray(data)
    logger.debug(f"Testing Embed 1D, bias_term=True, shape: {data.shape}")
    _test_embed(
        data,
        caffe_test_dir,
        num_output=30,
        input_dim=k,
        bias_term=True,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )
    logger.debug(f"Testing Embed 1D, bias_term=False, shape: {data.shape}")
    _test_embed(
        data,
        caffe_test_dir,
        num_output=30,
        input_dim=k,
        bias_term=False,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )
    data = np.reshape(data, [4, 5])
    logger.debug(f"Testing Embed 2D, bias_term=True, shape: {data.shape}")
    _test_embed(
        data,
        caffe_test_dir,
        num_output=30,
        input_dim=k,
        bias_term=True,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )
    logger.debug(f"Testing Embed 2D, bias_term=False, shape: {data.shape}")
    _test_embed(
        data,
        caffe_test_dir,
        num_output=30,
        input_dim=k,
        bias_term=False,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )
    data = np.reshape(data, [2, 2, 5])
    logger.debug(f"Testing Embed 3D, bias_term=True, shape: {data.shape}")
    _test_embed(
        data,
        caffe_test_dir,
        num_output=30,
        input_dim=k,
        bias_term=True,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )
    logger.debug(f"Testing Embed 3D, bias_term=False, shape: {data.shape}")
    _test_embed(
        data,
        caffe_test_dir,
        num_output=30,
        input_dim=k,
        bias_term=False,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )
    data = np.reshape(data, [2, 2, 5, 1])
    logger.debug(f"Testing Embed 4D, bias_term=True, shape: {data.shape}")
    _test_embed(
        data,
        caffe_test_dir,
        num_output=30,
        input_dim=k,
        bias_term=True,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )
    logger.debug(f"Testing Embed 4D, bias_term=False, shape: {data.shape}")
    _test_embed(
        data,
        caffe_test_dir,
        num_output=30,
        input_dim=k,
        bias_term=False,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )


@pytest.mark.correctness
def test_embed_correctness(caffe_test_dir):
    """Embed runs without crash and produces correct output shape."""
    logger.info("Running test_embed_correctness")
    np.random.seed(42)

    k = 10
    num_output = 8

    logger.debug("Testing Embed 1D shape: (10,) -> (10, 8)")
    data = np.arange(k, dtype=np.int32)
    caffe_out = _test_embed(
        data,
        caffe_test_dir,
        num_output=num_output,
        input_dim=k,
        bias_term=False,
        weight_filler=dict(type="xavier"),
    )
    expected_shape = data.shape + (num_output,)
    assert caffe_out[0].shape == expected_shape, f"Expected {expected_shape} got {caffe_out[0].shape}"

    logger.debug("Testing Embed 2D shape: (2, 5) -> (2, 5, 8)")
    data = np.arange(10, dtype=np.int32).reshape(2, 5)
    caffe_out = _test_embed(
        data,
        caffe_test_dir,
        num_output=num_output,
        input_dim=k,
        bias_term=True,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )
    expected_shape = data.shape + (num_output,)
    assert caffe_out[0].shape == expected_shape, f"Expected {expected_shape} got {caffe_out[0].shape}"

    logger.debug("Testing Embed 3D shape: (2, 2, 3) -> (2, 2, 3, 16)")
    k2 = 12
    num_output2 = 16
    data = np.arange(k2, dtype=np.int32).reshape(2, 2, 3)
    caffe_out = _test_embed(
        data,
        caffe_test_dir,
        num_output=num_output2,
        input_dim=k2,
        bias_term=False,
        weight_filler=dict(type="xavier"),
    )
    expected_shape = data.shape + (num_output2,)
    assert caffe_out[0].shape == expected_shape, f"Expected {expected_shape} got {caffe_out[0].shape}"


@pytest.mark.edge
def test_embed_edge_cases(caffe_test_dir):
    """Embed edge cases."""
    logger.info("Running test_embed_edge_cases")

    logger.debug("Testing Embed with single index")
    data = np.array([0], dtype=np.int32)
    caffe_out = _test_embed(
        data,
        caffe_test_dir,
        num_output=5,
        input_dim=10,
        bias_term=False,
        weight_filler=dict(type="xavier"),
    )
    assert caffe_out[0].shape == (1, 5), f"Expected (1, 5) got {caffe_out[0].shape}"

    logger.debug("Testing Embed with num_output=1")
    k = 5
    data = np.arange(k, dtype=np.int32)
    caffe_out = _test_embed(
        data,
        caffe_test_dir,
        num_output=1,
        input_dim=k,
        bias_term=False,
        weight_filler=dict(type="xavier"),
    )
    assert caffe_out[0].shape == (k, 1), f"Expected ({k}, 1) got {caffe_out[0].shape}"

    logger.debug("Testing Embed with bias_term=False and constant zero weight filler (output should be zero)")
    data = np.array([0, 1, 2], dtype=np.int32)
    caffe_out = _test_embed(
        data,
        caffe_test_dir,
        num_output=4,
        input_dim=5,
        bias_term=False,
        weight_filler=dict(type="constant", value=0.0),
    )
    assert caffe_out[0].shape == (3, 4), f"Expected (3, 4) got {caffe_out[0].shape}"
    np.testing.assert_allclose(caffe_out[0], np.zeros((3, 4), dtype=np.float32), atol=1e-6)

    logger.debug("Testing Embed with all same index")
    data = np.full((2, 3), 1, dtype=np.int32)
    caffe_out = _test_embed(
        data,
        caffe_test_dir,
        num_output=6,
        input_dim=5,
        bias_term=True,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="constant", value=0.0),
    )
    assert caffe_out[0].shape == (2, 3, 6), f"Expected (2, 3, 6) got {caffe_out[0].shape}"
