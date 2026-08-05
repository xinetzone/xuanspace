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


def _test_batchnorm(data, test_dir, moving_average_fraction=0.999, eps=1e-5, **kwargs):
    """One iteration of BatchNorm"""
    logger.info(f"Testing BatchNorm, input shape: {data.shape}")
    logger.debug(
        f"BatchNorm params - moving_average_fraction: {moving_average_fraction}, eps: {eps}, extra: {kwargs}"
    )
    return _test_op(data, L.BatchNorm, "BatchNorm", test_dir, moving_average_fraction=moving_average_fraction, eps=eps, **kwargs)


def test_forward_BatchNorm(caffe_test_dir):
    """BatchNorm"""
    logger.info("Running test_forward_BatchNorm")
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    logger.debug(f"Calling _test_batchnorm with data shape {data.shape}, default params")
    _test_batchnorm(data, caffe_test_dir)
    logger.debug(f"Calling _test_batchnorm with data shape {data.shape}, moving_average_fraction=0.88, eps=1e-4")
    _test_batchnorm(data, caffe_test_dir, moving_average_fraction=0.88, eps=1e-4)


@pytest.mark.correctness
def test_batchnorm_correctness(caffe_test_dir):
    """BatchNorm runs without crash and preserves input shape."""
    logger.info("Running test_batchnorm_correctness")
    np.random.seed(42)

    logger.debug("Testing BatchNorm on 4D data (standard case)")
    x = np.random.randn(2, 5, 8, 8).astype(np.float32)
    caffe_out = _test_batchnorm(x, caffe_test_dir)
    assert caffe_out[0].shape == x.shape, f"Expected shape {x.shape} got {caffe_out[0].shape}"

    logger.debug("Testing BatchNorm on 2D data")
    x = np.random.randn(4, 10).astype(np.float32)
    caffe_out = _test_batchnorm(x, caffe_test_dir)
    assert caffe_out[0].shape == x.shape, f"Expected shape {x.shape} got {caffe_out[0].shape}"

    logger.debug("Testing BatchNorm on 3D data")
    x = np.random.randn(3, 6, 7).astype(np.float32)
    caffe_out = _test_batchnorm(x, caffe_test_dir)
    assert caffe_out[0].shape == x.shape, f"Expected shape {x.shape} got {caffe_out[0].shape}"

    logger.debug("Testing BatchNorm with different eps values")
    x = np.random.randn(1, 3, 6, 6).astype(np.float32)
    caffe_out = _test_batchnorm(x, caffe_test_dir, eps=1e-3)
    assert caffe_out[0].shape == x.shape, f"Expected shape {x.shape} got {caffe_out[0].shape}"


@pytest.mark.edge
def test_batchnorm_edge_cases(caffe_test_dir):
    """BatchNorm edge cases."""
    logger.info("Running test_batchnorm_edge_cases")

    logger.debug("Testing BatchNorm with all zeros input")
    x = np.zeros((2, 3, 4, 4), dtype=np.float32)
    caffe_out = _test_batchnorm(x, caffe_test_dir)
    assert caffe_out[0].shape == x.shape, f"Expected shape {x.shape} got {caffe_out[0].shape}"

    logger.debug("Testing BatchNorm with all ones input")
    x = np.ones((1, 4, 5, 5), dtype=np.float32)
    caffe_out = _test_batchnorm(x, caffe_test_dir)
    assert caffe_out[0].shape == x.shape, f"Expected shape {x.shape} got {caffe_out[0].shape}"

    logger.debug("Testing BatchNorm with constant values across channels")
    x = np.full((1, 2, 3, 3), 5.0, dtype=np.float32)
    caffe_out = _test_batchnorm(x, caffe_test_dir)
    assert caffe_out[0].shape == x.shape, f"Expected shape {x.shape} got {caffe_out[0].shape}"

    logger.debug("Testing BatchNorm with large values")
    x = np.random.randn(1, 3, 4, 4).astype(np.float32) * 100
    caffe_out = _test_batchnorm(x, caffe_test_dir)
    assert caffe_out[0].shape == x.shape, f"Expected shape {x.shape} got {caffe_out[0].shape}"

    logger.debug("Testing BatchNorm batch_size=1")
    x = np.random.randn(1, 8, 6, 6).astype(np.float32)
    caffe_out = _test_batchnorm(x, caffe_test_dir)
    assert caffe_out[0].shape == x.shape, f"Expected shape {x.shape} got {caffe_out[0].shape}"
