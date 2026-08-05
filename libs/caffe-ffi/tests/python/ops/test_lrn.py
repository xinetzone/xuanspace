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


def _test_lrn(data, test_dir, local_size=5, alpha=1.0, beta=0.75, k=1.0):
    """One iteration of LRN"""
    logger.info(f"Testing LRN, input shape: {data.shape}")
    logger.debug(f"LRN params - local_size: {local_size}, alpha: {alpha}, beta: {beta}, k: {k}")
    return _test_op(data, L.LRN, "LRN", test_dir, local_size=local_size, alpha=alpha, beta=beta, k=k)


def test_forward_LRN(caffe_test_dir):
    """LRN"""
    logger.info("Running test_forward_LRN")
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    logger.debug(f"Calling _test_lrn with data shape {data.shape}, default params")
    _test_lrn(data, caffe_test_dir)
    logger.debug(f"Calling _test_lrn with data shape {data.shape}, local_size=3")
    _test_lrn(data, caffe_test_dir, local_size=3)
    logger.debug(f"Calling _test_lrn with data shape {data.shape}, local_size=3, alpha=2.0")
    _test_lrn(data, caffe_test_dir, local_size=3, alpha=2.0)
    logger.debug(f"Calling _test_lrn with data shape {data.shape}, local_size=3, alpha=2.0, beta=0.5")
    _test_lrn(data, caffe_test_dir, local_size=3, alpha=2.0, beta=0.5)
    logger.debug(f"Calling _test_lrn with data shape {data.shape}, local_size=3, alpha=2.0, beta=0.5, k=2.0")
    _test_lrn(data, caffe_test_dir, local_size=3, alpha=2.0, beta=0.5, k=2.0)


def _lrn_across_channels_ref(x, local_size=5, alpha=1.0, beta=0.75, k=1.0):
    """Numpy reference for Caffe LRN (ACROSS_CHANNELS mode, default)."""
    n, c, h, w = x.shape
    radius = (local_size - 1) // 2
    square_x = x * x
    out = np.zeros_like(x)
    for i in range(c):
        ch_start = max(0, i - radius)
        ch_end = min(c, i + radius + 1)
        sum_sq = np.sum(square_x[:, ch_start:ch_end, :, :], axis=1, keepdims=True)
        out[:, i:i+1, :, :] = x[:, i:i+1, :, :] / np.power(k + alpha / local_size * sum_sq, beta)
    return out.astype(np.float32)


@pytest.mark.correctness
def test_lrn_correctness(caffe_test_dir):
    """LRN runs without crash and produces correct output shape."""
    logger.info("Running test_lrn_correctness")
    np.random.seed(42)

    logger.debug("Testing LRN output shape preservation on random data")
    x = np.random.randn(1, 5, 8, 8).astype(np.float32)
    caffe_out = _test_lrn(x, caffe_test_dir, local_size=5, alpha=1.0, beta=0.75, k=1.0)
    assert caffe_out[0].shape == x.shape, f"Expected shape {x.shape} got {caffe_out[0].shape}"

    logger.debug("Testing LRN with all ones input (verifiable via formula)")
    x = np.ones((1, 3, 4, 4), dtype=np.float32)
    local_size, alpha, beta, k = 3, 1.0, 0.75, 1.0
    ref = _lrn_across_channels_ref(x, local_size=local_size, alpha=alpha, beta=beta, k=k)
    caffe_out = _test_lrn(x, caffe_test_dir, local_size=local_size, alpha=alpha, beta=beta, k=k)
    assert_op_correct(caffe_out, ref, op_name="LRN(all-ones)", atol=1e-5, rtol=1e-5)

    logger.debug("Testing LRN with k=2, beta=0.5 on all ones")
    x = np.ones((1, 5, 3, 3), dtype=np.float32)
    local_size, alpha, beta, k = 5, 2.0, 0.5, 2.0
    ref = _lrn_across_channels_ref(x, local_size=local_size, alpha=alpha, beta=beta, k=k)
    caffe_out = _test_lrn(x, caffe_test_dir, local_size=local_size, alpha=alpha, beta=beta, k=k)
    assert_op_correct(caffe_out, ref, op_name="LRN(all-ones-k=2)", atol=1e-5, rtol=1e-5)


@pytest.mark.edge
def test_lrn_edge_cases(caffe_test_dir):
    """LRN edge cases."""
    logger.info("Running test_lrn_edge_cases")

    logger.debug("Testing LRN with all zeros input (output should be zeros)")
    x = np.zeros((2, 3, 4, 4), dtype=np.float32)
    caffe_out = _test_lrn(x, caffe_test_dir)
    assert caffe_out[0].shape == x.shape, f"Expected shape {x.shape} got {caffe_out[0].shape}"
    np.testing.assert_allclose(caffe_out[0], np.zeros_like(x), atol=1e-6)

    logger.debug("Testing LRN with local_size=1 (essentially per-element scaling)")
    x = np.ones((1, 4, 3, 3), dtype=np.float32)
    caffe_out = _test_lrn(x, caffe_test_dir, local_size=1, alpha=0.0, beta=1.0, k=2.0)
    assert caffe_out[0].shape == x.shape, f"Expected shape {x.shape} got {caffe_out[0].shape}"
    expected = x / (2.0 ** 1.0)
    np.testing.assert_allclose(caffe_out[0], expected, atol=1e-5)

    logger.debug("Testing LRN with alpha=0 (no normalization, just x / k^beta)")
    x = np.ones((1, 3, 4, 4), dtype=np.float32)
    caffe_out = _test_lrn(x, caffe_test_dir, local_size=5, alpha=0.0, beta=1.0, k=2.0)
    assert caffe_out[0].shape == x.shape
    expected = x / (2.0 ** 1.0)
    np.testing.assert_allclose(caffe_out[0], expected, atol=1e-5)

    logger.debug("Testing LRN with large values")
    x = np.random.randn(1, 3, 4, 4).astype(np.float32) * 10
    caffe_out = _test_lrn(x, caffe_test_dir, local_size=3, alpha=1e-4, beta=0.75, k=1.0)
    assert caffe_out[0].shape == x.shape, f"Expected shape {x.shape} got {caffe_out[0].shape}"
