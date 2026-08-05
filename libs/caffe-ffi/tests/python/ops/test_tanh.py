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


def _tanh_ref(x):
    return np.tanh(x)


def _test_tanh(data, test_dir, **kwargs):
    """One iteration of TanH"""
    logger.info(f"Testing TanH, input shape: {data.shape}")
    logger.debug(f"TanH params: {kwargs}")
    return _test_op(data, L.TanH, "TanH", test_dir, **kwargs)


def test_forward_TanH(caffe_test_dir):
    """TanH"""
    logger.info("Running test_forward_TanH")
    logger.debug("Testing TanH 4D input")
    _test_tanh(np.random.rand(1, 3, 10, 10).astype(np.float32), caffe_test_dir)
    logger.debug("Testing TanH 3D input")
    _test_tanh(np.random.rand(3, 10, 10).astype(np.float32), caffe_test_dir)
    logger.debug("Testing TanH 2D input")
    _test_tanh(np.random.rand(10, 10).astype(np.float32), caffe_test_dir)
    logger.debug("Testing TanH 1D input")
    _test_tanh(np.random.rand(10).astype(np.float32), caffe_test_dir)


@pytest.mark.correctness
def test_tanh_correctness(caffe_test_dir):
    """TanH correctness test with multi-dimensional random data."""
    logger.info("Running test_tanh_correctness")
    np.random.seed(42)
    shapes = [
        (1, 3, 10, 10),
        (3, 10, 10),
        (10, 10),
        (100,),
    ]
    for shape in shapes:
        x = np.random.randn(*shape).astype(np.float32) * 3
        ref = _tanh_ref(x)
        caffe_out = _test_tanh(x, caffe_test_dir)
        assert_op_correct(caffe_out, ref, atol=1e-4, rtol=1e-4, op_name="TanH")


@pytest.mark.edge
def test_tanh_edge_cases(caffe_test_dir):
    """TanH edge cases."""
    logger.info("Running test_tanh_edge_cases")
    test_cases = [
        ("zeros", np.zeros((1, 3, 8, 8), dtype=np.float32)),
        ("large_positive", np.full((1, 3, 8, 8), 10.0, dtype=np.float32)),
        ("large_negative", np.full((1, 3, 8, 8), -10.0, dtype=np.float32)),
        ("one", np.full((1, 3, 8, 8), 1.0, dtype=np.float32)),
        ("neg_one", np.full((1, 3, 8, 8), -1.0, dtype=np.float32)),
        ("small_positive", np.full((1, 3, 8, 8), 1e-3, dtype=np.float32)),
        ("small_negative", np.full((1, 3, 8, 8), -1e-3, dtype=np.float32)),
    ]
    for name, x in test_cases:
        logger.debug(f"Testing TanH edge case: {name}")
        ref = _tanh_ref(x)
        caffe_out = _test_tanh(x, caffe_test_dir)
        assert_op_correct(caffe_out, ref, atol=1e-4, rtol=1e-4, op_name=f"TanH({name})")
