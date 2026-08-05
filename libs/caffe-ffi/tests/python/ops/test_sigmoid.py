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


def _sigmoid_ref(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def _test_sigmoid(data, test_dir, **kwargs):
    """One iteration of Sigmoid."""
    logger.info(f"Testing Sigmoid, input shape: {data.shape}")
    logger.debug(f"Sigmoid params: {kwargs}")
    return _test_op(data, L.Sigmoid, "Sigmoid", test_dir, **kwargs)


def test_forward_Sigmoid(caffe_test_dir):
    """Sigmoid"""
    logger.info("Running test_forward_Sigmoid")
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    logger.debug(f"Calling _test_sigmoid with data shape {data.shape}")
    _test_sigmoid(data, caffe_test_dir)


@pytest.mark.correctness
def test_sigmoid_correctness(caffe_test_dir):
    """Sigmoid correctness test with random data."""
    logger.info("Running test_sigmoid_correctness")
    np.random.seed(42)
    shapes = [
        (1, 3, 10, 10),
        (10, 20),
        (3, 10, 10),
        (100,),
    ]
    for shape in shapes:
        x = np.random.randn(*shape).astype(np.float32) * 5
        ref = _sigmoid_ref(x)
        caffe_out = _test_sigmoid(x, caffe_test_dir)
        assert_op_correct(caffe_out, ref, atol=1e-4, rtol=1e-4, op_name="Sigmoid")


@pytest.mark.edge
def test_sigmoid_edge_cases(caffe_test_dir):
    """Sigmoid edge cases."""
    logger.info("Running test_sigmoid_edge_cases")
    test_cases = [
        ("zeros", np.zeros((1, 3, 8, 8), dtype=np.float32)),
        ("large_positive", np.full((1, 3, 8, 8), 88.0, dtype=np.float32)),
        ("large_negative", np.full((1, 3, 8, 8), -88.0, dtype=np.float32)),
        ("medium_positive", np.full((1, 3, 8, 8), 5.0, dtype=np.float32)),
        ("medium_negative", np.full((1, 3, 8, 8), -5.0, dtype=np.float32)),
        ("one", np.full((1, 3, 8, 8), 1.0, dtype=np.float32)),
        ("neg_one", np.full((1, 3, 8, 8), -1.0, dtype=np.float32)),
    ]
    for name, x in test_cases:
        logger.debug(f"Testing Sigmoid edge case: {name}")
        ref = _sigmoid_ref(x)
        caffe_out = _test_sigmoid(x, caffe_test_dir)
        assert_op_correct(caffe_out, ref, atol=1e-4, rtol=1e-4, op_name=f"Sigmoid({name})")
