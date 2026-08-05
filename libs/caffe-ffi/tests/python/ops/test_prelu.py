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

DEFAULT_PRELU_SLOPE = 0.25


def _prelu_ref(x, slope=DEFAULT_PRELU_SLOPE):
    return np.where(x >= 0, x, slope * x)


def _get_slope_from_kwargs(kwargs):
    filler = kwargs.get("filler", None)
    if filler is not None and filler.get("type") == "constant":
        return filler.get("value", DEFAULT_PRELU_SLOPE)
    return DEFAULT_PRELU_SLOPE


def _test_prelu(data, test_dir, **kwargs):
    """One iteration of PReLU."""
    logger.info(f"Testing PReLU, input shape: {data.shape}")
    logger.debug(f"PReLU params: {kwargs}")
    return _test_op(data, L.PReLU, "PReLU", test_dir, **kwargs)


def test_forward_PReLU(caffe_test_dir):
    """PReLU"""
    logger.info("Running test_forward_PReLU")
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    logger.debug(f"Testing PReLU with constant filler=0.5, shape: {data.shape}")
    _test_prelu(data, caffe_test_dir, filler=dict(type="constant", value=0.5))
    logger.debug(f"Testing PReLU default params, shape: {data.shape}")
    _test_prelu(data, caffe_test_dir)
    data2 = np.random.rand(10, 20).astype(np.float32)
    logger.debug(f"Testing PReLU 2D input, shape: {data2.shape}")
    _test_prelu(data2, caffe_test_dir)


@pytest.mark.correctness
def test_prelu_correctness(caffe_test_dir):
    """PReLU correctness test with random data."""
    logger.info("Running test_prelu_correctness")
    np.random.seed(42)
    shapes = [
        (1, 3, 10, 10),
        (10, 20),
    ]
    for shape in shapes:
        x = np.random.randn(*shape).astype(np.float32)
        logger.debug(f"Testing PReLU default slope={DEFAULT_PRELU_SLOPE}, shape: {shape}")
        ref_default = _prelu_ref(x, DEFAULT_PRELU_SLOPE)
        caffe_out_default = _test_prelu(x, caffe_test_dir)
        assert_op_correct(caffe_out_default, ref_default, op_name="PReLU(default)")

        slope = 0.5
        logger.debug(f"Testing PReLU slope={slope}, shape: {shape}")
        ref_custom = _prelu_ref(x, slope)
        caffe_out_custom = _test_prelu(x, caffe_test_dir, filler=dict(type="constant", value=slope))
        assert_op_correct(caffe_out_custom, ref_custom, op_name=f"PReLU(slope={slope})")


@pytest.mark.edge
def test_prelu_edge_cases(caffe_test_dir):
    """PReLU edge cases."""
    logger.info("Running test_prelu_edge_cases")
    slope = DEFAULT_PRELU_SLOPE
    test_cases = [
        ("zeros", np.zeros((1, 3, 8, 8), dtype=np.float32)),
        ("all_negative", np.full((1, 3, 8, 8), -5.0, dtype=np.float32)),
        ("all_positive", np.full((1, 3, 8, 8), 5.0, dtype=np.float32)),
    ]
    for name, x in test_cases:
        logger.debug(f"Testing PReLU edge case: {name}")
        ref = _prelu_ref(x, slope)
        caffe_out = _test_prelu(x, caffe_test_dir)
        assert_op_correct(caffe_out, ref, op_name=f"PReLU({name})")
