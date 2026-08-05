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

DEFAULT_POWER_PARAMS = {"power": 1.0, "scale": 1.0, "shift": 0.0}


def _power_ref(x, power=1.0, scale=1.0, shift=0.0):
    return np.power(shift + scale * x, power)


def _get_power_params(kwargs):
    power_param = kwargs.get("power_param", {})
    params = DEFAULT_POWER_PARAMS.copy()
    params.update(power_param)
    return params


def _test_power(data, test_dir, **kwargs):
    """One iteration of Power."""
    logger.info(f"Testing Power, input shape: {data.shape}")
    logger.debug(f"Power params: {kwargs}")
    return _test_op(data, L.Power, "Power", test_dir, **kwargs)


def test_forward_Power(caffe_test_dir):
    """Power"""
    logger.info("Running test_forward_Power")
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    logger.debug("Testing Power with power=0.37, scale=0.83, shift=-2.4")
    _test_power(data, caffe_test_dir, power_param={"power": 0.37, "scale": 0.83, "shift": -2.4})
    logger.debug("Testing Power with power=0.37, scale=0.83, shift=0.0")
    _test_power(data, caffe_test_dir, power_param={"power": 0.37, "scale": 0.83, "shift": 0.0})
    logger.debug("Testing Power with power=0.0, scale=0.83, shift=-2.4")
    _test_power(data, caffe_test_dir, power_param={"power": 0.0, "scale": 0.83, "shift": -2.4})
    logger.debug("Testing Power with power=1.0, scale=0.83, shift=-2.4")
    _test_power(data, caffe_test_dir, power_param={"power": 1.0, "scale": 0.83, "shift": -2.4})
    logger.debug("Testing Power with power=2.0, scale=0.34, shift=-2.4")
    _test_power(data, caffe_test_dir, power_param={"power": 2.0, "scale": 0.34, "shift": -2.4})
    logger.debug("Testing Power with identity params (power=1.0, scale=1.0, shift=0.0)")
    _test_power(data, caffe_test_dir, power_param={"power": 1.0, "scale": 1.0, "shift": 0.0})


@pytest.mark.correctness
def test_power_correctness(caffe_test_dir):
    """Power correctness test with various parameter combinations."""
    logger.info("Running test_power_correctness")
    np.random.seed(42)
    shape = (1, 3, 10, 10)

    param_sets = [
        ("identity", {"power": 1.0, "scale": 1.0, "shift": 0.0}),
        ("square", {"power": 2.0, "scale": 1.0, "shift": 0.0}),
        ("linear_2x_plus_1", {"power": 1.0, "scale": 2.0, "shift": 1.0}),
        ("cubic", {"power": 3.0, "scale": 1.0, "shift": 0.0}),
    ]

    for name, params in param_sets:
        if params["power"] == 1.0:
            x = np.random.randn(*shape).astype(np.float32)
        else:
            x = np.abs(np.random.randn(*shape)).astype(np.float32) + 0.1
        logger.debug(f"Testing Power {name} with params {params}")
        ref = _power_ref(x, **params)
        caffe_out = _test_power(x, caffe_test_dir, power_param=params)
        assert_op_correct(caffe_out, ref, atol=1e-5, rtol=1e-4, op_name=f"Power({name})")


@pytest.mark.edge
def test_power_edge_cases(caffe_test_dir):
    """Power edge cases."""
    logger.info("Running test_power_edge_cases")
    shape = (1, 3, 8, 8)
    test_cases = [
        ("zeros_identity", np.zeros(shape, dtype=np.float32), {"power": 1.0, "scale": 1.0, "shift": 0.0}),
        ("zeros_square", np.zeros(shape, dtype=np.float32), {"power": 2.0, "scale": 1.0, "shift": 0.0}),
        ("ones_square", np.ones(shape, dtype=np.float32), {"power": 2.0, "scale": 1.0, "shift": 0.0}),
        ("pos_input_linear", np.full(shape, 2.0, dtype=np.float32), {"power": 1.0, "scale": 2.0, "shift": 1.0}),
        ("pos_input_even_power", np.full(shape, 3.0, dtype=np.float32), {"power": 2.0, "scale": 1.0, "shift": 0.0}),
    ]
    for name, x, params in test_cases:
        logger.debug(f"Testing Power edge case: {name}")
        ref = _power_ref(x, **params)
        caffe_out = _test_power(x, caffe_test_dir, power_param=params)
        assert_op_correct(caffe_out, ref, atol=1e-5, rtol=1e-4, op_name=f"Power({name})")
