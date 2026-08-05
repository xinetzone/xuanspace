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


def _test_eltwise(data_list, test_dir, **kwargs):
    """One iteration of Eltwise"""
    input_shapes = [d.shape for d in data_list]
    logger.info(f"Testing Eltwise, num inputs: {len(data_list)}, shapes: {input_shapes}")
    logger.debug(f"Eltwise params: {kwargs}")
    return _test_op(data_list, L.Eltwise, "Eltwise", test_dir, **kwargs)


def test_forward_Eltwise(caffe_test_dir):
    """Eltwise"""
    logger.info("Running test_forward_Eltwise")
    logger.debug("Testing Eltwise operation=0 (PROD), 2 inputs")
    _test_eltwise(
        [
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
        ],
        caffe_test_dir,
        operation=0,
    )
    logger.debug("Testing Eltwise operation=1 (SUM), 2 inputs")
    _test_eltwise(
        [
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
        ],
        caffe_test_dir,
        operation=1,
    )
    logger.debug("Testing Eltwise operation=2 (MAX), 2 inputs")
    _test_eltwise(
        [
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
        ],
        caffe_test_dir,
        operation=2,
    )
    logger.debug("Testing Eltwise operation=1 with coeff=[0.5, 1]")
    _test_eltwise(
        [
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
        ],
        caffe_test_dir,
        operation=1,
        coeff=[0.5, 1],
    )
    logger.debug("Testing Eltwise operation=0, 3 inputs")
    _test_eltwise(
        [
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
        ],
        caffe_test_dir,
        operation=0,
    )
    logger.debug("Testing Eltwise operation=1, 4 inputs")
    _test_eltwise(
        [
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
        ],
        caffe_test_dir,
        operation=1,
    )
    logger.debug("Testing Eltwise operation=2, 5 inputs")
    _test_eltwise(
        [
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
        ],
        caffe_test_dir,
        operation=2,
    )
    logger.debug("Testing Eltwise operation=1 with coeff for 6 inputs")
    _test_eltwise(
        [
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
            np.random.rand(1, 3, 10, 11).astype(np.float32),
        ],
        caffe_test_dir,
        operation=1,
        coeff=[0.5, 1, 0.2, 1.8, 3.1, 0.1],
    )


def _eltwise_ref(data_list, operation=1, coeff=None):
    if coeff is None:
        coeff = [1.0] * len(data_list)
    if operation == 0:
        ref = data_list[0].copy()
        for d in data_list[1:]:
            ref = ref * d
    elif operation == 1:
        ref = coeff[0] * data_list[0]
        for i, d in enumerate(data_list[1:], 1):
            ref = ref + coeff[i] * d
    elif operation == 2:
        ref = np.maximum.reduce(data_list)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    return ref.astype(np.float32)


@pytest.mark.correctness
def test_eltwise_correctness(caffe_test_dir):
    """Eltwise correctness test with numpy reference."""
    logger.info("Running test_eltwise_correctness")
    np.random.seed(42)
    shape = (1, 3, 8, 8)

    logger.debug("Testing PROD (operation=0) with 2 inputs")
    a = np.random.randn(*shape).astype(np.float32)
    b = np.random.randn(*shape).astype(np.float32)
    ref = _eltwise_ref([a, b], operation=0)
    caffe_out = _test_eltwise([a, b], caffe_test_dir, operation=0)
    assert_op_correct(caffe_out, ref, op_name="Eltwise(PROD-2in)")

    logger.debug("Testing SUM (operation=1) with 2 inputs, default coeff")
    a = np.random.randn(*shape).astype(np.float32)
    b = np.random.randn(*shape).astype(np.float32)
    ref = _eltwise_ref([a, b], operation=1)
    caffe_out = _test_eltwise([a, b], caffe_test_dir, operation=1)
    assert_op_correct(caffe_out, ref, op_name="Eltwise(SUM-2in)")

    logger.debug("Testing SUM (operation=1) with 2 inputs, custom coeff")
    a = np.random.randn(*shape).astype(np.float32)
    b = np.random.randn(*shape).astype(np.float32)
    coeff = [0.5, 1.5]
    ref = _eltwise_ref([a, b], operation=1, coeff=coeff)
    caffe_out = _test_eltwise([a, b], caffe_test_dir, operation=1, coeff=coeff)
    assert_op_correct(caffe_out, ref, op_name="Eltwise(SUM-coeff)")

    logger.debug("Testing MAX (operation=2) with 2 inputs")
    a = np.random.randn(*shape).astype(np.float32)
    b = np.random.randn(*shape).astype(np.float32)
    ref = _eltwise_ref([a, b], operation=2)
    caffe_out = _test_eltwise([a, b], caffe_test_dir, operation=2)
    assert_op_correct(caffe_out, ref, op_name="Eltwise(MAX-2in)")

    logger.debug("Testing PROD (operation=0) with 3 inputs")
    a = np.random.randn(*shape).astype(np.float32)
    b = np.random.randn(*shape).astype(np.float32)
    c = np.random.randn(*shape).astype(np.float32)
    ref = _eltwise_ref([a, b, c], operation=0)
    caffe_out = _test_eltwise([a, b, c], caffe_test_dir, operation=0)
    assert_op_correct(caffe_out, ref, op_name="Eltwise(PROD-3in)")

    logger.debug("Testing SUM (operation=1) with 3 inputs")
    a = np.random.randn(*shape).astype(np.float32)
    b = np.random.randn(*shape).astype(np.float32)
    c = np.random.randn(*shape).astype(np.float32)
    ref = _eltwise_ref([a, b, c], operation=1)
    caffe_out = _test_eltwise([a, b, c], caffe_test_dir, operation=1)
    assert_op_correct(caffe_out, ref, op_name="Eltwise(SUM-3in)")


@pytest.mark.edge
def test_eltwise_edge_cases(caffe_test_dir):
    """Eltwise edge cases."""
    logger.info("Running test_eltwise_edge_cases")
    shape = (1, 3, 6, 6)
    edge_cases = [
        ("all_zeros", [np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)]),
        ("all_ones", [np.ones(shape, dtype=np.float32), np.ones(shape, dtype=np.float32)]),
        ("all_negatives", [np.full(shape, -2.0, dtype=np.float32), np.full(shape, -3.0, dtype=np.float32)]),
        ("mixed_signs", [np.full(shape, 2.0, dtype=np.float32), np.full(shape, -1.0, dtype=np.float32)]),
        ("large_values", [np.full(shape, 1e2, dtype=np.float32), np.full(shape, 1e2, dtype=np.float32)]),
        ("small_values", [np.full(shape, 1e-3, dtype=np.float32), np.full(shape, 1e-3, dtype=np.float32)]),
    ]
    for name, inputs in edge_cases:
        for op_name, op_id in [("PROD", 0), ("SUM", 1), ("MAX", 2)]:
            logger.debug(f"Testing Eltwise edge case: {name} with {op_name}")
            ref = _eltwise_ref(inputs, operation=op_id)
            caffe_out = _test_eltwise(inputs, caffe_test_dir, operation=op_id)
            assert_op_correct(caffe_out, ref, op_name=f"Eltwise({op_name}-{name})")
