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

SUM = 1
ASUM = 2
SUMSQ = 3
MEAN = 4


def _test_reduction(data, test_dir, **kwargs):
    """One iteration of Reduction"""
    logger.info(f"Testing Reduction, input shape: {data.shape}")
    logger.debug(f"Reduction params: {kwargs}")
    return _test_op(data, L.Reduction, "Reduction", test_dir, **kwargs)


def test_forward_Reduction(caffe_test_dir):
    """Reduction"""
    logger.info("Running test_forward_Reduction")
    reduction_op = {"SUM": 1, "ASUM": 2, "SUMSQ": 3, "MEAN": 4}
    logger.debug("Testing Reduction SUM 1D, axis=0")
    _test_reduction(np.random.rand(10).astype(np.float32), caffe_test_dir, operation=reduction_op["SUM"], axis=0)
    logger.debug("Testing Reduction SUM 4D, axis=3")
    _test_reduction(
        np.random.rand(10, 20, 30, 40).astype(np.float32), caffe_test_dir, operation=reduction_op["SUM"], axis=3
    )
    logger.debug("Testing Reduction SUM 4D, axis=1")
    _test_reduction(
        np.random.rand(10, 20, 30, 40).astype(np.float32), caffe_test_dir, operation=reduction_op["SUM"], axis=1
    )
    logger.debug("Testing Reduction SUM 1D, axis=0, coeff=0.5")
    _test_reduction(
        np.random.rand(10).astype(np.float32), caffe_test_dir, operation=reduction_op["SUM"], axis=0, coeff=0.5
    )
    logger.debug("Testing Reduction SUM 4D, axis=3, coeff=5.0")
    _test_reduction(
        np.random.rand(10, 20, 30, 40).astype(np.float32),
        caffe_test_dir,
        operation=reduction_op["SUM"],
        axis=3,
        coeff=5.0,
    )
    logger.debug("Testing Reduction ASUM 1D")
    _test_reduction(np.random.rand(10).astype(np.float32), caffe_test_dir, operation=reduction_op["ASUM"])
    logger.debug("Testing Reduction ASUM 2D, axis=1")
    _test_reduction(
        np.random.rand(10, 20).astype(np.float32), caffe_test_dir, operation=reduction_op["ASUM"], axis=1
    )
    logger.debug("Testing Reduction ASUM 4D, axis=3")
    _test_reduction(
        np.random.rand(10, 20, 30, 40).astype(np.float32), caffe_test_dir, operation=reduction_op["ASUM"], axis=3
    )
    logger.debug("Testing Reduction ASUM 1D, axis=0, coeff=0.0")
    _test_reduction(
        np.random.rand(10).astype(np.float32), caffe_test_dir, operation=reduction_op["ASUM"], axis=0, coeff=0.0
    )
    logger.debug("Testing Reduction ASUM 3D, axis=2, coeff=7.0")
    _test_reduction(
        np.random.rand(10, 20, 30).astype(np.float32),
        caffe_test_dir,
        operation=reduction_op["ASUM"],
        axis=2,
        coeff=7.0,
    )
    logger.debug("Testing Reduction ASUM 5D, axis=3, coeff=1.0")
    _test_reduction(
        np.random.rand(10, 20, 30, 40, 10).astype(np.float32),
        caffe_test_dir,
        operation=reduction_op["ASUM"],
        axis=3,
        coeff=1.0,
    )
    logger.debug("Testing Reduction SUMSQ 1D, axis=0")
    _test_reduction(np.random.rand(10).astype(np.float32), caffe_test_dir, operation=reduction_op["SUMSQ"], axis=0)
    logger.debug("Testing Reduction SUMSQ 4D, axis=3")
    _test_reduction(
        np.random.rand(10, 20, 30, 40).astype(np.float32), caffe_test_dir, operation=reduction_op["SUMSQ"], axis=3
    )
    logger.debug("Testing Reduction SUMSQ 1D, axis=0, coeff=0.0")
    _test_reduction(
        np.random.rand(10).astype(np.float32), caffe_test_dir, operation=reduction_op["SUMSQ"], axis=0, coeff=0.0
    )
    logger.debug("Testing Reduction SUMSQ 5D, axis=4, coeff=2.0")
    _test_reduction(
        np.random.rand(10, 20, 30, 40, 50).astype(np.float32),
        caffe_test_dir,
        operation=reduction_op["SUMSQ"],
        axis=4,
        coeff=2.0,
    )
    logger.debug("Testing Reduction MEAN 1D, axis=0")
    _test_reduction(np.random.rand(10).astype(np.float32), caffe_test_dir, operation=reduction_op["MEAN"], axis=0)
    logger.debug("Testing Reduction MEAN 4D, axis=3")
    _test_reduction(
        np.random.rand(10, 20, 30, 40).astype(np.float32), caffe_test_dir, operation=reduction_op["MEAN"], axis=3
    )
    logger.debug("Testing Reduction MEAN 1D, axis=0, coeff=0.0")
    _test_reduction(
        np.random.rand(10).astype(np.float32), caffe_test_dir, operation=reduction_op["MEAN"], axis=0, coeff=0.0
    )
    logger.debug("Testing Reduction MEAN 4D, axis=3, coeff=2.0")
    _test_reduction(
        np.random.rand(10, 20, 30, 40).astype(np.float32),
        caffe_test_dir,
        operation=reduction_op["MEAN"],
        axis=3,
        coeff=2.0,
    )


def _reduction_ref_1d(x, operation, coeff=1.0):
    if operation == SUM:
        ref = np.sum(x)
    elif operation == ASUM:
        ref = np.sum(np.abs(x))
    elif operation == SUMSQ:
        ref = np.sum(x * x)
    elif operation == MEAN:
        ref = np.mean(x)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    return (coeff * ref).astype(np.float32)


def _reduction_ref_axis(x, operation, axis, coeff=1.0):
    if operation == SUM:
        ref = np.sum(x, axis=axis)
    elif operation == ASUM:
        ref = np.sum(np.abs(x), axis=axis)
    elif operation == SUMSQ:
        ref = np.sum(x * x, axis=axis)
    elif operation == MEAN:
        ref = np.mean(x, axis=axis)
    else:
        raise ValueError(f"Unknown operation: {operation}")
    return (coeff * ref).astype(np.float32)


@pytest.mark.correctness
def test_reduction_correctness(caffe_test_dir):
    """Reduction correctness test for simple cases (1D and last axis)."""
    logger.info("Running test_reduction_correctness")
    np.random.seed(42)

    logger.debug("Testing SUM on 1D")
    x = np.random.randn(20).astype(np.float32)
    ref = _reduction_ref_1d(x, SUM)
    caffe_out = _test_reduction(x, caffe_test_dir, operation=SUM, axis=0)
    assert_op_correct(caffe_out, ref, op_name="Reduction(SUM-1D)", atol=1e-4)

    logger.debug("Testing MEAN on 1D")
    x = np.random.randn(20).astype(np.float32)
    ref = _reduction_ref_1d(x, MEAN)
    caffe_out = _test_reduction(x, caffe_test_dir, operation=MEAN, axis=0)
    assert_op_correct(caffe_out, ref, op_name="Reduction(MEAN-1D)", atol=1e-4)

    logger.debug("Testing ASUM on 1D")
    x = np.random.randn(20).astype(np.float32)
    ref = _reduction_ref_1d(x, ASUM)
    caffe_out = _test_reduction(x, caffe_test_dir, operation=ASUM, axis=0)
    assert_op_correct(caffe_out, ref, op_name="Reduction(ASUM-1D)", atol=1e-4)

    logger.debug("Testing SUMSQ on 1D")
    x = np.random.randn(20).astype(np.float32)
    ref = _reduction_ref_1d(x, SUMSQ)
    caffe_out = _test_reduction(x, caffe_test_dir, operation=SUMSQ, axis=0)
    assert_op_correct(caffe_out, ref, op_name="Reduction(SUMSQ-1D)", atol=1e-4)

    logger.debug("Testing SUM with coeff on 1D")
    x = np.random.randn(15).astype(np.float32)
    coeff = 2.5
    ref = _reduction_ref_1d(x, SUM, coeff=coeff)
    caffe_out = _test_reduction(x, caffe_test_dir, operation=SUM, axis=0, coeff=coeff)
    assert_op_correct(caffe_out, ref, op_name="Reduction(SUM-coeff-1D)", atol=1e-4)

    logger.debug("Testing SUM on 2D axis=1 (last axis)")
    x = np.random.randn(4, 5).astype(np.float32)
    ref = _reduction_ref_axis(x, SUM, axis=1)
    caffe_out = _test_reduction(x, caffe_test_dir, operation=SUM, axis=1)
    assert_op_correct(caffe_out, ref, op_name="Reduction(SUM-2D-axis1)", atol=1e-4)

    logger.debug("Testing MEAN on 2D axis=1")
    x = np.random.randn(4, 5).astype(np.float32)
    ref = _reduction_ref_axis(x, MEAN, axis=1)
    caffe_out = _test_reduction(x, caffe_test_dir, operation=MEAN, axis=1)
    assert_op_correct(caffe_out, ref, op_name="Reduction(MEAN-2D-axis1)", atol=1e-4)


@pytest.mark.edge
def test_reduction_edge_cases(caffe_test_dir):
    """Reduction edge cases."""
    logger.info("Running test_reduction_edge_cases")

    logger.debug("Testing all zeros SUM")
    x = np.zeros(10, dtype=np.float32)
    caffe_out = _test_reduction(x, caffe_test_dir, operation=SUM, axis=0)
    assert np.allclose(caffe_out[0], 0.0, atol=1e-6)

    logger.debug("Testing all ones SUM")
    x = np.ones(10, dtype=np.float32)
    ref = _reduction_ref_1d(x, SUM)
    caffe_out = _test_reduction(x, caffe_test_dir, operation=SUM, axis=0)
    assert_op_correct(caffe_out, ref, op_name="Reduction(ones-SUM)", atol=1e-4)

    logger.debug("Testing all ones MEAN")
    x = np.ones(20, dtype=np.float32)
    ref = _reduction_ref_1d(x, MEAN)
    caffe_out = _test_reduction(x, caffe_test_dir, operation=MEAN, axis=0)
    assert_op_correct(caffe_out, ref, op_name="Reduction(ones-MEAN)", atol=1e-4)

    logger.debug("Testing coeff=0")
    x = np.random.randn(10).astype(np.float32)
    caffe_out = _test_reduction(x, caffe_test_dir, operation=SUM, axis=0, coeff=0.0)
    assert np.allclose(caffe_out[0], 0.0, atol=1e-6)

    logger.debug("Testing single element")
    x = np.array([42.0], dtype=np.float32)
    ref = _reduction_ref_1d(x, SUM)
    caffe_out = _test_reduction(x, caffe_test_dir, operation=SUM, axis=0)
    assert_op_correct(caffe_out, ref, op_name="Reduction(single-SUM)", atol=1e-4)
