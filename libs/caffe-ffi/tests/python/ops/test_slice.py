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


def _test_slice(data, test_dir, **kwargs):
    """One iteration of Slice"""
    logger.info(f"Testing Slice, input shape: {data.shape}")
    logger.debug(f"Slice params: {kwargs}")
    return _test_op(data, L.Slice, "Slice", test_dir, **kwargs)


def test_forward_Slice(caffe_test_dir):
    """Slice"""
    logger.info("Running test_forward_Slice")
    data = np.random.rand(1, 3, 10, 10).astype(np.float32)
    logger.debug(f"Testing Slice ntop=2, axis=1, slice_point=[1], shape: {data.shape}")
    _test_slice(data, caffe_test_dir, ntop=2, slice_param=dict(axis=1, slice_point=[1]))
    logger.debug(f"Testing Slice ntop=2, axis=-1, slice_point=[1], shape: {data.shape}")
    _test_slice(data, caffe_test_dir, ntop=2, slice_param=dict(axis=-1, slice_point=[1]))
    logger.debug(f"Testing Slice ntop=3, axis=2, slice_point=[1,6], shape: {data.shape}")
    _test_slice(data, caffe_test_dir, ntop=3, slice_param=dict(axis=2, slice_point=[1, 6]))
    logger.debug(f"Testing Slice ntop=3 default params, shape: {data.shape}")
    _test_slice(data, caffe_test_dir, ntop=3)


def _slice_ref(x, axis=1, slice_point=None, ntop=2):
    if axis < 0:
        axis = x.ndim + axis
    if slice_point is None:
        dim_size = x.shape[axis]
        slice_size = dim_size // ntop
        slice_point = [slice_size * (i + 1) for i in range(ntop - 1)]
    slices = []
    prev = 0
    slices_list = []
    for sp in slice_point:
        slicer = [slice(None)] * x.ndim
        slicer[axis] = slice(prev, sp)
        slices_list.append(tuple(slicer))
        prev = sp
    slicer = [slice(None)] * x.ndim
    slicer[axis] = slice(prev, None)
    slices_list.append(tuple(slicer))
    results = [x[s].astype(np.float32) for s in slices_list]
    return results


@pytest.mark.correctness
def test_slice_correctness(caffe_test_dir):
    """Slice correctness test with numpy slicing reference."""
    logger.info("Running test_slice_correctness")
    np.random.seed(42)

    logger.debug("Testing Slice ntop=2 axis=1 slice_point=[1] on 4D")
    x = np.random.randn(1, 3, 4, 4).astype(np.float32)
    ref = _slice_ref(x, axis=1, slice_point=[1], ntop=2)
    caffe_out = _test_slice(x, caffe_test_dir, ntop=2, slice_param=dict(axis=1, slice_point=[1]))
    assert len(caffe_out) == len(ref) == 2
    assert_op_correct(caffe_out[0], ref[0], op_name="Slice(4D-2out-part0)")
    assert_op_correct(caffe_out[1], ref[1], op_name="Slice(4D-2out-part1)")
    assert caffe_out[0].shape == (1, 1, 4, 4)
    assert caffe_out[1].shape == (1, 2, 4, 4)

    logger.debug("Testing Slice ntop=2 axis=0 slice_point=[2] on 3D")
    x = np.random.randn(4, 3, 5).astype(np.float32)
    ref = _slice_ref(x, axis=0, slice_point=[2], ntop=2)
    caffe_out = _test_slice(x, caffe_test_dir, ntop=2, slice_param=dict(axis=0, slice_point=[2]))
    assert len(caffe_out) == len(ref) == 2
    assert_op_correct(caffe_out[0], ref[0], op_name="Slice(3D-axis0-part0)")
    assert_op_correct(caffe_out[1], ref[1], op_name="Slice(3D-axis0-part1)")
    assert caffe_out[0].shape == (2, 3, 5)
    assert caffe_out[1].shape == (2, 3, 5)

    logger.debug("Testing Slice ntop=3 axis=2 slice_point=[1,6] on 4D")
    x = np.random.randn(1, 2, 8, 3).astype(np.float32)
    ref = _slice_ref(x, axis=2, slice_point=[1, 6], ntop=3)
    caffe_out = _test_slice(x, caffe_test_dir, ntop=3, slice_param=dict(axis=2, slice_point=[1, 6]))
    assert len(caffe_out) == len(ref) == 3
    for i in range(3):
        assert_op_correct(caffe_out[i], ref[i], op_name=f"Slice(3out-part{i})")
    assert caffe_out[0].shape[2] == 1
    assert caffe_out[1].shape[2] == 5
    assert caffe_out[2].shape[2] == 2

    logger.debug("Testing Slice ntop=2 axis=-1 (last dim)")
    x = np.random.randn(2, 3, 4, 6).astype(np.float32)
    ref = _slice_ref(x, axis=-1, slice_point=[3], ntop=2)
    caffe_out = _test_slice(x, caffe_test_dir, ntop=2, slice_param=dict(axis=-1, slice_point=[3]))
    assert len(caffe_out) == len(ref) == 2
    assert_op_correct(caffe_out[0], ref[0], op_name="Slice(axis-1-part0)")
    assert_op_correct(caffe_out[1], ref[1], op_name="Slice(axis-1-part1)")


@pytest.mark.edge
def test_slice_edge_cases(caffe_test_dir):
    """Slice edge cases."""
    logger.info("Running test_slice_edge_cases")

    logger.debug("Testing all zeros slice")
    x = np.zeros((1, 4, 3, 3), dtype=np.float32)
    caffe_out = _test_slice(x, caffe_test_dir, ntop=2, slice_param=dict(axis=1, slice_point=[2]))
    assert len(caffe_out) == 2
    assert caffe_out[0].shape == (1, 2, 3, 3)
    assert caffe_out[1].shape == (1, 2, 3, 3)
    assert np.all(caffe_out[0] == 0)
    assert np.all(caffe_out[1] == 0)

    logger.debug("Testing all ones slice")
    x = np.ones((2, 4, 2, 2), dtype=np.float32)
    caffe_out = _test_slice(x, caffe_test_dir, ntop=2, slice_param=dict(axis=1, slice_point=[1]))
    assert len(caffe_out) == 2
    assert caffe_out[0].shape == (2, 1, 2, 2)
    assert caffe_out[1].shape == (2, 3, 2, 2)
    assert np.all(caffe_out[0] == 1.0)
    assert np.all(caffe_out[1] == 1.0)

    logger.debug("Testing slice at boundary (first element)")
    x = np.random.randn(1, 3, 2, 2).astype(np.float32)
    caffe_out = _test_slice(x, caffe_test_dir, ntop=2, slice_param=dict(axis=1, slice_point=[1]))
    ref = _slice_ref(x, axis=1, slice_point=[1], ntop=2)
    assert_op_correct(caffe_out[0], ref[0], op_name="Slice(boundary1-part0)")
    assert caffe_out[0].shape == (1, 1, 2, 2)
