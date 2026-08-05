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


def _test_crop(data, test_dir, **kwargs):
    """One iteration of Crop"""
    input_shapes = [d.shape for d in data]
    logger.info(f"Testing Crop, num inputs: {len(data)}, shapes: {input_shapes}")
    logger.debug(f"Crop params: {kwargs}")
    return _test_op(data, L.Crop, "Crop", test_dir, **kwargs)


def test_forward_Crop(caffe_test_dir):
    """Crop"""
    logger.info("Running test_forward_Crop")
    logger.debug("Testing Crop 4D, default params")
    _test_crop([np.random.rand(10, 10, 120, 120), np.random.rand(10, 5, 50, 60)], caffe_test_dir)
    logger.debug("Testing Crop 4D, axis=1")
    _test_crop([np.random.rand(10, 10, 120, 120), np.random.rand(10, 5, 50, 60)], caffe_test_dir, axis=1)
    logger.debug("Testing Crop 4D, axis=1, offset=2")
    _test_crop([np.random.rand(10, 10, 120, 120), np.random.rand(10, 5, 50, 60)], caffe_test_dir, axis=1, offset=2)
    logger.debug("Testing Crop 4D, axis=1, offset=[1,2,4]")
    _test_crop(
        [np.random.rand(10, 10, 120, 120), np.random.rand(10, 5, 50, 60)], caffe_test_dir, axis=1, offset=[1, 2, 4]
    )
    logger.debug("Testing Crop 4D, axis=2, offset=[2,4]")
    _test_crop(
        [np.random.rand(10, 10, 120, 120), np.random.rand(10, 5, 50, 60)], caffe_test_dir, axis=2, offset=[2, 4]
    )
    logger.debug("Testing Crop 3D, axis=1, offset=[2,4]")
    _test_crop([np.random.rand(10, 120, 120), np.random.rand(5, 50, 60)], caffe_test_dir, axis=1, offset=[2, 4])
    logger.debug("Testing Crop 2D, axis=0, offset=[2,4]")
    _test_crop([np.random.rand(120, 120), np.random.rand(50, 60)], caffe_test_dir, axis=0, offset=[2, 4])


def _crop_ref(data, crop_to_shape, axis=2, offset=None):
    ndim = data.ndim
    if axis < 0:
        axis = ndim + axis
    if offset is None:
        offset = [0] * (ndim - axis)
    elif isinstance(offset, int):
        offset = [offset] * (ndim - axis)
    slicer = [slice(None)] * axis
    for i in range(ndim - axis):
        off = offset[i] if i < len(offset) else 0
        size = crop_to_shape[axis + i]
        slicer.append(slice(off, off + size))
    return data[tuple(slicer)].astype(np.float32)


@pytest.mark.correctness
def test_crop_correctness(caffe_test_dir):
    """Crop correctness test for simple explicit offset cases."""
    logger.info("Running test_crop_correctness")
    np.random.seed(42)

    logger.debug("Testing Crop 4D axis=2 offset=[2,3]")
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    crop_shape = (1, 3, 5, 6)
    crop_to = np.zeros(crop_shape, dtype=np.float32)
    axis = 2
    offset = [2, 3]
    ref = _crop_ref(x, crop_shape, axis=axis, offset=offset)
    caffe_out = _test_crop([x, crop_to], caffe_test_dir, axis=axis, offset=offset)
    assert caffe_out[0].shape == crop_shape
    assert_op_correct(caffe_out, ref, op_name="Crop(4D-axis2-offset)")

    logger.debug("Testing Crop 4D axis=1 offset=[1,0,0] (channel-only crop)")
    x = np.random.randn(1, 5, 8, 8).astype(np.float32)
    crop_shape = (1, 3, 8, 8)
    crop_to = np.zeros(crop_shape, dtype=np.float32)
    axis = 1
    offset = [1, 0, 0]
    ref = _crop_ref(x, crop_shape, axis=axis, offset=offset)
    caffe_out = _test_crop([x, crop_to], caffe_test_dir, axis=axis, offset=offset)
    assert caffe_out[0].shape == crop_shape
    assert_op_correct(caffe_out, ref, op_name="Crop(4D-axis1-offset1)")

    logger.debug("Testing Crop 4D axis=2 offset=[0,0] (identity crop for those dims)")
    x = np.random.randn(2, 3, 6, 6).astype(np.float32)
    crop_shape = (2, 3, 4, 4)
    crop_to = np.zeros(crop_shape, dtype=np.float32)
    axis = 2
    offset = [1, 1]
    ref = _crop_ref(x, crop_shape, axis=axis, offset=offset)
    caffe_out = _test_crop([x, crop_to], caffe_test_dir, axis=axis, offset=offset)
    assert caffe_out[0].shape == crop_shape
    assert_op_correct(caffe_out, ref, op_name="Crop(4D-axis2-center)")

    logger.debug("Testing Crop 3D axis=1 offset=[1,2]")
    x = np.random.randn(4, 10, 10).astype(np.float32)
    crop_shape = (4, 6, 5)
    crop_to = np.zeros(crop_shape, dtype=np.float32)
    axis = 1
    offset = [1, 2]
    ref = _crop_ref(x, crop_shape, axis=axis, offset=offset)
    caffe_out = _test_crop([x, crop_to], caffe_test_dir, axis=axis, offset=offset)
    assert caffe_out[0].shape == crop_shape
    assert_op_correct(caffe_out, ref, op_name="Crop(3D-axis1)")


@pytest.mark.edge
def test_crop_edge_cases(caffe_test_dir):
    """Crop edge cases."""
    logger.info("Running test_crop_edge_cases")

    logger.debug("Testing all zeros crop")
    x = np.zeros((1, 3, 8, 8), dtype=np.float32)
    crop_shape = (1, 3, 4, 4)
    crop_to = np.zeros(crop_shape, dtype=np.float32)
    caffe_out = _test_crop([x, crop_to], caffe_test_dir, axis=2, offset=[2, 2])
    assert caffe_out[0].shape == crop_shape
    assert np.all(caffe_out[0] == 0)

    logger.debug("Testing all ones crop")
    x = np.ones((2, 4, 10, 10), dtype=np.float32)
    crop_shape = (2, 2, 5, 5)
    crop_to = np.zeros(crop_shape, dtype=np.float32)
    caffe_out = _test_crop([x, crop_to], caffe_test_dir, axis=1, offset=[1, 2, 3])
    assert caffe_out[0].shape == crop_shape
    assert np.all(caffe_out[0] == 1.0)

    logger.debug("Testing crop at origin (offset=0)")
    x = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    crop_shape = (3, 2, 2)
    crop_to = np.zeros(crop_shape, dtype=np.float32)
    caffe_out = _test_crop([x, crop_to], caffe_test_dir, axis=1, offset=[0, 0])
    assert caffe_out[0].shape == crop_shape
    ref = _crop_ref(x, crop_shape, axis=1, offset=[0, 0])
    assert_op_correct(caffe_out, ref, op_name="Crop(origin)")
