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


def _test_deconvolution(data, test_dir, **kwargs):
    """One iteration of Deconvolution"""
    logger.info(f"Testing Deconvolution, input shape: {data.shape}")
    logger.debug(f"Deconvolution params: {kwargs}")
    return _test_op(data, L.Deconvolution, "Deconvolution", test_dir, **kwargs)


def test_forward_Deconvolution(caffe_test_dir):
    """Deconvolution"""
    logger.info("Running test_forward_Deconvolution")
    data = np.random.rand(1, 16, 32, 32).astype(np.float32)
    logger.debug("Testing Deconvolution with basic params")
    _test_deconvolution(
        data,
        caffe_test_dir,
        convolution_param=dict(
            num_output=20,
            bias_term=True,
            pad=0,
            kernel_size=3,
            stride=2,
            dilation=1,
            weight_filler=dict(type="xavier"),
            bias_filler=dict(type="xavier"),
        ),
    )
    logger.debug("Testing Deconvolution with pad=[1,2], bias_term=False")
    _test_deconvolution(
        data,
        caffe_test_dir,
        convolution_param=dict(
            num_output=20,
            bias_term=False,
            pad=[1, 2],
            kernel_size=3,
            stride=2,
            dilation=1,
            weight_filler=dict(type="xavier"),
            bias_filler=dict(type="xavier"),
        ),
    )
    logger.debug("Testing Deconvolution with explicit h/w params")
    _test_deconvolution(
        data,
        caffe_test_dir,
        convolution_param=dict(
            num_output=20,
            bias_term=True,
            pad_h=1,
            pad_w=2,
            kernel_h=3,
            kernel_w=5,
            stride_h=2,
            stride_w=1,
            dilation=1,
            weight_filler=dict(type="xavier"),
            bias_filler=dict(type="xavier"),
        ),
    )
    logger.debug("Testing Deconvolution with group=16")
    _test_deconvolution(
        data,
        caffe_test_dir,
        convolution_param=dict(
            num_output=16,
            bias_term=False,
            pad=0,
            kernel_size=2,
            stride=2,
            dilation=1,
            group=16,
            weight_filler=dict(type="xavier"),
            bias_filler=dict(type="xavier"),
        ),
    )
    data = np.random.rand(1, 100, 32, 32).astype(np.float32)
    logger.debug("Testing Deconvolution with group=100, 100 channels")
    _test_deconvolution(
        data,
        caffe_test_dir,
        convolution_param=dict(
            num_output=100,
            bias_term=False,
            pad=0,
            kernel_size=2,
            stride=2,
            dilation=1,
            group=100,
            weight_filler=dict(type="xavier"),
            bias_filler=dict(type="xavier"),
        ),
    )


@pytest.mark.correctness
def test_deconvolution_correctness(caffe_test_dir):
    """Deconvolution runs without crash and produces correct output shape."""
    logger.info("Running test_deconvolution_correctness")
    np.random.seed(42)

    logger.debug("Testing Deconvolution output shape: (1,16,4,4) -> (1,20,9,9) with kernel=3,stride=2,pad=0")
    x = np.random.randn(1, 16, 4, 4).astype(np.float32)
    caffe_out = _test_deconvolution(
        x,
        caffe_test_dir,
        convolution_param=dict(
            num_output=20,
            bias_term=True,
            pad=0,
            kernel_size=3,
            stride=2,
            dilation=1,
            weight_filler=dict(type="xavier"),
            bias_filler=dict(type="xavier"),
        ),
    )
    assert caffe_out[0].shape == (1, 20, 9, 9), f"Expected (1,20,9,9) got {caffe_out[0].shape}"

    logger.debug("Testing Deconvolution with pad=1, kernel=3, stride=1 - same padding")
    x = np.random.randn(1, 8, 8, 8).astype(np.float32)
    caffe_out = _test_deconvolution(
        x,
        caffe_test_dir,
        convolution_param=dict(
            num_output=16,
            bias_term=False,
            pad=1,
            kernel_size=3,
            stride=1,
            weight_filler=dict(type="xavier"),
        ),
    )
    assert caffe_out[0].shape == (1, 16, 8, 8), f"Expected (1,16,8,8) got {caffe_out[0].shape}"

    logger.debug("Testing Deconvolution with group convolution")
    x = np.random.randn(1, 4, 8, 8).astype(np.float32)
    caffe_out = _test_deconvolution(
        x,
        caffe_test_dir,
        convolution_param=dict(
            num_output=8,
            bias_term=True,
            pad=1,
            kernel_size=3,
            stride=1,
            group=2,
            weight_filler=dict(type="xavier"),
            bias_filler=dict(type="xavier"),
        ),
    )
    assert caffe_out[0].shape == (1, 8, 8, 8), f"Expected (1,8,8,8) got {caffe_out[0].shape}"


@pytest.mark.edge
def test_deconvolution_edge_cases(caffe_test_dir):
    """Deconvolution edge cases."""
    logger.info("Running test_deconvolution_edge_cases")

    logger.debug("Testing Deconvolution with all zeros input")
    x = np.zeros((1, 8, 4, 4), dtype=np.float32)
    caffe_out = _test_deconvolution(
        x,
        caffe_test_dir,
        convolution_param=dict(
            num_output=10,
            bias_term=True,
            pad=0,
            kernel_size=3,
            stride=1,
            weight_filler=dict(type="xavier"),
            bias_filler=dict(type="constant", value=0.5),
        ),
    )
    assert caffe_out[0].shape == (1, 10, 6, 6), f"Expected (1,10,6,6) got {caffe_out[0].shape}"

    logger.debug("Testing Deconvolution all ones input (runs without crash)")
    x = np.ones((1, 4, 4, 4), dtype=np.float32)
    caffe_out = _test_deconvolution(
        x,
        caffe_test_dir,
        convolution_param=dict(
            num_output=4,
            bias_term=False,
            pad=0,
            kernel_size=3,
            stride=1,
            weight_filler=dict(type="xavier"),
        ),
    )
    assert caffe_out[0].shape == (1, 4, 6, 6), f"Expected (1,4,6,6) got {caffe_out[0].shape}"

    logger.debug("Testing Deconvolution 1x1 kernel")
    x = np.random.randn(1, 8, 6, 6).astype(np.float32)
    caffe_out = _test_deconvolution(
        x,
        caffe_test_dir,
        convolution_param=dict(
            num_output=8,
            bias_term=False,
            pad=0,
            kernel_size=1,
            stride=1,
            weight_filler=dict(type="xavier"),
        ),
    )
    assert caffe_out[0].shape == (1, 8, 6, 6), f"Expected (1,8,6,6) got {caffe_out[0].shape}"

    logger.debug("Testing Deconvolution batch_size=1 single output channel")
    x = np.random.randn(1, 1, 4, 4).astype(np.float32)
    caffe_out = _test_deconvolution(
        x,
        caffe_test_dir,
        convolution_param=dict(
            num_output=1,
            bias_term=True,
            pad=0,
            kernel_size=2,
            stride=2,
            weight_filler=dict(type="xavier"),
            bias_filler=dict(type="xavier"),
        ),
    )
    assert caffe_out[0].shape == (1, 1, 8, 8), f"Expected (1,1,8,8) got {caffe_out[0].shape}"
