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


def _test_inner_product(data, test_dir, **kwargs):
    """One iteration of InnerProduct, returns Caffe output."""
    logger.info(f"Testing InnerProduct, input shape: {data.shape}")
    logger.debug(f"InnerProduct params: {kwargs}")
    return _test_op(data, L.InnerProduct, "InnerProduct", test_dir, **kwargs)


# ──────────────────────────────────────────────
# Original forward tests (3 basic cases)
# ──────────────────────────────────────────────


@pytest.mark.correctness
def test_forward_InnerProduct(caffe_test_dir):
    """InnerProduct — basic forward tests from original suite."""
    logger.info("Running test_forward_InnerProduct (basic forward tests)")
    data = np.random.rand(1, 3, 10, 10)
    logger.debug(f"Testing basic InnerProduct, bias_term=False, shape: {data.shape}")
    _test_inner_product(data, caffe_test_dir, num_output=20, bias_term=False, weight_filler=dict(type="xavier"))
    logger.debug(f"Testing basic InnerProduct, bias_term=True, shape: {data.shape}")
    _test_inner_product(
        data,
        caffe_test_dir,
        num_output=20,
        bias_term=True,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )
    data2 = np.random.rand(20, 10).astype(np.float32)
    logger.debug(f"Testing 2D InnerProduct, bias_term=True, shape: {data2.shape}")
    _test_inner_product(
        data2,
        caffe_test_dir,
        num_output=30,
        bias_term=True,
        weight_filler=dict(type="xavier"),
        bias_filler=dict(type="xavier"),
    )


# ──────────────────────────────────────────────
# Boundary: input dimensionality
# ──────────────────────────────────────────────


@pytest.mark.edge
class TestInnerProductDimensionality:
    """Test InnerProduct with various input tensor dimensions."""

    def test_1d_input(self, caffe_test_dir):
        """1D input (feature vector only, no batch dim) — axis=0 flattens everything."""
        logger.info("Running TestInnerProductDimensionality.test_1d_input")
        data = np.random.rand(128).astype(np.float32)
        logger.debug(f"1D input, shape: {data.shape}, axis=0")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=10,
            axis=0,
            bias_term=False,
            weight_filler=dict(type="xavier"),
        )
        assert out[0].shape == (10,), f"Expected (10,) got {out[0].shape}"

    def test_3d_input(self, caffe_test_dir):
        """3D input (N, C, L) with default axis=1."""
        logger.info("Running TestInnerProductDimensionality.test_3d_input")
        data = np.random.rand(4, 16, 8).astype(np.float32)
        logger.debug(f"3D input, shape: {data.shape}, default axis=1")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=32,
            bias_term=True,
            weight_filler=dict(type="xavier"),
            bias_filler=dict(type="xavier"),
        )
        assert out[0].shape == (4, 32), f"Expected (4, 32) got {out[0].shape}"

    def test_5d_input(self, caffe_test_dir):
        """5D input (N, C, D, H, W) with default axis=1."""
        logger.info("Running TestInnerProductDimensionality.test_5d_input")
        data = np.random.rand(2, 3, 4, 5, 6).astype(np.float32)
        logger.debug(f"5D input, shape: {data.shape}, default axis=1")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=64,
            bias_term=False,
            weight_filler=dict(type="xavier"),
        )
        assert out[0].shape == (2, 64), f"Expected (2, 64) got {out[0].shape}"

    def test_3d_input_axis2(self, caffe_test_dir):
        """3D input with axis=2 — only last dimension is flattened."""
        logger.info("Running TestInnerProductDimensionality.test_3d_input_axis2")
        data = np.random.rand(4, 16, 8).astype(np.float32)
        logger.debug(f"3D input, shape: {data.shape}, axis=2")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=32,
            axis=2,
            bias_term=False,
            weight_filler=dict(type="xavier"),
        )
        assert out[0].shape == (4, 16, 32), f"Expected (4, 16, 32) got {out[0].shape}"

    def test_4d_input_axis2(self, caffe_test_dir):
        """4D input (N, C, H, W) with axis=2 — flatten H*W, keep N*C as M_."""
        logger.info("Running TestInnerProductDimensionality.test_4d_input_axis2")
        data = np.random.rand(2, 3, 4, 5).astype(np.float32)
        logger.debug(f"4D input, shape: {data.shape}, axis=2")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=10,
            axis=2,
            bias_term=False,
            weight_filler=dict(type="xavier"),
        )
        assert out[0].shape == (2, 3, 10), f"Expected (2, 3, 10) got {out[0].shape}"

    def test_4d_input_axis3(self, caffe_test_dir):
        """4D input (N, C, H, W) with axis=3 — only W is the feature dim."""
        logger.info("Running TestInnerProductDimensionality.test_4d_input_axis3")
        data = np.random.rand(2, 3, 4, 5).astype(np.float32)
        logger.debug(f"4D input, shape: {data.shape}, axis=3")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=10,
            axis=3,
            bias_term=False,
            weight_filler=dict(type="xavier"),
        )
        assert out[0].shape == (2, 3, 4, 10), f"Expected (2, 3, 4, 10) got {out[0].shape}"


# ──────────────────────────────────────────────
# Boundary: axis parameter
# ──────────────────────────────────────────────


@pytest.mark.edge
class TestInnerProductAxis:
    """Test the axis parameter including negative indices and axis=0."""

    def test_axis0_flattens_all(self, caffe_test_dir):
        """axis=0 flattens the entire input into a single vector (M_=1)."""
        logger.info("Running TestInnerProductAxis.test_axis0_flattens_all")
        data = np.random.rand(3, 4, 5).astype(np.float32)
        logger.debug(f"3D input, shape: {data.shape}, axis=0")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=10,
            axis=0,
            bias_term=False,
            weight_filler=dict(type="xavier"),
        )
        assert out[0].shape == (10,), f"Expected (10,) got {out[0].shape}"

    def test_negative_axis_last(self, caffe_test_dir):
        """axis=-1 indexes the last dimension (equivalent to axis=ndim-1)."""
        logger.info("Running TestInnerProductAxis.test_negative_axis_last")
        data = np.random.rand(2, 3, 10).astype(np.float32)
        logger.debug(f"3D input, shape: {data.shape}, axis=2 (positive)")
        out_pos = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=5,
            axis=2,
            bias_term=False,
            weight_filler=dict(type="xavier"),
        )
        logger.debug(f"3D input, shape: {data.shape}, axis=-1 (negative)")
        out_neg = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=5,
            axis=-1,
            bias_term=False,
            weight_filler=dict(type="xavier"),
        )
        assert out_pos[0].shape == (2, 3, 5)
        assert out_neg[0].shape == (2, 3, 5), f"axis=-1 should equal axis=2 for 3D input, got {out_neg[0].shape}"

    def test_negative_axis_second_last(self, caffe_test_dir):
        """axis=-2 on 4D input is equivalent to axis=2."""
        logger.info("Running TestInnerProductAxis.test_negative_axis_second_last")
        data = np.random.rand(2, 3, 4, 5).astype(np.float32)
        logger.debug(f"4D input, shape: {data.shape}, axis=-2")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=7,
            axis=-2,
            bias_term=False,
            weight_filler=dict(type="xavier"),
        )
        assert out[0].shape == (2, 3, 7), f"Expected (2, 3, 7) got {out[0].shape}"

    def test_negative_axis_default(self, caffe_test_dir):
        """axis=1 on 4D should match axis=-3 on same shape."""
        logger.info("Running TestInnerProductAxis.test_negative_axis_default")
        data = np.random.rand(1, 3, 10, 10).astype(np.float32)
        logger.debug(f"4D input, shape: {data.shape}, axis=1 (positive)")
        out_pos = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=20,
            axis=1,
            bias_term=False,
            weight_filler=dict(type="xavier"),
        )
        logger.debug(f"4D input, shape: {data.shape}, axis=-3 (negative)")
        out_neg = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=20,
            axis=-3,
            bias_term=False,
            weight_filler=dict(type="xavier"),
        )
        assert out_pos[0].shape == out_neg[0].shape == (1, 20)


# ──────────────────────────────────────────────
# Boundary: transpose parameter
# ──────────────────────────────────────────────


@pytest.mark.edge
class TestInnerProductTranspose:
    """Test transpose=True weight layout."""

    def test_transpose_2d(self, caffe_test_dir):
        """transpose=True with 2D input should produce same output shape."""
        logger.info("Running TestInnerProductTranspose.test_transpose_2d")
        data = np.random.rand(8, 16).astype(np.float32)
        logger.debug(f"2D input, shape: {data.shape}, transpose=True")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=32,
            transpose=True,
            bias_term=False,
            weight_filler=dict(type="xavier"),
        )
        assert out[0].shape == (8, 32), f"Expected (8, 32) got {out[0].shape}"

    def test_transpose_4d(self, caffe_test_dir):
        """transpose=True with 4D input and default axis=1."""
        logger.info("Running TestInnerProductTranspose.test_transpose_4d")
        data = np.random.rand(2, 3, 4, 5).astype(np.float32)
        logger.debug(f"4D input, shape: {data.shape}, transpose=True, bias_term=True")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=10,
            transpose=True,
            bias_term=True,
            weight_filler=dict(type="xavier"),
            bias_filler=dict(type="xavier"),
        )
        assert out[0].shape == (2, 10), f"Expected (2, 10) got {out[0].shape}"

    def test_transpose_with_axis2(self, caffe_test_dir):
        """transpose=True combined with non-default axis."""
        logger.info("Running TestInnerProductTranspose.test_transpose_with_axis2")
        data = np.random.rand(4, 6, 8).astype(np.float32)
        logger.debug(f"3D input, shape: {data.shape}, transpose=True, axis=2")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=12,
            axis=2,
            transpose=True,
            bias_term=False,
            weight_filler=dict(type="xavier"),
        )
        assert out[0].shape == (4, 6, 12), f"Expected (4, 6, 12) got {out[0].shape}"


# ──────────────────────────────────────────────
# Boundary: num_output extremes
# ──────────────────────────────────────────────


@pytest.mark.edge
class TestInnerProductNumOutput:
    """Test extreme num_output values."""

    def test_num_output_1(self, caffe_test_dir):
        """Minimum num_output=1 (single output neuron)."""
        logger.info("Running TestInnerProductNumOutput.test_num_output_1")
        data = np.random.rand(4, 10).astype(np.float32)
        logger.debug(f"2D input, shape: {data.shape}, num_output=1, bias_term=True")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=1,
            bias_term=True,
            weight_filler=dict(type="xavier"),
            bias_filler=dict(type="xavier"),
        )
        assert out[0].shape == (4, 1), f"Expected (4, 1) got {out[0].shape}"

    def test_num_output_1_no_bias(self, caffe_test_dir):
        """num_output=1 without bias."""
        logger.info("Running TestInnerProductNumOutput.test_num_output_1_no_bias")
        data = np.random.rand(2, 3, 4, 5).astype(np.float32)
        logger.debug(f"4D input, shape: {data.shape}, num_output=1, bias_term=False")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=1,
            bias_term=False,
            weight_filler=dict(type="xavier"),
        )
        assert out[0].shape == (2, 1), f"Expected (2, 1) got {out[0].shape}"

    def test_large_num_output(self, caffe_test_dir):
        """Larger num_output to stress weight allocation."""
        logger.info("Running TestInnerProductNumOutput.test_large_num_output")
        data = np.random.rand(2, 64).astype(np.float32)
        logger.debug(f"2D input, shape: {data.shape}, num_output=256, bias_term=True")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=256,
            bias_term=True,
            weight_filler=dict(type="xavier"),
            bias_filler=dict(type="xavier"),
        )
        assert out[0].shape == (2, 256), f"Expected (2, 256) got {out[0].shape}"


# ──────────────────────────────────────────────
# Boundary: batch size extremes
# ──────────────────────────────────────────────


@pytest.mark.edge
class TestInnerProductBatchSize:
    """Test extreme batch sizes."""

    def test_batch_size_1(self, caffe_test_dir):
        """Single sample (batch size 1) — common inference edge case."""
        logger.info("Running TestInnerProductBatchSize.test_batch_size_1")
        data = np.random.rand(1, 100).astype(np.float32)
        logger.debug(f"2D input, shape: {data.shape}, batch_size=1, num_output=50")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=50,
            bias_term=True,
            weight_filler=dict(type="xavier"),
            bias_filler=dict(type="xavier"),
        )
        assert out[0].shape == (1, 50), f"Expected (1, 50) got {out[0].shape}"

    def test_batch_size_1_4d(self, caffe_test_dir):
        """Batch size 1 with 4D feature map input."""
        logger.info("Running TestInnerProductBatchSize.test_batch_size_1_4d")
        data = np.random.rand(1, 3, 224, 224).astype(np.float32)
        logger.debug(f"4D input, shape: {data.shape}, batch_size=1, num_output=1000")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=1000,
            bias_term=True,
            weight_filler=dict(type="xavier"),
            bias_filler=dict(type="xavier"),
        )
        assert out[0].shape == (1, 1000), f"Expected (1, 1000) got {out[0].shape}"

    def test_large_batch(self, caffe_test_dir):
        """Larger batch size to test batched GEMM."""
        logger.info("Running TestInnerProductBatchSize.test_large_batch")
        data = np.random.rand(64, 32).astype(np.float32)
        logger.debug(f"2D input, shape: {data.shape}, batch_size=64, num_output=16")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=16,
            bias_term=False,
            weight_filler=dict(type="xavier"),
        )
        assert out[0].shape == (64, 16), f"Expected (64, 16) got {out[0].shape}"


# ──────────────────────────────────────────────
# Boundary: bias and filler combinations
# ──────────────────────────────────────────────


@pytest.mark.edge
class TestInnerProductBiasAndFillers:
    """Test bias_term and weight_filler/bias_filler combinations."""

    def test_no_bias_default_filler(self, caffe_test_dir):
        """bias_term=False without explicit weight_filler (uses default filler)."""
        logger.info("Running TestInnerProductBiasAndFillers.test_no_bias_default_filler")
        data = np.random.rand(4, 10).astype(np.float32)
        logger.debug(f"2D input, shape: {data.shape}, bias_term=False, default filler")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=5,
            bias_term=False,
        )
        assert out[0].shape == (4, 5), f"Expected (4, 5) got {out[0].shape}"

    def test_bias_default_filler(self, caffe_test_dir):
        """bias_term=True without explicit fillers."""
        logger.info("Running TestInnerProductBiasAndFillers.test_bias_default_filler")
        data = np.random.rand(4, 10).astype(np.float32)
        logger.debug(f"2D input, shape: {data.shape}, bias_term=True, default filler")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=5,
            bias_term=True,
        )
        assert out[0].shape == (4, 5), f"Expected (4, 5) got {out[0].shape}"

    def test_gaussian_filler(self, caffe_test_dir):
        """Gaussian weight filler."""
        logger.info("Running TestInnerProductBiasAndFillers.test_gaussian_filler")
        data = np.random.rand(4, 10).astype(np.float32)
        logger.debug(f"2D input, shape: {data.shape}, gaussian filler, constant bias")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=5,
            bias_term=True,
            weight_filler=dict(type="gaussian", std=0.01),
            bias_filler=dict(type="constant", value=0.1),
        )
        assert out[0].shape == (4, 5), f"Expected (4, 5) got {out[0].shape}"

    def test_constant_filler_zero_weights(self, caffe_test_dir):
        """Constant filler with value=0 for weights — output should equal bias."""
        logger.info("Running TestInnerProductBiasAndFillers.test_constant_filler_zero_weights")
        data = np.random.rand(2, 3).astype(np.float32)
        logger.debug(f"2D input, shape: {data.shape}, zero weights, bias=1.0")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=4,
            bias_term=True,
            weight_filler=dict(type="constant", value=0.0),
            bias_filler=dict(type="constant", value=1.0),
        )
        assert out[0].shape == (2, 4), f"Expected (2, 4) got {out[0].shape}"
        np.testing.assert_allclose(out[0], np.ones((2, 4), dtype=np.float32), rtol=1e-5, atol=1e-5)

    def test_constant_filler_zero_bias(self, caffe_test_dir):
        """Constant filler with value=1 weights and value=0 bias: output = sum of inputs per sample."""
        logger.info("Running TestInnerProductBiasAndFillers.test_constant_filler_zero_bias")
        data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        logger.debug(f"2D input, shape: {data.shape}, weights=1.0, zero bias")
        out = _test_inner_product(
            data,
            caffe_test_dir,
            num_output=1,
            bias_term=True,
            weight_filler=dict(type="constant", value=1.0),
            bias_filler=dict(type="constant", value=0.0),
        )
        assert out[0].shape == (1, 1), f"Expected (1, 1) got {out[0].shape}"
        expected_sum = float(np.sum(data))
        np.testing.assert_allclose(out[0], np.array([[expected_sum]], dtype=np.float32), rtol=1e-5, atol=1e-5)
