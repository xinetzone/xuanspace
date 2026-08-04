---
title: CMake 重构 WSL 回归测试详细日志
date: 2026-08-04
category: caffe-ffi
task_type: test-log
tags: [caffe-ffi, cmake, regression, wsl, pytest]
status: verified
source: "caffe-ffi-tvm-integration/tasks.md#Task18"
---

# CMake 重构 WSL 回归测试详细日志

> 本文件为 CMake 原子化重构后，在 WSL docker 环境中全量 P3 回归测试的详细逐用例日志归档。

## 一、测试环境

| 项 | 值 |
|----|----|
| 容器 | caffe-ffi-jupyter (docker) |
| conda 环境 | caffe-ffi |
| Python | 3.14.6 (pytest-9.1.1, pluggy-1.6.0) |
| 构建工具 | cmake 4.4.1 / ninja 1.13.2 / gcc 14.3.0 |
| 运行目录 | /SpecWeave/projects/xuanspace/libs/caffe-ffi |
| 环境变量 | CAFFE_FFI_ENABLE_COW=1, CAFFE_FFI_ENABLE_COW_PHASE3=1 |
| 命令 | `pytest tests/python -v` |

## 二、结果汇总

| 指标 | 数值 |
|------|------|
| 收集用例 | 1647 |
| ✅ 通过 (PASSED) | 1646 |
| ⏭️ 跳过 (SKIPPED) | 1 |
| ❌ 失败 (FAILED) | 0 |
| ⚠️ 错误 (ERROR) | 0 |
| 结果 | **1646 passed, 1 skipped, 0 failures** |

## 三、按测试文件统计

| 测试文件 | 通过 | 跳过 | 失败 | 小计 |
|---------|:---:|:---:|:---:|:---:|
| test_activation_backward.py | 33 | 0 | 0 | 33 |
| test_batch_norm_backward.py | 11 | 0 | 0 | 11 |
| test_bias_backward.py | 19 | 0 | 0 | 19 |
| test_blob.py | 117 | 0 | 0 | 117 |
| test_complex_topologies.py | 28 | 0 | 0 | 28 |
| test_concat_backward.py | 24 | 0 | 0 | 24 |
| test_conv_backward.py | 30 | 0 | 0 | 30 |
| test_cow.py | 21 | 0 | 0 | 21 |
| test_crop_backward.py | 19 | 0 | 0 | 19 |
| test_deconv_backward.py | 10 | 0 | 0 | 10 |
| test_dropout_backward.py | 20 | 0 | 0 | 20 |
| test_eltwise_backward.py | 32 | 0 | 0 | 32 |
| test_elu_kink_stability.py | 24 | 0 | 0 | 24 |
| test_extreme_boundaries.py | 11 | 0 | 0 | 11 |
| test_extreme_inputs.py | 30 | 0 | 0 | 30 |
| test_ffi_set_shape_only.py | 17 | 0 | 0 | 17 |
| test_flatten_backward.py | 243 | 0 | 0 | 243 |
| test_grad_check_utils_selftest.py | 10 | 0 | 0 | 10 |
| test_inner_product_backward.py | 23 | 0 | 0 | 23 |
| test_insert_splits.py | 18 | 0 | 0 | 18 |
| test_layer_template_three_layer_validation.py | 26 | 0 | 0 | 26 |
| test_layers.py | 63 | 0 | 0 | 63 |
| test_lrn_backward.py | 13 | 0 | 0 | 13 |
| test_net.py | 67 | 1 | 0 | 68 |
| test_p2b_regression.py | 29 | 0 | 0 | 29 |
| test_p3a_conv_pool_bn.py | 24 | 0 | 0 | 24 |
| test_p3b_eltwise_scale.py | 50 | 0 | 0 | 50 |
| test_p3c_activations_ip.py | 68 | 0 | 0 | 68 |
| test_p3c_transformer.py | 13 | 0 | 0 | 13 |
| test_p3d_all_layers_e2e.py | 8 | 0 | 0 | 8 |
| test_p3d_slice_crop_deconv_lrn.py | 21 | 0 | 0 | 21 |
| test_phase3_log_aggregation.py | 12 | 0 | 0 | 12 |
| test_phase3_set_shape_only.py | 20 | 0 | 0 | 20 |
| test_pooling_backward.py | 28 | 0 | 0 | 28 |
| test_python_api.py | 66 | 0 | 0 | 66 |
| test_reshape_backward.py | 291 | 0 | 0 | 291 |
| test_scale_backward.py | 25 | 0 | 0 | 25 |
| test_slice_backward.py | 20 | 0 | 0 | 20 |
| test_softmax_backward.py | 22 | 0 | 0 | 22 |
| test_softmax_loss_backward.py | 12 | 0 | 0 | 12 |
| test_split_backward.py | 17 | 0 | 0 | 17 |
| test_split_concat_bench.py | 4 | 0 | 0 | 4 |
| test_split_topologies.py | 7 | 0 | 0 | 7 |

## 四、详细逐用例日志

> 按测试文件分组，列出每个用例及其结果。

### `test_activation_backward.py`

| 用例 | 结果 |
|------|------|
| `TestReLUGradient::test_relu_backward_analytical_positive` | ✅ PASSED |
| `TestReLUGradient::test_relu_backward_analytical_negative_slope` | ✅ PASSED |
| `TestReLUGradient::test_relu_backward_dead_neuron_zero` | ✅ PASSED |
| `TestReLUGradient::test_relu_backward_arbitrary_dy` | ✅ PASSED |
| `TestReLUGradient::test_relu_numerical_gradient` | ✅ PASSED |
| `TestReLUGradient::test_relu_numerical_gradient_mixed_signs` | ✅ PASSED |
| `TestReLUGradient::test_relu_backward_runs_without_crash` | ✅ PASSED |
| `TestSigmoidGradient::test_sigmoid_backward_analytical` | ✅ PASSED |
| `TestSigmoidGradient::test_sigmoid_backward_zero_input` | ✅ PASSED |
| `TestSigmoidGradient::test_sigmoid_backward_saturation_small` | ✅ PASSED |
| `TestSigmoidGradient::test_sigmoid_backward_saturation_large` | ✅ PASSED |
| `TestSigmoidGradient::test_sigmoid_numerical_gradient` | ✅ PASSED |
| `TestTanHGradient::test_tanh_backward_analytical` | ✅ PASSED |
| `TestTanHGradient::test_tanh_backward_zero_input` | ✅ PASSED |
| `TestTanHGradient::test_tanh_backward_saturation` | ✅ PASSED |
| `TestTanHGradient::test_tanh_backward_symmetry` | ✅ PASSED |
| `TestTanHGradient::test_tanh_numerical_gradient` | ✅ PASSED |
| `TestELUGradient::test_elu_backward_analytical` | ✅ PASSED |
| `TestELUGradient::test_elu_backward_positive_passthrough` | ✅ PASSED |
| `TestELUGradient::test_elu_backward_alpha05` | ✅ PASSED |
| `TestELUGradient::test_elu_backward_negative_saturation` | ✅ PASSED |
| `TestELUGradient::test_elu_numerical_gradient` | ✅ PASSED |
| `TestPReLUGradient::test_prelu_shared_analytical` | ✅ PASSED |
| `TestPReLUGradient::test_prelu_per_channel_analytical` | ✅ PASSED |
| `TestPReLUGradient::test_prelu_shared_dead_neuron_scaled` | ✅ PASSED |
| `TestPReLUGradient::test_prelu_shared_numerical_gradient` | ✅ PASSED |
| `TestPReLUGradient::test_prelu_slope_diff_shape_shared` | ✅ PASSED |
| `TestPReLUGradient::test_prelu_slope_diff_shape_per_channel` | ✅ PASSED |
| `TestActivationPerfLogs::test_backward_no_crash[ReLU]` | ✅ PASSED |
| `TestActivationPerfLogs::test_backward_no_crash[Sigmoid]` | ✅ PASSED |
| `TestActivationPerfLogs::test_backward_no_crash[TanH]` | ✅ PASSED |
| `TestActivationPerfLogs::test_backward_no_crash[ELU]` | ✅ PASSED |
| `TestActivationPerfLogs::test_prelu_backward_no_crash` | ✅ PASSED |

### `test_batch_norm_backward.py`

| 用例 | 结果 |
|------|------|
| `TestBatchNormBackward::test_bn_backward_known_values` | ✅ PASSED |
| `TestBatchNormBackward::test_bn_backward_analytical_dx` | ✅ PASSED |
| `TestBatchNormBackward::test_bn_numerical_gradient_dx` | ✅ PASSED |
| `TestBatchNormBackward::test_bn_backward_zero_dy_gives_zero_grads` | ✅ PASSED |
| `TestBatchNormBackward::test_bn_backward_shapes` | ✅ PASSED |
| `TestBatchNormBackward::test_bn_backward_deterministic` | ✅ PASSED |
| `TestBatchNormBackward::test_bn_backward_preserves_forward_output` | ✅ PASSED |
| `TestBatchNormBackward::test_bn_backward_eps_effect` | ✅ PASSED |
| `TestBatchNormBackwardMultiChannel::test_bn_per_channel_scaling` | ✅ PASSED |
| `TestBatchNormBackwardScaleFactor::test_bn_scale_factor_count` | ✅ PASSED |
| `TestBatchNormBackwardScaleFactor::test_bn_scale_factor_numerical` | ✅ PASSED |

### `test_bias_backward.py`

| 用例 | 结果 |
|------|------|
| `TestBiasBackwardKnownValues::test_forward_zero_bias` | ✅ PASSED |
| `TestBiasBackwardKnownValues::test_forward_known_bias` | ✅ PASSED |
| `TestBiasBackwardKnownValues::test_backward_dx_passes_through` | ✅ PASSED |
| `TestBiasBackwardKnownValues::test_backward_dbias_known_values` | ✅ PASSED |
| `TestBiasBackwardAnalytical::test_dx_vs_numpy[2-4]` | ✅ PASSED |
| `TestBiasBackwardAnalytical::test_dx_vs_numpy[4-8]` | ✅ PASSED |
| `TestBiasBackwardAnalytical::test_dx_vs_numpy[1-16]` | ✅ PASSED |
| `TestBiasBackwardAnalytical::test_dbias_vs_numpy[2-4]` | ✅ PASSED |
| `TestBiasBackwardAnalytical::test_dbias_vs_numpy[4-8]` | ✅ PASSED |
| `TestBiasBackwardAnalytical::test_4d_spatial_shape` | ✅ PASSED |
| `TestBiasBackwardAnalytical::test_multi_axis_bias` | ✅ PASSED |
| `TestBiasBackwardNumerical::test_numerical_grad_dx` | ✅ PASSED |
| `TestBiasBackwardNumerical::test_numerical_grad_dbias` | ✅ PASSED |
| `TestBiasBackwardProperties::test_zero_dy_gives_zero_gradients` | ✅ PASSED |
| `TestBiasBackwardProperties::test_gradient_shapes` | ✅ PASSED |
| `TestBiasBackwardProperties::test_determinism` | ✅ PASSED |
| `TestBiasBackwardProperties::test_forward_preserved_after_backward` | ✅ PASSED |
| `TestBiasBackwardProperties::test_dx_exact_copy_of_dy` | ✅ PASSED |
| `TestBiasBackwardProperties::test_finite_values` | ✅ PASSED |

### `test_blob.py`

| 用例 | 结果 |
|------|------|
| `TestBlobReshape::test_reshape_1d` | ✅ PASSED |
| `TestBlobReshape::test_reshape_2d` | ✅ PASSED |
| `TestBlobReshape::test_reshape_4d` | ✅ PASSED |
| `TestBlobReshape::test_reshape_changes_size` | ✅ PASSED |
| `TestBlobNumpy::test_from_numpy_to_numpy` | ✅ PASSED |
| `TestBlobNumpy::test_from_numpy_creates_copy` | ✅ PASSED |
| `TestBlobNumpy::test_to_numpy_creates_copy` | ✅ PASSED |
| `TestBlobNumpy::test_data_property` | ✅ PASSED |
| `TestBlobNumpy::test_data_setter_reshape` | ✅ PASSED |
| `TestBlobNumpy::test_diff_property` | ✅ PASSED |
| `TestBlobFill::test_fill` | ✅ PASSED |
| `TestBlobFill::test_zero` | ✅ PASSED |
| `TestBlobFill::test_fill_zeros_diff` | ✅ PASSED |
| `TestBlobFill::test_fill_returns_self` | ✅ PASSED |
| `TestBlobFill::test_fill_negative_value` | ✅ PASSED |
| `TestBlobFill::test_fill_zero_value` | ✅ PASSED |
| `TestBlobFill::test_fill_large_value` | ✅ PASSED |
| `TestBlobFill::test_fill_data_tensor_reflects_value` | ✅ PASSED |
| `TestBlobFill::test_fill_then_overwrite` | ✅ PASSED |
| `TestBlobFill::test_fill_int_coerced_to_float32` | ✅ PASSED |
| `TestBlobFromNumpyComprehensive::test_from_numpy_1d` | ✅ PASSED |
| `TestBlobFromNumpyComprehensive::test_from_numpy_2d` | ✅ PASSED |
| `TestBlobFromNumpyComprehensive::test_from_numpy_4d` | ✅ PASSED |
| `TestBlobFromNumpyComprehensive::test_from_numpy_int_converts_to_float32` | ✅ PASSED |
| `TestBlobFromNumpyComprehensive::test_from_numpy_float64_converts_to_float32` | ✅ PASSED |
| `TestBlobFromNumpyComprehensive::test_from_numpy_list_input` | ✅ PASSED |
| `TestBlobFromNumpyComprehensive::test_from_numpy_set_diff_true` | ✅ PASSED |
| `TestBlobFromNumpyComprehensive::test_from_numpy_set_diff_false_default` | ✅ PASSED |
| `TestBlobFromNumpyComprehensive::test_from_numpy_reshapes_existing_blob` | ✅ PASSED |
| `TestBlobFromNumpyComprehensive::test_from_numpy_returns_self` | ✅ PASSED |
| `TestBlobFromNumpyComprehensive::test_from_numpy_chain_fill` | ✅ PASSED |
| `TestBlobFromNumpyComprehensive::test_from_numpy_preserves_values_after_reshape` | ✅ PASSED |
| `TestBlobFromNumpyComprehensive::test_from_numpy_scalar_shape` | ✅ PASSED |
| `TestBlobFromNumpyComprehensive::test_data_setter_dtype_conversion` | ✅ PASSED |
| `TestBlobCopy::test_copy_from` | ✅ PASSED |
| `TestBlobCopy::test_copy_from_is_independent` | ✅ PASSED |
| `TestBlobProperties::test_shape` | ✅ PASSED |
| `TestBlobProperties::test_ndim` | ✅ PASSED |
| `TestBlobProperties::test_size` | ✅ PASSED |
| `TestBlobProperties::test_num_axes` | ✅ PASSED |
| `TestBlobRepr::test_repr` | ✅ PASSED |
| `TestBlobMemoryCounters::test_initial_alloc_counter` | ✅ PASSED |
| `TestBlobMemoryCounters::test_reshape_counter_delta` | ✅ PASSED |
| `TestBlobMemoryCounters::test_reshape_to_zero_frees_memory` | ✅ PASSED |
| `TestBlobMemoryCounters::test_reshape_same_shape_no_delta` | ✅ PASSED |
| `TestBlobMemoryCounters::test_destructor_frees_memory` | ✅ PASSED |
| `TestBlobMemoryCounters::test_multiple_blobs_additive` | ✅ PASSED |
| `TestBlobMemoryCounters::test_memory_info_dict` | ✅ PASSED |
| `TestBlobMemoryCounters::test_reshape_grow_shrink_cycle` | ✅ PASSED |
| `TestBlobMemoryCounters::test_live_blob_count` | ✅ PASSED |
| `TestBlobLifecycle::test_full_lifecycle_training_step` | ✅ PASSED |
| `TestBlobLifecycle::test_lifecycle_dtype_conversion_chain` | ✅ PASSED |
| `TestBlobZeroCopy::test_data_tensor_returns_ndarray` | ✅ PASSED |
| `TestBlobZeroCopy::test_data_tensor_zero_copy_write` | ✅ PASSED |
| `TestBlobZeroCopy::test_data_property_returns_copy` | ✅ PASSED |
| `TestBlobZeroCopy::test_diff_tensor_returns_ndarray` | ✅ PASSED |
| `TestBlobZeroCopy::test_diff_tensor_zero_copy_write` | ✅ PASSED |
| `TestBlobZeroCopy::test_diff_property_returns_copy` | ✅ PASSED |
| `TestBlobZeroCopy::test_update_subtracts_diff_from_data` | ✅ PASSED |
| `TestBlobZeroCopy::test_data_tensor_persists_across_calls` | ✅ PASSED |
| `TestBlobMemoryStress::test_create_destroy_loop_no_leak` | ✅ PASSED |
| `TestBlobMemoryStress::test_reshape_loop_no_leak` | ✅ PASSED |
| `TestBlobMemoryStress::test_copy_from_loop_no_leak` | ✅ PASSED |
| `TestBlobMemoryStress::test_from_numpy_to_numpy_loop_no_leak` | ✅ PASSED |
| `TestBlobMemoryStress::test_serialization_roundtrip_loop_no_leak` | ✅ PASSED |
| `TestBlobExceptionSafety::test_reshape_invalid_shape_no_leak` | ✅ PASSED |
| `TestBlobExceptionSafety::test_empty_lifecycle_no_leak` | ✅ PASSED |
| `TestBlobExceptionSafety::test_partial_update_gc_no_leak` | ✅ PASSED |
| `TestBlobInterleavedLifecycle::test_out_of_order_destruction` | ✅ PASSED |
| `TestBlobInterleavedLifecycle::test_nested_blob_references` | ✅ PASSED |
| `TestBlobInterleavedLifecycle::test_blob_list_append_pop` | ✅ PASSED |
| `TestBlobCountMethod::test_count_no_args_equals_size` | ✅ PASSED |
| `TestBlobCountMethod::test_count_full_range_explicit` | ✅ PASSED |
| `TestBlobCountMethod::test_count_single_axis` | ✅ PASSED |
| `TestBlobCountMethod::test_count_subrange_mid` | ✅ PASSED |
| `TestBlobCountMethod::test_count_negative_end_axis` | ✅ PASSED |
| `TestBlobCountMethod::test_count_against_numpy_prod` | ✅ PASSED |
| `TestBlobCountMethod::test_count_start_axis_only` | ✅ PASSED |
| `TestBlobCountMethod::test_count_1d` | ✅ PASSED |
| `TestBlobCountMethod::test_count_scalar_blob` | ✅ PASSED |
| `TestBlobGetSetData::test_get_data_returns_flat_list` | ✅ PASSED |
| `TestBlobGetSetData::test_get_data_values` | ✅ PASSED |
| `TestBlobGetSetData::test_get_data_after_fill` | ✅ PASSED |
| `TestBlobGetSetData::test_set_data_from_list` | ✅ PASSED |
| `TestBlobGetSetData::test_set_data_from_numpy` | ✅ PASSED |
| `TestBlobGetSetData::test_set_data_size_mismatch_raises` | ✅ PASSED |
| `TestBlobGetSetData::test_set_data_int_coerced_to_float32` | ✅ PASSED |
| `TestBlobGetSetData::test_set_data_then_numpy_roundtrip` | ✅ PASSED |
| `TestBlobGetSetData::test_get_diff_returns_flat_list` | ✅ PASSED |
| `TestBlobGetSetData::test_set_diff_from_list` | ✅ PASSED |
| `TestBlobGetSetData::test_set_diff_from_numpy` | ✅ PASSED |
| `TestBlobGetSetData::test_set_diff_size_mismatch_raises` | ✅ PASSED |
| `TestBlobGetSetData::test_get_data_matches_numpy` | ✅ PASSED |
| `TestBlobConstructionBacktrace::test_construction_backtrace_returns_string` | ✅ PASSED |
| `TestBlobConstructionBacktrace::test_construction_backtrace_is_callable_twice` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_empty_shape_list` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_zero_sized_dim` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_reshape_to_zero_then_back` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_zero_sized_blob_fill_no_crash` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_single_element_blob` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_scalar_1x1_blob` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_fill_negative_zero` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_fill_nan` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_fill_positive_inf` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_fill_negative_inf` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_fill_large_value` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_fill_small_value` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_from_numpy_float64_converts_to_float32_precision` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_from_numpy_int32_converts` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_from_numpy_bool_converts` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_from_numpy_float16_converts` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_from_numpy_transposed` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_from_numpy_sliced` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_from_numpy_fortran_order` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_copy_from_shape_mismatch_raises_or_resizes` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_data_tensor_fresh_after_reshape` | ✅ PASSED |
| `TestBlobExtremeBoundaries::test_moderate_large_blob` | ✅ PASSED |

### `test_complex_topologies.py`

| 用例 | 结果 |
|------|------|
| `TestNetTopologies::test_two_branch_concat_shape` | ✅ PASSED |
| `TestNetTopologies::test_two_branch_concat_values_deterministic` | ✅ PASSED |
| `TestNetTopologies::test_multi_input_concat` | ✅ PASSED |
| `TestNetTopologies::test_multi_input_different_batch_sizes_error` | ✅ PASSED |
| `TestNetTopologies::test_deep_mlp_forward` | ✅ PASSED |
| `TestNetTopologies::test_inplace_chain_forward` | ✅ PASSED |
| `TestNetTopologies::test_residual_eltwise_sum` | ✅ PASSED |
| `TestNetTopologies::test_lenet_like_classification` | ✅ PASSED |
| `TestNetTopologies::test_eltwise_coeff_sum` | ✅ PASSED |
| `TestNetTopologies::test_eltwise_prod` | ✅ PASSED |
| `TestNetReshapeDynamics::test_varying_batch_sizes[1]` | ✅ PASSED |
| `TestNetReshapeDynamics::test_varying_batch_sizes[2]` | ✅ PASSED |
| `TestNetReshapeDynamics::test_varying_batch_sizes[4]` | ✅ PASSED |
| `TestNetReshapeDynamics::test_varying_batch_sizes[8]` | ✅ PASSED |
| `TestNetReshapeDynamics::test_batch_size_1` | ✅ PASSED |
| `TestNetReshapeDynamics::test_large_batch_32` | ✅ PASSED |
| `TestNetReshapeDynamics::test_large_batch_128` | ✅ PASSED |
| `TestNetReshapeDynamics::test_forwards_with_increasing_batch_sizes` | ✅ PASSED |
| `TestNetReshapeDynamics::test_forwards_with_decreasing_batch_sizes` | ✅ PASSED |
| `TestNetReshapeDynamics::test_same_input_different_batch_layout` | ✅ PASSED |
| `TestNetReshapeDynamics::test_input_dimension_mismatch_no_crash` | ✅ PASSED |
| `TestNetReshapeDynamics::test_blob_reshape_between_forwards` | ✅ PASSED |
| `TestLargeScaleForward::test_100_forwards_no_memory_growth` | ✅ PASSED |
| `TestLargeScaleForward::test_500_forwards_deterministic` | ✅ PASSED |
| `TestLargeScaleForward::test_large_batch_256_stable` | ✅ PASSED |
| `TestLargeScaleForward::test_alternating_batch_sizes_stable` | ✅ PASSED |
| `TestLargeScaleForward::test_net_destruction_and_recreation` | ✅ PASSED |
| `TestLargeScaleForward::test_multi_net_parallel_usage` | ✅ PASSED |

### `test_concat_backward.py`

| 用例 | 结果 |
|------|------|
| `TestConcatBackwardKnownValues::test_concat_axis0_two_inputs` | ✅ PASSED |
| `TestConcatBackwardKnownValues::test_concat_axis1_2d` | ✅ PASSED |
| `TestConcatBackwardKnownValues::test_concat_axis1_nchw` | ✅ PASSED |
| `TestConcatBackwardKnownValues::test_concat_axis3_nchw` | ✅ PASSED |
| `TestConcatBackwardNumpy::test_concat_vs_numpy[shapes0-1]` | ✅ PASSED |
| `TestConcatBackwardNumpy::test_concat_vs_numpy[shapes1-1]` | ✅ PASSED |
| `TestConcatBackwardNumpy::test_concat_vs_numpy[shapes2-2]` | ✅ PASSED |
| `TestConcatBackwardNumpy::test_concat_vs_numpy[shapes3-1]` | ✅ PASSED |
| `TestConcatBackwardNumpy::test_concat_vs_numpy[shapes4-2]` | ✅ PASSED |
| `TestConcatBackwardNumpy::test_concat_vs_numpy[shapes5-3]` | ✅ PASSED |
| `TestConcatBackwardNumpy::test_three_inputs` | ✅ PASSED |
| `TestConcatBackwardNumpy::test_four_inputs_axis0` | ✅ PASSED |
| `TestConcatBackwardNumerical::test_numerical_grad[shapes0-1]` | ✅ PASSED |
| `TestConcatBackwardNumerical::test_numerical_grad[shapes1-2]` | ✅ PASSED |
| `TestConcatBackwardNumerical::test_numerical_grad[shapes2-1]` | ✅ PASSED |
| `TestConcatBackwardNumerical::test_numerical_grad_three_inputs` | ✅ PASSED |
| `TestConcatBackwardProperties::test_zero_dy_gives_zero_gradients` | ✅ PASSED |
| `TestConcatBackwardProperties::test_gradient_shapes` | ✅ PASSED |
| `TestConcatBackwardProperties::test_determinism` | ✅ PASSED |
| `TestConcatBackwardProperties::test_forward_preserved_after_backward` | ✅ PASSED |
| `TestConcatBackwardProperties::test_finite_values` | ✅ PASSED |
| `TestConcatBackwardProperties::test_round_trip_split_concat` | ✅ PASSED |
| `TestConcatBackwardProperties::test_axis2_3d_numerical` | ✅ PASSED |
| `TestConcatBackwardProperties::test_unequal_sizes_axis0` | ✅ PASSED |

### `test_conv_backward.py`

| 用例 | 结果 |
|------|------|
| `TestConvBackward1x1::test_conv1x1_known_identity` | ✅ PASSED |
| `TestConvBackward1x1::test_conv1x1_analytical_dx_dw_db` | ✅ PASSED |
| `TestConvBackward1x1::test_conv1x1_no_bias` | ✅ PASSED |
| `TestConvBackward1x1::test_conv1x1_numerical_dx` | ✅ PASSED |
| `TestConvBackward1x1::test_conv1x1_numerical_dw_db` | ✅ PASSED |
| `TestConvBackward3x3::test_conv3x3_pad1_analytical` | ✅ PASSED |
| `TestConvBackward3x3::test_conv3x3_stride2_analytical` | ✅ PASSED |
| `TestConvBackward3x3::test_conv3x3_numerical_dx` | ✅ PASSED |
| `TestConvBackward3x3::test_conv3x3_stride2_numerical_dx` | ✅ PASSED |
| `TestConvBackwardDilation::test_conv_dilation2_analytical` | ✅ PASSED |
| `TestConvBackwardGroups::test_conv_groups2_known_identity` | ✅ PASSED |
| `TestConvBackwardGroups::test_conv_groups2_analytical` | ✅ PASSED |
| `TestConvBackwardGroups::test_conv_groups2_3x3_analytical` | ✅ PASSED |
| `TestConvBackwardGroups::test_conv_groups2_numerical` | ✅ PASSED |
| `TestConvBackwardGroups::test_conv_groups2_3x3_numerical_dw_db` | ✅ PASSED |
| `TestConvBackwardGroups::test_conv_groups4_analytical` | ✅ PASSED |
| `TestConvBackwardGroups::test_conv_groups4_numerical` | ✅ PASSED |
| `TestConvBackwardGroups::test_conv_groups_stride2_numerical` | ✅ PASSED |
| `TestConvBackwardGroups::test_conv_groups_no_bias_numerical` | ✅ PASSED |
| `TestConvBackwardGroups::test_conv_groups_zero_dy` | ✅ PASSED |
| `TestConvBackwardDepthwise::test_depthwise_3x3_pad1_analytical` | ✅ PASSED |
| `TestConvBackwardDepthwise::test_depthwise_3x3_pad1_stride2_numerical` | ✅ PASSED |
| `TestConvBackwardDepthwise::test_depthwise_3x3_numerical_dx_dw` | ✅ PASSED |
| `TestConvBackwardDepthwise::test_depthwise_known_identity` | ✅ PASSED |
| `TestConvBackwardDepthwise::test_depthwise_zero_dy` | ✅ PASSED |
| `TestConvBackwardInvariants::test_conv_zero_dy_gives_zero_grads` | ✅ PASSED |
| `TestConvBackwardInvariants::test_conv_backward_shapes` | ✅ PASSED |
| `TestConvBackwardInvariants::test_conv_backward_deterministic` | ✅ PASSED |
| `TestConvBackwardInvariants::test_conv_backward_preserves_forward` | ✅ PASSED |
| `TestConvBackwardInvariants::test_conv_batch_gradient_accumulation` | ✅ PASSED |

### `test_cow.py`

| 用例 | 结果 |
|------|------|
| `TestBlobCOWApi::test_IsDataShared_false_for_standalone` | ✅ PASSED |
| `TestBlobCOWApi::test_IsDataShared_true_after_ShareData` | ✅ PASSED |
| `TestBlobCOWApi::test_IsDiffShared_false_for_standalone` | ✅ PASSED |
| `TestBlobCOWApi::test_IsDiffShared_true_after_ShareDiff` | ✅ PASSED |
| `TestBlobCOWApi::test_DataRefCount_zero_for_undefined` | ✅ PASSED |
| `TestBlobCOWApi::test_UnshareData_breaks_sharing` | ✅ PASSED |
| `TestBlobCOWApi::test_UnshareDiff_breaks_sharing` | ✅ PASSED |
| `TestBlobCOWApi::test_mutable_data_tensor_triggers_COW` | ✅ PASSED |
| `TestBlobCOWApi::test_mutable_diff_tensor_triggers_COW` | ✅ PASSED |
| `TestBlobCOWApi::test_cow_snapshot_helper` | ✅ PASSED |
| `TestBlobCOWApi::test_three_way_share_refcount` | ✅ PASSED |
| `TestBlobCOWApi::test_UnshareData_noop_when_not_shared` | ✅ PASSED |
| `TestBlobCOWApi::test_const_data_tensor_does_not_trigger_COW` | ✅ PASSED |
| `TestSplitCOWBehavior::test_n1_split_zero_copy_data_shared` | ✅ PASSED |
| `TestSplitCOWBehavior::test_n2_split_data_shared_before_write` | ✅ PASSED |
| `TestSplitCOWBehavior::test_n2_split_cow_isolation_after_write` | ✅ PASSED |
| `TestSplitCOWBehavior::test_n4_split_cow_isolation_after_write` | ✅ PASSED |
| `TestSplitCOWBehavior::test_n2_split_const_access_no_cow` | ✅ PASSED |
| `TestSplitCOWBehavior::test_n2_split_cow_after_inplace_relu` | ✅ PASSED |
| `TestSplitCOWBehavior::test_n2_split_cow_refcount_after_multiple_writes` | ✅ PASSED |
| `TestSplitCOWBehavior::test_cow_snapshot_before_after_forward` | ✅ PASSED |

### `test_crop_backward.py`

| 用例 | 结果 |
|------|------|
| `TestCropBackwardKnownValues::test_crop_1d_axis0_offset1` | ✅ PASSED |
| `TestCropBackwardKnownValues::test_crop_1d_axis0_offset0` | ✅ PASSED |
| `TestCropBackwardKnownValues::test_crop_nchw_axis2_offset` | ✅ PASSED |
| `TestCropBackwardNumpy::test_crop_vs_numpy[in_shape0-crop_shape0-0-offset0]` | ✅ PASSED |
| `TestCropBackwardNumpy::test_crop_vs_numpy[in_shape1-crop_shape1-0-offset1]` | ✅ PASSED |
| `TestCropBackwardNumpy::test_crop_vs_numpy[in_shape2-crop_shape2-2-None]` | ✅ PASSED |
| `TestCropBackwardNumpy::test_crop_vs_numpy[in_shape3-crop_shape3-2-offset3]` | ✅ PASSED |
| `TestCropBackwardNumpy::test_crop_vs_numpy[in_shape4-crop_shape4-2-None]` | ✅ PASSED |
| `TestCropBackwardNumpy::test_crop_vs_numpy[in_shape5-crop_shape5-2-offset5]` | ✅ PASSED |
| `TestCropBackwardNumerical::test_numerical_grad[in_shape0-crop_shape0-0-offset0]` | ✅ PASSED |
| `TestCropBackwardNumerical::test_numerical_grad[in_shape1-crop_shape1-2-None]` | ✅ PASSED |
| `TestCropBackwardNumerical::test_numerical_grad[in_shape2-crop_shape2-2-offset2]` | ✅ PASSED |
| `TestCropBackwardProperties::test_zero_gradient_outside_crop_region` | ✅ PASSED |
| `TestCropBackwardProperties::test_zero_dy_gives_zero_gradients` | ✅ PASSED |
| `TestCropBackwardProperties::test_gradient_shape` | ✅ PASSED |
| `TestCropBackwardProperties::test_determinism` | ✅ PASSED |
| `TestCropBackwardProperties::test_forward_preserved_after_backward` | ✅ PASSED |
| `TestCropBackwardProperties::test_finite_values` | ✅ PASSED |
| `TestCropBackwardProperties::test_round_trip_crop_backward` | ✅ PASSED |

### `test_deconv_backward.py`

| 用例 | 结果 |
|------|------|
| `TestDeconvBackward1x1::test_deconv1x1_known_values` | ✅ PASSED |
| `TestDeconvBackward1x1::test_deconv1x1_analytical_dx_dw_db` | ✅ PASSED |
| `TestDeconvBackward1x1::test_deconv1x1_numerical_dx_dw_db` | ✅ PASSED |
| `TestDeconvBackward1x1::test_deconv1x1_no_bias` | ✅ PASSED |
| `TestDeconvBackward1x1::test_deconv1x1_numerical_no_bias` | ✅ PASSED |
| `TestDeconvBackwardStride2::test_deconv_2x2_s2_numerical_dx_dw_db` | ✅ PASSED |
| `TestDeconvBackwardEdgeCases::test_zero_dy_zero_gradients` | ✅ PASSED |
| `TestDeconvBackwardEdgeCases::test_deterministic` | ✅ PASSED |
| `TestDeconvBackwardEdgeCases::test_shapes_dtypes` | ✅ PASSED |
| `TestDeconvBackwardEdgeCases::test_forward_preserved_after_backward` | ✅ PASSED |

### `test_dropout_backward.py`

| 用例 | 结果 |
|------|------|
| `TestDropoutIdentity::test_forward_is_identity[0.0]` | ✅ PASSED |
| `TestDropoutIdentity::test_forward_is_identity[0.3]` | ✅ PASSED |
| `TestDropoutIdentity::test_forward_is_identity[0.5]` | ✅ PASSED |
| `TestDropoutIdentity::test_forward_is_identity[0.7]` | ✅ PASSED |
| `TestDropoutIdentity::test_backward_dx_equals_dy[0.0]` | ✅ PASSED |
| `TestDropoutIdentity::test_backward_dx_equals_dy[0.3]` | ✅ PASSED |
| `TestDropoutIdentity::test_backward_dx_equals_dy[0.5]` | ✅ PASSED |
| `TestDropoutIdentity::test_backward_dx_equals_dy[0.7]` | ✅ PASSED |
| `TestDropoutIdentity::test_known_values_small` | ✅ PASSED |
| `TestDropout4DBackward::test_4d_analytical_dx` | ✅ PASSED |
| `TestDropout4DBackward::test_4d_numerical_dx` | ✅ PASSED |
| `TestDropout2DNumerical::test_2d_numerical_dx[0.0]` | ✅ PASSED |
| `TestDropout2DNumerical::test_2d_numerical_dx[0.5]` | ✅ PASSED |
| `TestDropoutEdgeCases::test_zero_dy_zero_dx` | ✅ PASSED |
| `TestDropoutEdgeCases::test_deterministic` | ✅ PASSED |
| `TestDropoutEdgeCases::test_dx_shape_dtype[shape0]` | ✅ PASSED |
| `TestDropoutEdgeCases::test_dx_shape_dtype[shape1]` | ✅ PASSED |
| `TestDropoutEdgeCases::test_dx_shape_dtype[shape2]` | ✅ PASSED |
| `TestDropoutEdgeCases::test_forward_preserved_after_backward` | ✅ PASSED |
| `TestDropoutEdgeCases::test_inplace_safe` | ✅ PASSED |

### `test_eltwise_backward.py`

| 用例 | 结果 |
|------|------|
| `TestEltwiseBackwardKnownValues::test_sum_two_inputs_simple` | ✅ PASSED |
| `TestEltwiseBackwardKnownValues::test_sum_with_coeffs` | ✅ PASSED |
| `TestEltwiseBackwardKnownValues::test_prod_two_inputs_simple` | ✅ PASSED |
| `TestEltwiseBackwardKnownValues::test_max_two_inputs_simple` | ✅ PASSED |
| `TestEltwiseBackwardNumpy::test_sum_vs_numpy[shape0-2]` | ✅ PASSED |
| `TestEltwiseBackwardNumpy::test_sum_vs_numpy[shape1-2]` | ✅ PASSED |
| `TestEltwiseBackwardNumpy::test_sum_vs_numpy[shape2-2]` | ✅ PASSED |
| `TestEltwiseBackwardNumpy::test_sum_vs_numpy[shape3-3]` | ✅ PASSED |
| `TestEltwiseBackwardNumpy::test_sum_coeffs_vs_numpy[shape0-coeffs0]` | ✅ PASSED |
| `TestEltwiseBackwardNumpy::test_sum_coeffs_vs_numpy[shape1-coeffs1]` | ✅ PASSED |
| `TestEltwiseBackwardNumpy::test_prod_vs_numpy[shape0-2]` | ✅ PASSED |
| `TestEltwiseBackwardNumpy::test_prod_vs_numpy[shape1-2]` | ✅ PASSED |
| `TestEltwiseBackwardNumpy::test_prod_vs_numpy[shape2-3]` | ✅ PASSED |
| `TestEltwiseBackwardNumpy::test_max_vs_numpy[shape0-2]` | ✅ PASSED |
| `TestEltwiseBackwardNumpy::test_max_vs_numpy[shape1-2]` | ✅ PASSED |
| `TestEltwiseBackwardNumpy::test_max_vs_numpy[shape2-3]` | ✅ PASSED |
| `TestEltwiseBackwardNumerical::test_sum_numerical_grad[0]` | ✅ PASSED |
| `TestEltwiseBackwardNumerical::test_sum_numerical_grad[1]` | ✅ PASSED |
| `TestEltwiseBackwardNumerical::test_prod_numerical_grad[0]` | ✅ PASSED |
| `TestEltwiseBackwardNumerical::test_prod_numerical_grad[1]` | ✅ PASSED |
| `TestEltwiseBackwardNumerical::test_max_numerical_grad[0]` | ✅ PASSED |
| `TestEltwiseBackwardNumerical::test_max_numerical_grad[1]` | ✅ PASSED |
| `TestEltwiseBackwardNumerical::test_sum_coeffs_numerical_grad` | ✅ PASSED |
| `TestEltwiseBackwardNumerical::test_three_inputs_sum_numerical_grad` | ✅ PASSED |
| `TestEltwiseBackwardProperties::test_zero_dy_gives_zero_gradients_sum` | ✅ PASSED |
| `TestEltwiseBackwardProperties::test_zero_dy_gives_zero_gradients_prod` | ✅ PASSED |
| `TestEltwiseBackwardProperties::test_zero_dy_gives_zero_gradients_max` | ✅ PASSED |
| `TestEltwiseBackwardProperties::test_gradient_shapes` | ✅ PASSED |
| `TestEltwiseBackwardProperties::test_determinism` | ✅ PASSED |
| `TestEltwiseBackwardProperties::test_forward_preserved_after_backward` | ✅ PASSED |
| `TestEltwiseBackwardProperties::test_finite_values` | ✅ PASSED |
| `TestEltwiseBackwardProperties::test_max_gradient_conservation` | ✅ PASSED |

### `test_elu_kink_stability.py`

| 用例 | 结果 |
|------|------|
| `TestELUKinkContinuity::test_c0_continuity_at_zero[0.1]` | ✅ PASSED |
| `TestELUKinkContinuity::test_c0_continuity_at_zero[0.5]` | ✅ PASSED |
| `TestELUKinkContinuity::test_c0_continuity_at_zero[1.0]` | ✅ PASSED |
| `TestELUKinkContinuity::test_c0_continuity_at_zero[2.0]` | ✅ PASSED |
| `TestELUKinkContinuity::test_c1_continuity_at_zero[0.1]` | ✅ PASSED |
| `TestELUKinkContinuity::test_c1_continuity_at_zero[0.5]` | ✅ PASSED |
| `TestELUKinkContinuity::test_c1_continuity_at_zero[1.0]` | ✅ PASSED |
| `TestELUKinkContinuity::test_c1_continuity_at_zero[2.0]` | ✅ PASSED |
| `TestELUKinkContinuity::test_c1_discontinuity_explained` | ✅ PASSED |
| `TestELUKinkNumericalGradient::test_gradient_error_at_exact_zero` | ✅ PASSED |
| `TestELUKinkNumericalGradient::test_gradient_error_at_exact_zero_alpha_01` | ✅ PASSED |
| `TestELUKinkNumericalGradient::test_gradient_away_from_kink_is_accurate[0.001]` | ✅ PASSED |
| `TestELUKinkNumericalGradient::test_gradient_away_from_kink_is_accurate[0.0005]` | ✅ PASSED |
| `TestELUKinkNumericalGradient::test_gradient_away_from_kink_is_accurate[0.0001]` | ✅ PASSED |
| `TestELUKinkNumericalGradient::test_threshold_5e3_is_robust` | ✅ PASSED |
| `TestELUKinkNumericalGradient::test_kink_element_requires_relaxed_threshold` | ✅ PASSED |
| `TestELUAlpha1Smooth::test_derivative_is_continuous_at_zero` | ✅ PASSED |
| `TestELUAlpha1Smooth::test_second_derivative_jump_at_zero` | ✅ PASSED |
| `TestELUSaturatedRegime::test_large_positive_saturates_to_linear[1.0]` | ✅ PASSED |
| `TestELUSaturatedRegime::test_large_positive_saturates_to_linear[0.1]` | ✅ PASSED |
| `TestELUSaturatedRegime::test_large_negative_saturates_to_minus_alpha[1.0]` | ✅ PASSED |
| `TestELUSaturatedRegime::test_large_negative_saturates_to_minus_alpha[0.1]` | ✅ PASSED |
| `TestELUSaturatedRegime::test_gradient_vanishes_for_large_negative[1.0]` | ✅ PASSED |
| `TestELUSaturatedRegime::test_gradient_vanishes_for_large_negative[0.1]` | ✅ PASSED |

### `test_extreme_boundaries.py`

| 用例 | 结果 |
|------|------|
| `TestExtremeBoundaries::test_large_input_2048` | ✅ PASSED |
| `TestExtremeBoundaries::test_split_large_input_1024` | ✅ PASSED |
| `TestExtremeBoundaries::test_nan_input_no_crash` | ✅ PASSED |
| `TestExtremeBoundaries::test_inf_input_no_crash` | ✅ PASSED |
| `TestExtremeBoundaries::test_zero_input_deterministic` | ✅ PASSED |
| `TestExtremeBoundaries::test_extreme_weights_large` | ✅ PASSED |
| `TestExtremeBoundaries::test_extreme_weights_tiny` | ✅ PASSED |
| `TestExtremeBoundaries::test_deep_network_20_layers` | ✅ PASSED |
| `TestExtremeBoundaries::test_lifecycle_stress_50_creates` | ✅ PASSED |
| `TestExtremeBoundaries::test_repeated_forward_100_times` | ✅ PASSED |
| `TestExtremeBoundaries::test_minimal_1x1` | ✅ PASSED |

### `test_extreme_inputs.py`

| 用例 | 结果 |
|------|------|
| `TestExtremeValues::test_nan_input_single_element` | ✅ PASSED |
| `TestExtremeValues::test_nan_input_all_elements` | ✅ PASSED |
| `TestExtremeValues::test_positive_inf_input` | ✅ PASSED |
| `TestExtremeValues::test_negative_inf_input` | ✅ PASSED |
| `TestExtremeValues::test_mixed_nan_inf` | ✅ PASSED |
| `TestExtremeValues::test_large_values_1e6` | ✅ PASSED |
| `TestExtremeValues::test_large_values_near_float32_max` | ✅ PASSED |
| `TestExtremeValues::test_denormal_values` | ✅ PASSED |
| `TestExtremeValues::test_all_zeros` | ✅ PASSED |
| `TestExtremeValues::test_negative_zero` | ✅ PASSED |
| `TestExtremeValues::test_alternating_extremes` | ✅ PASSED |
| `TestDTypeErrors::test_float_dtype_mismatch[float64]` | ✅ PASSED |
| `TestDTypeErrors::test_float_dtype_mismatch[float16]` | ✅ PASSED |
| `TestDTypeErrors::test_integer_dtype_raises[int32]` | ✅ PASSED |
| `TestDTypeErrors::test_integer_dtype_raises[int64]` | ✅ PASSED |
| `TestDTypeErrors::test_integer_dtype_raises[uint8]` | ✅ PASSED |
| `TestDTypeErrors::test_integer_dtype_raises[bool]` | ✅ PASSED |
| `TestDTypeErrors::test_complex_dtype_raises` | ✅ PASSED |
| `TestDTypeErrors::test_object_dtype_raises` | ✅ PASSED |
| `TestDTypeErrors::test_readonly_array` | ✅ PASSED |
| `TestNonContiguousArrays::test_transposed_array` | ✅ PASSED |
| `TestNonContiguousArrays::test_sliced_array_strided` | ✅ PASSED |
| `TestNonContiguousArrays::test_fortran_order_array` | ✅ PASSED |
| `TestNonContiguousArrays::test_reversed_array` | ✅ PASSED |
| `TestNonContiguousArrays::test_column_slice_non_contiguous` | ✅ PASSED |
| `TestRecoveryAfterError::test_normal_after_nan_input` | ✅ PASSED |
| `TestRecoveryAfterError::test_normal_after_type_error` | ✅ PASSED |
| `TestRecoveryAfterError::test_normal_after_large_values` | ✅ PASSED |
| `TestRecoveryAfterError::test_normal_after_non_contiguous_error` | ✅ PASSED |
| `TestRecoveryAfterError::test_repeated_error_recovery_cycles` | ✅ PASSED |

### `test_ffi_set_shape_only.py`

| 用例 | 结果 |
|------|------|
| `TestSetShapeOnlyFFI::test_set_shape_only_available` | ✅ PASSED |
| `TestSetShapeOnlyFFI::test_is_lazy_allocated_available` | ✅ PASSED |
| `TestSetShapeOnlyFFI::test_default_not_lazy` | ✅ PASSED |
| `TestSetShapeOnlyFFI::test_set_shape_only_shape_access` | ✅ PASSED |
| `TestSetShapeOnlyFFI::test_set_shape_only_count` | ✅ PASSED |
| `TestSetShapeOnlyFFI::test_set_shape_only_no_data_access` | ✅ PASSED |
| `TestSetShapeOnlyFFI::test_reshape_clears_lazy` | ✅ PASSED |
| `TestSetShapeOnlyFFI::test_set_shape_only_1d` | ✅ PASSED |
| `TestSetShapeOnlyFFI::test_set_shape_only_empty_raises` | ✅ PASSED |
| `TestSetShapeOnlyFFI::test_set_shape_only_negative_raises` | ✅ PASSED |
| `TestSetShapeOnlyFFI::test_set_shape_only_zero_raises` | ✅ PASSED |
| `TestSetShapeOnlyLifecycle::test_full_lazy_to_shared_cycle` | ✅ PASSED |
| `TestSetShapeOnlyLifecycle::test_lazy_share_data_simulation` | ✅ PASSED |
| `TestSetShapeOnlyLifecycle::test_lazy_blob_mutable_data_triggers_allocation` | ✅ PASSED |
| `TestSetShapeOnlySplitIntegration::test_split_n4_not_lazy` | ✅ PASSED |
| `TestSetShapeOnlySplitIntegration::test_split_n64_lazy_reshape` | ✅ PASSED |
| `TestSetShapeOnlySplitIntegration::test_split_n64_downstream_relu` | ✅ PASSED |

### `test_flatten_backward.py`

| 用例 | 结果 |
|------|------|
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape0]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape1]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape2]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape3]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape4]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape5]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape6]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape7]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape8]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape9]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape10]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape11]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape12]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape13]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape14]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape15]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape16]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape17]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape18]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape19]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape20]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape21]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape22]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape23]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape24]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape25]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape26]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape27]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape28]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape29]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape30]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape31]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape32]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape33]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape34]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape35]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape36]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape37]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape38]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape39]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape40]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape41]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape42]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape43]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape44]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape45]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape46]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape47]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape48]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape49]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape50]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape51]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape52]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape53]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape54]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape55]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape56]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape57]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape58]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape59]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape60]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape61]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape62]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape63]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape64]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape65]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape66]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape67]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape68]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape69]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_default_axis_passthrough[input_shape70]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape0]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape1]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape2]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape3]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape4]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape5]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape6]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape7]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape8]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape9]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape10]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape11]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape12]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape13]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape14]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape15]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape16]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape17]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape18]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape19]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape20]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape21]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape22]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape23]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape24]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_2d_input[input_shape25]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_1d_input[1]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_1d_input[2]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_1d_input[5]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_1d_input[10]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_1d_input[100]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_1d_input[128]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_1d_input[256]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_1d_input[512]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_1d_input[784]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_1d_input[1000]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_1d_input[4096]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_5d_input[input_shape0]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_5d_input[input_shape1]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_5d_input[input_shape2]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_5d_input[input_shape3]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_5d_input[input_shape4]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_5d_input[input_shape5]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_3d_input[input_shape0]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_3d_input[input_shape1]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_3d_input[input_shape2]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_3d_input[input_shape3]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_3d_input[input_shape4]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_3d_input[input_shape5]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_3d_input[input_shape6]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_3d_input[input_shape7]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_3d_input[input_shape8]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_4d[0--1]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_4d[1--1]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_4d[0-0]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_4d[0-1]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_4d[0-2]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_4d[0-3]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_4d[1-1]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_4d[1-2]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_4d[1-3]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_4d[2-2]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_4d[2-3]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_4d[3-3]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_5d[0--1]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_5d[0-0]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_5d[0-1]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_5d[0-2]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_5d[0-3]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_5d[0-4]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_5d[1--1]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_5d[1-1]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_5d[1-2]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_5d[1-3]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_5d[1-4]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_5d[2--1]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_5d[2-2]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_5d[2-3]` | ✅ PASSED |
| `TestFlattenBackwardIdentity::test_flatten_custom_axes_5d[2-4]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient[input_shape0]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient[input_shape1]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient[input_shape2]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient[input_shape3]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient[input_shape4]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient[input_shape5]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient[input_shape6]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient[input_shape7]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient[input_shape8]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient[input_shape9]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient[input_shape10]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient[input_shape11]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient[input_shape12]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient[input_shape13]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient[input_shape14]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient[input_shape15]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient[input_shape16]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient[input_shape17]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient[input_shape18]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient[input_shape19]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient[input_shape20]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient_2d[input_shape0]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient_2d[input_shape1]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient_2d[input_shape2]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient_2d[input_shape3]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient_2d[input_shape4]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient_2d[input_shape5]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient_2d[input_shape6]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient_custom_axes[input_shape0-0--1]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient_custom_axes[input_shape1-1-2]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient_custom_axes[input_shape2-2-3]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient_custom_axes[input_shape3-0--1]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient_custom_axes[input_shape4-1-2]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient_custom_axes[input_shape5-0--1]` | ✅ PASSED |
| `TestFlattenBackwardNumerical::test_flatten_numerical_gradient_custom_axes[input_shape6-1-3]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_zero_dy_zero_dx[input_shape0]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_zero_dy_zero_dx[input_shape1]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_zero_dy_zero_dx[input_shape2]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_zero_dy_zero_dx[input_shape3]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_zero_dy_zero_dx[input_shape4]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_zero_dy_zero_dx[input_shape5]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_zero_dy_zero_dx[input_shape6]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_zero_dy_zero_dx[input_shape7]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_zero_dy_zero_dx[input_shape8]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_zero_dy_zero_dx[input_shape9]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_zero_dy_zero_dx[input_shape10]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_zero_dy_zero_dx[input_shape11]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_zero_dy_zero_dx[input_shape12]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_zero_dy_zero_dx[input_shape13]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_zero_dy_zero_dx[input_shape14]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_zero_dy_zero_dx[input_shape15]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_zero_dy_zero_dx[input_shape16]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_zero_dy_zero_dx[input_shape17]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_zero_dy_zero_dx[input_shape18]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_zero_dy_zero_dx[input_shape19]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_zero_dy_zero_dx[input_shape20]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_deterministic` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_shapes_dtypes[input_shape0]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_shapes_dtypes[input_shape1]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_shapes_dtypes[input_shape2]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_shapes_dtypes[input_shape3]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_shapes_dtypes[input_shape4]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_shapes_dtypes[input_shape5]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_forward_preserved_after_backward` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_flatten_in_chain[4-3-2]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_flatten_in_chain[8-3-4]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_flatten_in_chain[2-1-2]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_flatten_in_chain[16-3-10]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_no_learnable_params` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_special_values_forward_backward[0.0]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_special_values_forward_backward[1.0]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_special_values_forward_backward[-1.0]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_special_values_forward_backward[2.0]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_special_values_forward_backward[-2.0]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_special_values_forward_backward[0.5]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_special_values_forward_backward[-0.5]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_special_values_forward_backward[100.0]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_special_values_forward_backward[-100.0]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_degenerate_shapes[input_shape0]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_degenerate_shapes[input_shape1]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_degenerate_shapes[input_shape2]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_degenerate_shapes[input_shape3]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_degenerate_shapes[input_shape4]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_degenerate_shapes[input_shape5]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_degenerate_shapes[input_shape6]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_degenerate_shapes[input_shape7]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_degenerate_shapes[input_shape8]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_degenerate_shapes[input_shape9]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_small_and_large_dy[input_shape0-0.001]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_small_and_large_dy[input_shape1--0.001]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_small_and_large_dy[input_shape2-1e-06]` | ✅ PASSED |
| `TestFlattenBackwardEdgeCases::test_small_and_large_dy[input_shape3-3.14159]` | ✅ PASSED |
| `TestFlattenInplaceProtection::test_inplace_forbidden` | ✅ PASSED |

### `test_grad_check_utils_selftest.py`

| 用例 | 结果 |
|------|------|
| `test_compare_gradients_matching` | ✅ PASSED |
| `test_compare_gradients_noisy` | ✅ PASSED |
| `test_compare_gradients_mismatch` | ✅ PASSED |
| `test_numerical_gradient_quadratic` | ✅ PASSED |
| `test_numerical_gradient_matmul` | ✅ PASSED |
| `test_assert_grad_close_passes` | ✅ PASSED |
| `test_assert_grad_close_raises` | ✅ PASSED |
| `test_assert_backward_matches_reference_passes` | ✅ PASSED |
| `test_assert_backward_matches_reference_fails` | ✅ PASSED |
| `test_assert_backward_matches_reference_skip_numerical` | ✅ PASSED |

### `test_inner_product_backward.py`

| 用例 | 结果 |
|------|------|
| `TestInnerProductBackward::test_ip_backward_known_values` | ✅ PASSED |
| `TestInnerProductBackward::test_ip_backward_known_values_with_bias` | ✅ PASSED |
| `TestInnerProductBackward::test_ip_backward_analytical_dx` | ✅ PASSED |
| `TestInnerProductBackward::test_ip_backward_analytical_dw` | ✅ PASSED |
| `TestInnerProductBackward::test_ip_backward_analytical_db` | ✅ PASSED |
| `TestInnerProductBackward::test_ip_numerical_gradient_dx` | ✅ PASSED |
| `TestInnerProductBackward::test_ip_numerical_gradient_dw` | ✅ PASSED |
| `TestInnerProductBackward::test_ip_numerical_gradient_db` | ✅ PASSED |
| `TestInnerProductBackward::test_ip_backward_no_bias` | ✅ PASSED |
| `TestInnerProductBackward::test_ip_numerical_gradient_no_bias` | ✅ PASSED |
| `TestInnerProductBackward::test_ip_backward_shapes` | ✅ PASSED |
| `TestInnerProductBackward::test_ip_backward_finite` | ✅ PASSED |
| `TestInnerProductBackward::test_ip_backward_zero_dy_gives_zero_grads` | ✅ PASSED |
| `TestInnerProductBackward::test_ip_backward_deterministic` | ✅ PASSED |
| `TestInnerProductBackward::test_ip_backward_preserves_forward_output` | ✅ PASSED |
| `TestInnerProductBackwardNCHW::test_ip_backward_nchw_analytical` | ✅ PASSED |
| `TestInnerProductBackwardNCHW::test_ip_backward_nchw_dx_numerical` | ✅ PASSED |
| `TestInnerProductBackwardTranspose::test_ip_transpose_backward_analytical` | ✅ PASSED |
| `TestInnerProductBackwardTranspose::test_ip_transpose_numerical_gradient_dx` | ✅ PASSED |
| `TestInnerProductBackwardTranspose::test_ip_transpose_numerical_gradient_dw` | ✅ PASSED |
| `TestInnerProductBackwardIdentity::test_ip_identity_weights_dx_equals_dy` | ✅ PASSED |
| `TestInnerProductBackwardIdentity::test_ip_ones_weights_dw` | ✅ PASSED |
| `TestInnerProductBackwardIdentity::test_ip_db_is_column_sum` | ✅ PASSED |

### `test_insert_splits.py`

| 用例 | 结果 |
|------|------|
| `TestInsertSplits::test_dead_end_no_split` | ✅ PASSED |
| `TestInsertSplits::test_single_consumer_no_split` | ✅ PASSED |
| `TestInsertSplits::test_inplace_relu_split_named_after_last_producer` | ✅ PASSED |
| `TestInsertSplits::test_loss_weight_triggers_split` | ✅ PASSED |
| `TestInsertSplits::test_chained_splits` | ✅ PASSED |
| `TestInsertSplits::test_idempotent_no_duplicate_splits` | ✅ PASSED |
| `TestInsertSplits::test_forward_correctness_inplace_split` | ✅ PASSED |
| `TestInsertSplits::test_multiple_external_inputs_order` | ✅ PASSED |
| `TestInsertSplits::test_linear_chain_zero_splits` | ✅ PASSED |
| `TestInsertSplits::test_double_inplace_split_after_last_producer` | ✅ PASSED |
| `TestInsertSplits::test_mixed_input_layer_and_param_input` | ✅ PASSED |
| `TestInsertSplits::test_split_output_names_match_caffe_native_convention` | ✅ PASSED |
| `TestInsertSplits::test_split_concat_split_nested` | ✅ PASSED |
| `TestInsertSplits::test_multiple_layers_need_splits_positions` | ✅ PASSED |
| `TestInsertSplits::test_empty_network_no_crash` | ✅ PASSED |
| `TestInsertSplits::test_input_layer_three_consumers` | ✅ PASSED |
| `TestInsertSplits::test_loss_weight_plus_multiple_consumers` | ✅ PASSED |
| `TestInsertSplits::test_unknown_bottom_raises_error` | ✅ PASSED |

### `test_layer_template_three_layer_validation.py`

| 用例 | 结果 |
|------|------|
| `TestReLULayers::test_known_values_identity` | ✅ PASSED |
| `TestReLULayers::test_known_values_zero_negative` | ✅ PASSED |
| `TestReLULayers::test_known_values_leaky_relu` | ✅ PASSED |
| `TestReLULayers::test_known_values_mixed_signs` | ✅ PASSED |
| `TestReLULayers::test_random_numpy_match[shape0-0.0]` | ✅ PASSED |
| `TestReLULayers::test_random_numpy_match[shape0-0.01]` | ✅ PASSED |
| `TestReLULayers::test_random_numpy_match[shape0-0.1]` | ✅ PASSED |
| `TestReLULayers::test_random_numpy_match[shape0-0.5]` | ✅ PASSED |
| `TestReLULayers::test_random_numpy_match[shape1-0.0]` | ✅ PASSED |
| `TestReLULayers::test_random_numpy_match[shape1-0.01]` | ✅ PASSED |
| `TestReLULayers::test_random_numpy_match[shape1-0.1]` | ✅ PASSED |
| `TestReLULayers::test_random_numpy_match[shape1-0.5]` | ✅ PASSED |
| `TestReLULayers::test_random_numpy_match[shape2-0.0]` | ✅ PASSED |
| `TestReLULayers::test_random_numpy_match[shape2-0.01]` | ✅ PASSED |
| `TestReLULayers::test_random_numpy_match[shape2-0.1]` | ✅ PASSED |
| `TestReLULayers::test_random_numpy_match[shape2-0.5]` | ✅ PASSED |
| `TestReLULayers::test_random_numpy_match[shape3-0.0]` | ✅ PASSED |
| `TestReLULayers::test_random_numpy_match[shape3-0.01]` | ✅ PASSED |
| `TestReLULayers::test_random_numpy_match[shape3-0.1]` | ✅ PASSED |
| `TestReLULayers::test_random_numpy_match[shape3-0.5]` | ✅ PASSED |
| `TestReLULayers::test_repeated_forward_determinism` | ✅ PASSED |
| `TestReLULayers::test_weights_invariance` | ✅ PASSED |
| `TestReLULayers::test_multi_round_stability` | ✅ PASSED |
| `TestReLULayers::test_edge_all_zeros` | ✅ PASSED |
| `TestReLULayers::test_edge_large_values` | ✅ PASSED |
| `TestReLULayers::test_edge_1d_input` | ✅ PASSED |

### `test_layers.py`

| 用例 | 结果 |
|------|------|
| `TestLayerType::test_layer_type_property` | ✅ PASSED |
| `TestLayerType::test_layer_repr` | ✅ PASSED |
| `TestInputLayer::test_input_layer_in_net` | ✅ PASSED |
| `TestInputLayer::test_input_layer_forward` | ✅ PASSED |
| `TestReLU::test_relu_positive_unchanged` | ✅ PASSED |
| `TestReLU::test_relu_negative_zero` | ✅ PASSED |
| `TestReLU::test_relu_numpy_reference_positive` | ✅ PASSED |
| `TestReLU::test_relu_numpy_reference_negative` | ✅ PASSED |
| `TestReLU::test_relu_numpy_reference_negative_slope` | ✅ PASSED |
| `TestInnerProduct::test_inner_product_matmul_bias` | ✅ PASSED |
| `TestInnerProduct::test_inner_product_numpy_reference` | ✅ PASSED |
| `TestSoftmax::test_softmax_sums_to_one` | ✅ PASSED |
| `TestSoftmax::test_softmax_all_zero_uniform` | ✅ PASSED |
| `TestSoftmax::test_softmax_numpy_reference_sum_one` | ✅ PASSED |
| `TestSoftmax::test_softmax_numpy_reference_uniform` | ✅ PASSED |
| `TestFlatten::test_flatten_shape` | ✅ PASSED |
| `TestFlatten::test_flatten_numpy_reference` | ✅ PASSED |
| `TestFlatten::test_flatten_numpy_reference_end_axis` | ✅ PASSED |
| `TestSigmoid::test_sigmoid_numpy_output_range` | ✅ PASSED |
| `TestSigmoid::test_sigmoid_numpy_zero` | ✅ PASSED |
| `TestSigmoid::test_sigmoid_numpy_symmetric` | ✅ PASSED |
| `TestTanH::test_tanh_numpy_output_range` | ✅ PASSED |
| `TestTanH::test_tanh_numpy_zero` | ✅ PASSED |
| `TestTanH::test_tanh_numpy_symmetric` | ✅ PASSED |
| `TestPReLU::test_prelu_numpy_channel_shared` | ✅ PASSED |
| `TestPReLU::test_prelu_numpy_positive_unchanged` | ✅ PASSED |
| `TestPReLU::test_prelu_numpy_per_channel` | ✅ PASSED |
| `TestELU::test_elu_numpy_positive_identity` | ✅ PASSED |
| `TestELU::test_elu_numpy_negative` | ✅ PASSED |
| `TestELU::test_elu_numpy_alpha` | ✅ PASSED |
| `TestDropout::test_dropout_numpy_identity` | ✅ PASSED |
| `TestConcat::test_concat_numpy_axis0` | ✅ PASSED |
| `TestConcat::test_concat_numpy_axis1` | ✅ PASSED |
| `TestConcat::test_concat_numpy_three_inputs` | ✅ PASSED |
| `TestConcat::test_concat_numpy_axis2_3d` | ✅ PASSED |
| `TestEltwise::test_eltwise_sum_numpy` | ✅ PASSED |
| `TestEltwise::test_eltwise_sum_with_coeffs` | ✅ PASSED |
| `TestEltwise::test_eltwise_prod_numpy` | ✅ PASSED |
| `TestEltwise::test_eltwise_max_numpy` | ✅ PASSED |
| `TestReshape::test_reshape_numpy_basic` | ✅ PASSED |
| `TestReshape::test_reshape_numpy_infer` | ✅ PASSED |
| `TestReshape::test_reshape_numpy_infer_2d` | ✅ PASSED |
| `TestReshape::test_reshape_numpy_data_unchanged` | ✅ PASSED |
| `TestReshape::test_reshape_numpy_same_data` | ✅ PASSED |
| `TestLayerStandalone::test_standalone_layer_type_empty` | ✅ PASSED |
| `TestLayerStandalone::test_standalone_layer_name_empty` | ✅ PASSED |
| `TestLayerStandalone::test_standalone_layer_blobs_empty` | ✅ PASSED |
| `TestLayerStandalone::test_standalone_layer_repr` | ✅ PASSED |
| `TestLayerStandalone::test_standalone_layer_not_native` | ✅ PASSED |
| `TestLayerStandalone::test_standalone_layer_multiple_calls_safe` | ✅ PASSED |
| `TestLayerFromNet::test_input_layer_has_no_blobs` | ✅ PASSED |
| `TestLayerFromNet::test_relu_layer_has_no_blobs` | ✅ PASSED |
| `TestLayerFromNet::test_softmax_layer_has_no_blobs` | ✅ PASSED |
| `TestLayerFromNet::test_inner_product_with_bias_has_two_blobs` | ✅ PASSED |
| `TestLayerFromNet::test_inner_product_no_bias_has_one_blob` | ✅ PASSED |
| `TestLayerFromNet::test_layer_types_match` | ✅ PASSED |
| `TestLayerFromNet::test_layer_names_match` | ✅ PASSED |
| `TestLayerFromNet::test_layer_blobs_are_blob_instances` | ✅ PASSED |
| `TestLayerFromNet::test_layer_repr_contains_name_and_type` | ✅ PASSED |
| `TestLayerFromNet::test_weight_blobs_persist_after_forward` | ✅ PASSED |
| `TestLayerFromNet::test_multiple_forwards_do_not_corrupt_weights` | ✅ PASSED |
| `TestLayerFromNet::test_layer_blobs_return_new_list_each_time` | ✅ PASSED |
| `TestLayerFromNet::test_input_layer_blobs_mutable_no_crash` | ✅ PASSED |

### `test_lrn_backward.py`

| 用例 | 结果 |
|------|------|
| `TestLRNBackwardNumpy::test_lrn_vs_numpy[4-3-0.001-0.5-2.0]` | ✅ PASSED |
| `TestLRNBackwardNumpy::test_lrn_vs_numpy[6-5-0.0001-0.75-1.0]` | ✅ PASSED |
| `TestLRNBackwardNumpy::test_lrn_vs_numpy[8-3-0.005-0.5-1.0]` | ✅ PASSED |
| `TestLRNBackwardNumpy::test_lrn_vs_numpy[5-5-0.0001-0.75-1.0]` | ✅ PASSED |
| `TestLRNBackwardNumpy::test_lrn_multibatch` | ✅ PASSED |
| `TestLRNBackwardNumerical::test_numerical_grad[4-3-0.001-0.5-2.0]` | ✅ PASSED |
| `TestLRNBackwardNumerical::test_numerical_grad[6-5-0.0001-0.75-1.0]` | ✅ PASSED |
| `TestLRNBackwardProperties::test_zero_dy_gives_zero_gradients` | ✅ PASSED |
| `TestLRNBackwardProperties::test_gradient_shape` | ✅ PASSED |
| `TestLRNBackwardProperties::test_determinism` | ✅ PASSED |
| `TestLRNBackwardProperties::test_forward_preserved_after_backward` | ✅ PASSED |
| `TestLRNBackwardProperties::test_finite_values` | ✅ PASSED |
| `TestLRNBackwardProperties::test_forward_consistency` | ✅ PASSED |

### `test_net.py`

| 用例 | 结果 |
|------|------|
| `TestNetParse::test_parse_prototxt_string` | ✅ PASSED |
| `TestNetParse::test_parse_multilayer_prototxt` | ✅ PASSED |
| `TestNetBuild::test_build_from_param` | ✅ PASSED |
| `TestNetBuild::test_net_name_property` | ✅ PASSED |
| `TestNetBlobAccess::test_has_blob` | ✅ PASSED |
| `TestNetBlobAccess::test_has_layer` | ✅ PASSED |
| `TestNetBlobAccess::test_blob_by_name` | ✅ PASSED |
| `TestNetBlobAccess::test_layer_by_name` | ✅ PASSED |
| `TestNetBlobAccess::test_blob_by_name_keyerror` | ✅ PASSED |
| `TestNetBlobAccess::test_layer_by_name_keyerror` | ✅ PASSED |
| `TestNetBlobAccess::test_getitem` | ✅ PASSED |
| `TestNetBlobAccess::test_getitem_keyerror` | ✅ PASSED |
| `TestNetBlobAccess::test_contains` | ✅ PASSED |
| `TestNetBlobAccess::test_blobs_dict` | ✅ PASSED |
| `TestNetBlobAccess::test_layers_dict` | ✅ PASSED |
| `TestNetBlobAccess::test_iter` | ✅ PASSED |
| `TestNetBlobAccess::test_len` | ✅ PASSED |
| `TestNetForward::test_forward_executes` | ✅ PASSED |
| `TestNetForward::test_forward_output_shape` | ✅ PASSED |
| `TestNetForward::test_forward_all` | ✅ PASSED |
| `TestNetForward::test_forward_pure_python_reference` | ⏭️ SKIPPED |
| `TestNetRepr::test_repr` | ✅ PASSED |
| `TestNetEmptyConstructor::test_empty_net_name` | ✅ PASSED |
| `TestNetEmptyConstructor::test_empty_net_no_blobs` | ✅ PASSED |
| `TestNetEmptyConstructor::test_empty_net_no_layers` | ✅ PASSED |
| `TestNetEmptyConstructor::test_empty_net_zero_inputs` | ✅ PASSED |
| `TestNetEmptyConstructor::test_empty_net_zero_outputs` | ✅ PASSED |
| `TestNetEmptyConstructor::test_empty_net_input_names_empty` | ✅ PASSED |
| `TestNetEmptyConstructor::test_empty_net_output_names_empty` | ✅ PASSED |
| `TestNetEmptyConstructor::test_empty_net_blob_names_empty` | ✅ PASSED |
| `TestNetEmptyConstructor::test_empty_net_layer_names_empty` | ✅ PASSED |
| `TestNetEmptyConstructor::test_empty_net_blobs_dict_empty` | ✅ PASSED |
| `TestNetEmptyConstructor::test_empty_net_layers_dict_empty` | ✅ PASSED |
| `TestNetEmptyConstructor::test_empty_net_len_zero` | ✅ PASSED |
| `TestNetEmptyConstructor::test_empty_net_iter_empty` | ✅ PASSED |
| `TestNetEmptyConstructor::test_empty_net_contains_false` | ✅ PASSED |
| `TestNetEmptyConstructor::test_empty_net_repr` | ✅ PASSED |
| `TestNetEmptyConstructor::test_empty_net_has_blob_false` | ✅ PASSED |
| `TestNetEmptyConstructor::test_empty_net_has_layer_false` | ✅ PASSED |
| `TestNetConstructorErrors::test_empty_string_prototxt_raises` | ✅ PASSED |
| `TestNetConstructorErrors::test_invalid_prototxt_raises` | ✅ PASSED |
| `TestNetConstructorErrors::test_whitespace_only_prototxt` | ✅ PASSED |
| `TestNetForwardBoundaries::test_forward_no_args` | ✅ PASSED |
| `TestNetForwardBoundaries::test_forward_empty_dict` | ✅ PASSED |
| `TestNetForwardBoundaries::test_forward_none_explicit` | ✅ PASSED |
| `TestNetForwardBoundaries::test_forward_wrong_input_name_silently_ignored` | ✅ PASSED |
| `TestNetForwardBoundaries::test_forward_list_input_auto_converts` | ✅ PASSED |
| `TestNetForwardBoundaries::test_forward_float64_converts_to_float32` | ✅ PASSED |
| `TestNetForwardBoundaries::test_forward_int_input_converts` | ✅ PASSED |
| `TestNetForwardBoundaries::test_forward_returns_dict_of_numpy_arrays` | ✅ PASSED |
| `TestNetForwardBoundaries::test_forward_all_no_kwargs` | ✅ PASSED |
| `TestNetForwardBoundaries::test_forward_deterministic` | ✅ PASSED |
| `TestNetForwardBoundaries::test_forward_output_not_input_reference` | ✅ PASSED |
| `TestNetForwardBoundaries::test_forward_nan_input_no_crash` | ✅ PASSED |
| `TestNetForwardBoundaries::test_forward_inf_input_no_crash` | ✅ PASSED |
| `TestNetForwardBoundaries::test_forward_zero_input` | ✅ PASSED |
| `TestNetConsistency::test_blob_names_count_matches_blobs_array` | ✅ PASSED |
| `TestNetConsistency::test_layer_names_count_matches_layers_array` | ✅ PASSED |
| `TestNetConsistency::test_input_blobs_count_matches_num_inputs` | ✅ PASSED |
| `TestNetConsistency::test_output_blobs_count_matches_num_outputs` | ✅ PASSED |
| `TestNetConsistency::test_blobs_dict_keys_match_blob_names` | ✅ PASSED |
| `TestNetConsistency::test_layers_dict_keys_match_layer_names` | ✅ PASSED |
| `TestNetConsistency::test_layer_by_name_consistent_with_layers_array` | ✅ PASSED |
| `TestNetConsistency::test_blob_by_name_consistent_with_blobs_array` | ✅ PASSED |
| `TestNetConsistency::test_iter_yields_blob_names` | ✅ PASSED |
| `TestNetConsistency::test_len_matches_blobs_array` | ✅ PASSED |
| `TestNetConsistency::test_input_names_consistent_with_input_blobs` | ✅ PASSED |
| `TestNetConsistency::test_output_names_consistent_with_output_blobs` | ✅ PASSED |

### `test_p2b_regression.py`

| 用例 | 结果 |
|------|------|
| `TestSplitTopologies::test_split_1to2_copies_data` | ✅ PASSED |
| `TestSplitTopologies::test_split_1to3_concat_roundtrip` | ✅ PASSED |
| `TestSplitTopologies::test_residual_with_split` | ✅ PASSED |
| `TestSplitTopologies::test_split_inplace_branch_isolation` | ✅ PASSED |
| `TestSplitTopologies::test_n1_split_passthrough` | ✅ PASSED |
| `TestSplitTopologies::test_split_deterministic_repeated_forward` | ✅ PASSED |
| `TestSplitPerformanceScaling::test_split_perf_scaling[1-128-2]` | ✅ PASSED |
| `TestSplitPerformanceScaling::test_split_perf_scaling[1-128-4]` | ✅ PASSED |
| `TestSplitPerformanceScaling::test_split_perf_scaling[32-256-2]` | ✅ PASSED |
| `TestSplitPerformanceScaling::test_split_perf_scaling[32-256-4]` | ✅ PASSED |
| `TestSplitPerformanceScaling::test_split_perf_scaling[32-512-2]` | ✅ PASSED |
| `TestSplitPerformanceScaling::test_split_perf_scaling[32-1024-2]` | ✅ PASSED |
| `TestSplitPerformanceScaling::test_split_perf_scaling[16-2048-2]` | ✅ PASSED |
| `TestSplitPerformanceScaling::test_split_perf_scaling[16-2048-4]` | ✅ PASSED |
| `TestExtremeBoundaries::test_large_input_2048` | ✅ PASSED |
| `TestExtremeBoundaries::test_split_large_input_1024` | ✅ PASSED |
| `TestExtremeBoundaries::test_nan_input_no_crash` | ✅ PASSED |
| `TestExtremeBoundaries::test_inf_input_no_crash` | ✅ PASSED |
| `TestExtremeBoundaries::test_zero_input_deterministic` | ✅ PASSED |
| `TestExtremeBoundaries::test_extreme_weights_large` | ✅ PASSED |
| `TestExtremeBoundaries::test_extreme_weights_tiny` | ✅ PASSED |
| `TestExtremeBoundaries::test_deep_network_20_layers` | ✅ PASSED |
| `TestExtremeBoundaries::test_minimal_1x1` | ✅ PASSED |
| `TestSplitMemoryStability::test_lifecycle_stress_50_creates` | ✅ PASSED |
| `TestSplitMemoryStability::test_repeated_forward_100_times` | ✅ PASSED |
| `TestSplitMemoryStability::test_split_high_fanout_8` | ✅ PASSED |
| `TestSplitMemoryStability::test_split_lifecycle_stress` | ✅ PASSED |
| `TestSplitMemoryStability::test_multi_level_split_chain` | ✅ PASSED |
| `TestSplitMemoryStability::test_concurrent_net_creation_stress` | ✅ PASSED |

### `test_p3a_conv_pool_bn.py`

| 用例 | 结果 |
|------|------|
| `TestConvolutionLayers::test_conv_1x1_identity` | ✅ PASSED |
| `TestConvolutionLayers::test_conv_1x1_with_bias` | ✅ PASSED |
| `TestConvolutionLayers::test_conv_3x3_no_padding` | ✅ PASSED |
| `TestConvolutionLayers::test_conv_3x3_with_padding` | ✅ PASSED |
| `TestConvolutionLayers::test_conv_stride_2` | ✅ PASSED |
| `TestConvolutionLayers::test_conv_group_2` | ✅ PASSED |
| `TestConvolutionLayers::test_conv_repeated_forward_stable` | ✅ PASSED |
| `TestConvolutionLayers::test_conv_weights_unchanged_after_forward` | ✅ PASSED |
| `TestPoolingLayers::test_max_pooling_2x2_stride2` | ✅ PASSED |
| `TestPoolingLayers::test_ave_pooling_2x2_stride2` | ✅ PASSED |
| `TestPoolingLayers::test_max_pooling_3x3_pad1` | ✅ PASSED |
| `TestPoolingLayers::test_global_max_pooling` | ✅ PASSED |
| `TestPoolingLayers::test_global_ave_pooling` | ✅ PASSED |
| `TestPoolingLayers::test_ave_pooling_padding_boundary` | ✅ PASSED |
| `TestPoolingLayers::test_pooling_repeated_forward_stable` | ✅ PASSED |
| `TestBatchNormLayers::test_batchnorm_zero_mean_unit_var` | ✅ PASSED |
| `TestBatchNormLayers::test_batchnorm_normalizes` | ✅ PASSED |
| `TestBatchNormLayers::test_batchnorm_epsilon_stability` | ✅ PASSED |
| `TestBatchNormLayers::test_batchnorm_scale_factor` | ✅ PASSED |
| `TestBatchNormLayers::test_batchnorm_2d_input` | ✅ PASSED |
| `TestBatchNormLayers::test_batchnorm_repeated_forward_stable` | ✅ PASSED |
| `TestBatchNormLayers::test_batchnorm_weights_unchanged_after_forward` | ✅ PASSED |
| `TestConvPoolBNCombination::test_conv_pool_bn_pipeline` | ✅ PASSED |
| `TestConvPoolBNCombination::test_pipeline_repeated_forward_no_crash` | ✅ PASSED |

### `test_p3b_eltwise_scale.py`

| 用例 | 结果 |
|------|------|
| `TestScaleLayers::test_scale_identity_default` | ✅ PASSED |
| `TestScaleLayers::test_scale_per_channel` | ✅ PASSED |
| `TestScaleLayers::test_scale_with_bias` | ✅ PASSED |
| `TestScaleLayers::test_scale_axis0` | ✅ PASSED |
| `TestScaleLayers::test_scale_repeated_forward` | ✅ PASSED |
| `TestScaleLayers::test_scale_weights_unchanged` | ✅ PASSED |
| `TestBiasLayers::test_bias_zero_default` | ✅ PASSED |
| `TestBiasLayers::test_bias_per_channel` | ✅ PASSED |
| `TestBiasLayers::test_bias_known_values` | ✅ PASSED |
| `TestBiasLayers::test_bias_axis0` | ✅ PASSED |
| `TestBiasLayers::test_bias_repeated_forward` | ✅ PASSED |
| `TestBiasLayers::test_bias_weights_unchanged` | ✅ PASSED |
| `TestEltwiseLayers::test_eltwise_sum_two_inputs` | ✅ PASSED |
| `TestEltwiseLayers::test_eltwise_sum_with_coeffs` | ✅ PASSED |
| `TestEltwiseLayers::test_eltwise_prod_two_inputs` | ✅ PASSED |
| `TestEltwiseLayers::test_eltwise_max_two_inputs` | ✅ PASSED |
| `TestEltwiseLayers::test_eltwise_sum_three_inputs` | ✅ PASSED |
| `TestEltwiseLayers::test_eltwise_known_values_sum` | ✅ PASSED |
| `TestEltwiseLayers::test_eltwise_known_values_prod` | ✅ PASSED |
| `TestEltwiseLayers::test_eltwise_known_values_max` | ✅ PASSED |
| `TestEltwiseLayers::test_eltwise_repeated_forward` | ✅ PASSED |
| `TestConcatLayers::test_concat_axis1_channel` | ✅ PASSED |
| `TestConcatLayers::test_concat_axis0_batch` | ✅ PASSED |
| `TestConcatLayers::test_concat_axis2_height` | ✅ PASSED |
| `TestConcatLayers::test_concat_three_inputs` | ✅ PASSED |
| `TestConcatLayers::test_concat_known_values` | ✅ PASSED |
| `TestConcatLayers::test_concat_repeated_forward` | ✅ PASSED |
| `TestDropoutLayers::test_dropout_identity_ratio_0` | ✅ PASSED |
| `TestDropoutLayers::test_dropout_identity_ratio_05` | ✅ PASSED |
| `TestDropoutLayers::test_dropout_identity_ratio_09` | ✅ PASSED |
| `TestDropoutLayers::test_dropout_1d_input` | ✅ PASSED |
| `TestDropoutLayers::test_dropout_preserves_special_values` | ✅ PASSED |
| `TestDropoutLayers::test_dropout_repeated_forward` | ✅ PASSED |
| `TestSoftmaxWithLossLayers::test_softmax_probs_only` | ✅ PASSED |
| `TestSoftmaxWithLossLayers::test_softmax_loss_perfect_predictions` | ✅ PASSED |
| `TestSoftmaxWithLossLayers::test_softmax_loss_uniform` | ✅ PASSED |
| `TestSoftmaxWithLossLayers::test_softmax_loss_numpy_match` | ✅ PASSED |
| `TestSoftmaxWithLossLayers::test_softmax_loss_with_probs_top` | ✅ PASSED |
| `TestSoftmaxWithLossLayers::test_softmax_repeated_forward` | ✅ PASSED |
| `TestAccuracyLayers::test_accuracy_perfect` | ✅ PASSED |
| `TestAccuracyLayers::test_accuracy_zero` | ✅ PASSED |
| `TestAccuracyLayers::test_accuracy_partial` | ✅ PASSED |
| `TestAccuracyLayers::test_accuracy_topk` | ✅ PASSED |
| `TestAccuracyLayers::test_accuracy_spatial` | ✅ PASSED |
| `TestAccuracyLayers::test_accuracy_numpy_match` | ✅ PASSED |
| `TestAccuracyLayers::test_accuracy_repeated_forward` | ✅ PASSED |
| `TestScaleBiasEltwiseCombination::test_scale_then_bias_pipeline` | ✅ PASSED |
| `TestScaleBiasEltwiseCombination::test_eltwise_then_scale_pipeline` | ✅ PASSED |
| `TestScaleBiasEltwiseCombination::test_classification_pipeline` | ✅ PASSED |
| `TestScaleBiasEltwiseCombination::test_stability_20_iters` | ✅ PASSED |

### `test_p3c_activations_ip.py`

| 用例 | 结果 |
|------|------|
| `TestReLULayers::test_relu_basic_known_values` | ✅ PASSED |
| `TestReLULayers::test_relu_leaky_negative_slope` | ✅ PASSED |
| `TestReLULayers::test_relu_numpy_match` | ✅ PASSED |
| `TestReLULayers::test_relu_leaky_numpy_match` | ✅ PASSED |
| `TestReLULayers::test_relu_preserves_shape` | ✅ PASSED |
| `TestReLULayers::test_relu_repeated_forward` | ✅ PASSED |
| `TestSigmoidLayers::test_sigmoid_known_values` | ✅ PASSED |
| `TestSigmoidLayers::test_sigmoid_numpy_match` | ✅ PASSED |
| `TestSigmoidLayers::test_sigmoid_output_range` | ✅ PASSED |
| `TestSigmoidLayers::test_sigmoid_float32_saturation_exact` | ✅ PASSED |
| `TestSigmoidLayers::test_sigmoid_saturation_transition_zone` | ✅ PASSED |
| `TestSigmoidLayers::test_sigmoid_extreme_large_tensor` | ✅ PASSED |
| `TestSigmoidLayers::test_sigmoid_zero_input` | ✅ PASSED |
| `TestSigmoidLayers::test_sigmoid_symmetric` | ✅ PASSED |
| `TestSigmoidLayers::test_sigmoid_repeated_forward` | ✅ PASSED |
| `TestSigmoidBackward::test_sigmoid_backward_gradient_values` | ✅ PASSED |
| `TestSigmoidBackward::test_sigmoid_backward_with_arbitrary_dy` | ✅ PASSED |
| `TestSigmoidBackward::test_sigmoid_backward_saturation_counter_zero_input` | ✅ PASSED |
| `TestSigmoidBackward::test_sigmoid_backward_saturation_counter_all_saturated_positive` | ✅ PASSED |
| `TestSigmoidBackward::test_sigmoid_backward_saturation_counter_all_saturated_negative` | ✅ PASSED |
| `TestSigmoidBackward::test_sigmoid_backward_saturation_counter_mixed` | ✅ PASSED |
| `TestSigmoidBackward::test_sigmoid_backward_saturation_boundary_threshold` | ✅ PASSED |
| `TestSigmoidBackward::test_sigmoid_backward_large_tensor_saturation_stats` | ✅ PASSED |
| `TestSigmoidBackward::test_sigmoid_backward_deterministic` | ✅ PASSED |
| `TestSigmoidBackward::test_sigmoid_backward_preserves_forward_output` | ✅ PASSED |
| `TestTanHLayers::test_tanh_known_values` | ✅ PASSED |
| `TestTanHLayers::test_tanh_numpy_match` | ✅ PASSED |
| `TestTanHLayers::test_tanh_output_range` | ✅ PASSED |
| `TestTanHLayers::test_tanh_odd_function` | ✅ PASSED |
| `TestTanHLayers::test_tanh_repeated_forward` | ✅ PASSED |
| `TestELULayers::test_elu_default_alpha_known_values` | ✅ PASSED |
| `TestELULayers::test_elu_custom_alpha` | ✅ PASSED |
| `TestELULayers::test_elu_numpy_match_default_alpha` | ✅ PASSED |
| `TestELULayers::test_elu_numpy_match_custom_alpha` | ✅ PASSED |
| `TestELULayers::test_elu_continuity_at_zero` | ✅ PASSED |
| `TestELULayers::test_elu_repeated_forward` | ✅ PASSED |
| `TestPReLULayers::test_prelu_channel_shared_default` | ✅ PASSED |
| `TestPReLULayers::test_prelu_per_channel_default` | ✅ PASSED |
| `TestPReLULayers::test_prelu_per_channel_custom_slopes` | ✅ PASSED |
| `TestPReLULayers::test_prelu_channel_shared_custom_slope` | ✅ PASSED |
| `TestPReLULayers::test_prelu_numpy_match_per_channel` | ✅ PASSED |
| `TestPReLULayers::test_prelu_repeated_forward` | ✅ PASSED |
| `TestInnerProductLayers::test_ip_known_values_no_bias` | ✅ PASSED |
| `TestInnerProductLayers::test_ip_known_values_with_bias` | ✅ PASSED |
| `TestInnerProductLayers::test_ip_numpy_match_with_bias` | ✅ PASSED |
| `TestInnerProductLayers::test_ip_numpy_match_no_bias` | ✅ PASSED |
| `TestInnerProductLayers::test_ip_output_shape` | ✅ PASSED |
| `TestInnerProductLayers::test_ip_weights_unchanged_after_forward` | ✅ PASSED |
| `TestInnerProductLayers::test_ip_repeated_forward` | ✅ PASSED |
| `TestSoftmaxLayers::test_softmax_known_values` | ✅ PASSED |
| `TestSoftmaxLayers::test_softmax_one_hot_large_input` | ✅ PASSED |
| `TestSoftmaxLayers::test_softmax_numpy_match` | ✅ PASSED |
| `TestSoftmaxLayers::test_softmax_sums_to_one` | ✅ PASSED |
| `TestSoftmaxLayers::test_softmax_preserves_shape` | ✅ PASSED |
| `TestSoftmaxLayers::test_softmax_repeated_forward` | ✅ PASSED |
| `TestFlattenLayers::test_flatten_default_all` | ✅ PASSED |
| `TestFlattenLayers::test_flatten_axis1_to_2` | ✅ PASSED |
| `TestFlattenLayers::test_flatten_numpy_match` | ✅ PASSED |
| `TestFlattenLayers::test_flatten_preserves_values` | ✅ PASSED |
| `TestFlattenLayers::test_flatten_repeated_forward` | ✅ PASSED |
| `TestReshapeLayers::test_reshape_simple` | ✅ PASSED |
| `TestReshapeLayers::test_reshape_with_inferred_dim` | ✅ PASSED |
| `TestReshapeLayers::test_reshape_numpy_match` | ✅ PASSED |
| `TestReshapeLayers::test_reshape_preserves_values` | ✅ PASSED |
| `TestReshapeLayers::test_reshape_repeated_forward` | ✅ PASSED |
| `TestActivationIPCombination::test_mlp_pipeline_ip_relu_softmax` | ✅ PASSED |
| `TestActivationIPCombination::test_ip_sigmoid_sigmoid_numpy_chain` | ✅ PASSED |
| `TestActivationIPCombination::test_stability_20_iters` | ✅ PASSED |

### `test_p3c_transformer.py`

| 用例 | 结果 |
|------|------|
| `TestPositionalEncoding::test_sinusoidal_pe_eltwise_sum` | ✅ PASSED |
| `TestPositionalEncoding::test_learnable_pe_bias_layer` | ✅ PASSED |
| `TestPositionalEncoding::test_pe_addition_2d_flattened` | ✅ PASSED |
| `TestPositionalEncoding::test_pe_repeated_forward_deterministic` | ✅ PASSED |
| `TestSelfAttentionComponents::test_qkv_linear_projection_dimensions` | ✅ PASSED |
| `TestSelfAttentionComponents::test_attention_scale_factor` | ✅ PASSED |
| `TestSelfAttentionComponents::test_softmax_attention_weights` | ✅ PASSED |
| `TestSelfAttentionComponents::test_residual_connection_split_eltwise` | ✅ PASSED |
| `TestSelfAttentionComponents::test_attention_output_via_innerproduct` | ✅ PASSED |
| `TestScaledDotProductAttention::test_sdp_attention_pipeline` | ✅ PASSED |
| `TestScaledDotProductAttention::test_sdp_attention_identity_case` | ✅ PASSED |
| `TestMultiHeadProjection::test_multi_head_concat` | ✅ PASSED |
| `TestTransformerEncoderBlock::test_encoder_block_forward` | ✅ PASSED |

### `test_p3d_all_layers_e2e.py`

| 用例 | 结果 |
|------|------|
| `TestP3DAllLayersEndToEnd::test_p3d_all_layers_forward_backward_no_crash` | ✅ PASSED |
| `TestP3DAllLayersEndToEnd::test_p3d_all_param_gradients_finite` | ✅ PASSED |
| `TestP3DAllLayersEndToEnd::test_p3d_loss_decreases_with_training` | ✅ PASSED |
| `TestP3DAllLayersEndToEnd::test_p3d_softmax_independent_layer_probabilities` | ✅ PASSED |
| `TestP3DAllLayersEndToEnd::test_p3d_eltwise_sum_gradient_routes` | ✅ PASSED |
| `TestP3DAllLayersEndToEnd::test_p3d_concat_gradient_splits` | ✅ PASSED |
| `TestP3DAllLayersEndToEnd::test_p3d_dropout_identity_gradient_passthrough` | ✅ PASSED |
| `TestP3DAllLayersEndToEnd::test_p3d_scale_bias_gradient_shapes_correct` | ✅ PASSED |

### `test_p3d_slice_crop_deconv_lrn.py`

| 用例 | 结果 |
|------|------|
| `TestSliceLayers::test_slice_equal_2way_channel` | ✅ PASSED |
| `TestSliceLayers::test_slice_equal_3way_channel` | ✅ PASSED |
| `TestSliceLayers::test_slice_explicit_points` | ✅ PASSED |
| `TestSliceLayers::test_slice_n1_identity` | ✅ PASSED |
| `TestSliceLayers::test_slice_output_shapes` | ✅ PASSED |
| `TestCropLayers::test_crop_center_hw` | ✅ PASSED |
| `TestCropLayers::test_crop_no_offset` | ✅ PASSED |
| `TestCropLayers::test_crop_axis1_channels` | ✅ PASSED |
| `TestCropLayers::test_crop_single_offset_all_dims` | ✅ PASSED |
| `TestCropLayers::test_crop_output_shape` | ✅ PASSED |
| `TestLRNLayers::test_lrn_alexnet_defaults` | ✅ PASSED |
| `TestLRNLayers::test_lrn_custom_params` | ✅ PASSED |
| `TestLRNLayers::test_lrn_uniform_input` | ✅ PASSED |
| `TestLRNLayers::test_lrn_zero_input` | ✅ PASSED |
| `TestLRNLayers::test_lrn_output_shape` | ✅ PASSED |
| `TestDeconvolutionLayers::test_deconv_1x1_identity_uprojection` | ✅ PASSED |
| `TestDeconvolutionLayers::test_deconv_1x1_channel_projection` | ✅ PASSED |
| `TestDeconvolutionLayers::test_deconv_1x1_with_bias` | ✅ PASSED |
| `TestDeconvolutionLayers::test_deconv_output_shape_1x1` | ✅ PASSED |
| `TestDeconvolutionLayers::test_deconv_stride2_shape` | ✅ PASSED |
| `TestSliceConcatRoundtrip::test_slice_concat_roundtrip_3way` | ✅ PASSED |

### `test_phase3_log_aggregation.py`

| 用例 | 结果 |
|------|------|
| `TestLogAggregationN100::test_n100_split_perf_lines_bounded` | ✅ PASSED |
| `TestLogAggregationN100::test_n100_forward_summary_present` | ✅ PASSED |
| `TestLogAggregationN100::test_n100_reshape_summary_present` | ✅ PASSED |
| `TestLogAggregationN100::test_n4_split_output_correct` | ✅ PASSED |
| `TestLogAggregationN100::test_n100_split_output_correct` | ✅ PASSED |
| `TestLogAggregationN100::test_n100_forward_deterministic` | ✅ PASSED |
| `TestLogAggregationBoundary::test_threshold_boundary[4-False]` | ✅ PASSED |
| `TestLogAggregationBoundary::test_threshold_boundary[31-False]` | ✅ PASSED |
| `TestLogAggregationBoundary::test_threshold_boundary[32-True]` | ✅ PASSED |
| `TestLogAggregationBoundary::test_threshold_boundary[33-True]` | ✅ PASSED |
| `TestLogAggregationBoundary::test_threshold_boundary[64-True]` | ✅ PASSED |
| `TestLogAggregationCorrectness::test_n100_split_forward_plus_relu` | ✅ PASSED |

### `test_phase3_set_shape_only.py`

| 用例 | 结果 |
|------|------|
| `TestSetShapeOnly::test_basic_shape_storage` | ✅ PASSED |
| `TestSetShapeOnly::test_no_data_allocation` | ✅ PASSED |
| `TestSetShapeOnly::test_lazy_flag_indicates_no_data` | ✅ PASSED |
| `TestSetShapeOnly::test_reshape_clears_lazy_flag` | ✅ PASSED |
| `TestSetShapeOnly::test_share_data_clears_lazy_flag` | ✅ PASSED |
| `TestSetShapeOnly::test_mutable_data_triggers_allocation` | ✅ PASSED |
| `TestSetShapeOnly::test_1d_shape` | ✅ PASSED |
| `TestSplitLazyReshape::test_large_n_triggers_lazy_allocation` | ✅ PASSED |
| `TestSplitLazyReshape::test_forward_transitions_to_normal` | ✅ PASSED |
| `TestSplitLazyReshape::test_small_n_stays_normal` | ✅ PASSED |
| `TestSplitLazyReshape::test_downstream_layer_compatibility_relu` | ✅ PASSED |
| `TestSetShapeOnlyExtended::test_count_after_set_shape_only` | ✅ PASSED |
| `TestSetShapeOnlyExtended::test_set_shape_only_then_reshape` | ✅ PASSED |
| `TestSetShapeOnlyExtended::test_empty_shape_rejected` | ✅ PASSED |
| `TestSetShapeOnlyExtended::test_lazy_blob_share_data_then_write` | ✅ PASSED |
| `TestSetShapeOnlyExtended::test_n1_split_no_lazy_allocation` | ✅ PASSED |
| `TestSetShapeOnlyExtended::test_n16_boundary` | ✅ PASSED |
| `TestDownstreamLayerCompatibility::test_activation_after_lazy_split[ReLU-downstream_out]` | ✅ PASSED |
| `TestDownstreamLayerCompatibility::test_activation_after_lazy_split[Sigmoid-downstream_out]` | ✅ PASSED |
| `TestDownstreamLayerCompatibility::test_activation_after_lazy_split[TanH-downstream_out]` | ✅ PASSED |

### `test_pooling_backward.py`

| 用例 | 结果 |
|------|------|
| `TestMaxPoolBackward2x2::test_maxpool_2x2_known_values` | ✅ PASSED |
| `TestMaxPoolBackward2x2::test_maxpool_2x2_analytical_dx` | ✅ PASSED |
| `TestMaxPoolBackward2x2::test_maxpool_2x2_numerical_dx` | ✅ PASSED |
| `TestMaxPoolBackward2x2::test_maxpool_zero_dy_zero_dx` | ✅ PASSED |
| `TestAvePoolBackward2x2::test_avepool_2x2_known_values` | ✅ PASSED |
| `TestAvePoolBackward2x2::test_avepool_2x2_analytical_dx` | ✅ PASSED |
| `TestAvePoolBackward2x2::test_avepool_2x2_numerical_dx` | ✅ PASSED |
| `TestMaxPoolBackwardOverlapping::test_maxpool_3x3_pad1_analytical_dx` | ✅ PASSED |
| `TestMaxPoolBackwardOverlapping::test_maxpool_3x3_pad1_numerical_dx` | ✅ PASSED |
| `TestAvePoolBackwardOverlapping::test_avepool_3x3_s2_analytical_dx` | ✅ PASSED |
| `TestAvePoolBackwardOverlapping::test_avepool_3x3_s2_numerical_dx` | ✅ PASSED |
| `TestGlobalPoolBackward::test_global_maxpool_analytical_dx` | ✅ PASSED |
| `TestGlobalPoolBackward::test_global_avepool_analytical_dx` | ✅ PASSED |
| `TestGlobalPoolBackward::test_global_maxpool_numerical_dx` | ✅ PASSED |
| `TestPoolBackwardDeterminism::test_deterministic` | ✅ PASSED |
| `TestPoolBackwardDeterminism::test_dx_shape_dtype` | ✅ PASSED |
| `TestPoolBackwardDeterminism::test_forward_preserved_after_backward` | ✅ PASSED |
| `TestMaxPoolTieBreaking::test_tie_2x2_s2_all_equal` | ✅ PASSED |
| `TestMaxPoolTieBreaking::test_tie_2x2_s2_partial_equal` | ✅ PASSED |
| `TestMaxPoolTieBreaking::test_tie_vs_numpy_reference_random_ties` | ✅ PASSED |
| `TestMaxPoolTieBreaking::test_tie_deterministic_across_runs` | ✅ PASSED |
| `TestPoolOverlapAccumulation::test_ave_3x3_s1_overlap_accumulation_known_values` | ✅ PASSED |
| `TestPoolOverlapAccumulation::test_ave_3x3_s1_pad1_overlap_center_accumulates_more` | ✅ PASSED |
| `TestPoolOverlapAccumulation::test_ave_boundary_pool_size_correction` | ✅ PASSED |
| `TestPoolOverlapAccumulation::test_max_overlap_same_pixel_wins_multiple_windows` | ✅ PASSED |
| `TestPoolOverlapAccumulation::test_ave_stride1_full_overlap_random_vs_numpy` | ✅ PASSED |
| `TestPoolOverlapAccumulation::test_max_stride1_overlap_random_vs_numpy` | ✅ PASSED |
| `TestPoolOverlapAccumulation::test_gradient_sum_conservation_ave` | ✅ PASSED |

### `test_python_api.py`

| 用例 | 结果 |
|------|------|
| `TestModuleAPI::test_dunder_version` | ✅ PASSED |
| `TestModuleAPI::test_enable_disable_debug_logging` | ✅ PASSED |
| `TestModuleAPI::test_ffi_api_is_available` | ✅ PASSED |
| `TestModuleAPI::test_live_blob_count_non_negative` | ✅ PASSED |
| `TestModuleAPI::test_log_level_roundtrip` | ✅ PASSED |
| `TestModuleAPI::test_memory_info_returns_dict` | ✅ PASSED |
| `TestModuleAPI::test_python_only_fallback_when_native_lib_missing` | ✅ PASSED |
| `TestModuleAPI::test_total_allocated_bytes_non_negative` | ✅ PASSED |
| `TestModuleAPI::test_version_returns_string` | ✅ PASSED |
| `TestBlobNativeAPI::test_copy_from_blob` | ✅ PASSED |
| `TestBlobNativeAPI::test_copy_from_numpy_array` | ✅ PASSED |
| `TestBlobNativeAPI::test_data_setter_from_numpy` | ✅ PASSED |
| `TestBlobNativeAPI::test_data_tensor_zero_copy` | ✅ PASSED |
| `TestBlobNativeAPI::test_default_constructor_creates_valid_blob` | ✅ PASSED |
| `TestBlobNativeAPI::test_diff_setter_from_numpy` | ✅ PASSED |
| `TestBlobNativeAPI::test_diff_tensor_zero_copy` | ✅ PASSED |
| `TestBlobNativeAPI::test_fill_sets_data_values` | ✅ PASSED |
| `TestBlobNativeAPI::test_from_numpy_to_numpy_roundtrip` | ✅ PASSED |
| `TestBlobNativeAPI::test_name_property_roundtrip` | ✅ PASSED |
| `TestBlobNativeAPI::test_repr` | ✅ PASSED |
| `TestBlobNativeAPI::test_reshape_changes_shape` | ✅ PASSED |
| `TestBlobNativeAPI::test_reshape_negative_dimension_raises` | ✅ PASSED |
| `TestBlobNativeAPI::test_set_data_get_data_roundtrip` | ✅ PASSED |
| `TestBlobNativeAPI::test_set_diff_get_diff_roundtrip` | ✅ PASSED |
| `TestBlobNativeAPI::test_shape_constructor` | ✅ PASSED |
| `TestBlobNativeAPI::test_size_alias_for_count` | ✅ PASSED |
| `TestBlobNativeAPI::test_to_numpy_get_diff` | ✅ PASSED |
| `TestBlobNativeAPI::test_update_subtracts_diff` | ✅ PASSED |
| `TestBlobNativeAPI::test_zero_resets_data_and_diff` | ✅ PASSED |
| `TestNetConstructor::test_create_from_protoString_mlp` | ✅ PASSED |
| `TestNetConstructor::test_create_from_protoString_simple_input` | ✅ PASSED |
| `TestNetConstructor::test_invalid_protoString_raises` | ✅ PASSED |
| `TestNetConstructor::test_net_from_protoString_has_correct_counts` | ✅ PASSED |
| `TestNetConstructor::test_net_input_output_blobs` | ✅ PASSED |
| `TestNetConstructor::test_net_input_output_names` | ✅ PASSED |
| `TestNetConstructor::test_no_args_constructor` | ✅ PASSED |
| `TestNetAccess::test_blob_by_name_raises_on_missing` | ✅ PASSED |
| `TestNetAccess::test_blob_by_name_returns_blob` | ✅ PASSED |
| `TestNetAccess::test_blob_names` | ✅ PASSED |
| `TestNetAccess::test_blobs_array_returns_list` | ✅ PASSED |
| `TestNetAccess::test_blobs_dict` | ✅ PASSED |
| `TestNetAccess::test_contains` | ✅ PASSED |
| `TestNetAccess::test_getitem_access` | ✅ PASSED |
| `TestNetAccess::test_getitem_keyerror` | ✅ PASSED |
| `TestNetAccess::test_has_blob_false` | ✅ PASSED |
| `TestNetAccess::test_has_blob_true` | ✅ PASSED |
| `TestNetAccess::test_has_layer_false` | ✅ PASSED |
| `TestNetAccess::test_has_layer_true` | ✅ PASSED |
| `TestNetAccess::test_iter` | ✅ PASSED |
| `TestNetAccess::test_layer_by_name_raises_on_missing` | ✅ PASSED |
| `TestNetAccess::test_layer_by_name_returns_layer` | ✅ PASSED |
| `TestNetAccess::test_layer_names` | ✅ PASSED |
| `TestNetAccess::test_layers_array_returns_list` | ✅ PASSED |
| `TestNetAccess::test_layers_dict` | ✅ PASSED |
| `TestNetAccess::test_len` | ✅ PASSED |
| `TestNetAccess::test_repr` | ✅ PASSED |
| `TestNetForward::test_forward_all_kwargs` | ✅ PASSED |
| `TestNetForward::test_forward_mlp` | ✅ PASSED |
| `TestNetForward::test_forward_simple_input` | ✅ PASSED |
| `TestLayerAccess::test_layer_blobs_array_via_reflection` | ✅ PASSED |
| `TestLayerAccess::test_layer_blobs_for_inner_product` | ✅ PASSED |
| `TestLayerAccess::test_layer_name_property` | ✅ PASSED |
| `TestLayerAccess::test_layer_repr` | ✅ PASSED |
| `TestLayerAccess::test_layer_type_property` | ✅ PASSED |
| `TestConstructorEquivalence::test_mlp_layer_count_equivalent` | ✅ PASSED |
| `TestConstructorEquivalence::test_simple_input_equivalent` | ✅ PASSED |

### `test_reshape_backward.py`

| 用例 | 结果 |
|------|------|
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape0]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape1]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape2]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape3]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape4]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape5]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape6]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape7]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape8]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape9]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape10]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape11]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape12]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape13]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape14]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape15]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape16]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape17]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape18]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape19]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape20]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape21]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape22]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape23]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape24]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape25]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape26]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape27]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape28]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape29]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape30]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape31]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape32]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape33]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape34]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape35]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape36]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape37]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape38]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape39]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape40]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape41]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape42]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape43]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape44]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape45]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape46]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape47]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape48]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape49]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape50]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape51]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape52]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape53]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape54]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape55]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape56]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_flatten_all[input_shape57]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape0]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape1]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape2]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape3]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape4]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape5]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape6]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape7]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape8]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape9]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape10]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape11]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape12]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape13]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape14]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape15]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape16]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape17]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape18]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape19]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape20]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape21]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape22]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape23]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape24]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape25]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape26]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape27]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape28]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape29]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape30]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape31]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape32]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape33]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape34]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape35]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape36]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape37]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape38]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape39]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape40]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape41]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape42]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape43]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape44]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape45]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape46]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape47]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape48]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape49]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape50]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape51]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape52]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape53]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape54]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape55]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape56]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_to_batch_features[input_shape57]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape0]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape1]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape2]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape3]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape4]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape5]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape6]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape7]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape8]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape9]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape10]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape11]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape12]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape13]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape14]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape15]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape16]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape17]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape18]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape19]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape20]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_2d_flatten[input_shape21]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_1d_identity[1]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_1d_identity[2]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_1d_identity[5]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_1d_identity[10]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_1d_identity[100]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_1d_identity[128]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_1d_identity[256]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_1d_identity[512]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_1d_identity[784]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_5d_flatten[input_shape0]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_5d_flatten[input_shape1]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_5d_flatten[input_shape2]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_5d_flatten[input_shape3]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_5d_flatten[input_shape4]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_3d_flatten[input_shape0]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_3d_flatten[input_shape1]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_3d_flatten[input_shape2]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_3d_flatten[input_shape3]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_3d_flatten[input_shape4]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_3d_flatten[input_shape5]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_3d_flatten[input_shape6]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_3d_flatten[input_shape7]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_3d_flatten[input_shape8]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_3d_to_batch_features[input_shape0]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_3d_to_batch_features[input_shape1]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_3d_to_batch_features[input_shape2]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_3d_to_batch_features[input_shape3]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_3d_to_batch_features[input_shape4]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_3d_to_batch_features[input_shape5]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_3d_to_batch_features[input_shape6]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_3d_to_batch_features[input_shape7]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_3d_to_batch_features[input_shape8]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_partial[input_shape0-shape_dims0]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_partial[input_shape1-shape_dims1]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_partial[input_shape2-shape_dims2]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_partial[input_shape3-shape_dims3]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_partial[input_shape4-shape_dims4]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_1d_to_nd[input_shape0-shape_dims0]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_1d_to_nd[input_shape1-shape_dims1]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_1d_to_nd[input_shape2-shape_dims2]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_1d_to_nd[input_shape3-shape_dims3]` | ✅ PASSED |
| `TestReshapeBackwardIdentity::test_reshape_1d_to_nd[input_shape4-shape_dims4]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_flatten[input_shape0]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_flatten[input_shape1]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_flatten[input_shape2]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_flatten[input_shape3]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_flatten[input_shape4]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_flatten[input_shape5]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_flatten[input_shape6]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_flatten[input_shape7]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_flatten[input_shape8]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_flatten[input_shape9]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_flatten[input_shape10]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_flatten[input_shape11]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_flatten[input_shape12]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_flatten[input_shape13]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_flatten[input_shape14]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_flatten[input_shape15]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_flatten[input_shape16]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_flatten[input_shape17]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_flatten[input_shape18]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_flatten[input_shape19]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_flatten[input_shape20]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_batch_feat[input_shape0]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_batch_feat[input_shape1]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_batch_feat[input_shape2]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_batch_feat[input_shape3]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_batch_feat[input_shape4]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_batch_feat[input_shape5]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_batch_feat[input_shape6]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_batch_feat[input_shape7]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_batch_feat[input_shape8]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_batch_feat[input_shape9]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_2d_numerical_gradient[input_shape0]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_2d_numerical_gradient[input_shape1]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_2d_numerical_gradient[input_shape2]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_2d_numerical_gradient[input_shape3]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_2d_numerical_gradient[input_shape4]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_2d_numerical_gradient[input_shape5]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_2d_numerical_gradient[input_shape6]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_multi_dim[input_shape0-shape_dims0]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_multi_dim[input_shape1-shape_dims1]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_multi_dim[input_shape2-shape_dims2]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_multi_dim[input_shape3-shape_dims3]` | ✅ PASSED |
| `TestReshapeBackwardNumerical::test_reshape_numerical_gradient_multi_dim[input_shape4-shape_dims4]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_zero_dy_zero_dx[input_shape0]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_zero_dy_zero_dx[input_shape1]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_zero_dy_zero_dx[input_shape2]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_zero_dy_zero_dx[input_shape3]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_zero_dy_zero_dx[input_shape4]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_zero_dy_zero_dx[input_shape5]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_zero_dy_zero_dx[input_shape6]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_zero_dy_zero_dx[input_shape7]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_zero_dy_zero_dx[input_shape8]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_zero_dy_zero_dx[input_shape9]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_zero_dy_zero_dx[input_shape10]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_zero_dy_zero_dx[input_shape11]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_zero_dy_zero_dx[input_shape12]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_zero_dy_zero_dx[input_shape13]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_zero_dy_zero_dx[input_shape14]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_zero_dy_zero_dx[input_shape15]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_zero_dy_zero_dx[input_shape16]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_zero_dy_zero_dx[input_shape17]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_zero_dy_zero_dx[input_shape18]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_zero_dy_zero_dx[input_shape19]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_zero_dy_zero_dx[input_shape20]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_deterministic` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_shapes_dtypes[input_shape0-shape_dims0]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_shapes_dtypes[input_shape1-shape_dims1]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_shapes_dtypes[input_shape2-shape_dims2]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_shapes_dtypes[input_shape3-shape_dims3]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_forward_preserved_after_backward` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_reshape_in_chain[4-3-2]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_reshape_in_chain[8-3-4]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_reshape_in_chain[2-1-2]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_reshape_in_chain[16-3-10]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_no_learnable_params` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_reshape_flatten_equivalence` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_special_values_forward_backward[0.0]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_special_values_forward_backward[1.0]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_special_values_forward_backward[-1.0]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_special_values_forward_backward[2.0]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_special_values_forward_backward[-2.0]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_special_values_forward_backward[0.5]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_special_values_forward_backward[-0.5]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_special_values_forward_backward[100.0]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_special_values_forward_backward[-100.0]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_1d_5d_inputs[input_shape0]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_1d_5d_inputs[input_shape1]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_1d_5d_inputs[input_shape2]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_1d_5d_inputs[input_shape3]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_1d_5d_inputs[input_shape4]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_1d_5d_inputs[input_shape5]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_1d_5d_inputs[input_shape6]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_1d_5d_inputs[input_shape7]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_reshape_with_axis_num_axes[input_shape0-1-2-shape_dims0-expected_shape0]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_reshape_with_axis_num_axes[input_shape1-2--1-shape_dims1-expected_shape1]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_reshape_with_axis_num_axes[input_shape2-1--1-shape_dims2-expected_shape2]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_reshape_with_axis_num_axes[input_shape3-0-2-shape_dims3-expected_shape3]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_reshape_with_axis_num_axes[input_shape4-1-3-shape_dims4-expected_shape4]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_reshape_with_axis_num_axes[input_shape5-1-2-shape_dims5-expected_shape5]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_reshape_with_axis_num_axes[input_shape6-1--1-shape_dims6-expected_shape6]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_reshape_with_axis_num_axes[input_shape7-2-3-shape_dims7-expected_shape7]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_degenerate_shapes[input_shape0-shape_dims0]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_degenerate_shapes[input_shape1-shape_dims1]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_degenerate_shapes[input_shape2-shape_dims2]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_degenerate_shapes[input_shape3-shape_dims3]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_small_and_large_dy[input_shape0-0.001]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_small_and_large_dy[input_shape1--0.001]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_small_and_large_dy[input_shape2-1e-06]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_small_and_large_dy[input_shape3-3.14159]` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_copy_dim_zero_preserves_axis` | ✅ PASSED |
| `TestReshapeBackwardEdgeCases::test_reshape_inferred_dim_middle` | ✅ PASSED |

### `test_scale_backward.py`

| 用例 | 结果 |
|------|------|
| `TestScaleBackwardKnownValues::test_forward_identity_scale` | ✅ PASSED |
| `TestScaleBackwardKnownValues::test_forward_scale_only` | ✅ PASSED |
| `TestScaleBackwardKnownValues::test_forward_scale_plus_bias` | ✅ PASSED |
| `TestScaleBackwardKnownValues::test_backward_dx_known_values` | ✅ PASSED |
| `TestScaleBackwardKnownValues::test_backward_dscale_known_values` | ✅ PASSED |
| `TestScaleBackwardKnownValues::test_backward_dbias_known_values` | ✅ PASSED |
| `TestScaleBackwardAnalytical::test_dx_vs_numpy[2-4]` | ✅ PASSED |
| `TestScaleBackwardAnalytical::test_dx_vs_numpy[4-8]` | ✅ PASSED |
| `TestScaleBackwardAnalytical::test_dx_vs_numpy[1-16]` | ✅ PASSED |
| `TestScaleBackwardAnalytical::test_dscale_vs_numpy[2-4]` | ✅ PASSED |
| `TestScaleBackwardAnalytical::test_dscale_vs_numpy[4-8]` | ✅ PASSED |
| `TestScaleBackwardAnalytical::test_dbias_vs_numpy[2-4]` | ✅ PASSED |
| `TestScaleBackwardAnalytical::test_dbias_vs_numpy[4-8]` | ✅ PASSED |
| `TestScaleBackwardAnalytical::test_4d_spatial_shape` | ✅ PASSED |
| `TestScaleBackwardAnalytical::test_alpha_equals_zero` | ✅ PASSED |
| `TestScaleBackwardNumerical::test_numerical_grad_dx` | ✅ PASSED |
| `TestScaleBackwardNumerical::test_numerical_grad_dscale` | ✅ PASSED |
| `TestScaleBackwardNumerical::test_numerical_grad_dbias` | ✅ PASSED |
| `TestScaleBackwardProperties::test_zero_dy_gives_zero_gradients` | ✅ PASSED |
| `TestScaleBackwardProperties::test_gradient_shapes` | ✅ PASSED |
| `TestScaleBackwardProperties::test_determinism` | ✅ PASSED |
| `TestScaleBackwardProperties::test_forward_preserved_after_backward` | ✅ PASSED |
| `TestScaleBackwardProperties::test_bias_term_false_no_bias_blob` | ✅ PASSED |
| `TestScaleBackwardProperties::test_no_bias_dalpha_only` | ✅ PASSED |
| `TestScaleBackwardProperties::test_finite_values` | ✅ PASSED |

### `test_slice_backward.py`

| 用例 | 结果 |
|------|------|
| `TestSliceBackwardKnownValues::test_slice_axis0_2way` | ✅ PASSED |
| `TestSliceBackwardKnownValues::test_slice_axis1_2d` | ✅ PASSED |
| `TestSliceBackwardKnownValues::test_slice_axis1_nchw` | ✅ PASSED |
| `TestSliceBackwardKnownValues::test_slice_explicit_points` | ✅ PASSED |
| `TestSliceBackwardKnownValues::test_slice_n1_identity` | ✅ PASSED |
| `TestSliceBackwardNumpy::test_slice_vs_numpy[shape0-2-1]` | ✅ PASSED |
| `TestSliceBackwardNumpy::test_slice_vs_numpy[shape1-3-1]` | ✅ PASSED |
| `TestSliceBackwardNumpy::test_slice_vs_numpy[shape2-2-1]` | ✅ PASSED |
| `TestSliceBackwardNumpy::test_slice_vs_numpy[shape3-2-2]` | ✅ PASSED |
| `TestSliceBackwardNumpy::test_slice_vs_numpy[shape4-3-3]` | ✅ PASSED |
| `TestSliceBackwardNumerical::test_numerical_grad[shape0-2-0]` | ✅ PASSED |
| `TestSliceBackwardNumerical::test_numerical_grad[shape1-2-1]` | ✅ PASSED |
| `TestSliceBackwardNumerical::test_numerical_grad[shape2-3-1]` | ✅ PASSED |
| `TestSliceBackwardNumerical::test_numerical_grad[shape3-2-2]` | ✅ PASSED |
| `TestSliceBackwardProperties::test_zero_dy_gives_zero_gradients` | ✅ PASSED |
| `TestSliceBackwardProperties::test_gradient_shape` | ✅ PASSED |
| `TestSliceBackwardProperties::test_determinism` | ✅ PASSED |
| `TestSliceBackwardProperties::test_forward_preserved_after_backward` | ✅ PASSED |
| `TestSliceBackwardProperties::test_finite_values` | ✅ PASSED |
| `TestSliceBackwardProperties::test_round_trip_slice_concat` | ✅ PASSED |

### `test_softmax_backward.py`

| 用例 | 结果 |
|------|------|
| `TestSoftmaxBackwardKnownValues::test_uniform_input_gives_uniform_output` | ✅ PASSED |
| `TestSoftmaxBackwardKnownValues::test_two_class_confident` | ✅ PASSED |
| `TestSoftmaxBackwardKnownValues::test_three_class_known_values` | ✅ PASSED |
| `TestSoftmaxBackwardKnownValues::test_one_hot_dy_on_correct_class` | ✅ PASSED |
| `TestSoftmaxBackwardNumpy::test_softmax_vs_numpy[shape0-1]` | ✅ PASSED |
| `TestSoftmaxBackwardNumpy::test_softmax_vs_numpy[shape1-1]` | ✅ PASSED |
| `TestSoftmaxBackwardNumpy::test_softmax_vs_numpy[shape2-1]` | ✅ PASSED |
| `TestSoftmaxBackwardNumpy::test_softmax_vs_numpy[shape3-1]` | ✅ PASSED |
| `TestSoftmaxBackwardNumpy::test_softmax_vs_numpy[shape4-1]` | ✅ PASSED |
| `TestSoftmaxBackwardNumpy::test_softmax_vs_numpy[shape5-2]` | ✅ PASSED |
| `TestSoftmaxBackwardNumerical::test_numerical_grad[shape0-1]` | ✅ PASSED |
| `TestSoftmaxBackwardNumerical::test_numerical_grad[shape1-1]` | ✅ PASSED |
| `TestSoftmaxBackwardNumerical::test_numerical_grad[shape2-1]` | ✅ PASSED |
| `TestSoftmaxBackwardNumerical::test_numerical_grad[shape3-1]` | ✅ PASSED |
| `TestSoftmaxBackwardProperties::test_zero_dy_gives_zero_gradients` | ✅ PASSED |
| `TestSoftmaxBackwardProperties::test_gradient_shapes` | ✅ PASSED |
| `TestSoftmaxBackwardProperties::test_determinism` | ✅ PASSED |
| `TestSoftmaxBackwardProperties::test_forward_preserved_after_backward` | ✅ PASSED |
| `TestSoftmaxBackwardProperties::test_finite_values` | ✅ PASSED |
| `TestSoftmaxBackwardProperties::test_probability_sums_to_one` | ✅ PASSED |
| `TestSoftmaxBackwardProperties::test_gradient_sums_to_zero_per_position` | ✅ PASSED |
| `TestSoftmaxBackwardProperties::test_gradient_when_dy_equals_y` | ✅ PASSED |

### `test_softmax_loss_backward.py`

| 用例 | 结果 |
|------|------|
| `TestSoftmaxWithLossBackward::test_sml_known_perfect_predictions` | ✅ PASSED |
| `TestSoftmaxWithLossBackward::test_sml_known_uniform` | ✅ PASSED |
| `TestSoftmaxWithLossBackward::test_sml_gradient_sums_to_zero` | ✅ PASSED |
| `TestSoftmaxWithLossBackward::test_sml_analytical_vs_numpy` | ✅ PASSED |
| `TestSoftmaxWithLossBackward::test_sml_numerical_gradient` | ✅ PASSED |
| `TestSoftmaxWithLossBackward::test_sml_numerical_gradient_spatial` | ✅ PASSED |
| `TestSoftmaxWithLossBackward::test_sml_loss_weight_scaling` | ✅ PASSED |
| `TestSoftmaxWithLossBackward::test_sml_ignore_label` | ✅ PASSED |
| `TestSoftmaxWithLossBackward::test_sml_deterministic` | ✅ PASSED |
| `TestSoftmaxWithLossBackward::test_sml_no_nan_inf` | ✅ PASSED |
| `TestSoftmaxWithLossBackward::test_sml_forward_preserved` | ✅ PASSED |
| `TestSoftmaxWithLossBackward::test_sml_multi_sample_consistency` | ✅ PASSED |

### `test_split_backward.py`

| 用例 | 结果 |
|------|------|
| `TestSplitBackwardKnownValues::test_split_n2_identity_accumulation` | ✅ PASSED |
| `TestSplitBackwardKnownValues::test_split_n3_identity_accumulation` | ✅ PASSED |
| `TestSplitBackwardKnownValues::test_split_n1_identity_passthrough` | ✅ PASSED |
| `TestSplitBackwardNumpy::test_split_vs_numpy[shape0-2]` | ✅ PASSED |
| `TestSplitBackwardNumpy::test_split_vs_numpy[shape1-3]` | ✅ PASSED |
| `TestSplitBackwardNumpy::test_split_vs_numpy[shape2-2]` | ✅ PASSED |
| `TestSplitBackwardNumpy::test_split_vs_numpy[shape3-2]` | ✅ PASSED |
| `TestSplitBackwardNumpy::test_split_vs_numpy[shape4-4]` | ✅ PASSED |
| `TestSplitBackwardNumerical::test_numerical_grad[shape0-2]` | ✅ PASSED |
| `TestSplitBackwardNumerical::test_numerical_grad[shape1-2]` | ✅ PASSED |
| `TestSplitBackwardNumerical::test_numerical_grad[shape2-3]` | ✅ PASSED |
| `TestSplitBackwardProperties::test_zero_dy_gives_zero_gradients` | ✅ PASSED |
| `TestSplitBackwardProperties::test_gradient_shape` | ✅ PASSED |
| `TestSplitBackwardProperties::test_determinism` | ✅ PASSED |
| `TestSplitBackwardProperties::test_forward_preserved_after_backward` | ✅ PASSED |
| `TestSplitBackwardProperties::test_finite_values` | ✅ PASSED |
| `TestSplitBackwardProperties::test_round_trip_split_concat` | ✅ PASSED |

### `test_split_concat_bench.py`

| 用例 | 结果 |
|------|------|
| `TestSplitConcatBenchmark::test_construction_overhead_scales_linearly` | ✅ PASSED |
| `TestSplitConcatBenchmark::test_forward_correctness_all_scenarios` | ✅ PASSED |
| `TestSplitConcatBenchmark::test_split_count_matches_expected` | ✅ PASSED |
| `TestSplitConcatBenchmark::test_print_benchmark_table` | ✅ PASSED |

### `test_split_topologies.py`

| 用例 | 结果 |
|------|------|
| `TestSplitTopologies::test_split_1to2_copies_data` | ✅ PASSED |
| `TestSplitTopologies::test_split_1to3_concat_roundtrip` | ✅ PASSED |
| `TestSplitTopologies::test_residual_with_split` | ✅ PASSED |
| `TestSplitTopologies::test_split_inplace_branch_isolation` | ✅ PASSED |
| `TestSplitTopologies::test_n1_split_passthrough` | ✅ PASSED |
| `TestSplitTopologies::test_split_deterministic_repeated_forward` | ✅ PASSED |
| `TestSplitTopologies::test_split_perf_scaling` | ✅ PASSED |

---

## 附：验证说明

- 本次回归在 WSL docker 容器内执行，确认 CMake 原子化重构（10 个模块化 cmake 文件）构建与运行正常。
- 修复了 editable install 路径下 stale `_caffe_ffi.so` 问题（将 `build/python/caffe_ffi/_caffe_ffi.so` 复制到源码树 `python/caffe_ffi/`）。
- 确认 lazy allocation 触发（N≥16 时 Split 层使用 `SetShapeOnly`）。
- 对应规划：[tasks.md#Task18](caffe-ffi-tvm-integration/tasks.md) 中的完整 P3 回归验证记录。
