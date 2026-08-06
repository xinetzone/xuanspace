#include "caffe_ffi/layers/batch_norm_layer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

#include <tvm/ffi/memory.h>

#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/math_utils.hpp"

namespace caffe_ffi {

void BatchNormLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                                 const std::vector<Blob*>& top) {
  const caffe::BatchNormParameter& param = this->layer_param_.batch_norm_param();
  use_global_stats_ = param.use_global_stats();
  moving_average_fraction_ = param.moving_average_fraction();
  eps_ = param.eps();

  CAFFE_FFI_LAYER_LOG << "BatchNorm LayerSetUp: use_global_stats_=" << use_global_stats_
                      << " moving_average_fraction_=" << moving_average_fraction_
                      << " eps_=" << eps_;

  if (this->blobs_.size() > 0) {
    CAFFE_FFI_LAYER_LOG << "BatchNorm: using pre-loaded weights, blobs_.size=" << this->blobs_.size();
    CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_.size(), 3U)
        << "Incorrect number of batch norm blobs.";
  } else {
    this->blobs_.resize(3);
    channels_ = static_cast<int>(bottom[0]->shape(1));
    std::vector<int64_t> sz = {channels_};
    this->blobs_[0] = make_object<Blob>(sz);
    this->blobs_[1] = make_object<Blob>(sz);
    std::vector<int64_t> one = {1};
    this->blobs_[2] = make_object<Blob>(one);
    caffe_set_fp32(static_cast<size_t>(this->blobs_[2]->count()), 1.0f, this->blobs_[2]->cpu_mutable_data());
    CAFFE_FFI_TENSOR_LOG << "BatchNorm: created mean blob shape=[" << channels_ << "]";
    CAFFE_FFI_TENSOR_LOG << "BatchNorm: created variance blob shape=[" << channels_ << "]";
    CAFFE_FFI_TENSOR_LOG << "BatchNorm: created scale factor blob shape=[1] (initialized to 1.0)";
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

void BatchNormLayer::Reshape(const std::vector<Blob*>& bottom,
                              const std::vector<Blob*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  channels_ = static_cast<int>(bottom[0]->shape(1));
  if (bottom[0]->num_axes() == 1) {
    channels_ = 1;
  }

  std::ostringstream input_shape_ss;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) input_shape_ss << ", ";
    input_shape_ss << bottom[0]->shape(i);
  }
  std::ostringstream output_shape_ss;
  for (int i = 0; i < top[0]->num_axes(); ++i) {
    if (i > 0) output_shape_ss << ", ";
    output_shape_ss << top[0]->shape(i);
  }

  std::vector<int64_t> sz = {channels_};
  if (!this->blobs_[0] || this->blobs_[0]->count() != channels_) {
    this->blobs_[0] = make_object<Blob>(sz);
    this->blobs_[1] = make_object<Blob>(sz);
    CAFFE_FFI_TENSOR_LOG << "BatchNorm Reshape: recreated mean/variance blobs shape=[" << channels_ << "]";
  }

  int spatial_dim = static_cast<int>(bottom[0]->count(2));
  if (bottom[0]->num_axes() == 1) {
    spatial_dim = 1;
  }

  CAFFE_FFI_LAYER_LOG << "BatchNorm Reshape: input=[" << input_shape_ss.str()
                      << "] output=[" << output_shape_ss.str()
                      << "] channels_=" << channels_
                      << " spatial_dim=" << spatial_dim;
}

void BatchNormLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                  const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int num = static_cast<int>(bottom[0]->shape(0));
  const int channels = channels_;
  int spatial_dim = static_cast<int>(bottom[0]->count(2));
  if (bottom[0]->num_axes() == 1) {
    spatial_dim = 1;
  }

  const float* mean = this->blobs_[0]->cpu_data();
  const float* variance = this->blobs_[1]->cpu_data();
  const float scale_factor = this->blobs_[2]->cpu_data()[0] == 0.0f
      ? 0.0f
      : 1.0f / this->blobs_[2]->cpu_data()[0];

  CAFFE_FFI_LAYER_LOG << "BatchNorm Forward: num=" << num
                      << " channels=" << channels
                      << " spatial_dim=" << spatial_dim
                      << " use_global_stats_=" << use_global_stats_
                      << " scale_factor=" << scale_factor;

  const float scale_factor_use = scale_factor == 0.0f ? 1.0f : scale_factor;
  const int64_t count = bottom[0]->count();

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  float in_min = std::numeric_limits<float>::max();
  float in_max = -std::numeric_limits<float>::max();
  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();
#endif

  for (int i = 0; i < count; ++i) {
    int c = (i / spatial_dim) % channels;
    float x = bottom_data[i];
    float y = (x - mean[c] * scale_factor_use)
        / std::sqrt(std::max(variance[c] * scale_factor_use, 0.0f) + eps_);
    top_data[i] = y;
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
    in_min = std::min(in_min, x);
    in_max = std::max(in_max, x);
    out_min = std::min(out_min, y);
    out_max = std::max(out_max, y);
#endif
  }

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  float mean_min = std::numeric_limits<float>::max();
  float mean_max = -std::numeric_limits<float>::max();
  float var_min = std::numeric_limits<float>::max();
  float var_max = -std::numeric_limits<float>::max();
  for (int c = 0; c < channels; ++c) {
    float m = mean[c] * scale_factor_use;
    float v = variance[c] * scale_factor_use;
    mean_min = std::min(mean_min, m);
    mean_max = std::max(mean_max, m);
    var_min = std::min(var_min, v);
    var_max = std::max(var_max, v);
  }

  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[BN-PERF] " << this->name()
                       << " BatchNorm forward: num=" << num
                       << " channels=" << channels
                       << " spatial_dim=" << spatial_dim
                       << " use_global_stats=" << use_global_stats_
                       << " eps=" << eps_
                       << " in=[" << in_min << ", " << in_max << "]"
                       << " out=[" << out_min << ", " << out_max << "]"
                       << " mean=[" << mean_min << ", " << mean_max << "]"
                       << " var=[" << var_min << ", " << var_max << "]"
                       << " time=" << elapsed_us << "us";
#endif
}

void BatchNormLayer::Backward_cpu(const std::vector<Blob*>& top,
                                   const std::vector<bool>& propagate_down,
                                   const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    CAFFE_FFI_LAYER_LOG << "BatchNorm Backward_cpu: propagate_down[0]=false, skipping";
    return;
  }

  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const int num = static_cast<int>(bottom[0]->shape(0));
  const int channels = channels_;
  int spatial_dim = static_cast<int>(bottom[0]->count(2));
  if (bottom[0]->num_axes() == 1) {
    spatial_dim = 1;
  }

  const float* variance = this->blobs_[1]->cpu_data();
  const float scale_factor = this->blobs_[2]->cpu_data()[0] == 0.0f
      ? 0.0f
      : 1.0f / this->blobs_[2]->cpu_data()[0];
  const float scale_factor_use = scale_factor == 0.0f ? 1.0f : scale_factor;
  const int64_t count = bottom[0]->count();

  CAFFE_FFI_LAYER_LOG << "BatchNorm Backward: num=" << num
                      << " channels=" << channels
                      << " spatial_dim=" << spatial_dim
                      << " scale_factor_use=" << scale_factor_use
                      << " eps=" << eps_;

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  float inv_std_min = std::numeric_limits<float>::max();
  float inv_std_max = -std::numeric_limits<float>::max();
#endif

  std::vector<float> inv_std(channels);
  for (int c = 0; c < channels; ++c) {
    float var_c = std::max(variance[c] * scale_factor_use, 0.0f);
    inv_std[c] = 1.0f / std::sqrt(var_c + eps_);
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
    inv_std_min = std::min(inv_std_min, inv_std[c]);
    inv_std_max = std::max(inv_std_max, inv_std[c]);
#endif
  }

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  float diff_in_min = std::numeric_limits<float>::max();
  float diff_in_max = -std::numeric_limits<float>::max();
  float diff_out_min = std::numeric_limits<float>::max();
  float diff_out_max = -std::numeric_limits<float>::max();
#endif

  for (int64_t i = 0; i < count; ++i) {
    int c = static_cast<int>((i / spatial_dim) % channels);
    float dy = top_diff[i];
    float dx = dy * inv_std[c];
    bottom_diff[i] = dx;
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
    diff_in_min = std::min(diff_in_min, dy);
    diff_in_max = std::max(diff_in_max, dy);
    diff_out_min = std::min(diff_out_min, dx);
    diff_out_max = std::max(diff_out_max, dx);
#endif
  }

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[BN-PERF] " << this->name()
                       << " BatchNorm backward: num=" << num
                       << " channels=" << channels
                       << " spatial_dim=" << spatial_dim
                       << " inv_std=[" << inv_std_min << ", " << inv_std_max << "]"
                       << " diff_in=[" << diff_in_min << ", " << diff_in_max << "]"
                       << " diff_out=[" << diff_out_min << ", " << diff_out_max << "]"
                       << " time=" << elapsed_us << "us";
#endif
}

REGISTER_LAYER_CLASS(BatchNorm);

}  // namespace caffe_ffi
