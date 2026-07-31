#include "caffe_ffi/layers/prelu_layer.hpp"

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

void PReLULayer::LayerSetUp(const std::vector<Blob*>& bottom,
                             const std::vector<Blob*>& top) {
  const caffe::PReLUParameter& param = this->layer_param_.prelu_param();
  channel_shared_ = param.channel_shared();

  CAFFE_FFI_LAYER_LOG << "PReLU LayerSetUp: channel_shared=" << channel_shared_;

  if (this->blobs_.size() > 0) {
    CAFFE_FFI_LAYER_LOG << "PReLU: using pre-loaded weights, blobs_.size=" << this->blobs_.size();
    CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_.size(), 1U)
        << "PReLU takes exactly one blob for slope.";
  } else {
    this->blobs_.resize(1);
    if (channel_shared_) {
      this->blobs_[0] = make_object<Blob>(std::vector<int64_t>{1});
      CAFFE_FFI_TENSOR_LOG << "PReLU: created slope blob (channel_shared) shape=[1]";
    } else {
      if (bottom[0]->num_axes() == 1) {
        channels_ = 1;
      } else {
        channels_ = bottom[0]->shape(1);
      }
      this->blobs_[0] = make_object<Blob>(std::vector<int64_t>{channels_});
      CAFFE_FFI_TENSOR_LOG << "PReLU: created slope blob (per-channel) shape=[" << channels_ << "]";
    }
    caffe_set_fp32(static_cast<size_t>(this->blobs_[0]->count()), 0.25f, this->blobs_[0]->cpu_mutable_data());
    if (param.has_filler()) {
      const caffe::FillerParameter& filler = param.filler();
      float value = filler.value();
      CAFFE_FFI_LAYER_LOG << "PReLU: using custom filler value=" << value;
      caffe_set_fp32(static_cast<size_t>(this->blobs_[0]->count()), value, this->blobs_[0]->cpu_mutable_data());
    }
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

void PReLULayer::Reshape(const std::vector<Blob*>& bottom,
                          const std::vector<Blob*>& top) {
  top[0]->ReshapeLike(*bottom[0]);

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

  if (!channel_shared_) {
    if (bottom[0]->num_axes() == 1) {
      channels_ = 1;
      inner_dim_ = 1;
    } else {
      channels_ = bottom[0]->shape(1);
      inner_dim_ = bottom[0]->count(2);
    }
    CAFFE_FFI_LAYER_LOG << "PReLU Reshape: input=[" << input_shape_ss.str()
                        << "] output=[" << output_shape_ss.str()
                        << "] channels_=" << channels_
                        << " inner_dim_=" << inner_dim_;
  } else {
    CAFFE_FFI_LAYER_LOG << "PReLU Reshape: input=[" << input_shape_ss.str()
                        << "] output=[" << output_shape_ss.str()
                        << "] (channel_shared mode)";
  }
}

void PReLULayer::Forward_cpu(const std::vector<Blob*>& bottom,
                              const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int64_t count = bottom[0]->count();
  const float* slope_data = this->blobs_[0]->cpu_data();

  CAFFE_FFI_LAYER_LOG << "PReLU Forward: count=" << count
                      << " channel_shared=" << channel_shared_;

  auto t_start = std::chrono::high_resolution_clock::now();

  float in_min = std::numeric_limits<float>::max();
  float in_max = -std::numeric_limits<float>::max();
  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();
  float slope_min = std::numeric_limits<float>::max();
  float slope_max = -std::numeric_limits<float>::max();

  if (channel_shared_) {
    const float slope = slope_data[0];
    slope_min = slope_max = slope;
    for (int64_t i = 0; i < count; ++i) {
      float x = bottom_data[i];
      float y = std::max(x, 0.0f) + slope * std::min(x, 0.0f);
      top_data[i] = y;
      in_min = std::min(in_min, x);
      in_max = std::max(in_max, x);
      out_min = std::min(out_min, y);
      out_max = std::max(out_max, y);
    }
  } else {
    for (int c = 0; c < channels_; ++c) {
      float s = slope_data[c];
      slope_min = std::min(slope_min, s);
      slope_max = std::max(slope_max, s);
    }
    for (int64_t i = 0; i < count; ++i) {
      float x = bottom_data[i];
      int c = static_cast<int>((i / inner_dim_) % channels_);
      float y = std::max(x, 0.0f) + slope_data[c] * std::min(x, 0.0f);
      top_data[i] = y;
      in_min = std::min(in_min, x);
      in_max = std::max(in_max, x);
      out_min = std::min(out_min, y);
      out_max = std::max(out_max, y);
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[ACTIVATION-PERF] " << this->name()
                       << " PReLU forward: count=" << count
                       << " channel_shared=" << (channel_shared_ ? "true" : "false")
                       << " slope=[" << slope_min << ", " << slope_max << "]"
                       << " in=[" << in_min << ", " << in_max << "]"
                       << " out=[" << out_min << ", " << out_max << "]"
                       << " time=" << elapsed_us << "us";
}

REGISTER_LAYER_CLASS(PReLU);

}  // namespace caffe_ffi
