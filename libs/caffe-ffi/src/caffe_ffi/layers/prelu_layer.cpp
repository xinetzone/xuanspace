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
  NeuronLayer::Reshape(bottom, top);

  if (!channel_shared_) {
    if (bottom[0]->num_axes() == 1) {
      channels_ = 1;
      inner_dim_ = 1;
    } else {
      channels_ = bottom[0]->shape(1);
      inner_dim_ = bottom[0]->count(2);
    }
    CAFFE_FFI_LAYER_LOG << "PReLU Reshape: channels_=" << channels_
                        << " inner_dim_=" << inner_dim_;
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

void PReLULayer::Backward_cpu(const std::vector<Blob*>& top,
                               const std::vector<bool>& propagate_down,
                               const std::vector<Blob*>& bottom) {
  const float* bottom_data = bottom[0]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  const float* slope_data = this->blobs_[0]->cpu_data();
  const int64_t count = bottom[0]->count();
  const bool prop_down = propagate_down[0];
  const bool prop_slope = this->param_propagate_down(0);

  CAFFE_FFI_LAYER_LOG << "PReLU Backward_cpu: count=" << count
                      << " channel_shared=" << channel_shared_
                      << " prop_down=" << prop_down
                      << " prop_slope=" << prop_slope;

  if (!prop_down && !prop_slope) {
    CAFFE_FFI_LAYER_LOG << "PReLU Backward_cpu: nothing to propagate";
    return;
  }

  float* bottom_diff = nullptr;
  if (prop_down) {
    bottom_diff = bottom[0]->cpu_mutable_diff();
  }

  float* slope_diff = nullptr;
  if (prop_slope) {
    slope_diff = this->blobs_[0]->cpu_mutable_diff();
    caffe_set_fp32(static_cast<size_t>(this->blobs_[0]->count()), 0.0f, slope_diff);
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  float diff_in_min = std::numeric_limits<float>::max();
  float diff_in_max = -std::numeric_limits<float>::max();
  float diff_out_min = std::numeric_limits<float>::max();
  float diff_out_max = -std::numeric_limits<float>::max();
  float slope_diff_min = std::numeric_limits<float>::max();
  float slope_diff_max = -std::numeric_limits<float>::max();
  int64_t dead_count = 0;

  if (channel_shared_) {
    const float slope = slope_data[0];
    for (int64_t i = 0; i < count; ++i) {
      float dy = top_diff[i];
      float x = bottom_data[i];
      float dx;
      if (x > 0.0f) {
        dx = dy;
      } else {
        dx = dy * slope;
        if (prop_slope) {
          slope_diff[0] += dy * x;
        }
        dead_count++;
      }
      if (prop_down) {
        bottom_diff[i] = dx;
      }
      diff_in_min = std::min(diff_in_min, dy);
      diff_in_max = std::max(diff_in_max, dy);
      if (prop_down) {
        diff_out_min = std::min(diff_out_min, dx);
        diff_out_max = std::max(diff_out_max, dx);
      }
    }
    if (prop_slope) {
      slope_diff_min = slope_diff_max = slope_diff[0];
    }
  } else {
    for (int64_t i = 0; i < count; ++i) {
      float dy = top_diff[i];
      float x = bottom_data[i];
      int c = static_cast<int>((i / inner_dim_) % channels_);
      float slope = slope_data[c];
      float dx;
      if (x > 0.0f) {
        dx = dy;
      } else {
        dx = dy * slope;
        if (prop_slope) {
          slope_diff[c] += dy * x;
        }
        dead_count++;
      }
      if (prop_down) {
        bottom_diff[i] = dx;
      }
      diff_in_min = std::min(diff_in_min, dy);
      diff_in_max = std::max(diff_in_max, dy);
      if (prop_down) {
        diff_out_min = std::min(diff_out_min, dx);
        diff_out_max = std::max(diff_out_max, dx);
      }
    }
    if (prop_slope) {
      slope_diff_min = std::numeric_limits<float>::max();
      slope_diff_max = -std::numeric_limits<float>::max();
      for (int c = 0; c < channels_; ++c) {
        slope_diff_min = std::min(slope_diff_min, slope_diff[c]);
        slope_diff_max = std::max(slope_diff_max, slope_diff[c]);
      }
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  float dead_ratio = static_cast<float>(dead_count) / static_cast<float>(count);

  std::ostringstream perf_ss;
  perf_ss << "[ACTIVATION-PERF] " << this->name()
          << " PReLU backward: count=" << count
          << " channel_shared=" << (channel_shared_ ? "true" : "false")
          << " diff_in=[" << diff_in_min << ", " << diff_in_max << "]";
  if (prop_down) {
    perf_ss << " diff_out=[" << diff_out_min << ", " << diff_out_max << "]";
  }
  if (prop_slope) {
    perf_ss << " slope_diff=[" << slope_diff_min << ", " << slope_diff_max << "]";
  }
  perf_ss << " dead=" << dead_count << "/" << count
          << " (" << dead_ratio << ")"
          << " time=" << elapsed_us << "us";
  CAFFE_FFI_LOG_INFO() << perf_ss.str();
}

REGISTER_LAYER_CLASS(PReLU);

}  // namespace caffe_ffi
