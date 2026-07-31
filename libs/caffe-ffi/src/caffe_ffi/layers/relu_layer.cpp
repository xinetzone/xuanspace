#include "caffe_ffi/layers/relu_layer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include "caffe_ffi/error.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void ReLULayer::Forward_cpu(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int64_t count = bottom[0]->count();
  float negative_slope = this->layer_param_.relu_param().negative_slope();
  CAFFE_FFI_LAYER_LOG << "ReLU Forward_cpu: count=" << count
                      << " negative_slope=" << negative_slope;

  auto t_start = std::chrono::high_resolution_clock::now();

  float in_min = std::numeric_limits<float>::max();
  float in_max = -std::numeric_limits<float>::max();
  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();

  for (int64_t i = 0; i < count; ++i) {
    float x = bottom_data[i];
    float y = std::max(x, 0.0f) + negative_slope * std::min(x, 0.0f);
    top_data[i] = y;
    in_min = std::min(in_min, x);
    in_max = std::max(in_max, x);
    out_min = std::min(out_min, y);
    out_max = std::max(out_max, y);
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[ACTIVATION-PERF] " << this->name()
                       << " ReLU forward: count=" << count
                       << " negative_slope=" << negative_slope
                       << " in=[" << in_min << ", " << in_max << "]"
                       << " out=[" << out_min << ", " << out_max << "]"
                       << " time=" << elapsed_us << "us";
}

void ReLULayer::Backward_cpu(const std::vector<Blob*>& top,
                              const std::vector<bool>& propagate_down,
                              const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    CAFFE_FFI_LAYER_LOG << "ReLU Backward_cpu: propagate_down[0]=false, skipping";
    return;
  }

  const float* bottom_data = bottom[0]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const int64_t count = bottom[0]->count();
  float negative_slope = this->layer_param_.relu_param().negative_slope();
  CAFFE_FFI_LAYER_LOG << "ReLU Backward_cpu: count=" << count
                      << " negative_slope=" << negative_slope;

  auto t_start = std::chrono::high_resolution_clock::now();

  float diff_in_min = std::numeric_limits<float>::max();
  float diff_in_max = -std::numeric_limits<float>::max();
  float diff_out_min = std::numeric_limits<float>::max();
  float diff_out_max = -std::numeric_limits<float>::max();
  int64_t dead_count = 0;

  for (int64_t i = 0; i < count; ++i) {
    float dy = top_diff[i];
    float x = bottom_data[i];
    float dx = dy * (x > 0.0f ? 1.0f : negative_slope);
    bottom_diff[i] = dx;

    diff_in_min = std::min(diff_in_min, dy);
    diff_in_max = std::max(diff_in_max, dy);
    diff_out_min = std::min(diff_out_min, dx);
    diff_out_max = std::max(diff_out_max, dx);

    if (x <= 0.0f) {
      dead_count++;
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  float dead_ratio = static_cast<float>(dead_count) / static_cast<float>(count);

  CAFFE_FFI_LOG_INFO() << "[ACTIVATION-PERF] " << this->name()
                       << " ReLU backward: count=" << count
                       << " negative_slope=" << negative_slope
                       << " diff_in=[" << diff_in_min << ", " << diff_in_max << "]"
                       << " diff_out=[" << diff_out_min << ", " << diff_out_max << "]"
                       << " dead=" << dead_count << "/" << count
                       << " (" << dead_ratio << ")"
                       << " time=" << elapsed_us << "us";
}

REGISTER_LAYER_CLASS(ReLU);

}  // namespace caffe_ffi
