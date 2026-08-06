#include "caffe_ffi/layers/softsign_layer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void SoftsignLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int64_t count = bottom[0]->count();
  CAFFE_FFI_LAYER_LOG << "Softsign Forward_cpu: count=" << count;

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  float in_min = std::numeric_limits<float>::max();
  float in_max = -std::numeric_limits<float>::max();
  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();
#endif

  for (int64_t i = 0; i < count; ++i) {
    float x = bottom_data[i];
    float y = x / (1.0f + std::abs(x));
    top_data[i] = y;
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
    in_min = std::min(in_min, x);
    in_max = std::max(in_max, x);
    out_min = std::min(out_min, y);
    out_max = std::max(out_max, y);
#endif
  }

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[ACTIVATION-PERF] " << this->name()
                       << " Softsign forward: count=" << count
                       << " in=[" << in_min << ", " << in_max << "]"
                       << " out=[" << out_min << ", " << out_max << "]"
                       << " time=" << elapsed_us << "us";
#endif
}

void SoftsignLayer::Backward_cpu(const std::vector<Blob*>& top,
                                 const std::vector<bool>& propagate_down,
                                 const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    CAFFE_FFI_LAYER_LOG << "Softsign Backward_cpu: propagate_down[0]=false, skipping";
    return;
  }

  const float* bottom_data = bottom[0]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const int64_t count = bottom[0]->count();
  CAFFE_FFI_LAYER_LOG << "Softsign Backward_cpu: count=" << count;

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  float diff_in_min = std::numeric_limits<float>::max();
  float diff_in_max = -std::numeric_limits<float>::max();
  float diff_out_min = std::numeric_limits<float>::max();
  float diff_out_max = -std::numeric_limits<float>::max();
#endif

  for (int64_t i = 0; i < count; ++i) {
    float dy = top_diff[i];
    float x = bottom_data[i];
    float denom = 1.0f + std::abs(x);
    float dx = dy / (denom * denom);
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

  CAFFE_FFI_LOG_INFO() << "[ACTIVATION-PERF] " << this->name()
                       << " Softsign backward: count=" << count
                       << " diff_in=[" << diff_in_min << ", " << diff_in_max << "]"
                       << " diff_out=[" << diff_out_min << ", " << diff_out_max << "]"
                       << " time=" << elapsed_us << "us";
#endif
}

REGISTER_LAYER_CLASS(Softsign);

}  // namespace caffe_ffi
