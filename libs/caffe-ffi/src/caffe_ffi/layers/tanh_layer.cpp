#include "caffe_ffi/layers/tanh_layer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void TanHLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int64_t count = bottom[0]->count();
  CAFFE_FFI_LAYER_LOG << "TanH Forward_cpu: count=" << count;

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
    float y = std::tanh(x);
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
                       << " TanH forward: count=" << count
                       << " in=[" << in_min << ", " << in_max << "]"
                       << " out=[" << out_min << ", " << out_max << "]"
                       << " time=" << elapsed_us << "us";
#endif
}

void TanHLayer::Backward_cpu(const std::vector<Blob*>& top,
                              const std::vector<bool>& propagate_down,
                              const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    CAFFE_FFI_LAYER_LOG << "TanH Backward_cpu: propagate_down[0]=false, skipping";
    return;
  }

  const float* top_data = top[0]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const int64_t count = bottom[0]->count();
  CAFFE_FFI_LAYER_LOG << "TanH Backward_cpu: count=" << count;

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  float diff_in_min = std::numeric_limits<float>::max();
  float diff_in_max = -std::numeric_limits<float>::max();
  float diff_out_min = std::numeric_limits<float>::max();
  float diff_out_max = -std::numeric_limits<float>::max();
  int64_t saturated_count = 0;
#endif

  for (int64_t i = 0; i < count; ++i) {
    float dy = top_diff[i];
    float y = top_data[i];
    float dx = dy * (1.0f - y * y);
    bottom_diff[i] = dx;

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
    diff_in_min = std::min(diff_in_min, dy);
    diff_in_max = std::max(diff_in_max, dy);
    diff_out_min = std::min(diff_out_min, dx);
    diff_out_max = std::max(diff_out_max, dx);

    if (1.0f - y * y < 1e-4f) {
      saturated_count++;
    }
#endif
  }

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  float saturate_ratio = static_cast<float>(saturated_count) / static_cast<float>(count);

  CAFFE_FFI_LOG_INFO() << "[ACTIVATION-PERF] " << this->name()
                       << " TanH backward: count=" << count
                       << " diff_in=[" << diff_in_min << ", " << diff_in_max << "]"
                       << " diff_out=[" << diff_out_min << ", " << diff_out_max << "]"
                       << " saturate=" << saturated_count << "/" << count
                       << " (" << saturate_ratio << ")"
                       << " time=" << elapsed_us << "us";
#endif
}

REGISTER_LAYER_CLASS(TanH);

}  // namespace caffe_ffi
