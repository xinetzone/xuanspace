#include "caffe_ffi/layers/softplus_layer.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void SoftplusLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int64_t count = bottom[0]->count();
  CAFFE_FFI_LAYER_LOG << "Softplus Forward_cpu: count=" << count;

  auto t_start = std::chrono::high_resolution_clock::now();

  float in_min = std::numeric_limits<float>::max();
  float in_max = -std::numeric_limits<float>::max();
  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();

  // y = log(1 + exp(x)). Numerically stable branch: for x > 0 compute
  // x + log1p(exp(-x)) to avoid exp(x) overflow for large x; for x <= 0 the
  // naive log1p(exp(x)) is well-behaved (|exp(x)| <= 1).
  for (int64_t i = 0; i < count; ++i) {
    float x = bottom_data[i];
    float y = (x > 0.0f) ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x));
    top_data[i] = y;
    in_min = std::min(in_min, x);
    in_max = std::max(in_max, x);
    out_min = std::min(out_min, y);
    out_max = std::max(out_max, y);
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[ACTIVATION-PERF] " << this->name()
                       << " Softplus forward: count=" << count
                       << " in=[" << in_min << ", " << in_max << "]"
                       << " out=[" << out_min << ", " << out_max << "]"
                       << " time=" << elapsed_us << "us";
}

void SoftplusLayer::Backward_cpu(const std::vector<Blob*>& top,
                                 const std::vector<bool>& propagate_down,
                                 const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    CAFFE_FFI_LAYER_LOG << "Softplus Backward_cpu: propagate_down[0]=false, skipping";
    return;
  }

  const float* bottom_data = bottom[0]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const int64_t count = bottom[0]->count();
  CAFFE_FFI_LAYER_LOG << "Softplus Backward_cpu: count=" << count;

  auto t_start = std::chrono::high_resolution_clock::now();

  float diff_in_min = std::numeric_limits<float>::max();
  float diff_in_max = -std::numeric_limits<float>::max();
  float diff_out_min = std::numeric_limits<float>::max();
  float diff_out_max = -std::numeric_limits<float>::max();

  // d/dx [ log(1 + exp(x)) ] = 1 / (1 + exp(-x)) = logistic sigmoid(x).
  // Compute the sigmoid in the overflow-safe form: for x < 0 use exp(x)/(1+exp(x)).
  for (int64_t i = 0; i < count; ++i) {
    float dy = top_diff[i];
    float x = bottom_data[i];
    float sigmoid = (x >= 0.0f) ? 1.0f / (1.0f + std::exp(-x)) : std::exp(x) / (1.0f + std::exp(x));
    float dx = dy * sigmoid;
    bottom_diff[i] = dx;

    diff_in_min = std::min(diff_in_min, dy);
    diff_in_max = std::max(diff_in_max, dy);
    diff_out_min = std::min(diff_out_min, dx);
    diff_out_max = std::max(diff_out_max, dx);
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[ACTIVATION-PERF] " << this->name()
                       << " Softplus backward: count=" << count
                       << " diff_in=[" << diff_in_min << ", " << diff_in_max << "]"
                       << " diff_out=[" << diff_out_min << ", " << diff_out_max << "]"
                       << " time=" << elapsed_us << "us";
}

REGISTER_LAYER_CLASS(Softplus);

}  // namespace caffe_ffi