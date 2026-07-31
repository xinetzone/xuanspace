#include "caffe_ffi/layers/tanh_layer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void TanHLayer::Reshape(const std::vector<Blob*>& bottom,
                         const std::vector<Blob*>& top) {
  std::ostringstream bottom_shape_ss;
  bottom_shape_ss << "[";
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) bottom_shape_ss << ", ";
    bottom_shape_ss << bottom[0]->shape(i);
  }
  bottom_shape_ss << "]";
  CAFFE_FFI_LAYER_LOG << "TanH Reshape: bottom shape=" << bottom_shape_ss.str();
  top[0]->ReshapeLike(*bottom[0]);
  std::ostringstream top_shape_ss;
  top_shape_ss << "[";
  for (int i = 0; i < top[0]->num_axes(); ++i) {
    if (i > 0) top_shape_ss << ", ";
    top_shape_ss << top[0]->shape(i);
  }
  top_shape_ss << "]";
  CAFFE_FFI_LAYER_LOG << "TanH Reshape: top shape=" << top_shape_ss.str();
}

void TanHLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int64_t count = bottom[0]->count();
  CAFFE_FFI_LAYER_LOG << "TanH Forward_cpu: count=" << count;

  auto t_start = std::chrono::high_resolution_clock::now();

  float in_min = std::numeric_limits<float>::max();
  float in_max = -std::numeric_limits<float>::max();
  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();

  for (int64_t i = 0; i < count; ++i) {
    float x = bottom_data[i];
    float y = std::tanh(x);
    top_data[i] = y;
    in_min = std::min(in_min, x);
    in_max = std::max(in_max, x);
    out_min = std::min(out_min, y);
    out_max = std::max(out_max, y);
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[ACTIVATION-PERF] " << this->name()
                       << " TanH forward: count=" << count
                       << " in=[" << in_min << ", " << in_max << "]"
                       << " out=[" << out_min << ", " << out_max << "]"
                       << " time=" << elapsed_us << "us";
}

REGISTER_LAYER_CLASS(TanH);

}  // namespace caffe_ffi
