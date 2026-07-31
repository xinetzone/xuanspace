#include "caffe_ffi/layers/relu_layer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void ReLULayer::Reshape(const std::vector<Blob*>& bottom,
                         const std::vector<Blob*>& top) {
  std::ostringstream bottom_shape_ss;
  bottom_shape_ss << "[";
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) bottom_shape_ss << ", ";
    bottom_shape_ss << bottom[0]->shape(i);
  }
  bottom_shape_ss << "]";
  CAFFE_FFI_LAYER_LOG << "ReLU Reshape: bottom shape=" << bottom_shape_ss.str();
  top[0]->ReshapeLike(*bottom[0]);
  std::ostringstream top_shape_ss;
  top_shape_ss << "[";
  for (int i = 0; i < top[0]->num_axes(); ++i) {
    if (i > 0) top_shape_ss << ", ";
    top_shape_ss << top[0]->shape(i);
  }
  top_shape_ss << "]";
  CAFFE_FFI_LAYER_LOG << "ReLU Reshape: top shape=" << top_shape_ss.str();
}

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

REGISTER_LAYER_CLASS(ReLU);

}  // namespace caffe_ffi
