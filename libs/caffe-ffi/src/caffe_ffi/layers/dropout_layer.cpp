#include "caffe_ffi/layers/dropout_layer.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void DropoutLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                               const std::vector<Blob*>& top) {
  const float dropout_ratio = this->layer_param_.dropout_param().dropout_ratio();
  CAFFE_FFI_LAYER_LOG << "Dropout LayerSetUp: dropout_ratio=" << dropout_ratio
                      << " (inference mode: pass-through)";
}

void DropoutLayer::Reshape(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  top[0]->ReshapeLike(*bottom[0]);

  std::ostringstream shape_ss;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) shape_ss << ", ";
    shape_ss << bottom[0]->shape(i);
  }
  CAFFE_FFI_LAYER_LOG << "Dropout Reshape: input/output shape=[" << shape_ss.str() << "]"
                      << " count=" << bottom[0]->count();
}

void DropoutLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int64_t count = bottom[0]->count();
  const float dropout_ratio = this->layer_param_.dropout_param().dropout_ratio();
  CAFFE_FFI_LAYER_LOG << "Dropout Forward: count=" << count
                      << " dropout_ratio=" << dropout_ratio
                      << " inplace=" << (bottom[0] == top[0] ? "true" : "false")
                      << " (inference: identity copy)";

  auto t_start = std::chrono::high_resolution_clock::now();
  if (bottom[0] != top[0]) {
    std::memcpy(top_data, bottom_data, sizeof(float) * count);
  }
  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[DROPOUT-PERF] " << this->name()
                       << " Dropout forward (inference): count=" << count
                       << " dropout_ratio=" << dropout_ratio
                       << " inplace=" << (bottom[0] == top[0] ? "true" : "false")
                       << " time=" << elapsed_us << "us";
}

void DropoutLayer::Backward_cpu(const std::vector<Blob*>& top,
                                 const std::vector<bool>& propagate_down,
                                 const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    CAFFE_FFI_LAYER_LOG << "Dropout Backward_cpu: propagate_down[0]=false, skipping";
    return;
  }

  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const int64_t count = bottom[0]->count();
  const float dropout_ratio = this->layer_param_.dropout_param().dropout_ratio();
  CAFFE_FFI_LAYER_LOG << "Dropout Backward_cpu: count=" << count
                      << " dropout_ratio=" << dropout_ratio
                      << " inplace=" << (bottom[0] == top[0] ? "true" : "false")
                      << " (inference: identity copy)";

  auto t_start = std::chrono::high_resolution_clock::now();

  // In inference mode Dropout is identity (y = x), therefore backward is also identity: dx = dy
  if (bottom[0] != top[0]) {
    std::memcpy(bottom_diff, top_diff, sizeof(float) * count);
  }
  // else: inplace operation, bottom_diff already points to top_diff memory, no copy needed

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[DROPOUT-PERF] " << this->name()
                       << " Dropout backward (inference): count=" << count
                       << " dropout_ratio=" << dropout_ratio
                       << " inplace=" << (bottom[0] == top[0] ? "true" : "false")
                       << " time=" << elapsed_us << "us";
}

REGISTER_LAYER_CLASS(Dropout);

}  // namespace caffe_ffi
