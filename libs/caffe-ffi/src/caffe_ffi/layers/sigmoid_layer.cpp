#include "caffe_ffi/layers/sigmoid_layer.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void SigmoidLayer::Reshape(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  std::ostringstream bottom_shape_ss;
  bottom_shape_ss << "[";
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) bottom_shape_ss << ", ";
    bottom_shape_ss << bottom[0]->shape(i);
  }
  bottom_shape_ss << "]";
  CAFFE_FFI_LAYER_LOG << "Sigmoid Reshape: bottom shape=" << bottom_shape_ss.str();
  top[0]->ReshapeLike(*bottom[0]);
  std::ostringstream top_shape_ss;
  top_shape_ss << "[";
  for (int i = 0; i < top[0]->num_axes(); ++i) {
    if (i > 0) top_shape_ss << ", ";
    top_shape_ss << top[0]->shape(i);
  }
  top_shape_ss << "]";
  CAFFE_FFI_LAYER_LOG << "Sigmoid Reshape: top shape=" << top_shape_ss.str();
}

void SigmoidLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                               const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_data();
  const int64_t count = bottom[0]->count();
  CAFFE_FFI_LAYER_LOG << "Sigmoid Forward_cpu: count=" << count;
  for (int64_t i = 0; i < count; ++i) {
    top_data[i] = 1.0f / (1.0f + std::exp(-bottom_data[i]));
  }
}

REGISTER_LAYER_CLASS(Sigmoid);

}  // namespace caffe_ffi
