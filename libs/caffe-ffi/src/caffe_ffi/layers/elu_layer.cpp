#include "caffe_ffi/layers/elu_layer.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void ELULayer::LayerSetUp(const std::vector<Blob*>& bottom,
                           const std::vector<Blob*>& top) {
  alpha_ = this->layer_param_.elu_param().alpha();
  CAFFE_FFI_LAYER_LOG << "ELU LayerSetUp: alpha=" << alpha_;
}

void ELULayer::Reshape(const std::vector<Blob*>& bottom,
                        const std::vector<Blob*>& top) {
  std::ostringstream bottom_shape_ss;
  bottom_shape_ss << "[";
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) bottom_shape_ss << ", ";
    bottom_shape_ss << bottom[0]->shape(i);
  }
  bottom_shape_ss << "]";
  CAFFE_FFI_LAYER_LOG << "ELU Reshape: bottom shape=" << bottom_shape_ss.str();
  top[0]->ReshapeLike(*bottom[0]);
  std::ostringstream top_shape_ss;
  top_shape_ss << "[";
  for (int i = 0; i < top[0]->num_axes(); ++i) {
    if (i > 0) top_shape_ss << ", ";
    top_shape_ss << top[0]->shape(i);
  }
  top_shape_ss << "]";
  CAFFE_FFI_LAYER_LOG << "ELU Reshape: top shape=" << top_shape_ss.str();
}

void ELULayer::Forward_cpu(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_data();
  const int64_t count = bottom[0]->count();
  CAFFE_FFI_LAYER_LOG << "ELU Forward_cpu: count=" << count
                      << " alpha=" << alpha_;
  for (int64_t i = 0; i < count; ++i) {
    if (bottom_data[i] >= 0.0f) {
      top_data[i] = bottom_data[i];
    } else {
      top_data[i] = alpha_ * (std::exp(bottom_data[i]) - 1.0f);
    }
  }
}

REGISTER_LAYER_CLASS(ELU);

}  // namespace caffe_ffi
