#include "caffe_ffi/layers/neuron_layer.hpp"

#include <sstream>

#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void NeuronLayer::Reshape(const std::vector<Blob*>& bottom,
                           const std::vector<Blob*>& top) {
  std::ostringstream shape_ss;
  shape_ss << "[";
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) shape_ss << ", ";
    shape_ss << bottom[0]->shape(i);
  }
  shape_ss << "]";
  CAFFE_FFI_LAYER_LOG << this->type() << " Reshape: bottom shape=" << shape_ss.str();
  top[0]->ReshapeLike(*bottom[0]);
}

}  // namespace caffe_ffi
