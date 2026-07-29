#include "caffe_ffi/layers/input_layer.hpp"

#include <sstream>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

void InputLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  const int num_top = static_cast<int>(top.size());
  const caffe::InputParameter& param = this->layer_param_.input_param();
  const int num_shape = param.shape_size();
  CAFFE_FFI_CHECK_VALUE(num_shape == 0 || num_shape == 1 || num_shape == num_top)
      << "Must specify 'shape' once, once per top blob, or not at all: "
      << num_top << " tops vs. " << num_shape << " shapes.";
  if (num_shape > 0) {
    for (int i = 0; i < num_top; ++i) {
      const int shape_index = (param.shape_size() == 1) ? 0 : i;
      top[i]->Reshape(param.shape(shape_index));
    }
  }

  CAFFE_FFI_LAYER_LOG << "Input LayerSetUp: num_top=" << num_top
                      << " num_shape=" << num_shape;
  for (int i = 0; i < num_top; ++i) {
    std::ostringstream shape_ss;
    for (int j = 0; j < top[i]->num_axes(); ++j) {
      if (j > 0) shape_ss << ", ";
      shape_ss << top[i]->shape(j);
    }
    CAFFE_FFI_LAYER_LOG << "Input: top[" << i << "] shape=[" << shape_ss.str() << "]";
  }
}

void InputLayer::Reshape(const std::vector<Blob*>& bottom,
                          const std::vector<Blob*>& top) {
  const int num_top = static_cast<int>(top.size());
  CAFFE_FFI_LAYER_LOG << "Input Reshape: num_top=" << num_top;
  for (int i = 0; i < num_top; ++i) {
    std::ostringstream shape_ss;
    for (int j = 0; j < top[i]->num_axes(); ++j) {
      if (j > 0) shape_ss << ", ";
      shape_ss << top[i]->shape(j);
    }
    CAFFE_FFI_LAYER_LOG << "Input: top[" << i << "] shape=[" << shape_ss.str() << "]";
  }
}

void InputLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                              const std::vector<Blob*>& top) {
  const int num_top = static_cast<int>(top.size());
  CAFFE_FFI_LAYER_LOG << "Input Forward: num_top=" << num_top << " (data pass-through)";
}

REGISTER_LAYER_CLASS(Input);

}  // namespace caffe_ffi
