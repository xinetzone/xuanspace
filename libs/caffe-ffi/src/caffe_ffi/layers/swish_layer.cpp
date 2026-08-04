#include "caffe_ffi/layers/swish_layer.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void SwishLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                             const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int64_t count = bottom[0]->count();
  const float beta = this->layer_param_.swish_param().beta();
  CAFFE_FFI_LAYER_LOG << "Swish Forward_cpu: count=" << count << " beta=" << beta;

  for (int64_t i = 0; i < count; ++i) {
    const float x = bottom_data[i];
    const float sigmoid = 1.0f / (1.0f + std::exp(-beta * x));
    top_data[i] = x * sigmoid;
  }
}

void SwishLayer::Backward_cpu(const std::vector<Blob*>& top,
                              const std::vector<bool>& propagate_down,
                              const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const float* bottom_data = bottom[0]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const int64_t count = bottom[0]->count();
  const float beta = this->layer_param_.swish_param().beta();
  CAFFE_FFI_LAYER_LOG << "Swish Backward_cpu: count=" << count << " beta=" << beta;

  for (int64_t i = 0; i < count; ++i) {
    const float dy = top_diff[i];
    const float x = bottom_data[i];
    const float sigmoid = 1.0f / (1.0f + std::exp(-beta * x));
    const float y = x * sigmoid;
    bottom_diff[i] = dy * (beta * y + sigmoid * (1.0f - beta * y));
  }
}

REGISTER_LAYER_CLASS(Swish);

}  // namespace caffe_ffi