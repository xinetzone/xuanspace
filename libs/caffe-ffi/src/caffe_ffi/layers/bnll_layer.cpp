#include "caffe_ffi/layers/bnll_layer.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void BNLLLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int64_t count = bottom[0]->count();
  CAFFE_FFI_LAYER_LOG << "BNLL Forward_cpu: count=" << count;

  for (int64_t i = 0; i < count; ++i) {
    const float x = bottom_data[i];
    if (x > 0.0f) {
      top_data[i] = x + std::log(1.0f + std::exp(-x));
    } else {
      top_data[i] = std::log(1.0f + std::exp(x));
    }
  }
}

void BNLLLayer::Backward_cpu(const std::vector<Blob*>& top,
                             const std::vector<bool>& propagate_down,
                             const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const float* bottom_data = bottom[0]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const int64_t count = bottom[0]->count();
  CAFFE_FFI_LAYER_LOG << "BNLL Backward_cpu: count=" << count;

  for (int64_t i = 0; i < count; ++i) {
    const float dy = top_diff[i];
    const float x = bottom_data[i];
    const float expval = std::exp(std::min(x, 50.0f));
    bottom_diff[i] = dy * expval / (expval + 1.0f);
  }
}

REGISTER_LAYER_CLASS(BNLL);

}  // namespace caffe_ffi