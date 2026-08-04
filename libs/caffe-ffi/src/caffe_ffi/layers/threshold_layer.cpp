#include "caffe_ffi/layers/threshold_layer.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void ThresholdLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                 const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int64_t count = bottom[0]->count();
  const float threshold = this->layer_param_.threshold_param().threshold();
  CAFFE_FFI_LAYER_LOG << "Threshold Forward_cpu: count=" << count
                      << " threshold=" << threshold;

  for (int64_t i = 0; i < count; ++i) {
    top_data[i] = (bottom_data[i] > threshold) ? 1.0f : 0.0f;
  }
}

void ThresholdLayer::Backward_cpu(const std::vector<Blob*>& top,
                                  const std::vector<bool>& propagate_down,
                                  const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  CAFFE_FFI_LOG_WARN() << "Threshold Backward_cpu: layer is not differentiable, "
                       << "skipping gradient computation (no-op)";
}

REGISTER_LAYER_CLASS(Threshold);

}  // namespace caffe_ffi