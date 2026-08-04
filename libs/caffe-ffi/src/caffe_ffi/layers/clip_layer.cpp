#include "caffe_ffi/layers/clip_layer.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void ClipLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int64_t count = bottom[0]->count();
  const float min = this->layer_param_.clip_param().min();
  const float max = this->layer_param_.clip_param().max();
  CAFFE_FFI_LAYER_LOG << "Clip Forward_cpu: count=" << count
                      << " min=" << min << " max=" << max;

  for (int64_t i = 0; i < count; ++i) {
    top_data[i] = std::max(min, std::min(bottom_data[i], max));
  }
}

void ClipLayer::Backward_cpu(const std::vector<Blob*>& top,
                             const std::vector<bool>& propagate_down,
                             const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const float* bottom_data = bottom[0]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const int64_t count = bottom[0]->count();
  const float min = this->layer_param_.clip_param().min();
  const float max = this->layer_param_.clip_param().max();
  CAFFE_FFI_LAYER_LOG << "Clip Backward_cpu: count=" << count
                      << " min=" << min << " max=" << max;

  for (int64_t i = 0; i < count; ++i) {
    const float dy = top_diff[i];
    const float x = bottom_data[i];
    bottom_diff[i] = dy * ((x >= min && x <= max) ? 1.0f : 0.0f);
  }
}

REGISTER_LAYER_CLASS(Clip);

}  // namespace caffe_ffi