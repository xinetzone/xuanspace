#include "caffe_ffi/layers/log_layer.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void LogLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                          const std::vector<Blob*>& top) {
  const float base = this->layer_param_.log_param().base();
  const float scale = this->layer_param_.log_param().scale();
  const float shift = this->layer_param_.log_param().shift();
  const float log_base = (base == -1.0f) ? 1.0f : std::log(base);
  base_scale_ = 1.0f / log_base;
  input_scale_ = scale;
  input_shift_ = shift;
  backward_num_scale_ = scale / log_base;
  CAFFE_FFI_LAYER_LOG << "Log LayerSetUp: base=" << base << " scale=" << scale
                      << " shift=" << shift << " base_scale=" << base_scale_
                      << " input_scale=" << input_scale_ << " input_shift=" << input_shift_;
}

void LogLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                           const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int64_t count = bottom[0]->count();
  CAFFE_FFI_LAYER_LOG << "Log Forward_cpu: count=" << count
                      << " base_scale=" << base_scale_ << " input_scale=" << input_scale_
                      << " input_shift=" << input_shift_;

  for (int64_t i = 0; i < count; ++i) {
    top_data[i] = base_scale_ * std::log(input_scale_ * bottom_data[i] + input_shift_);
  }
}

void LogLayer::Backward_cpu(const std::vector<Blob*>& top,
                            const std::vector<bool>& propagate_down,
                            const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const float* bottom_data = bottom[0]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const int64_t count = bottom[0]->count();
  CAFFE_FFI_LAYER_LOG << "Log Backward_cpu: count=" << count
                      << " backward_num_scale=" << backward_num_scale_;

  for (int64_t i = 0; i < count; ++i) {
    const float dy = top_diff[i];
    const float t = input_scale_ * bottom_data[i] + input_shift_;
    bottom_diff[i] = dy * backward_num_scale_ * std::pow(t, -1.0f);
  }
}

REGISTER_LAYER_CLASS(Log);

}  // namespace caffe_ffi