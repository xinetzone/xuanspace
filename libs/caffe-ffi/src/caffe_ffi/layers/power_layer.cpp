#include "caffe_ffi/layers/power_layer.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void PowerLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  power_ = this->layer_param_.power_param().power();
  scale_ = this->layer_param_.power_param().scale();
  shift_ = this->layer_param_.power_param().shift();
  diff_scale_ = power_ * scale_;
  CAFFE_FFI_LAYER_LOG << "Power LayerSetUp: power=" << power_
                      << " scale=" << scale_ << " shift=" << shift_;
}

void PowerLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                             const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int64_t count = bottom[0]->count();
  CAFFE_FFI_LAYER_LOG << "Power Forward_cpu: count=" << count
                      << " power=" << power_ << " scale=" << scale_ << " shift=" << shift_;

  if (scale_ == 0.0f || power_ == 0.0f) {
    const float constant = std::pow(shift_, power_);
    for (int64_t i = 0; i < count; ++i) {
      top_data[i] = constant;
    }
    return;
  }

  for (int64_t i = 0; i < count; ++i) {
    const float t = shift_ + scale_ * bottom_data[i];
    top_data[i] = std::pow(t, power_);
  }
}

void PowerLayer::Backward_cpu(const std::vector<Blob*>& top,
                              const std::vector<bool>& propagate_down,
                              const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const float* bottom_data = bottom[0]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const int64_t count = bottom[0]->count();
  CAFFE_FFI_LAYER_LOG << "Power Backward_cpu: count=" << count
                      << " power=" << power_ << " scale=" << scale_ << " shift=" << shift_;

  if (power_ == 0.0f || scale_ == 0.0f) {
    for (int64_t i = 0; i < count; ++i) {
      bottom_diff[i] = 0.0f;
    }
    return;
  }

  for (int64_t i = 0; i < count; ++i) {
    const float dy = top_diff[i];
    const float t = shift_ + scale_ * bottom_data[i];
    bottom_diff[i] = dy * diff_scale_ * std::pow(t, power_ - 1.0f);
  }
}

REGISTER_LAYER_CLASS(Power);

}  // namespace caffe_ffi