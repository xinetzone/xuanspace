#include "caffe_ffi/layers/exp_layer.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void ExpLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                          const std::vector<Blob*>& top) {
  const float base = this->layer_param_.exp_param().base();
  const float scale = this->layer_param_.exp_param().scale();
  const float shift = this->layer_param_.exp_param().shift();
  const float log_base = (base == -1.0f) ? 1.0f : std::log(base);
  inner_scale_ = log_base * scale;
  outer_scale_ = (shift == 0.0f) ? 1.0f
                                 : ((base == -1.0f) ? std::exp(shift) : std::pow(base, shift));
  CAFFE_FFI_LAYER_LOG << "Exp LayerSetUp: base=" << base << " scale=" << scale
                      << " shift=" << shift << " inner_scale=" << inner_scale_
                      << " outer_scale=" << outer_scale_;
}

void ExpLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                           const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int64_t count = bottom[0]->count();
  CAFFE_FFI_LAYER_LOG << "Exp Forward_cpu: count=" << count
                      << " inner_scale=" << inner_scale_ << " outer_scale=" << outer_scale_;

  for (int64_t i = 0; i < count; ++i) {
    top_data[i] = outer_scale_ * std::exp(inner_scale_ * bottom_data[i]);
  }
}

void ExpLayer::Backward_cpu(const std::vector<Blob*>& top,
                            const std::vector<bool>& propagate_down,
                            const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const float* top_data = top[0]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const int64_t count = bottom[0]->count();
  CAFFE_FFI_LAYER_LOG << "Exp Backward_cpu: count=" << count
                      << " inner_scale=" << inner_scale_;

  for (int64_t i = 0; i < count; ++i) {
    const float dy = top_diff[i];
    bottom_diff[i] = dy * top_data[i] * inner_scale_;
  }
}

REGISTER_LAYER_CLASS(Exp);

}  // namespace caffe_ffi