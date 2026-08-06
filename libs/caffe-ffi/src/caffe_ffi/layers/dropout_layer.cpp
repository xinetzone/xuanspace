#include "caffe_ffi/layers/dropout_layer.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"

namespace caffe_ffi {

void DropoutLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                               const std::vector<Blob*>& top) {
  const float dropout_ratio = this->layer_param_.dropout_param().dropout_ratio();
  CAFFE_FFI_CHECK_VALUE_GE(dropout_ratio, 0.0f)
      << "Dropout dropout_ratio must be >= 0.";
  CAFFE_FFI_CHECK_VALUE_LT(dropout_ratio, 1.0f)
      << "Dropout dropout_ratio must be < 1 (need a positive keep probability).";
  ratio_ = dropout_ratio;
  scale_ = 1.0f / (1.0f - ratio_);
  CAFFE_FFI_LAYER_LOG << "Dropout LayerSetUp: dropout_ratio=" << ratio_
                      << " scale_=" << scale_
                      << " mode=" << (train_mode() ? "train" : "inference");
}

void DropoutLayer::Reshape(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  if (mask_ == nullptr) {
    mask_ = make_object<Blob>();
  }
  mask_->ReshapeLike(*bottom[0]);

  std::ostringstream shape_ss;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) shape_ss << ", ";
    shape_ss << bottom[0]->shape(i);
  }
  CAFFE_FFI_LAYER_LOG << "Dropout Reshape: input/output shape=[" << shape_ss.str() << "]"
                      << " count=" << bottom[0]->count();
}

void DropoutLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                const std::vector<Blob*>& top) {
  const int64_t count = bottom[0]->count();
  const bool inplace = (bottom[0] == top[0]);
  const bool train = train_mode();

  CAFFE_FFI_LAYER_LOG << "Dropout Forward: count=" << count
                      << " dropout_ratio=" << ratio_
                      << " mode=" << (train ? "train" : "inference")
                      << " inplace=" << (inplace ? "true" : "false");

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();
#endif

  if (!train) {
    if (!inplace) {
      top[0]->ShareData(bottom[0]);
    }
  } else {
    float* mask_data = mask_->cpu_mutable_data();
    std::bernoulli_distribution dist(1.0f - ratio_);
    for (int64_t i = 0; i < count; ++i) {
      mask_data[i] = dist(rng_) ? 1.0f : 0.0f;
    }
    const float* bottom_data = bottom[0]->cpu_data();
    float* top_data = top[0]->cpu_mutable_data();
    for (int64_t i = 0; i < count; ++i) {
      top_data[i] = bottom_data[i] * mask_data[i] * scale_;
    }
  }

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[DROPOUT-PERF] " << this->name()
                       << " Dropout forward (" << (train ? "train" : "inference") << "):"
                       << " count=" << count
                       << " dropout_ratio=" << ratio_
                       << " scale=" << (train ? scale_ : 1.0f)
                       << " inplace=" << (inplace ? "true" : "false")
                       << " zerocopy=" << ((!train && !inplace) ? "yes" : "no")
                       << " time=" << elapsed_us << "us";
#endif
}

void DropoutLayer::Backward_cpu(const std::vector<Blob*>& top,
                                 const std::vector<bool>& propagate_down,
                                 const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    CAFFE_FFI_LAYER_LOG << "Dropout Backward_cpu: propagate_down[0]=false, skipping";
    return;
  }

  const int64_t count = bottom[0]->count();
  const bool inplace = (bottom[0] == top[0]);
  const bool train = train_mode();

  CAFFE_FFI_LAYER_LOG << "Dropout Backward_cpu: count=" << count
                      << " dropout_ratio=" << ratio_
                      << " mode=" << (train ? "train" : "inference")
                      << " inplace=" << (inplace ? "true" : "false");

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();
#endif

  if (!train) {
    if (!inplace) {
      bottom[0]->ShareDiff(top[0]);
    }
  } else {
    const float* mask_data = mask_->cpu_data();
    const float* top_diff = top[0]->cpu_diff();
    float* bottom_diff = bottom[0]->cpu_mutable_diff();
    for (int64_t i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * mask_data[i] * scale_;
    }
  }

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[DROPOUT-PERF] " << this->name()
                       << " Dropout backward (" << (train ? "train" : "inference") << "):"
                       << " count=" << count
                       << " dropout_ratio=" << ratio_
                       << " scale=" << (train ? scale_ : 1.0f)
                       << " inplace=" << (inplace ? "true" : "false")
                       << " zerocopy=" << ((!train && !inplace) ? "yes" : "no")
                       << " time=" << elapsed_us << "us";
#endif
}

REGISTER_LAYER_CLASS(Dropout);

}  // namespace caffe_ffi
