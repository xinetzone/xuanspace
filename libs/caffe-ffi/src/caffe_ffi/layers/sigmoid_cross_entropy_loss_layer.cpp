#include "caffe_ffi/layers/sigmoid_cross_entropy_loss_layer.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/math_utils.hpp"

namespace caffe_ffi {

namespace {

inline float SigmoidStable(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

}  // namespace

void SigmoidCrossEntropyLossLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                                               const std::vector<Blob*>& top) {
  has_ignore_label_ = this->layer_param_.has_loss_param() &&
                      this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (this->layer_param_.has_loss_param() &&
      this->layer_param_.loss_param().has_normalization()) {
    normalization_ = this->layer_param_.loss_param().normalization();
  } else if (this->layer_param_.has_loss_param() &&
             this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize()
                         ? caffe::LossParameter_NormalizationMode_VALID
                         : caffe::LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = caffe::LossParameter_NormalizationMode_BATCH_SIZE;
  }
  CAFFE_FFI_LAYER_LOG << "SigmoidCrossEntropyLoss LayerSetUp:"
                      << " has_ignore_label_=" << has_ignore_label_
                      << " normalization_=" << normalization_;
}

void SigmoidCrossEntropyLossLayer::Reshape(const std::vector<Blob*>& bottom,
                                            const std::vector<Blob*>& top) {
  outer_num_ = static_cast<int>(bottom[0]->shape(0));
  inner_num_ = static_cast<int>(bottom[0]->count(1));
  CAFFE_FFI_CHECK_VALUE_EQ(bottom[0]->count(), bottom[1]->count())
      << "SigmoidCrossEntropyLoss inputs must have the same count.";
  std::vector<int64_t> sigmoid_shape;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    sigmoid_shape.push_back(bottom[0]->shape(i));
  }
  sigmoid_output_ = make_object<Blob>(sigmoid_shape);
  std::vector<int64_t> loss_shape = {1};
  top[0]->Reshape(loss_shape);
  CAFFE_FFI_LAYER_LOG << "SigmoidCrossEntropyLoss Reshape: outer_num_=" << outer_num_
                      << " inner_num_=" << inner_num_
                      << " count=" << bottom[0]->count();
}

void SigmoidCrossEntropyLossLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                                const std::vector<Blob*>& top) {
  const float* input_data = bottom[0]->cpu_data();
  const float* target = bottom[1]->cpu_data();
  float* sigmoid_output = sigmoid_output_->cpu_mutable_data();

  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    sigmoid_output[i] = SigmoidStable(input_data[i]);
  }

  int valid_count = 0;
  float loss = 0.0f;
  for (int i = 0; i < count; ++i) {
    const int target_value = static_cast<int>(target[i]);
    if (has_ignore_label_ && target_value == ignore_label_) {
      continue;
    }
    // Stable loss: -x*(y - (x>=0)) + log(1 + exp(x - 2*x*(x>=0)))
    const float x = input_data[i];
    const float y = target[i];
    loss -= x * (y - (x >= 0 ? 1.0f : 0.0f)) -
            std::log(1.0f + std::exp(x - 2.0f * x * (x >= 0 ? 1.0f : 0.0f)));
    ++valid_count;
  }

  switch (normalization_) {
    case caffe::LossParameter_NormalizationMode_FULL:
      normalizer_ = static_cast<float>(outer_num_ * inner_num_);
      break;
    case caffe::LossParameter_NormalizationMode_VALID:
      normalizer_ = (valid_count == -1) ? static_cast<float>(outer_num_ * inner_num_)
                                        : static_cast<float>(valid_count);
      break;
    case caffe::LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer_ = static_cast<float>(outer_num_);
      break;
    case caffe::LossParameter_NormalizationMode_NONE:
    default:
      normalizer_ = 1.0f;
      break;
  }
  normalizer_ = std::max(1.0f, normalizer_);

  top[0]->cpu_mutable_data()[0] = loss / normalizer_;
  CAFFE_FFI_LAYER_LOG << "SigmoidCrossEntropyLoss Forward: valid_count=" << valid_count
                      << " loss=" << loss << " normalizer=" << normalizer_
                      << " avg_loss=" << top[0]->cpu_data()[0];
}

void SigmoidCrossEntropyLossLayer::Backward_cpu(
    const std::vector<Blob*>& top, const std::vector<bool>& propagate_down,
    const std::vector<Blob*>& bottom) {
  if (propagate_down[1]) {
    CAFFE_FFI_LOG_WARN() << "SigmoidCrossEntropyLoss cannot backpropagate to label inputs.";
  }
  if (!propagate_down[0]) {
    return;
  }
  const int count = bottom[0]->count();
  const float* sigmoid_output = sigmoid_output_->cpu_data();
  const float* target = bottom[1]->cpu_data();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();

  // dL/dx = sigmoid(x) - target
  for (int i = 0; i < count; ++i) {
    bottom_diff[i] = sigmoid_output[i] - target[i];
  }
  if (has_ignore_label_) {
    for (int i = 0; i < count; ++i) {
      if (static_cast<int>(target[i]) == ignore_label_) {
        bottom_diff[i] = 0.0f;
      }
    }
  }
  const float loss_weight = top[0]->cpu_diff()[0];
  const float scale = loss_weight / normalizer_;
  caffe_scal_fp32(static_cast<size_t>(count), scale, bottom_diff);
  CAFFE_FFI_LAYER_LOG << "SigmoidCrossEntropyLoss Backward: loss_weight=" << loss_weight
                      << " normalizer=" << normalizer_ << " scale=" << scale;
}

REGISTER_LAYER_CLASS(SigmoidCrossEntropyLoss);

}  // namespace caffe_ffi