#include "caffe_ffi/layers/multinomial_logistic_loss_layer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/math_utils.hpp"

namespace caffe_ffi {

void MultinomialLogisticLossLayer::LayerSetUp(const std::vector<Blob*>& bottom,
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
  CAFFE_FFI_LAYER_LOG << "MultinomialLogisticLoss LayerSetUp:"
                      << " has_ignore_label_=" << has_ignore_label_
                      << " normalization_=" << normalization_;
}

void MultinomialLogisticLossLayer::Reshape(const std::vector<Blob*>& bottom,
                                           const std::vector<Blob*>& top) {
  outer_num_ = static_cast<int>(bottom[0]->shape(0));
  inner_num_ = static_cast<int>(bottom[0]->count(2));
  channels_ = static_cast<int>(bottom[0]->shape(1));
  CAFFE_FFI_CHECK_VALUE_EQ(bottom[1]->count(), outer_num_ * inner_num_)
      << "MultinomialLogisticLoss label count must match data count.";

  std::vector<int64_t> loss_shape = {1};
  top[0]->Reshape(loss_shape);
  CAFFE_FFI_LAYER_LOG << "MultinomialLogisticLoss Reshape: outer_num_=" << outer_num_
                      << " inner_num_=" << inner_num_
                      << " channels_=" << channels_;
}

void MultinomialLogisticLossLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                               const std::vector<Blob*>& top) {
  const int outer_num = outer_num_;
  const int inner_num = inner_num_;
  const int channels = channels_;
  const float* prob = bottom[0]->cpu_data();
  const float* label = bottom[1]->cpu_data();

  double total = 0.0;
  int valid_count = 0;
  for (int i = 0; i < outer_num; ++i) {
    for (int j = 0; j < inner_num; ++j) {
      const int label_value = static_cast<int>(label[i * inner_num + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      CAFFE_FFI_CHECK_VALUE_GE(label_value, 0);
      CAFFE_FFI_CHECK_VALUE_LT(label_value, channels);
      const float p = prob[i * channels * inner_num + label_value * inner_num + j];
      total -= std::log(std::max(p, std::numeric_limits<float>::min()));
      ++valid_count;
    }
  }

  switch (normalization_) {
    case caffe::LossParameter_NormalizationMode_FULL:
      normalizer_ = static_cast<float>(outer_num * inner_num);
      break;
    case caffe::LossParameter_NormalizationMode_VALID:
      normalizer_ = static_cast<float>(valid_count);
      break;
    case caffe::LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer_ = static_cast<float>(outer_num);
      break;
    case caffe::LossParameter_NormalizationMode_NONE:
    default:
      normalizer_ = 1.0f;
      break;
  }
  normalizer_ = std::max(1.0f, normalizer_);

  top[0]->cpu_mutable_data()[0] = static_cast<float>(total) / normalizer_;
  CAFFE_FFI_LAYER_LOG << "MultinomialLogisticLoss Forward: valid_count=" << valid_count
                      << " total_loss=" << total << " normalizer=" << normalizer_
                      << " avg_loss=" << top[0]->cpu_data()[0];
}

void MultinomialLogisticLossLayer::Backward_cpu(
    const std::vector<Blob*>& top, const std::vector<bool>& propagate_down,
    const std::vector<Blob*>& bottom) {
  if (propagate_down.size() > 1 && propagate_down[1]) {
    CAFFE_FFI_LOG_WARN() << "MultinomialLogisticLoss Backward: cannot backpropagate to label inputs.";
  }
  if (!propagate_down[0]) {
    return;
  }

  const int outer_num = outer_num_;
  const int inner_num = inner_num_;
  const int channels = channels_;
  const float* prob = bottom[0]->cpu_data();
  const float* label = bottom[1]->cpu_data();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();

  const float loss_weight = top[0]->cpu_diff()[0];
  const float scale = loss_weight / normalizer_;

  // dL/dp[i, k] = -scale / p[i, gt] for k == gt, else 0.
  for (int i = 0; i < outer_num; ++i) {
    for (int j = 0; j < inner_num; ++j) {
      const int label_value = static_cast<int>(label[i * inner_num + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        for (int c = 0; c < channels; ++c) {
          bottom_diff[i * channels * inner_num + c * inner_num + j] = 0.0f;
        }
        continue;
      }
      CAFFE_FFI_CHECK_VALUE_GE(label_value, 0);
      CAFFE_FFI_CHECK_VALUE_LT(label_value, channels);
      for (int c = 0; c < channels; ++c) {
        bottom_diff[i * channels * inner_num + c * inner_num + j] = 0.0f;
      }
      const float p = prob[i * channels * inner_num + label_value * inner_num + j];
      bottom_diff[i * channels * inner_num + label_value * inner_num + j] =
          -scale / std::max(p, std::numeric_limits<float>::min());
    }
  }
  CAFFE_FFI_LAYER_LOG << "MultinomialLogisticLoss Backward: loss_weight=" << loss_weight
                      << " normalizer=" << normalizer_ << " scale=" << scale;
}

REGISTER_LAYER_CLASS(MultinomialLogisticLoss);

}  // namespace caffe_ffi