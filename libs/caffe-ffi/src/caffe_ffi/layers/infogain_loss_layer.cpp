#include "caffe_ffi/layers/infogain_loss_layer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

#include <tvm/ffi/memory.h>

#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/math_utils.hpp"

namespace caffe_ffi {

void InfogainLossLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                                   const std::vector<Blob*>& top) {
  has_ignore_label_ = this->layer_param_.has_loss_param() &&
                      this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }

  softmax_axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.infogain_loss_param().axis());

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

  CAFFE_FFI_LAYER_LOG << "InfogainLoss LayerSetUp: softmax_axis_=" << softmax_axis_
                      << " has_ignore_label_=" << has_ignore_label_
                      << " ignore_label_=" << (has_ignore_label_ ? ignore_label_ : -1)
                      << " normalization_=" << normalization_;
}

void InfogainLossLayer::Reshape(const std::vector<Blob*>& bottom,
                                const std::vector<Blob*>& top) {
  std::vector<int64_t> prob_shape;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    prob_shape.push_back(bottom[0]->shape(i));
  }
  prob_ = make_object<Blob>(prob_shape);

  // The infogain matrix is provided via an extra bottom blob (bottom[2]).
  has_infogain_ = (bottom.size() > 2);
  if (has_infogain_) {
    CAFFE_FFI_CHECK_VALUE_EQ(bottom[2]->num_axes(), 2)
        << "InfogainLoss infogain matrix must be 2D.";
    CAFFE_FFI_CHECK_VALUE_EQ(bottom[2]->shape(0), bottom[0]->shape(softmax_axis_))
        << "InfogainLoss infogain matrix rows must match the number of classes.";
    CAFFE_FFI_CHECK_VALUE_EQ(bottom[2]->shape(1), bottom[0]->shape(softmax_axis_))
        << "InfogainLoss infogain matrix must be square.";
  }

  outer_num_ = static_cast<int>(bottom[0]->count(0, softmax_axis_));
  inner_num_ = static_cast<int>(bottom[0]->count(softmax_axis_ + 1));
  channels_ = static_cast<int>(bottom[0]->shape(softmax_axis_));

  std::vector<int64_t> mult_dims = {channels_};
  sum_multiplier_ = make_object<Blob>(mult_dims);
  caffe_set_fp32(static_cast<size_t>(sum_multiplier_->count()), 1.0f,
                 sum_multiplier_->cpu_mutable_data());

  std::vector<int64_t> scale_dims;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i == softmax_axis_) {
      scale_dims.push_back(1);
    } else {
      scale_dims.push_back(bottom[0]->shape(i));
    }
  }
  scale_ = make_object<Blob>(scale_dims);

  std::vector<int64_t> loss_shape = {1};
  top[0]->Reshape(loss_shape);

  CAFFE_FFI_LAYER_LOG << "InfogainLoss Reshape: outer_num_=" << outer_num_
                      << " inner_num_=" << inner_num_
                      << " channels_=" << channels_
                      << " has_infogain_=" << has_infogain_;
}

void InfogainLossLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                    const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* prob_data = prob_->cpu_mutable_data();
  float* scale_data = scale_->cpu_mutable_data();

  const int channels = channels_;
  const int dim = channels * inner_num_;

  // Stable softmax over the softmax axis (max-subtraction).
  caffe_copy_fp32(static_cast<size_t>(bottom[0]->count()), bottom_data, prob_data);
  for (int i = 0; i < outer_num_; ++i) {
    float* prob_data_i = prob_data + i * dim;
    float* scale_data_i = scale_data + i * inner_num_;
    for (int k = 0; k < inner_num_; ++k) {
      scale_data_i[k] = prob_data_i[k];
      for (int j = 1; j < channels; ++j) {
        scale_data_i[k] = std::max(scale_data_i[k], prob_data_i[j * inner_num_ + k]);
      }
    }
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < inner_num_; ++k) {
        prob_data_i[j * inner_num_ + k] -= scale_data_i[k];
      }
    }
    caffe_exp_fp32(static_cast<size_t>(dim), prob_data_i, prob_data_i);
    for (int k = 0; k < inner_num_; ++k) {
      scale_data_i[k] = 0;
      for (int j = 0; j < channels; ++j) {
        scale_data_i[k] += prob_data_i[j * inner_num_ + k];
      }
    }
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < inner_num_; ++k) {
        prob_data_i[j * inner_num_ + k] /= scale_data_i[k];
      }
    }
  }

  const float* label = bottom[1]->cpu_data();
  const float* h = has_infogain_ ? bottom[2]->cpu_data() : nullptr;

  double total = 0.0;
  int valid_count = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      CAFFE_FFI_CHECK_VALUE_GE(label_value, 0);
      CAFFE_FFI_CHECK_VALUE_LT(label_value, channels);
      if (has_infogain_) {
        const float* h_row = h + label_value * channels;
        for (int k = 0; k < channels; ++k) {
          const float p = prob_data[i * dim + k * inner_num_ + j];
          total -= static_cast<double>(h_row[k]) *
                   std::log(std::max(p, std::numeric_limits<float>::min()));
        }
      } else {
        const float p = prob_data[i * dim + label_value * inner_num_ + j];
        total -= std::log(std::max(p, std::numeric_limits<float>::min()));
      }
      ++valid_count;
    }
  }

  switch (normalization_) {
    case caffe::LossParameter_NormalizationMode_FULL:
      normalizer_ = static_cast<float>(outer_num_ * inner_num_);
      break;
    case caffe::LossParameter_NormalizationMode_VALID:
      normalizer_ = static_cast<float>(valid_count);
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

  top[0]->cpu_mutable_data()[0] = static_cast<float>(total) / normalizer_;
  CAFFE_FFI_LAYER_LOG << "InfogainLoss Forward: valid_count=" << valid_count
                      << " total_loss=" << total << " normalizer=" << normalizer_
                      << " avg_loss=" << top[0]->cpu_data()[0];
}

void InfogainLossLayer::Backward_cpu(const std::vector<Blob*>& top,
                                     const std::vector<bool>& propagate_down,
                                     const std::vector<Blob*>& bottom) {
  if (propagate_down.size() > 1 && propagate_down[1]) {
    CAFFE_FFI_LOG_WARN() << "InfogainLoss Backward: cannot backpropagate to label inputs.";
  }
  if (propagate_down.size() > 2 && propagate_down[2]) {
    CAFFE_FFI_LOG_WARN() << "InfogainLoss Backward: cannot backpropagate to infogain matrix.";
  }
  if (!propagate_down[0]) {
    return;
  }

  const int channels = channels_;
  const int dim = channels * inner_num_;
  const int count = bottom[0]->count();
  const float* prob_data = prob_->cpu_data();
  const float* label = bottom[1]->cpu_data();
  const float* h = has_infogain_ ? bottom[2]->cpu_data() : nullptr;
  float* bottom_diff = bottom[0]->cpu_mutable_diff();

  const float loss_weight = top[0]->cpu_diff()[0];
  const float scale = loss_weight / normalizer_;

  // dL/dx_j = p_j * H_row_sum - H[gt, j]; identity case: p_j - delta_{gt,j}.
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        for (int c = 0; c < channels; ++c) {
          bottom_diff[i * dim + c * inner_num_ + j] = 0.0f;
        }
        continue;
      }
      CAFFE_FFI_CHECK_VALUE_GE(label_value, 0);
      CAFFE_FFI_CHECK_VALUE_LT(label_value, channels);
      float h_row_sum = 1.0f;
      if (has_infogain_) {
        const float* h_row = h + label_value * channels;
        h_row_sum = 0.0f;
        for (int k = 0; k < channels; ++k) {
          h_row_sum += h_row[k];
        }
      }
      for (int c = 0; c < channels; ++c) {
        float h_gt_c = has_infogain_ ? h[label_value * channels + c]
                                     : ((c == label_value) ? 1.0f : 0.0f);
        const float p = prob_data[i * dim + c * inner_num_ + j];
        bottom_diff[i * dim + c * inner_num_ + j] = p * h_row_sum - h_gt_c;
      }
    }
  }

  caffe_scal_fp32(static_cast<size_t>(count), scale, bottom_diff);
  CAFFE_FFI_LAYER_LOG << "InfogainLoss Backward: loss_weight=" << loss_weight
                      << " normalizer=" << normalizer_ << " scale=" << scale
                      << " has_infogain_=" << has_infogain_;
}

REGISTER_LAYER_CLASS(InfogainLoss);

}  // namespace caffe_ffi