#include "caffe_ffi/layers/softmax_loss_layer.hpp"

#include <algorithm>
#include <chrono>
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

void SoftmaxWithLossLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                                       const std::vector<Blob*>& top) {
  has_ignore_label_ = this->layer_param_.has_loss_param() &&
                      this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }

  softmax_axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.softmax_param().axis());
  label_axis_ = softmax_axis_;

  CAFFE_FFI_LAYER_LOG << "SoftmaxWithLoss LayerSetUp: softmax_axis_=" << softmax_axis_
                      << " label_axis_=" << label_axis_
                      << " has_ignore_label_=" << has_ignore_label_
                      << " ignore_label_=" << (has_ignore_label_ ? ignore_label_ : -1);
}

void SoftmaxWithLossLayer::Reshape(const std::vector<Blob*>& bottom,
                                    const std::vector<Blob*>& top) {
  std::vector<int64_t> prob_shape;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    prob_shape.push_back(bottom[0]->shape(i));
  }
  prob_ = make_object<Blob>(prob_shape);

  std::ostringstream prob_shape_ss;
  for (size_t i = 0; i < prob_shape.size(); ++i) {
    if (i > 0) prob_shape_ss << ", ";
    prob_shape_ss << prob_shape[i];
  }
  CAFFE_FFI_TENSOR_LOG << "SoftmaxWithLoss: created prob_ blob shape=[" << prob_shape_ss.str() << "]";

  if (bottom.size() == 2) {
    CAFFE_FFI_CHECK_VALUE_EQ(bottom[0]->num_axes(), bottom[1]->num_axes())
        << "Data and label must have same number of axes.";
    for (int i = 0; i < bottom[0]->num_axes(); ++i) {
      if (i == softmax_axis_) {
        CAFFE_FFI_CHECK_VALUE_EQ(bottom[1]->shape(i), 1)
            << "Label channel dimension must be 1.";
      } else {
        CAFFE_FFI_CHECK_VALUE_EQ(bottom[0]->shape(i), bottom[1]->shape(i))
            << "Data and label dimensions mismatch at axis " << i;
      }
    }
  }

  outer_num_ = static_cast<int>(bottom[0]->count(0, softmax_axis_));
  inner_num_ = static_cast<int>(bottom[0]->count(softmax_axis_ + 1));

  std::vector<int64_t> mult_dims = {bottom[0]->shape(softmax_axis_)};
  sum_multiplier_ = make_object<Blob>(mult_dims);
  caffe_set_fp32(static_cast<size_t>(sum_multiplier_->count()), 1.0f, sum_multiplier_->cpu_mutable_data());
  CAFFE_FFI_TENSOR_LOG << "SoftmaxWithLoss: created sum_multiplier_ shape=[" << mult_dims[0] << "] (initialized to 1.0)";

  std::vector<int64_t> scale_dims;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i == softmax_axis_) {
      scale_dims.push_back(1);
    } else {
      scale_dims.push_back(bottom[0]->shape(i));
    }
  }
  scale_ = make_object<Blob>(scale_dims);

  std::ostringstream scale_shape_ss;
  for (size_t i = 0; i < scale_dims.size(); ++i) {
    if (i > 0) scale_shape_ss << ", ";
    scale_shape_ss << scale_dims[i];
  }
  CAFFE_FFI_TENSOR_LOG << "SoftmaxWithLoss: created scale_ blob shape=[" << scale_shape_ss.str() << "]";

  std::ostringstream bottom0_shape_ss;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) bottom0_shape_ss << ", ";
    bottom0_shape_ss << bottom[0]->shape(i);
  }

  CAFFE_FFI_LAYER_LOG << "SoftmaxWithLoss Reshape: bottom[0]=[" << bottom0_shape_ss.str()
                      << "] outer_num_=" << outer_num_
                      << " inner_num_=" << inner_num_
                      << " softmax_axis_=" << softmax_axis_;

  if (bottom.size() == 2) {
    std::vector<int64_t> loss_shape = {1};
    top[0]->Reshape(loss_shape);
    CAFFE_FFI_TENSOR_LOG << "SoftmaxWithLoss: created top[0] (loss) shape=[1]";
    if (top.size() == 2) {
      top[1]->ReshapeLike(*bottom[0]);
      CAFFE_FFI_LAYER_LOG << "SoftmaxWithLoss: top[1] (probs) shape matches bottom[0]";
    }
  } else {
    top[0]->ReshapeLike(*bottom[0]);
    CAFFE_FFI_LAYER_LOG << "SoftmaxWithLoss: top[0] (probs) shape matches bottom[0]";
  }
}

void SoftmaxWithLossLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                        const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* prob_data = prob_->cpu_mutable_data();
  float* top_data = top[0]->cpu_mutable_data();
  float* scale_data = scale_->cpu_mutable_data();

  int channels = static_cast<int>(bottom[0]->shape(softmax_axis_));
  int dim = channels * inner_num_;

  CAFFE_FFI_LAYER_LOG << "SoftmaxWithLoss Forward: outer_num_=" << outer_num_
                      << " inner_num_=" << inner_num_
                      << " channels=" << channels
                      << " dim=" << dim
                      << " has_labels=" << (bottom.size() == 2);

  auto t_start = std::chrono::high_resolution_clock::now();

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

  // 概率分布统计
  float prob_min = std::numeric_limits<float>::max();
  float prob_max = -std::numeric_limits<float>::max();
  double sum_max_prob = 0.0;
  double sum_entropy = 0.0;
  int n_samples = outer_num_ * inner_num_;

  for (int i = 0; i < outer_num_; ++i) {
    const float* prob_data_i = prob_data + i * dim;
    for (int k = 0; k < inner_num_; ++k) {
      float sample_max = 0.0f;
      double sample_entropy = 0.0;
      for (int j = 0; j < channels; ++j) {
        float p = prob_data_i[j * inner_num_ + k];
        prob_min = std::min(prob_min, p);
        prob_max = std::max(prob_max, p);
        sample_max = std::max(sample_max, p);
        if (p > 0.0f) {
          sample_entropy -= static_cast<double>(p) * std::log(static_cast<double>(p));
        }
      }
      sum_max_prob += sample_max;
      sum_entropy += sample_entropy;
    }
  }
  float avg_max_prob = static_cast<float>(sum_max_prob / static_cast<double>(n_samples));
  float avg_entropy = static_cast<float>(sum_entropy / static_cast<double>(n_samples));
  float max_entropy = std::log(static_cast<float>(channels));

  float loss = 0.0f;
  int valid_count = 0;
  if (bottom.size() == 2) {
    const float* label = bottom[1]->cpu_data();
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          continue;
        }
        CAFFE_FFI_CHECK_VALUE_GE(label_value, 0);
        CAFFE_FFI_CHECK_VALUE_LT(label_value, channels);
        loss -= std::log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                                  std::numeric_limits<float>::min()));
        ++valid_count;
      }
    }
    top_data[0] = (valid_count > 0) ? loss / valid_count : 0.0f;
    CAFFE_FFI_LAYER_LOG << "SoftmaxWithLoss Forward: count=" << valid_count
                        << " total_loss=" << loss
                        << " avg_loss=" << top_data[0];
    if (top.size() == 2) {
      caffe_copy_fp32(static_cast<size_t>(prob_->count()), prob_data, top[1]->cpu_mutable_data());
    }
  } else {
    caffe_copy_fp32(static_cast<size_t>(prob_->count()), prob_data, top_data);
    CAFFE_FFI_LAYER_LOG << "SoftmaxWithLoss Forward: outputting probabilities only";
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  float uncertainty = (max_entropy > 0.0f) ? avg_entropy / max_entropy : 0.0f;
  float avg_loss = (valid_count > 0) ? loss / valid_count : 0.0f;

  CAFFE_FFI_LOG_INFO() << "[LOSS-PERF] " << this->name()
                       << " SoftmaxWithLoss forward: outer_num=" << outer_num_
                       << " channels=" << channels
                       << " inner_num=" << inner_num_
                       << " valid_count=" << valid_count
                       << " prob=[" << prob_min << ", " << prob_max << "]"
                       << " avg_max_prob=" << avg_max_prob
                       << " avg_entropy=" << avg_entropy
                       << " uncertainty=" << uncertainty
                       << " avg_loss=" << avg_loss
                       << " time=" << elapsed_us << "us";
}

void SoftmaxWithLossLayer::Backward_cpu(const std::vector<Blob*>& top,
                                        const std::vector<bool>& propagate_down,
                                        const std::vector<Blob*>& bottom) {
  // Labels (bottom[1]) never need gradients.
  if (bottom.size() == 2 && propagate_down[1]) {
    CAFFE_FFI_LOG_WARN() << "SoftmaxWithLoss Backward: cannot backpropagate to label inputs.";
  }

  if (!propagate_down[0]) {
    CAFFE_FFI_LAYER_LOG << "SoftmaxWithLoss Backward: propagate_down[0]=false, skipping";
    return;
  }

  if (bottom.size() < 2) {
    // Probability-only mode (no labels): cannot compute loss gradient.
    CAFFE_FFI_LOG_WARN() << "SoftmaxWithLoss Backward: no labels provided; cannot compute loss gradient.";
    float* bottom_diff = bottom[0]->cpu_mutable_diff();
    caffe_set_fp32(static_cast<size_t>(bottom[0]->count()), 0.0f, bottom_diff);
    return;
  }

  int channels = static_cast<int>(bottom[0]->shape(softmax_axis_));
  int dim = channels * inner_num_;
  int count = bottom[0]->count();
  const float* prob_data = prob_->cpu_data();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const float* label = bottom[1]->cpu_data();

  // loss_weight: scalar gradient from upstream (Net sets to layer's loss_weight, typically 1.0)
  float loss_weight = top[0]->cpu_diff()[0];

  CAFFE_FFI_LAYER_LOG << "SoftmaxWithLoss Backward: outer_num=" << outer_num_
                      << " inner_num=" << inner_num_
                      << " channels=" << channels
                      << " loss_weight=" << loss_weight;

  // Step 1: Copy probabilities to bottom_diff: d_bottom = prob
  caffe_copy_fp32(static_cast<size_t>(count), prob_data, bottom_diff);

  // Step 2: Subtract 1 at ground-truth class positions: d_bottom[j] = prob[j] - 1{j==label}
  //         Zero out ignored label positions.
  int valid_count = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        // Zero gradient for ignored positions across all channels
        for (int c = 0; c < channels; ++c) {
          bottom_diff[i * dim + c * inner_num_ + j] = 0.0f;
        }
      } else {
        CAFFE_FFI_CHECK_VALUE_GE(label_value, 0);
        CAFFE_FFI_CHECK_VALUE_LT(label_value, channels);
        bottom_diff[i * dim + label_value * inner_num_ + j] -= 1.0f;
        ++valid_count;
      }
    }
  }

  // Step 3: Scale by loss_weight / valid_count (average loss gradient)
  //         If no valid samples, gradient is zero.
  float scale = (valid_count > 0) ? (loss_weight / static_cast<float>(valid_count)) : 0.0f;
  if (scale != 1.0f) {
    caffe_scal_fp32(static_cast<size_t>(count), scale, bottom_diff);
  }

  // Gradient statistics for logging
  if (valid_count > 0) {
    double sum_sq = 0.0;
    float grad_max = -std::numeric_limits<float>::max();
    float grad_min = std::numeric_limits<float>::max();
    for (int i = 0; i < count; ++i) {
      float v = bottom_diff[i];
      sum_sq += static_cast<double>(v) * static_cast<double>(v);
      grad_max = std::max(grad_max, v);
      grad_min = std::min(grad_min, v);
    }
    float grad_norm = static_cast<float>(std::sqrt(sum_sq));
    CAFFE_FFI_LAYER_LOG << "SoftmaxWithLoss Backward: valid_count=" << valid_count
                        << " scale=" << scale
                        << " grad_range=[" << grad_min << ", " << grad_max << "]"
                        << " grad_l2norm=" << grad_norm;
  } else {
    CAFFE_FFI_LOG_WARN() << "SoftmaxWithLoss Backward: no valid labels, all gradients zero";
  }
}

REGISTER_LAYER_CLASS(SoftmaxWithLoss);

}  // namespace caffe_ffi
