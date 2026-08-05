#include "caffe_ffi/layers/contrastive_loss_layer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/math_utils.hpp"

namespace caffe_ffi {

void ContrastiveLossLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                                      const std::vector<Blob*>& top) {
  const caffe::ContrastiveLossParameter& param = this->layer_param_.contrastive_loss_param();
  margin_ = param.margin();
  legacy_version_ = param.legacy_version();

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

  CAFFE_FFI_LAYER_LOG << "ContrastiveLoss LayerSetUp: margin_=" << margin_
                      << " legacy_version_=" << legacy_version_
                      << " normalization_=" << normalization_;
}

void ContrastiveLossLayer::Reshape(const std::vector<Blob*>& bottom,
                                   const std::vector<Blob*>& top) {
  CAFFE_FFI_CHECK_VALUE_EQ(bottom[0]->count(), bottom[1]->count())
      << "ContrastiveLoss feature inputs must have the same count.";
  CAFFE_FFI_CHECK_VALUE_EQ(bottom[2]->count(), bottom[0]->shape(0))
      << "ContrastiveLoss label count must match the batch size.";

  num_ = static_cast<int>(bottom[0]->shape(0));
  dim_ = static_cast<int>(bottom[0]->count(1));

  diff_ = make_object<Blob>();
  diff_->ReshapeLike(*bottom[0]);

  std::vector<int64_t> loss_shape = {1};
  top[0]->Reshape(loss_shape);

  std::ostringstream feat_shape_ss;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) feat_shape_ss << ", ";
    feat_shape_ss << bottom[0]->shape(i);
  }
  CAFFE_FFI_LAYER_LOG << "ContrastiveLoss Reshape: features=[" << feat_shape_ss.str()
                      << "] num_=" << num_ << " dim_=" << dim_;
}

void ContrastiveLossLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                       const std::vector<Blob*>& top) {
  const int num = num_;
  const int dim = dim_;
  const float margin = margin_;
  const bool legacy_version = legacy_version_;
  const float* bottom0 = bottom[0]->cpu_data();
  const float* bottom1 = bottom[1]->cpu_data();
  const float* label = bottom[2]->cpu_data();
  float* diff_data = diff_->cpu_mutable_data();

  double total = 0.0;
  int valid_count = 0;
  for (int i = 0; i < num; ++i) {
    const float* f0 = bottom0 + i * dim;
    const float* f1 = bottom1 + i * dim;
    float* d = diff_data + i * dim;
    for (int j = 0; j < dim; ++j) {
      d[j] = f0[j] - f1[j];
    }
    const float dist_sq = caffe_cpu_dot_fp32(static_cast<size_t>(dim), d, d);
    const float y = label[i];
    if (y == 1.0f) {
      total += static_cast<double>(dist_sq);
    } else {
      if (legacy_version) {
        const float dist = std::sqrt(dist_sq);
        if (dist < margin) {
          const float m = margin - dist;
          total += static_cast<double>(m * m);
        }
      } else {
        if (dist_sq < margin) {
          const float m = margin - dist_sq;
          total += static_cast<double>(m * m);
        }
      }
    }
    ++valid_count;
  }

  switch (normalization_) {
    case caffe::LossParameter_NormalizationMode_FULL:
      normalizer_ = static_cast<float>(num * dim);
      break;
    case caffe::LossParameter_NormalizationMode_VALID:
      normalizer_ = static_cast<float>(valid_count);
      break;
    case caffe::LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer_ = static_cast<float>(num);
      break;
    case caffe::LossParameter_NormalizationMode_NONE:
    default:
      normalizer_ = 1.0f;
      break;
  }
  normalizer_ = std::max(1.0f, normalizer_);
  dist_weight_ = 1.0f / normalizer_;

  top[0]->cpu_mutable_data()[0] = static_cast<float>(total) * dist_weight_;
  CAFFE_FFI_LAYER_LOG << "ContrastiveLoss Forward: valid_count=" << valid_count
                      << " total_loss=" << total << " normalizer=" << normalizer_
                      << " avg_loss=" << top[0]->cpu_data()[0];
}

void ContrastiveLossLayer::Backward_cpu(const std::vector<Blob*>& top,
                                        const std::vector<bool>& propagate_down,
                                        const std::vector<Blob*>& bottom) {
  if (propagate_down.size() > 2 && propagate_down[2]) {
    CAFFE_FFI_LOG_WARN() << "ContrastiveLoss Backward: cannot backpropagate to label inputs.";
  }
  const bool need_dx0 = propagate_down[0];
  const bool need_dx1 = propagate_down[1];
  if (!need_dx0 && !need_dx1) {
    CAFFE_FFI_LAYER_LOG << "ContrastiveLoss Backward: no data gradients needed, skipping";
    return;
  }

  const int num = num_;
  const int dim = dim_;
  const float margin = margin_;
  const bool legacy_version = legacy_version_;
  const float* label = bottom[2]->cpu_data();
  const float* diff_data = diff_->cpu_data();
  const float loss_weight = top[0]->cpu_diff()[0];
  const float scale = loss_weight * dist_weight_;
  float* x0_diff = need_dx0 ? bottom[0]->cpu_mutable_diff() : nullptr;
  float* x1_diff = need_dx1 ? bottom[1]->cpu_mutable_diff() : nullptr;

  CAFFE_FFI_LAYER_LOG << "ContrastiveLoss Backward: num=" << num
                      << " dim=" << dim << " loss_weight=" << loss_weight
                      << " scale=" << scale
                      << " need_dx0=" << need_dx0 << " need_dx1=" << need_dx1;

  for (int i = 0; i < num; ++i) {
    const float* d = diff_data + i * dim;
    float alpha = 0.0f;
    const float y = label[i];
    if (y == 1.0f) {
      alpha = 2.0f * scale;
    } else {
      const float dist_sq = caffe_cpu_dot_fp32(static_cast<size_t>(dim), d, d);
      if (legacy_version) {
        const float dist = std::sqrt(dist_sq);
        if (dist < margin && dist > 0.0f) {
          alpha = -2.0f * (margin - dist) / dist * scale;
        }
      } else {
        if (dist_sq < margin) {
          alpha = -2.0f * (margin - dist_sq) * scale;
        }
      }
    }
    if (need_dx0) {
      float* x0 = x0_diff + i * dim;
      for (int j = 0; j < dim; ++j) {
        x0[j] = alpha * d[j];
      }
    }
    if (need_dx1) {
      float* x1 = x1_diff + i * dim;
      for (int j = 0; j < dim; ++j) {
        x1[j] = -alpha * d[j];
      }
    }
  }
}

REGISTER_LAYER_CLASS(ContrastiveLoss);

}  // namespace caffe_ffi