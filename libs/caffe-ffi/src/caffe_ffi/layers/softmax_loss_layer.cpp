#include "caffe_ffi/layers/softmax_loss_layer.hpp"

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
  caffe_set_fp32(static_cast<size_t>(sum_multiplier_->count()), 1.0f, sum_multiplier_->cpu_data());
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
  float* prob_data = prob_->cpu_data();
  float* top_data = top[0]->cpu_data();
  float* scale_data = scale_->cpu_data();

  int channels = static_cast<int>(bottom[0]->shape(softmax_axis_));
  int dim = channels * inner_num_;

  CAFFE_FFI_LAYER_LOG << "SoftmaxWithLoss Forward: outer_num_=" << outer_num_
                      << " inner_num_=" << inner_num_
                      << " channels=" << channels
                      << " dim=" << dim
                      << " has_labels=" << (bottom.size() == 2);

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

  float loss = 0.0f;
  int count = 0;
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
        ++count;
      }
    }
    top_data[0] = (count > 0) ? loss / count : 0.0f;
    CAFFE_FFI_LAYER_LOG << "SoftmaxWithLoss Forward: count=" << count
                        << " total_loss=" << loss
                        << " avg_loss=" << top_data[0];
    if (top.size() == 2) {
      caffe_copy_fp32(static_cast<size_t>(prob_->count()), prob_data, top[1]->cpu_data());
    }
  } else {
    caffe_copy_fp32(static_cast<size_t>(prob_->count()), prob_data, top_data);
    CAFFE_FFI_LAYER_LOG << "SoftmaxWithLoss Forward: outputting probabilities only";
  }
}

REGISTER_LAYER_CLASS(SoftmaxWithLoss);

}  // namespace caffe_ffi
