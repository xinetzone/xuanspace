#include "caffe_ffi/layers/softmax_layer.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

#include <tvm/ffi/memory.h>

#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void SoftmaxLayer::Reshape(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  softmax_axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  std::vector<int64_t> mult_dims = {bottom[0]->shape(softmax_axis_)};
  sum_multiplier_ = make_object<Blob>(mult_dims);
  float* multiplier_data = sum_multiplier_->cpu_data();
  caffe_set_fp32(static_cast<size_t>(sum_multiplier_->count()), 1.0f, multiplier_data);
  outer_num_ = static_cast<int>(bottom[0]->count(0, softmax_axis_));
  inner_num_ = static_cast<int>(bottom[0]->count(softmax_axis_ + 1));
  std::vector<int64_t> scale_dims;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i == softmax_axis_) {
      scale_dims.push_back(1);
    } else {
      scale_dims.push_back(bottom[0]->shape(i));
    }
  }
  scale_ = make_object<Blob>(scale_dims);

  std::ostringstream bottom_shape_ss;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) bottom_shape_ss << ", ";
    bottom_shape_ss << bottom[0]->shape(i);
  }
  std::ostringstream scale_shape_ss;
  for (int i = 0; i < scale_->num_axes(); ++i) {
    if (i > 0) scale_shape_ss << ", ";
    scale_shape_ss << scale_->shape(i);
  }
  CAFFE_FFI_LAYER_LOG << "Softmax Reshape: softmax_axis=" << softmax_axis_
                      << " outer_num=" << outer_num_ << " inner_num=" << inner_num_
                      << " bottom_shape=[" << bottom_shape_ss.str() << "]"
                      << " sum_multiplier shape=[" << mult_dims[0] << "]"
                      << " scale shape=[" << scale_shape_ss.str() << "]";
}

void SoftmaxLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_data();
  float* scale_data = scale_->cpu_data();
  int channels = static_cast<int>(bottom[0]->shape(softmax_axis_));
  int dim = channels * inner_num_;
  caffe_copy_fp32(static_cast<size_t>(bottom[0]->count()), bottom_data, top_data);
  for (int i = 0; i < outer_num_; ++i) {
    float* top_data_i = top_data + i * dim;
    float* scale_data_i = scale_data + i * inner_num_;
    for (int k = 0; k < inner_num_; ++k) {
      scale_data_i[k] = top_data_i[k];
      for (int j = 1; j < channels; ++j) {
        scale_data_i[k] = std::max(scale_data_i[k], top_data_i[j * inner_num_ + k]);
      }
    }
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < inner_num_; ++k) {
        top_data_i[j * inner_num_ + k] -= scale_data_i[k];
      }
    }
    caffe_exp_fp32(static_cast<size_t>(dim), top_data_i, top_data_i);
    for (int k = 0; k < inner_num_; ++k) {
      scale_data_i[k] = 0;
      for (int j = 0; j < channels; ++j) {
        scale_data_i[k] += top_data_i[j * inner_num_ + k];
      }
    }
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < inner_num_; ++k) {
        top_data_i[j * inner_num_ + k] /= scale_data_i[k];
      }
    }
  }
}

REGISTER_LAYER_CLASS(Softmax);

}  // namespace caffe_ffi
