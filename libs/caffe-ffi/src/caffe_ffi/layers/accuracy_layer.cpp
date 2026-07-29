#include "caffe_ffi/layers/accuracy_layer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/math_utils.hpp"

namespace caffe_ffi {

void AccuracyLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                                const std::vector<Blob*>& top) {
  const caffe::AccuracyParameter& param = this->layer_param_.accuracy_param();
  top_k_ = static_cast<int>(param.top_k());
  has_ignore_label_ = param.has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = param.ignore_label();
  }

  CAFFE_FFI_LAYER_LOG << "Accuracy LayerSetUp: top_k_=" << top_k_
                      << " has_ignore_label_=" << has_ignore_label_
                      << " ignore_label_=" << (has_ignore_label_ ? ignore_label_ : -1);
}

void AccuracyLayer::Reshape(const std::vector<Blob*>& bottom,
                             const std::vector<Blob*>& top) {
  label_axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.accuracy_param().axis());
  outer_num_ = static_cast<int>(bottom[0]->count(0, label_axis_));
  inner_num_ = static_cast<int>(bottom[0]->count(label_axis_ + 1));

  int dim = static_cast<int>(bottom[0]->count() / outer_num_);
  CAFFE_FFI_CHECK_VALUE_LE(top_k_, dim) << "top_k must be <= number of classes.";

  std::vector<int64_t> top_shape = {1};
  top[0]->Reshape(top_shape);
  CAFFE_FFI_TENSOR_LOG << "Accuracy: created top[0] (accuracy) shape=[1]";

  std::ostringstream bottom0_shape_ss;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) bottom0_shape_ss << ", ";
    bottom0_shape_ss << bottom[0]->shape(i);
  }
  std::ostringstream bottom1_shape_ss;
  for (int i = 0; i < bottom[1]->num_axes(); ++i) {
    if (i > 0) bottom1_shape_ss << ", ";
    bottom1_shape_ss << bottom[1]->shape(i);
  }

  CAFFE_FFI_LAYER_LOG << "Accuracy Reshape: bottom[0]=[" << bottom0_shape_ss.str()
                      << "] bottom[1]=[" << bottom1_shape_ss.str()
                      << "] label_axis_=" << label_axis_
                      << " outer_num_=" << outer_num_
                      << " inner_num_=" << inner_num_
                      << " num_classes=" << dim;

  if (top.size() > 1) {
    std::vector<int64_t> per_class_shape = {static_cast<int64_t>(bottom[0]->shape(label_axis_))};
    top[1]->Reshape(per_class_shape);
    caffe_set_fp32(static_cast<size_t>(top[1]->count()), 0.0f, top[1]->cpu_data());

    std::ostringstream per_class_shape_ss;
    for (size_t i = 0; i < per_class_shape.size(); ++i) {
      if (i > 0) per_class_shape_ss << ", ";
      per_class_shape_ss << per_class_shape[i];
    }
    CAFFE_FFI_TENSOR_LOG << "Accuracy: created top[1] (per-class) shape=[" << per_class_shape_ss.str() << "]";
  }
}

void AccuracyLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                 const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  const float* bottom_label = bottom[1]->cpu_data();
  float* top_data = top[0]->cpu_data();

  int channels = static_cast<int>(bottom[0]->shape(label_axis_));
  int dim = channels * inner_num_;

  CAFFE_FFI_LAYER_LOG << "Accuracy Forward: outer_num_=" << outer_num_
                      << " inner_num_=" << inner_num_
                      << " channels=" << channels
                      << " dim=" << dim
                      << " top_k_=" << top_k_;

  int count = 0;
  float accuracy = 0.0f;

  std::vector<std::pair<float, int>> bottom_data_vector(channels);

  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const int label_value = static_cast<int>(bottom_label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      CAFFE_FFI_CHECK_VALUE_GE(label_value, 0);
      CAFFE_FFI_CHECK_VALUE_LT(label_value, channels);

      for (int k = 0; k < channels; ++k) {
        bottom_data_vector[k] = std::make_pair(
            bottom_data[i * dim + k * inner_num_ + j], k);
      }
      std::partial_sort(
          bottom_data_vector.begin(),
          bottom_data_vector.begin() + top_k_,
          bottom_data_vector.end(),
          std::greater<std::pair<float, int>>());

      for (int k = 0; k < top_k_; ++k) {
        if (bottom_data_vector[k].second == label_value) {
          ++accuracy;
          break;
        }
      }
      ++count;
    }
  }

  top_data[0] = (count > 0) ? accuracy / count : 0.0f;
  CAFFE_FFI_LAYER_LOG << "Accuracy Forward: count=" << count
                      << " correct=" << accuracy
                      << " accuracy=" << top_data[0];
}

REGISTER_LAYER_CLASS(Accuracy);

}  // namespace caffe_ffi
