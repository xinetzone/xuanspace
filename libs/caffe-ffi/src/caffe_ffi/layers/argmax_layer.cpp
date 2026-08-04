#include "caffe_ffi/layers/argmax_layer.hpp"

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"

namespace caffe_ffi {

void ArgMaxLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                             const std::vector<Blob*>& top) {
  const caffe::ArgMaxParameter& argmax_param = this->layer_param_.argmax_param();
  out_max_val_ = argmax_param.out_max_val();
  top_k_ = argmax_param.top_k();
  has_axis_ = argmax_param.has_axis();
  CAFFE_FFI_CHECK_VALUE_GE(top_k_, 1) << "top k must not be less than 1.";
  if (has_axis_) {
    axis_ = bottom[0]->CanonicalAxisIndex(argmax_param.axis());
    CAFFE_FFI_CHECK_VALUE_GE(axis_, 0) << "axis must not be less than 0.";
    CAFFE_FFI_CHECK_VALUE_LE(axis_, bottom[0]->num_axes())
        << "axis must be less than or equal to the number of axis.";
    CAFFE_FFI_CHECK_VALUE_LE(top_k_, bottom[0]->shape(axis_))
        << "top_k must be less than or equal to the dimension of the axis.";
  } else {
    CAFFE_FFI_CHECK_VALUE_LE(top_k_, bottom[0]->count(1))
        << "top_k must be less than or equal to the dimension of the "
           "flattened bottom blob per instance.";
  }
  CAFFE_FFI_LAYER_LOG << "ArgMax LayerSetUp: out_max_val=" << out_max_val_
                      << " top_k=" << top_k_ << " has_axis=" << has_axis_
                      << " axis=" << axis_;
}

void ArgMaxLayer::Reshape(const std::vector<Blob*>& bottom,
                          const std::vector<Blob*>& top) {
  int num_top_axes = bottom[0]->num_axes();
  if (num_top_axes < 3) {
    num_top_axes = 3;
  }
  std::vector<int64_t> shape(num_top_axes, 1);
  if (has_axis_) {
    for (int i = 0; i < num_top_axes; ++i) {
      shape[i] = bottom[0]->shape(i);
    }
    shape[axis_] = top_k_;
  } else {
    shape[0] = bottom[0]->shape(0);
    shape[2] = top_k_;
    if (out_max_val_) {
      shape[1] = 2;
    }
  }
  top[0]->Reshape(shape);
  CAFFE_FFI_LAYER_LOG << "ArgMax Reshape: top_shape=[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) CAFFE_FFI_LAYER_LOG << ", ";
    CAFFE_FFI_LAYER_LOG << shape[i];
  }
  CAFFE_FFI_LAYER_LOG << "]";
}

void ArgMaxLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                              const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  int dim, axis_dist;
  if (has_axis_) {
    dim = static_cast<int>(bottom[0]->shape(axis_));
    axis_dist = static_cast<int>(bottom[0]->count(axis_) / dim);
  } else {
    dim = static_cast<int>(bottom[0]->count(1));
    axis_dist = 1;
  }
  const int num = static_cast<int>(bottom[0]->count() / dim);
  std::vector<std::pair<float, int>> bottom_data_vector(dim);
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < dim; ++j) {
      bottom_data_vector[j] = std::make_pair(
          bottom_data[(i / axis_dist * dim + j) * axis_dist + i % axis_dist], j);
    }
    std::partial_sort(
        bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
        bottom_data_vector.end(), std::greater<std::pair<float, int>>());
    for (int j = 0; j < top_k_; ++j) {
      if (out_max_val_) {
        if (has_axis_) {
          top_data[(i / axis_dist * top_k_ + j) * axis_dist + i % axis_dist] =
              bottom_data_vector[j].first;
        } else {
          top_data[2 * i * top_k_ + j] = bottom_data_vector[j].second;
          top_data[2 * i * top_k_ + top_k_ + j] = bottom_data_vector[j].first;
        }
      } else {
        top_data[(i / axis_dist * top_k_ + j) * axis_dist + i % axis_dist] =
            bottom_data_vector[j].second;
      }
    }
  }
  CAFFE_FFI_LAYER_LOG << "ArgMax Forward_cpu: num=" << num << " dim=" << dim
                      << " top_k=" << top_k_ << " out_max_val=" << out_max_val_;
}

REGISTER_LAYER_CLASS(ArgMax);

}  // namespace caffe_ffi