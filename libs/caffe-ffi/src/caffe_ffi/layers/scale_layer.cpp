#include "caffe_ffi/layers/scale_layer.hpp"

#include <algorithm>
#include <chrono>
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

void ScaleLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                             const std::vector<Blob*>& top) {
  const caffe::ScaleParameter& param = this->layer_param_.scale_param();
  axis_ = bottom[0]->CanonicalAxisIndex(param.axis());
  num_axes_ = param.num_axes();
  bias_term_ = param.bias_term();

  CAFFE_FFI_LAYER_LOG << "Scale LayerSetUp: axis_=" << axis_
                      << " num_axes_=" << num_axes_
                      << " bias_term_=" << bias_term_;

  CAFFE_FFI_CHECK_VALUE_GE(num_axes_, 1) << "num_axes should be >= 1";
  CAFFE_FFI_CHECK_VALUE_LE(axis_ + num_axes_, bottom[0]->num_axes())
      << "axis + num_axes exceeds blob dimensions";

  std::vector<int64_t> scale_shape;
  for (int i = 0; i < num_axes_; ++i) {
    scale_shape.push_back(bottom[0]->shape(axis_ + i));
  }
  const int64_t scale_dim = Count(ShapeView(scale_shape.data(), scale_shape.size()));

  std::ostringstream scale_shape_ss;
  for (size_t i = 0; i < scale_shape.size(); ++i) {
    if (i > 0) scale_shape_ss << ", ";
    scale_shape_ss << scale_shape[i];
  }
  CAFFE_FFI_LAYER_LOG << "Scale: scale_shape=[" << scale_shape_ss.str()
                      << "] scale_dim=" << scale_dim;

  if (bottom.size() == 1 && this->blobs_.size() > 0) {
    CAFFE_FFI_LAYER_LOG << "Scale: using pre-loaded weights, blobs_.size=" << this->blobs_.size();
    CAFFE_FFI_CHECK_VALUE_GE(this->blobs_.size(), 1U);
    if (bias_term_) {
      CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_.size(), 2U);
      CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_[1]->count(), scale_dim);
    }
    CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_[0]->count(), scale_dim);
  } else if (bottom.size() == 1) {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    this->blobs_[0] = make_object<Blob>(scale_shape);
    caffe_set_fp32(static_cast<size_t>(this->blobs_[0]->count()), 1.0f, this->blobs_[0]->cpu_mutable_data());
    CAFFE_FFI_TENSOR_LOG << "Scale: created scale blob shape=[" << scale_shape_ss.str() << "] (initialized to 1.0)";
    if (bias_term_) {
      this->blobs_[1] = make_object<Blob>(scale_shape);
      caffe_set_fp32(static_cast<size_t>(this->blobs_[1]->count()), 0.0f, this->blobs_[1]->cpu_mutable_data());
      CAFFE_FFI_TENSOR_LOG << "Scale: created bias blob shape=[" << scale_shape_ss.str() << "] (initialized to 0.0)";
    }
  } else {
    CAFFE_FFI_LAYER_LOG << "Scale: scale factor from bottom[1]";
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

void ScaleLayer::Reshape(const std::vector<Blob*>& bottom,
                          const std::vector<Blob*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  outer_dim_ = static_cast<int>(bottom[0]->count(0, axis_));
  scale_dim_ = static_cast<int>((bottom.size() > 1) ? bottom[1]->count()
                               : this->blobs_[0]->count());
  inner_dim_ = static_cast<int>(bottom[0]->count(axis_ + num_axes_));

  std::ostringstream input_shape_ss;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) input_shape_ss << ", ";
    input_shape_ss << bottom[0]->shape(i);
  }
  std::ostringstream output_shape_ss;
  for (int i = 0; i < top[0]->num_axes(); ++i) {
    if (i > 0) output_shape_ss << ", ";
    output_shape_ss << top[0]->shape(i);
  }

  CAFFE_FFI_LAYER_LOG << "Scale Reshape: input=[" << input_shape_ss.str()
                      << "] output=[" << output_shape_ss.str()
                      << "] outer_dim_=" << outer_dim_
                      << " scale_dim_=" << scale_dim_
                      << " inner_dim_=" << inner_dim_;
}

void ScaleLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                              const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const float* scale_data = (bottom.size() > 1) ? bottom[1]->cpu_data()
                           : this->blobs_[0]->cpu_data();
  const int count = static_cast<int>(bottom[0]->count());

  CAFFE_FFI_LAYER_LOG << "Scale Forward: count=" << count
                      << " outer_dim_=" << outer_dim_
                      << " scale_dim_=" << scale_dim_
                      << " inner_dim_=" << inner_dim_
                      << " bias_term_=" << bias_term_
                      << " scale_from_bottom=" << (bottom.size() > 1);

  auto t_start = std::chrono::high_resolution_clock::now();

  float in_min = std::numeric_limits<float>::max();
  float in_max = -std::numeric_limits<float>::max();
  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();
  float s_min = std::numeric_limits<float>::max();
  float s_max = -std::numeric_limits<float>::max();
  float b_min = std::numeric_limits<float>::max();
  float b_max = -std::numeric_limits<float>::max();

  // scale值域
  for (int i = 0; i < scale_dim_; ++i) {
    s_min = std::min(s_min, scale_data[i]);
    s_max = std::max(s_max, scale_data[i]);
  }

  const float* bias_data = (bias_term_ && this->blobs_.size() > 1) ? this->blobs_[1]->cpu_data() : nullptr;
  if (bias_data) {
    for (int i = 0; i < scale_dim_; ++i) {
      b_min = std::min(b_min, bias_data[i]);
      b_max = std::max(b_max, bias_data[i]);
    }
  }

  // 单次遍历：scale+bias+in/out值域统计（融合，无二次遍历）
  for (int n = 0; n < outer_dim_; ++n) {
    for (int d = 0; d < scale_dim_; ++d) {
      const float factor = scale_data[d];
      const float b = bias_data ? bias_data[d] : 0.0f;
      for (int i = 0; i < inner_dim_; ++i) {
        const int idx = n * scale_dim_ * inner_dim_ + d * inner_dim_ + i;
        float x = bottom_data[idx];
        float y = x * factor + b;
        top_data[idx] = y;
        in_min = std::min(in_min, x);
        in_max = std::max(in_max, x);
        out_min = std::min(out_min, y);
        out_max = std::max(out_max, y);
      }
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  std::string b_str;
  if (bias_data) {
    b_str = " bias=[" + std::to_string(b_min) + ", " + std::to_string(b_max) + "]";
  }

  CAFFE_FFI_LOG_INFO() << "[SCALE-PERF] " << this->name()
                       << " Scale forward: outer_dim=" << outer_dim_
                       << " scale_dim=" << scale_dim_
                       << " inner_dim=" << inner_dim_
                       << " bias_term=" << bias_term_
                       << " in=[" << in_min << ", " << in_max << "]"
                       << " out=[" << out_min << ", " << out_max << "]"
                       << " scale=[" << s_min << ", " << s_max << "]"
                       << b_str
                       << " time=" << elapsed_us << "us";
}

REGISTER_LAYER_CLASS(Scale);

}  // namespace caffe_ffi
