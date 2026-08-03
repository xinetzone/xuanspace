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

#include <cstring>

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

void ScaleLayer::Backward_cpu(const std::vector<Blob*>& top,
                               const std::vector<bool>& propagate_down,
                               const std::vector<Blob*>& bottom) {
  const float* top_diff = top[0]->cpu_diff();
  const float* bottom_data = bottom[0]->cpu_data();
  const float* scale_data = (bottom.size() > 1) ? bottom[1]->cpu_data()
                           : this->blobs_[0]->cpu_data();

  const bool scale_from_bottom = (bottom.size() > 1);
  const bool need_dx = propagate_down[0];
  const bool need_dscale = scale_from_bottom
      ? propagate_down[1]
      : (this->blobs_.size() >= 1 && this->param_propagate_down_[0]);

  // Determine bias parameter index: when scale is from bottom[1], bias is blobs_[0] (index 0);
  // when scale is from blobs_[0], bias is blobs_[1] (index 1).
  int bias_param_idx = -1;
  bool need_dbias = false;
  if (bias_term_ && this->blobs_.size() >= 1) {
    if (scale_from_bottom && this->blobs_.size() >= 1) {
      bias_param_idx = 0;
      need_dbias = this->param_propagate_down_[0];
    } else if (!scale_from_bottom && this->blobs_.size() >= 2) {
      bias_param_idx = 1;
      need_dbias = this->param_propagate_down_[1];
    }
  }

  const int count = bottom[0]->count();
  CAFFE_FFI_LAYER_LOG << "Scale Backward_cpu: count=" << count
                      << " outer_dim=" << outer_dim_
                      << " scale_dim=" << scale_dim_
                      << " inner_dim=" << inner_dim_
                      << " need_dx=" << need_dx
                      << " need_dscale=" << need_dscale
                      << " scale_from_bottom=" << scale_from_bottom
                      << " need_dbias=" << need_dbias
                      << " bias_param_idx=" << bias_param_idx;

  if (!need_dx && !need_dscale && !need_dbias) {
    CAFFE_FFI_LAYER_LOG << "Scale Backward_cpu: no gradients needed, skipping";
    return;
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  float* bottom_diff = need_dx ? bottom[0]->cpu_mutable_diff() : nullptr;
  float* scale_diff = nullptr;
  if (need_dscale) {
    scale_diff = scale_from_bottom
        ? bottom[1]->cpu_mutable_diff()
        : this->blobs_[0]->cpu_mutable_diff();
    std::memset(scale_diff, 0, sizeof(float) * scale_dim_);
  }
  float* bias_diff = nullptr;
  if (need_dbias) {
    bias_diff = this->blobs_[bias_param_idx]->cpu_mutable_diff();
    std::memset(bias_diff, 0, sizeof(float) * scale_dim_);
  }

  // Value-range tracking for diagnostics
  float dx_min = std::numeric_limits<float>::max();
  float dx_max = -std::numeric_limits<float>::max();
  float ds_min = std::numeric_limits<float>::max();
  float ds_max = -std::numeric_limits<float>::max();
  float db_min = std::numeric_limits<float>::max();
  float db_max = -std::numeric_limits<float>::max();

  // Single-pass triple loop: compute dX, d_scale, d_bias simultaneously
  // Forward: y[n,d,i] = x[n,d,i] * s[d] + b[d]
  // Backward:
  //   dX[n,d,i] = dy[n,d,i] * s[d]
  //   d_scale[d] += dy[n,d,i] * x[n,d,i]   (sum over broadcast dims n,i)
  //   d_bias[d]  += dy[n,d,i]               (sum over broadcast dims n,i)
  for (int n = 0; n < outer_dim_; ++n) {
    for (int d = 0; d < scale_dim_; ++d) {
      const float factor = scale_data[d];
      float ds_acc = 0.0f;
      float db_acc = 0.0f;
      for (int i = 0; i < inner_dim_; ++i) {
        const int idx = n * scale_dim_ * inner_dim_ + d * inner_dim_ + i;
        const float dy_val = top_diff[idx];
        const float x_val = bottom_data[idx];

        // dX = dy * s
        if (need_dx) {
          float dx_val = dy_val * factor;
          bottom_diff[idx] = dx_val;
          dx_min = std::min(dx_min, dx_val);
          dx_max = std::max(dx_max, dx_val);
        }

        // Accumulate d_scale and d_bias
        ds_acc += dy_val * x_val;
        db_acc += dy_val;
      }
      if (need_dscale) {
        scale_diff[d] += ds_acc;
      }
      if (need_dbias) {
        bias_diff[d] += db_acc;
      }
    }
  }

  // Compute scale_diff and bias_diff ranges
  if (need_dscale) {
    for (int d = 0; d < scale_dim_; ++d) {
      ds_min = std::min(ds_min, scale_diff[d]);
      ds_max = std::max(ds_max, scale_diff[d]);
    }
  }
  if (need_dbias) {
    for (int d = 0; d < scale_dim_; ++d) {
      db_min = std::min(db_min, bias_diff[d]);
      db_max = std::max(db_max, bias_diff[d]);
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  std::string extra;
  if (need_dx) {
    extra += " dx=[" + std::to_string(dx_min) + ", " + std::to_string(dx_max) + "]";
  }
  if (need_dscale) {
    extra += " dscale=[" + std::to_string(ds_min) + ", " + std::to_string(ds_max) + "]";
  }
  if (need_dbias) {
    extra += " dbias=[" + std::to_string(db_min) + ", " + std::to_string(db_max) + "]";
  }

  CAFFE_FFI_LOG_INFO() << "[SCALE-PERF] " << this->name()
                       << " Scale backward: outer_dim=" << outer_dim_
                       << " scale_dim=" << scale_dim_
                       << " inner_dim=" << inner_dim_
                       << extra
                       << " time=" << elapsed_us << "us";
}

REGISTER_LAYER_CLASS(Scale);

}  // namespace caffe_ffi
