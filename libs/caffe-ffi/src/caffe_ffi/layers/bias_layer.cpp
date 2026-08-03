#include "caffe_ffi/layers/bias_layer.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
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

void BiasLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  const caffe::BiasParameter& param = this->layer_param_.bias_param();
  axis_ = bottom[0]->CanonicalAxisIndex(param.axis());
  num_axes_ = param.num_axes();

  CAFFE_FFI_LAYER_LOG << "Bias LayerSetUp: axis_=" << axis_
                      << " num_axes_=" << num_axes_;

  CAFFE_FFI_CHECK_VALUE_GE(num_axes_, 1) << "num_axes should be >= 1";
  CAFFE_FFI_CHECK_VALUE_LE(axis_ + num_axes_, bottom[0]->num_axes())
      << "axis + num_axes exceeds blob dimensions";

  if (bottom.size() == 1 && this->blobs_.size() == 0) {
    this->blobs_.resize(1);
    std::vector<int64_t> bias_shape;
    for (int i = 0; i < num_axes_; ++i) {
      bias_shape.push_back(bottom[0]->shape(axis_ + i));
    }
    this->blobs_[0] = make_object<Blob>(bias_shape);
    caffe_set_fp32(static_cast<size_t>(this->blobs_[0]->count()), 0.0f, this->blobs_[0]->cpu_mutable_data());

    std::ostringstream bias_shape_ss;
    for (size_t i = 0; i < bias_shape.size(); ++i) {
      if (i > 0) bias_shape_ss << ", ";
      bias_shape_ss << bias_shape[i];
    }
    CAFFE_FFI_TENSOR_LOG << "Bias: created bias blob shape=[" << bias_shape_ss.str() << "] (initialized to 0.0)";
  } else if (this->blobs_.size() > 0) {
    CAFFE_FFI_LAYER_LOG << "Bias: using pre-loaded weights, blobs_.size=" << this->blobs_.size();
  } else {
    CAFFE_FFI_LAYER_LOG << "Bias: bias from bottom[1]";
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

void BiasLayer::Reshape(const std::vector<Blob*>& bottom,
                         const std::vector<Blob*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  outer_dim_ = static_cast<int>(bottom[0]->count(0, axis_));
  bias_dim_ = static_cast<int>((bottom.size() > 1) ? bottom[1]->count()
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

  CAFFE_FFI_LAYER_LOG << "Bias Reshape: input=[" << input_shape_ss.str()
                      << "] output=[" << output_shape_ss.str()
                      << "] outer_dim_=" << outer_dim_
                      << " bias_dim_=" << bias_dim_
                      << " inner_dim_=" << inner_dim_;

  int dim = bias_dim_;
  for (int i = 0; i < num_axes_; ++i) {
    int expected = bottom[0]->shape(axis_ + i);
    int actual = (bottom.size() > 1) ? static_cast<int>(bottom[1]->shape(i))
                                     : static_cast<int>(this->blobs_[0]->shape(i));
    if (expected != actual) {
      // Format bias shape
      std::ostringstream bias_shape_ss;
      const Blob* bias_blob = (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();
      bias_shape_ss << "(";
      for (int bi = 0; bi < bias_blob->num_axes(); ++bi) {
        if (bi > 0) bias_shape_ss << ", ";
        bias_shape_ss << bias_blob->shape(bi);
      }
      bias_shape_ss << ")";
      std::ostringstream input_shape_ss;
      input_shape_ss << "(";
      for (int si = 0; si < bottom[0]->num_axes(); ++si) {
        if (si > 0) input_shape_ss << ", ";
        input_shape_ss << bottom[0]->shape(si);
      }
      input_shape_ss << ")";
      CAFFE_FFI_LOG_ERROR() << "[BIAS-SHAPE-MISMATCH] layer='" << this->name()
                            << "' Bias dimension mismatch at axis[" << axis_ + i
                            << "] (bias_axis=" << i << "):"
                            << " input.shape(" << axis_ + i << ")=" << expected
                            << " but bias.shape(" << i << ")=" << actual
                            << ". input_shape=" << input_shape_ss.str()
                            << " bias_shape=" << bias_shape_ss.str()
                            << " axis_=" << axis_ << " num_axes_=" << num_axes_
                            << " bias_source=" << ((bottom.size() > 1) ? "bottom[1]" : "learnable blob")
                            << "\n  *** HINT: Bias shape must match input's dimensions starting at axis."
                            << "\n      For learnable positional encoding with axis=1 num_axes=2:"
                            << " bias shape must be (seq_len, d_model), same as input's axes 1..2."
                            << "\n      For per-channel bias (e.g. CNN bias): use axis=1 num_axes=1"
                            << " with bias shape (channels,).";
    }
    CAFFE_FFI_CHECK_VALUE_EQ(bottom[0]->shape(axis_ + i), (bottom.size() > 1)
                 ? bottom[1]->shape(i) : this->blobs_[0]->shape(i))
        << "Dimensions mismatch for bias (layer '" << this->name()
        << "', axis " << (axis_ + i) << "). See [BIAS-SHAPE-MISMATCH] above.";
    dim /= static_cast<int>(bottom[0]->shape(axis_ + i));
  }
}

void BiasLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                             const std::vector<Blob*>& top) {
  const float* bias_data = (bottom.size() > 1) ? bottom[1]->cpu_data()
                          : this->blobs_[0]->cpu_data();
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int count = static_cast<int>(bottom[0]->count());

  CAFFE_FFI_LAYER_LOG << "Bias Forward: count=" << count
                      << " outer_dim_=" << outer_dim_
                      << " bias_dim_=" << bias_dim_
                      << " inner_dim_=" << inner_dim_
                      << " bias_from_bottom=" << (bottom.size() > 1);

  auto t_start = std::chrono::high_resolution_clock::now();

  float in_min = std::numeric_limits<float>::max();
  float in_max = -std::numeric_limits<float>::max();
  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();
  float b_min = std::numeric_limits<float>::max();
  float b_max = -std::numeric_limits<float>::max();

  // bias值域
  for (int i = 0; i < bias_dim_; ++i) {
    b_min = std::min(b_min, bias_data[i]);
    b_max = std::max(b_max, bias_data[i]);
  }

  // 单次遍历：copy + bias + in/out值域统计（融合）
  for (int n = 0; n < outer_dim_; ++n) {
    for (int d = 0; d < bias_dim_; ++d) {
      const float b = bias_data[d];
      for (int i = 0; i < inner_dim_; ++i) {
        const int idx = n * bias_dim_ * inner_dim_ + d * inner_dim_ + i;
        float x = bottom_data[idx];
        float y = x + b;
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

  CAFFE_FFI_LOG_INFO() << "[BIAS-PERF] " << this->name()
                       << " Bias forward: outer_dim=" << outer_dim_
                       << " bias_dim=" << bias_dim_
                       << " inner_dim=" << inner_dim_
                       << " in=[" << in_min << ", " << in_max << "]"
                       << " out=[" << out_min << ", " << out_max << "]"
                       << " bias=[" << b_min << ", " << b_max << "]"
                       << " time=" << elapsed_us << "us";
}

void BiasLayer::Backward_cpu(const std::vector<Blob*>& top,
                              const std::vector<bool>& propagate_down,
                              const std::vector<Blob*>& bottom) {
  const float* top_diff = top[0]->cpu_diff();
  const int count = static_cast<int>(bottom[0]->count());
  const bool bias_from_bottom = (bottom.size() > 1);
  const bool need_dx = propagate_down[0];
  const bool need_dbias = bias_from_bottom
      ? propagate_down[1]
      : (this->blobs_.size() >= 1 && this->param_propagate_down_[0]);

  CAFFE_FFI_LAYER_LOG << "Bias Backward_cpu: count=" << count
                      << " outer_dim_=" << outer_dim_
                      << " bias_dim_=" << bias_dim_
                      << " inner_dim_=" << inner_dim_
                      << " need_dx=" << need_dx
                      << " need_dbias=" << need_dbias
                      << " bias_from_bottom=" << bias_from_bottom;

  if (!need_dx && !need_dbias) {
    CAFFE_FFI_LAYER_LOG << "Bias Backward_cpu: no gradients needed, skipping";
    return;
  }

  auto t_start = std::chrono::high_resolution_clock::now();

  float* bottom_diff = need_dx ? bottom[0]->cpu_mutable_diff() : nullptr;
  float* bias_diff = nullptr;
  if (need_dbias) {
    bias_diff = bias_from_bottom ? bottom[1]->cpu_mutable_diff()
                                 : this->blobs_[0]->cpu_mutable_diff();
    std::memset(bias_diff, 0, sizeof(float) * bias_dim_);
  }

  // Value-range tracking for diagnostics
  float dx_min = std::numeric_limits<float>::max();
  float dx_max = -std::numeric_limits<float>::max();
  float db_min = std::numeric_limits<float>::max();
  float db_max = -std::numeric_limits<float>::max();

  // Single-pass triple loop: compute dX and d_bias simultaneously
  // Forward: y[n,d,i] = x[n,d,i] + b[d]
  // Backward:
  //   dX[n,d,i] = dy[n,d,i]           (∂y/∂x = 1, gradient passes through)
  //   d_bias[d] += dy[n,d,i]           (∂y/∂b = 1, sum over broadcast dims n,i)
  for (int n = 0; n < outer_dim_; ++n) {
    for (int d = 0; d < bias_dim_; ++d) {
      float db_acc = 0.0f;
      for (int i = 0; i < inner_dim_; ++i) {
        const int idx = n * bias_dim_ * inner_dim_ + d * inner_dim_ + i;
        const float dy_val = top_diff[idx];

        // dX = dy (identity, gradient passes through directly)
        if (need_dx) {
          bottom_diff[idx] = dy_val;
          dx_min = std::min(dx_min, dy_val);
          dx_max = std::max(dx_max, dy_val);
        }

        // Accumulate d_bias
        db_acc += dy_val;
      }
      if (need_dbias) {
        bias_diff[d] += db_acc;
      }
    }
  }

  // Compute bias_diff range
  if (need_dbias) {
    for (int d = 0; d < bias_dim_; ++d) {
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
  if (need_dbias) {
    extra += " dbias=[" + std::to_string(db_min) + ", " + std::to_string(db_max) + "]";
  }

  CAFFE_FFI_LOG_INFO() << "[BIAS-PERF] " << this->name()
                       << " Bias backward: outer_dim=" << outer_dim_
                       << " bias_dim=" << bias_dim_
                       << " inner_dim=" << inner_dim_
                       << extra
                       << " time=" << elapsed_us << "us";
}

REGISTER_LAYER_CLASS(Bias);

}  // namespace caffe_ffi
