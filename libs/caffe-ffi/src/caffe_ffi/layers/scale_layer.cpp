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
    // Apply scale_param.filler() if specified (default to 1.0, the identity scale).
    float scale_value = 1.0f;
    if (param.has_filler()) {
      const caffe::FillerParameter& filler = param.filler();
      if (filler.type() == "constant") {
        scale_value = filler.value();
      } else {
        CAFFE_FFI_LOG_WARN() << "[SCALE-FILLER] filler type '" << filler.type()
                             << "' not implemented, using constant 1.0 for scale";
      }
    }
    caffe_set_fp32(static_cast<size_t>(this->blobs_[0]->count()), scale_value,
                   this->blobs_[0]->cpu_mutable_data());
    CAFFE_FFI_TENSOR_LOG << "Scale: created scale blob shape=[" << scale_shape_ss.str()
                         << "] (filler value=" << scale_value << ")";
    if (bias_term_) {
      this->blobs_[1] = make_object<Blob>(scale_shape);
      // Apply scale_param.bias_filler() if specified (default to 0.0, the identity bias).
      float bias_value = 0.0f;
      if (param.has_bias_filler()) {
        const caffe::FillerParameter& filler = param.bias_filler();
        if (filler.type() == "constant") {
          bias_value = filler.value();
        }
      }
      caffe_set_fp32(static_cast<size_t>(this->blobs_[1]->count()), bias_value,
                     this->blobs_[1]->cpu_mutable_data());
      CAFFE_FFI_TENSOR_LOG << "Scale: created bias blob shape=[" << scale_shape_ss.str()
                           << "] (filler value=" << bias_value << ")";
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
  const float* scale_data = (bottom.size() > 1) ? bottom[1]->cpu_data()
                           : this->blobs_[0]->cpu_data();
  const int count = static_cast<int>(bottom[0]->count());
  const float* bias_data = (bias_term_ && this->blobs_.size() > 1) ? this->blobs_[1]->cpu_data() : nullptr;

  CAFFE_FFI_LAYER_LOG << "Scale Forward: count=" << count
                      << " outer_dim_=" << outer_dim_
                      << " scale_dim_=" << scale_dim_
                      << " inner_dim_=" << inner_dim_
                      << " bias_term_=" << bias_term_
                      << " scale_from_bottom=" << (bottom.size() > 1);

  // TS31-B4 COW promotion: when Scale degenerates to identity (scale all 1.0
  // and no bias / bias all 0.0), replace the O(n) memcpy with O(1) refcount
  // sharing. The COW clone is deferred to the first downstream mutable access,
  // preserving the isolation semantics of the original memcpy. NOTE: the
  // identity check must run BEFORE cpu_mutable_data(), which would otherwise
  // trigger a COW clone on a shared tensor.
  const bool inplace = (bottom[0] == top[0]);
  bool identity = !inplace;
  if (identity) {
    for (int i = 0; i < scale_dim_; ++i) {
      if (scale_data[i] != 1.0f) { identity = false; break; }
    }
  }
  if (identity && bias_data) {
    for (int i = 0; i < scale_dim_; ++i) {
      if (bias_data[i] != 0.0f) { identity = false; break; }
    }
  }
  if (identity) {
    top[0]->ShareData(bottom[0]);
    cow_identity_ = true;
    CAFFE_FFI_LOG_INFO() << "[SCALE-COW] " << this->name()
                         << " Scale forward: IDENTITY (scale=1, bias=0) -> COW zero-copy"
                         << " count=" << count
                         << " shared_ptr=" << static_cast<const void*>(top[0]->cpu_data())
                         << " bottom_ptr=" << static_cast<const void*>(bottom_data);
    return;
  }
  cow_identity_ = false;

  float* top_data = top[0]->cpu_mutable_data();

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  float in_min = std::numeric_limits<float>::max();
  float in_max = -std::numeric_limits<float>::max();
  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();
  float s_min = std::numeric_limits<float>::max();
  float s_max = -std::numeric_limits<float>::max();
  float b_min = std::numeric_limits<float>::max();
  float b_max = -std::numeric_limits<float>::max();
#endif

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  // scale值域
  for (int i = 0; i < scale_dim_; ++i) {
    s_min = std::min(s_min, scale_data[i]);
    s_max = std::max(s_max, scale_data[i]);
  }

  if (bias_data) {
    for (int i = 0; i < scale_dim_; ++i) {
      b_min = std::min(b_min, bias_data[i]);
      b_max = std::max(b_max, bias_data[i]);
    }
  }
#endif

  for (int n = 0; n < outer_dim_; ++n) {
    for (int d = 0; d < scale_dim_; ++d) {
      const float factor = scale_data[d];
      const float b = bias_data ? bias_data[d] : 0.0f;
      for (int i = 0; i < inner_dim_; ++i) {
        const int idx = n * scale_dim_ * inner_dim_ + d * inner_dim_ + i;
        float x = bottom_data[idx];
        float y = x * factor + b;
        top_data[idx] = y;
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
        in_min = std::min(in_min, x);
        in_max = std::max(in_max, x);
        out_min = std::min(out_min, y);
        out_max = std::max(out_max, y);
#endif
      }
    }
  }

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  auto t_end = clock::now();
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
#endif
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

  // TS31-B4 COW promotion: when the forward used the identity short-circuit
  // (scale=1, bias=0), the input gradient is a pure identity pass-through
  // (dX = dy). Reuse the O(1) ShareDiff zero-copy for the *input* gradient
  // instead of an O(n) memcpy. Note: d_scale and d_bias are NOT zero in
  // general (they are sums over broadcast dims), so they must still be
  // computed below.
  const bool inplace = (bottom[0] == top[0]);
  const bool identity_dx = cow_identity_ && need_dx && !inplace;
  if (identity_dx) {
    bottom[0]->ShareDiff(top[0]);
    CAFFE_FFI_LOG_INFO() << "[SCALE-COW] " << this->name()
                         << " Scale backward: IDENTITY -> COW zero-copy diff (dx=dy)"
                         << " count=" << count
                         << " shared_ptr=" << static_cast<const void*>(bottom[0]->cpu_diff())
                         << " top_ptr=" << static_cast<const void*>(top[0]->cpu_diff());
  }

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();
#endif

  // When identity_dx, bottom's diff is already shared with top's diff
  // (zero-copy), so we must NOT write to it via cpu_mutable_diff() (which
  // would trigger a COW unshare). d_scale/d_bias are still accumulated.
  float* bottom_diff = (need_dx && !identity_dx) ? bottom[0]->cpu_mutable_diff() : nullptr;
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

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  // Value-range tracking for diagnostics
  float dx_min = std::numeric_limits<float>::max();
  float dx_max = -std::numeric_limits<float>::max();
  float ds_min = std::numeric_limits<float>::max();
  float ds_max = -std::numeric_limits<float>::max();
  float db_min = std::numeric_limits<float>::max();
  float db_max = -std::numeric_limits<float>::max();
#endif

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

        // dX = dy * s. When identity_dx, the gradient is already shared via
        // ShareDiff (zero-copy), so no write is needed here.
        if (need_dx && !identity_dx) {
          float dx_val = dy_val * factor;
          bottom_diff[idx] = dx_val;
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
          dx_min = std::min(dx_min, dx_val);
          dx_max = std::max(dx_max, dx_val);
#endif
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

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
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

  auto t_end = clock::now();
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
#endif
}

REGISTER_LAYER_CLASS(Scale);

}  // namespace caffe_ffi
