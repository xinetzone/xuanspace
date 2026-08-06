#include "caffe_ffi/layers/l2_norm_layer.hpp"

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

void L2NormLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                             const std::vector<Blob*>& top) {
  const caffe::L2NormParameter& param = this->layer_param_.l2_norm_param();
  axis_ = bottom[0]->CanonicalAxisIndex(param.axis());
  eps_ = param.eps();

  CAFFE_FFI_LAYER_LOG << "L2Norm LayerSetUp: axis_=" << axis_ << " eps_=" << eps_;
}

void L2NormLayer::Reshape(const std::vector<Blob*>& bottom,
                          const std::vector<Blob*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  outer_dim_ = static_cast<int>(bottom[0]->count(0, axis_));
  inner_dim_ = static_cast<int>(bottom[0]->count(axis_));

  std::ostringstream input_shape_ss;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) input_shape_ss << ", ";
    input_shape_ss << bottom[0]->shape(i);
  }

  CAFFE_FFI_LAYER_LOG << "L2Norm Reshape: input=[" << input_shape_ss.str()
                      << "] outer_dim_=" << outer_dim_
                      << " inner_dim_=" << inner_dim_
                      << " axis_=" << axis_
                      << " eps_=" << eps_;
}

void L2NormLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                              const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int outer_dim = outer_dim_;
  const int inner_dim = inner_dim_;

  CAFFE_FFI_LAYER_LOG << "L2Norm Forward: outer_dim=" << outer_dim
                      << " inner_dim=" << inner_dim;

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  float in_min = std::numeric_limits<float>::max();
  float in_max = -std::numeric_limits<float>::max();
  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();
  float norm_min = std::numeric_limits<float>::max();
  float norm_max = -std::numeric_limits<float>::max();
#endif

  for (int o = 0; o < outer_dim; ++o) {
    const float* x = bottom_data + o * inner_dim;
    float* y = top_data + o * inner_dim;

    // Compute the L2 norm over the group: sqrt(sum x^2 + eps).
    double sum_sq = 0.0;
    for (int i = 0; i < inner_dim; ++i) {
      sum_sq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }
    float norm = std::sqrt(static_cast<float>(sum_sq) + eps_);
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
    norm_min = std::min(norm_min, norm);
    norm_max = std::max(norm_max, norm);
#endif

    const float inv_norm = 1.0f / norm;
    for (int i = 0; i < inner_dim; ++i) {
      float v = x[i] * inv_norm;
      y[i] = v;
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
      in_min = std::min(in_min, x[i]);
      in_max = std::max(in_max, x[i]);
      out_min = std::min(out_min, v);
      out_max = std::max(out_max, v);
#endif
    }
  }

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[L2NORM-PERF] " << this->name()
                       << " L2Norm forward: outer_dim=" << outer_dim
                       << " inner_dim=" << inner_dim
                       << " eps=" << eps_
                       << " in=[" << in_min << ", " << in_max << "]"
                       << " out=[" << out_min << ", " << out_max << "]"
                       << " norm=[" << norm_min << ", " << norm_max << "]"
                       << " time=" << elapsed_us << "us";
#endif
}

void L2NormLayer::Backward_cpu(const std::vector<Blob*>& top,
                               const std::vector<bool>& propagate_down,
                               const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    CAFFE_FFI_LAYER_LOG << "L2Norm Backward_cpu: propagate_down[0]=false, skipping";
    return;
  }

  const float* bottom_data = bottom[0]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const int outer_dim = outer_dim_;
  const int inner_dim = inner_dim_;

  CAFFE_FFI_LAYER_LOG << "L2Norm Backward: outer_dim=" << outer_dim
                      << " inner_dim=" << inner_dim;

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  float diff_in_min = std::numeric_limits<float>::max();
  float diff_in_max = -std::numeric_limits<float>::max();
  float diff_out_min = std::numeric_limits<float>::max();
  float diff_out_max = -std::numeric_limits<float>::max();
#endif

  for (int o = 0; o < outer_dim; ++o) {
    const float* x = bottom_data + o * inner_dim;
    const float* dy = top_diff + o * inner_dim;
    float* dx = bottom_diff + o * inner_dim;

    // Compute norm and the dot product sum(dy * x).
    double sum_sq = 0.0;
    double dot = 0.0;
    for (int i = 0; i < inner_dim; ++i) {
      sum_sq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
      dot += static_cast<double>(dy[i]) * static_cast<double>(x[i]);
    }
    float norm = std::sqrt(static_cast<float>(sum_sq) + eps_);
    const float inv_norm = 1.0f / norm;
    const float coef = static_cast<float>(dot) / (norm * norm * norm);

    for (int i = 0; i < inner_dim; ++i) {
      float v = dy[i] * inv_norm - x[i] * coef;
      dx[i] = v;
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
      diff_in_min = std::min(diff_in_min, dy[i]);
      diff_in_max = std::max(diff_in_max, dy[i]);
      diff_out_min = std::min(diff_out_min, v);
      diff_out_max = std::max(diff_out_max, v);
#endif
    }
  }

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[L2NORM-PERF] " << this->name()
                       << " L2Norm backward: outer_dim=" << outer_dim
                       << " inner_dim=" << inner_dim
                       << " diff_in=[" << diff_in_min << ", " << diff_in_max << "]"
                       << " diff_out=[" << diff_out_min << ", " << diff_out_max << "]"
                       << " time=" << elapsed_us << "us";
#endif
}

REGISTER_LAYER_CLASS(L2Norm);

}  // namespace caffe_ffi