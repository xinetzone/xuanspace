#include "caffe_ffi/layers/instance_norm_layer.hpp"

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

void InstanceNormLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                                   const std::vector<Blob*>& top) {
  const caffe::InstanceNormParameter& param = this->layer_param_.instance_norm_param();
  eps_ = param.eps();
  affine_ = param.affine();
  use_global_stats_ = param.use_global_stats();

  CAFFE_FFI_LAYER_LOG << "InstanceNorm LayerSetUp: eps_=" << eps_
                      << " affine_=" << affine_
                      << " use_global_stats_=" << use_global_stats_;

  if (affine_) {
    if (this->blobs_.size() > 0) {
      CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_.size(), 2U)
          << "Incorrect number of InstanceNorm affine blobs (expected 2).";
    } else {
      this->blobs_.resize(2);
      channels_ = static_cast<int>(bottom[0]->shape(1));
      std::vector<int64_t> sz = {channels_};
      this->blobs_[0] = make_object<Blob>(sz);
      this->blobs_[1] = make_object<Blob>(sz);
      caffe_set_fp32(static_cast<size_t>(this->blobs_[0]->count()), 1.0f,
                     this->blobs_[0]->cpu_mutable_data());
      caffe_set_fp32(static_cast<size_t>(this->blobs_[1]->count()), 0.0f,
                     this->blobs_[1]->cpu_mutable_data());
      CAFFE_FFI_TENSOR_LOG << "InstanceNorm: created gamma blob shape=[" << channels_ << "] (=1.0)";
      CAFFE_FFI_TENSOR_LOG << "InstanceNorm: created beta blob shape=[" << channels_ << "] (=0.0)";
    }
    this->param_propagate_down_.resize(this->blobs_.size(), true);
  }
}

void InstanceNormLayer::Reshape(const std::vector<Blob*>& bottom,
                                const std::vector<Blob*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  num_ = static_cast<int>(bottom[0]->shape(0));
  channels_ = static_cast<int>(bottom[0]->shape(1));
  spatial_dim_ = static_cast<int>(bottom[0]->count(2));

  std::ostringstream input_shape_ss;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) input_shape_ss << ", ";
    input_shape_ss << bottom[0]->shape(i);
  }

  CAFFE_FFI_LAYER_LOG << "InstanceNorm Reshape: input=[" << input_shape_ss.str()
                      << "] num_=" << num_
                      << " channels_=" << channels_
                      << " spatial_dim_=" << spatial_dim_
                      << " affine_=" << affine_;
}

void InstanceNormLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                    const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int num = num_;
  const int channels = channels_;
  const int spatial_dim = spatial_dim_;

  const float* gamma = affine_ ? this->blobs_[0]->cpu_data() : nullptr;
  const float* beta = affine_ ? this->blobs_[1]->cpu_data() : nullptr;

  CAFFE_FFI_LAYER_LOG << "InstanceNorm Forward: num=" << num
                      << " channels=" << channels
                      << " spatial_dim=" << spatial_dim
                      << " affine_=" << affine_;

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  const int count = num * channels * spatial_dim;

  float in_min = std::numeric_limits<float>::max();
  float in_max = -std::numeric_limits<float>::max();
  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();
  float mean_min = std::numeric_limits<float>::max();
  float mean_max = -std::numeric_limits<float>::max();
  float std_min = std::numeric_limits<float>::max();
  float std_max = -std::numeric_limits<float>::max();
#endif

  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      const float* x = bottom_data + (n * channels + c) * spatial_dim;
      float* y = top_data + (n * channels + c) * spatial_dim;

      // Compute mean and variance over the spatial plane.
      double sum = 0.0;
      for (int i = 0; i < spatial_dim; ++i) {
        sum += static_cast<double>(x[i]);
      }
      float mean = static_cast<float>(sum / static_cast<double>(spatial_dim));
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
      mean_min = std::min(mean_min, mean);
      mean_max = std::max(mean_max, mean);
#endif

      double sum_sq = 0.0;
      for (int i = 0; i < spatial_dim; ++i) {
        float d = x[i] - mean;
        sum_sq += static_cast<double>(d) * static_cast<double>(d);
      }
      float var = static_cast<float>(sum_sq / static_cast<double>(spatial_dim));
      float inv_std = 1.0f / std::sqrt(var + eps_);
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
      std_min = std::min(std_min, inv_std);
      std_max = std::max(std_max, inv_std);
#endif

      const float g = gamma ? gamma[c] : 1.0f;
      const float b = beta ? beta[c] : 0.0f;
      for (int i = 0; i < spatial_dim; ++i) {
        float v = (x[i] - mean) * inv_std * g + b;
        y[i] = v;
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
        in_min = std::min(in_min, x[i]);
        in_max = std::max(in_max, x[i]);
        out_min = std::min(out_min, v);
        out_max = std::max(out_max, v);
#endif
      }
    }
  }

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[INSTNORM-PERF] " << this->name()
                       << " InstanceNorm forward: num=" << num
                       << " channels=" << channels
                       << " spatial_dim=" << spatial_dim
                       << " affine=" << affine_
                       << " eps=" << eps_
                       << " in=[" << in_min << ", " << in_max << "]"
                       << " out=[" << out_min << ", " << out_max << "]"
                       << " mean=[" << mean_min << ", " << mean_max << "]"
                       << " inv_std=[" << std_min << ", " << std_max << "]"
                       << " count=" << count
                       << " time=" << elapsed_us << "us";
#endif
}

void InstanceNormLayer::Backward_cpu(const std::vector<Blob*>& top,
                                     const std::vector<bool>& propagate_down,
                                     const std::vector<Blob*>& bottom) {
  const float* bottom_data = bottom[0]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const int num = num_;
  const int channels = channels_;
  const int spatial_dim = spatial_dim_;

  const float* gamma = affine_ ? this->blobs_[0]->cpu_data() : nullptr;
  const bool need_dx = propagate_down[0];
  const bool need_dgamma = affine_ && this->param_propagate_down_[0];
  const bool need_dbeta = affine_ && this->param_propagate_down_[1];

  CAFFE_FFI_LAYER_LOG << "InstanceNorm Backward: num=" << num
                      << " channels=" << channels
                      << " spatial_dim=" << spatial_dim
                      << " need_dx=" << need_dx
                      << " need_dgamma=" << need_dgamma
                      << " need_dbeta=" << need_dbeta;

  if (!need_dx && !need_dgamma && !need_dbeta) {
    CAFFE_FFI_LAYER_LOG << "InstanceNorm Backward_cpu: no gradients needed, skipping";
    return;
  }

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();
#endif

  float* gamma_diff = nullptr;
  float* beta_diff = nullptr;
  if (need_dgamma) {
    gamma_diff = this->blobs_[0]->cpu_mutable_diff();
    caffe_set_fp32(static_cast<size_t>(channels), 0.0f, gamma_diff);
  }
  if (need_dbeta) {
    beta_diff = this->blobs_[1]->cpu_mutable_diff();
    caffe_set_fp32(static_cast<size_t>(channels), 0.0f, beta_diff);
  }

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  float diff_in_min = std::numeric_limits<float>::max();
  float diff_in_max = -std::numeric_limits<float>::max();
  float diff_out_min = std::numeric_limits<float>::max();
  float diff_out_max = -std::numeric_limits<float>::max();
#endif

  const float m = static_cast<float>(spatial_dim);

  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      const int nc = n * channels + c;
      const float* x = bottom_data + nc * spatial_dim;
      const float* dy = top_diff + nc * spatial_dim;
      float* dx = bottom_diff + nc * spatial_dim;

      // Recompute mean and variance.
      double sum = 0.0;
      for (int i = 0; i < spatial_dim; ++i) {
        sum += static_cast<double>(x[i]);
      }
      float mean = static_cast<float>(sum / static_cast<double>(spatial_dim));
      double sum_sq = 0.0;
      for (int i = 0; i < spatial_dim; ++i) {
        float d = x[i] - mean;
        sum_sq += static_cast<double>(d) * static_cast<double>(d);
      }
      float var = static_cast<float>(sum_sq / static_cast<double>(spatial_dim));
      float inv_std = 1.0f / std::sqrt(var + eps_);

      const float g = gamma ? gamma[c] : 1.0f;

      // Effective upstream gradient (after affine scaling).
      double sum_dy = 0.0;
      double sum_dy_x = 0.0;
      for (int i = 0; i < spatial_dim; ++i) {
        float dy_g = dy[i] * g;
        sum_dy += static_cast<double>(dy_g);
        sum_dy_x += static_cast<double>(dy_g) * static_cast<double>(x[i] - mean);
      }
      float mean_dy = static_cast<float>(sum_dy / static_cast<double>(m));
      float mean_dy_x = static_cast<float>(sum_dy_x / static_cast<double>(m));

      // Batchnorm-style backward
      const float inv_std_sq = inv_std * inv_std;
      for (int i = 0; i < spatial_dim; ++i) {
        float v = inv_std * (dy[i] * g - mean_dy - (x[i] - mean) * inv_std_sq * mean_dy_x);
        if (need_dx) {
          dx[i] = v;
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
          diff_in_min = std::min(diff_in_min, dy[i]);
          diff_in_max = std::max(diff_in_max, dy[i]);
          diff_out_min = std::min(diff_out_min, v);
          diff_out_max = std::max(diff_out_max, v);
#endif
        }
        // Accumulate dgamma / dbeta
        if (need_dgamma) {
          gamma_diff[c] += dy[i] * (x[i] - mean) * inv_std;
        }
        if (need_dbeta) {
          beta_diff[c] += dy[i];
        }
      }
    }
  }

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[INSTNORM-PERF] " << this->name()
                       << " InstanceNorm backward: num=" << num
                       << " channels=" << channels
                       << " spatial_dim=" << spatial_dim
                       << " need_dx=" << need_dx
                       << " need_dgamma=" << need_dgamma
                       << " need_dbeta=" << need_dbeta
                       << " diff_in=[" << diff_in_min << ", " << diff_in_max << "]"
                       << " diff_out=[" << diff_out_min << ", " << diff_out_max << "]"
                       << " time=" << elapsed_us << "us";
#endif
}

REGISTER_LAYER_CLASS(InstanceNorm);

}  // namespace caffe_ffi