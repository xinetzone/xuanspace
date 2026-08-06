#include "caffe_ffi/layers/deconv_layer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void DeconvolutionLayer::compute_output_shape() {
  const int kernel_extent_h = dilation_h_ * (kernel_h_ - 1) + 1;
  const int kernel_extent_w = dilation_w_ * (kernel_w_ - 1) + 1;
  output_h_ = stride_h_ * (height_ - 1) + kernel_extent_h - 2 * pad_h_;
  output_w_ = stride_w_ * (width_ - 1) + kernel_extent_w - 2 * pad_w_;
}

void DeconvolutionLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                      const std::vector<Blob*>& top) {
  const float* weight = this->blobs_[0]->cpu_data();
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  using clock = std::chrono::high_resolution_clock;
  auto t_total_start = clock::now();

  const int M = conv_out_channels_ / group_;
  const int K = kernel_dim_;
  const int64_t top_count = top[0]->count();
  const int64_t weight_count = this->blobs_[0]->count();

  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();
  float w_min = std::numeric_limits<float>::max();
  float w_max = -std::numeric_limits<float>::max();
  float b_min = std::numeric_limits<float>::max();
  float b_max = -std::numeric_limits<float>::max();
  double w_norm_sq = 0.0;

  double t_gemm_us = 0, t_bias_us = 0;
#endif

  // Deconv forward = Conv backward (transposed GEMM + col2im)
  for (int n = 0; n < num_; ++n) {
    const float* input = bottom_data + n * bottom_dim_;
    float* output = top_data + n * top_dim_;

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
    auto t_gemm_start = clock::now();
#endif
    backward_cpu_gemm(input, weight, output);
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
    auto t_gemm_end = clock::now();
    t_gemm_us += std::chrono::duration<double, std::micro>(t_gemm_end - t_gemm_start).count();
#endif

    if (bias_term_) {
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
      auto t_bias_start = clock::now();
#endif
      const float* bias = this->blobs_[1]->cpu_data();
      forward_cpu_bias(output, bias);
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
      auto t_bias_end = clock::now();
      t_bias_us += std::chrono::duration<double, std::micro>(t_bias_end - t_bias_start).count();
#endif
    }
  }

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  for (int64_t i = 0; i < top_count; ++i) {
    out_min = std::min(out_min, top_data[i]);
    out_max = std::max(out_max, top_data[i]);
  }
  for (int64_t i = 0; i < weight_count; ++i) {
    float w = weight[i];
    w_min = std::min(w_min, w);
    w_max = std::max(w_max, w);
    w_norm_sq += static_cast<double>(w) * static_cast<double>(w);
  }
  float w_norm = static_cast<float>(std::sqrt(w_norm_sq));
  if (bias_term_) {
    int64_t bias_count = this->blobs_[1]->count();
    const float* bias_data = this->blobs_[1]->cpu_data();
    for (int64_t i = 0; i < bias_count; ++i) {
      b_min = std::min(b_min, bias_data[i]);
      b_max = std::max(b_max, bias_data[i]);
    }
  }

  auto t_total_end = clock::now();
  double total_us = std::chrono::duration<double, std::micro>(t_total_end - t_total_start).count();

  CAFFE_FFI_LOG_INFO() << "[DECONV-PERF] " << this->name()
                       << " Deconvolution forward: num=" << num_
                       << " group=" << group_
                       << " M=" << M << " N=" << conv_out_spatial_dim_ << " K=" << K
                       << " kernel=[" << kernel_h_ << "," << kernel_w_ << "]"
                       << " stride=[" << stride_h_ << "," << stride_w_ << "]"
                       << " pad=[" << pad_h_ << "," << pad_w_ << "]"
                       << " is_1x1=" << is_1x1_
                       << " bias_term=" << bias_term_
                       << " out=[" << out_min << ", " << out_max << "]"
                       << " w=[" << w_min << ", " << w_max << "]"
                       << " w_norm=" << w_norm
                       << (bias_term_ ? " b=[" + std::to_string(b_min) + ", " + std::to_string(b_max) + "]" : "")
                       << " t_gemm=" << t_gemm_us << "us"
                       << (bias_term_ ? " t_bias=" + std::to_string(t_bias_us) + "us" : "")
                       << " time=" << total_us << "us";
#endif
}

void DeconvolutionLayer::Backward_cpu(const std::vector<Blob*>& top,
                                       const std::vector<bool>& propagate_down,
                                       const std::vector<Blob*>& bottom) {
  const float* weight = this->blobs_[0]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  const float* bottom_data = bottom[0]->cpu_data();
  float* weight_diff = this->param_propagate_down_[0] ? this->blobs_[0]->cpu_mutable_diff() : nullptr;
  float* bottom_diff = propagate_down[0] ? bottom[0]->cpu_mutable_diff() : nullptr;

  const int64_t weight_count = this->blobs_[0]->count();

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  using clock = std::chrono::high_resolution_clock;
  auto t_total_start = clock::now();

  const int M = conv_out_channels_ / group_;
  const int N = conv_out_spatial_dim_;
  const int K = kernel_dim_;
  const int64_t bottom_count = bottom[0]->count();

  double t_zero_us = 0;
  {
    auto t0 = clock::now();
#endif
    if (this->param_propagate_down_[0]) {
      caffe_set_fp32(static_cast<size_t>(weight_count), 0.0f, weight_diff);
    }
    if (bias_term_ && this->param_propagate_down_[1]) {
      caffe_set_fp32(static_cast<size_t>(this->blobs_[1]->count()), 0.0f,
                     this->blobs_[1]->cpu_mutable_diff());
    }
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
    t_zero_us = std::chrono::duration<double, std::micro>(clock::now() - t0).count();
  }
#endif

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  float top_diff_min = std::numeric_limits<float>::max();
  float top_diff_max = -std::numeric_limits<float>::max();
  float bottom_diff_min = std::numeric_limits<float>::max();
  float bottom_diff_max = -std::numeric_limits<float>::max();
  float w_diff_min = std::numeric_limits<float>::max();
  float w_diff_max = -std::numeric_limits<float>::max();
  float b_diff_min = std::numeric_limits<float>::max();
  float b_diff_max = -std::numeric_limits<float>::max();

  double t_gemm_filter_us = 0, t_gemm_data_us = 0, t_gemm_bias_us = 0;
#endif

  bool skip_im2col = false;

  // Deconv backward:
  for (int n = 0; n < num_; ++n) {
    const float* out_data = top_diff + n * top_dim_;
    const float* in_data = bottom_data + n * bottom_dim_;
    float* out_diff = propagate_down[0] ? bottom_diff + n * bottom_dim_ : nullptr;

    // Bias gradient
    if (bias_term_ && this->param_propagate_down_[1]) {
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
      auto tgb = clock::now();
#endif
      backward_cpu_bias(this->blobs_[1]->cpu_mutable_diff(), out_data);
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
      t_gemm_bias_us += std::chrono::duration<double, std::micro>(clock::now() - tgb).count();
#endif
    }

    if (this->param_propagate_down_[0] || propagate_down[0]) {
      // Weight gradient
      if (this->param_propagate_down_[0]) {
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
        auto tgf = clock::now();
#endif
        weight_cpu_gemm(out_data, in_data, weight_diff);
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
        t_gemm_filter_us += std::chrono::duration<double, std::micro>(clock::now() - tgf).count();
#endif
        skip_im2col = true;
      }

      // Bottom gradient
      if (propagate_down[0]) {
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
        auto tgd = clock::now();
#endif
        forward_cpu_gemm(out_data, weight, out_diff,
                         this->param_propagate_down_[0] ? skip_im2col : false);
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
        t_gemm_data_us += std::chrono::duration<double, std::micro>(clock::now() - tgd).count();
#endif
      }
    }
  }

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  if (propagate_down[0]) {
    for (int64_t i = 0; i < bottom_count; ++i) {
      bottom_diff_min = std::min(bottom_diff_min, bottom_diff[i]);
      bottom_diff_max = std::max(bottom_diff_max, bottom_diff[i]);
    }
  }
  double w_diff_norm_sq = 0.0;
  float w_diff_norm = 0.0f;
  if (this->param_propagate_down_[0]) {
    for (int64_t i = 0; i < weight_count; ++i) {
      float dw = weight_diff[i];
      w_diff_min = std::min(w_diff_min, dw);
      w_diff_max = std::max(w_diff_max, dw);
      w_diff_norm_sq += static_cast<double>(dw) * static_cast<double>(dw);
    }
    w_diff_norm = static_cast<float>(std::sqrt(w_diff_norm_sq));
  }
  {
    int64_t top_count = top[0]->count();
    const float* td = top_diff;
    for (int64_t i = 0; i < top_count; ++i) {
      top_diff_min = std::min(top_diff_min, td[i]);
      top_diff_max = std::max(top_diff_max, td[i]);
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    int64_t bd_count = this->blobs_[1]->count();
    const float* bd = this->blobs_[1]->cpu_diff();
    for (int64_t i = 0; i < bd_count; ++i) {
      b_diff_min = std::min(b_diff_min, bd[i]);
      b_diff_max = std::max(b_diff_max, bd[i]);
    }
  }

  double total_us = std::chrono::duration<double, std::micro>(clock::now() - t_total_start).count();

  std::string w_diff_str;
  if (this->param_propagate_down_[0]) {
    w_diff_str = " w_diff=[" + std::to_string(w_diff_min) + ", " + std::to_string(w_diff_max) + "]"
               + " w_diff_norm=" + std::to_string(w_diff_norm);
  }
  std::string b_diff_str;
  if (bias_term_ && this->param_propagate_down_[1]) {
    b_diff_str = " b_diff=[" + std::to_string(b_diff_min) + ", " + std::to_string(b_diff_max) + "]";
  }
  std::string bottom_diff_str;
  if (propagate_down[0]) {
    bottom_diff_str = " bottom_diff=[" + std::to_string(bottom_diff_min) + ", " + std::to_string(bottom_diff_max) + "]"
                    + " t_gemm_data=" + std::to_string(t_gemm_data_us) + "us";
  }
  std::string b_bias_str;
  if (bias_term_ && this->param_propagate_down_[1]) {
    b_bias_str = " t_gemm_bias=" + std::to_string(t_gemm_bias_us) + "us";
  }

  CAFFE_FFI_LOG_INFO() << "[DECONV-PERF] " << this->name()
                       << " Deconvolution backward: num=" << num_
                       << " group=" << group_
                       << " M=" << M << " N=" << N << " K=" << K
                       << " kernel=[" << kernel_h_ << "," << kernel_w_ << "]"
                       << " stride=[" << stride_h_ << "," << stride_w_ << "]"
                       << " pad=[" << pad_h_ << "," << pad_w_ << "]"
                       << " is_1x1=" << is_1x1_
                       << " bias_term=" << bias_term_
                       << " prop_down=" << (propagate_down[0] ? "true" : "false")
                       << " prop_w=" << this->param_propagate_down_[0]
                       << " top_diff=[" << top_diff_min << ", " << top_diff_max << "]"
                       << bottom_diff_str
                       << w_diff_str
                       << b_diff_str
                       << " t_zero=" << t_zero_us << "us"
                       << " t_gemm_filter=" << t_gemm_filter_us << "us"
                       << b_bias_str
                       << " time=" << total_us << "us";
#endif
}

REGISTER_LAYER_CLASS(Deconvolution);

}  // namespace caffe_ffi
