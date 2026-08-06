#include "caffe_ffi/layers/conv_layer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

#ifdef CAFFE_USE_OPENMP
#include <omp.h>
#endif

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void ConvolutionLayer::compute_output_shape() {
  output_h_ = (height_ + 2 * pad_h_ - dilation_h_ * (kernel_h_ - 1) - 1) / stride_h_ + 1;
  output_w_ = (width_ + 2 * pad_w_ - dilation_w_ * (kernel_w_ - 1) - 1) / stride_w_ + 1;
}

void ConvolutionLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                    const std::vector<Blob*>& top) {
  const float* weight = this->blobs_[0]->cpu_data();
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int M = conv_out_channels_ / group_;
  const int K = kernel_dim_;

  CAFFE_FFI_LAYER_LOG << "Convolution Forward: num=" << num_
                      << " group=" << group_
                      << " M=" << M
                      << " N=" << conv_out_spatial_dim_
                      << " K=" << K
                      << " is_1x1=" << is_1x1_
                      << " bias_term=" << bias_term_;

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  using clock = std::chrono::high_resolution_clock;
  auto t_total_start = clock::now();

  const int64_t top_count = top[0]->count();
  const int64_t weight_count = this->blobs_[0]->count();
  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();
  float w_min = std::numeric_limits<float>::max();
  float w_max = -std::numeric_limits<float>::max();
  float b_min = std::numeric_limits<float>::max();
  float b_max = -std::numeric_limits<float>::max();
  double w_norm_sq = 0.0;
#endif

#ifdef CAFFE_USE_OPENMP
  // ── OpenMP parallel path v4: output-channel (M) parallelism ──
  // Design rationale:
  //   - Serial path when OMP=1: BLAS can use its own threading without interference.
  //   - Multi-threaded path: split output channels (M dimension of GEMM) across threads.
  //   - Each thread gets exactly one contiguous channel range: GEMM → bias fused,
  //     eliminating inter-phase barriers.
  //   - Minimum chunk = 8 channels to keep OpenBLAS SGEMM efficient while maximizing
  //     parallelism for small-channel layers (e.g. ResNet50 conv1=64ch → 8 chunks).
  //   - Chunk count = min(max_threads, M/kMinChunk) to avoid creating chunks too small,
  //     which causes GEMM inefficiency and barrier skew.
  //   - im2col is single-threaded (memory-bound, ~5% of compute).
  //   - BLAS MUST be single-threaded (OPENBLAS_NUM_THREADS=1) to prevent oversubscription.
  const int M_total = conv_out_channels_;
  const int N_spat = conv_out_spatial_dim_;
  const int M_per_group = M_total / group_;
  const int K_per_group = kernel_dim_;
  const int max_omp_threads = omp_get_max_threads();

  if (max_omp_threads <= 1) {
    // ── Serial path (OMP=1): no parallel region, BLAS may multi-thread ──
    for (int n = 0; n < num_; ++n) {
      const float* input = bottom_data + static_cast<int64_t>(n) * bottom_dim_;
      float* output = top_data + static_cast<int64_t>(n) * top_dim_;

      forward_cpu_gemm(input, weight, output);
      if (bias_term_) {
        const float* bias = this->blobs_[1]->cpu_data();
        forward_cpu_bias(output, bias);
      }
    }
  } else {
    // ── Multi-threaded path ──
    const int kMinChunk = 8;
    // Number of chunks: no more than M_total/kMinChunk, no more than max_omp_threads.
    // This ensures each chunk is ≥ kMinChunk channels and each thread gets ≤1 chunk.
    const int max_chunks_from_channels = std::max(1, M_total / kMinChunk);
    const int num_chunks = std::min(max_omp_threads, max_chunks_from_channels);
    const int chunk_size = (M_total + num_chunks - 1) / num_chunks;

    #pragma omp parallel num_threads(num_chunks)
    {
      for (int n = 0; n < num_; ++n) {
        const float* input = bottom_data + static_cast<int64_t>(n) * bottom_dim_;
        float* output = top_data + static_cast<int64_t>(n) * top_dim_;

        // im2col: single thread executes; implicit barrier ensures all threads
        // see col_buffer_ writes before GEMM reads it.
        if (!is_1x1_) {
          #pragma omp single
          {
            im2col_cpu(input, conv_in_channels_, conv_input_h(), conv_input_w(),
                       kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
                       dilation_h_, dilation_w_, col_buffer_.cpu_mutable_data());
          }
        }

        const float* col_buff = is_1x1_ ? input : col_buffer_.cpu_data();

        // Each thread processes one channel chunk: GEMM + bias fused (no barrier between)
        #pragma omp for schedule(static)
        for (int mc = 0; mc < num_chunks; ++mc) {
          const int m_start = mc * chunk_size;
          const int m_end = std::min(m_start + chunk_size, M_total);

          // GEMM for this chunk
          for (int g = 0; g < group_; ++g) {
            const int g_m_start = g * M_per_group;
            const int g_m_end = (g + 1) * M_per_group;
            const int lo = std::max(m_start, g_m_start);
            const int hi = std::min(m_end, g_m_end);
            if (lo >= hi) continue;
            const int loc_m = lo - g_m_start;
            const int loc_cnt = hi - lo;

            caffe_cpu_gemm(false, false, loc_cnt, N_spat, K_per_group, 1.F,
                           weight + static_cast<int64_t>(weight_offset_) * g
                                  + static_cast<int64_t>(loc_m) * K_per_group,
                           col_buff + static_cast<int64_t>(col_offset_) * g,
                           0.F,
                           output + static_cast<int64_t>(output_offset_) * g
                                  + static_cast<int64_t>(loc_m) * N_spat);
          }

          // Bias for this chunk (fused: no barrier between GEMM and bias)
          if (bias_term_) {
            const int m_count = m_end - m_start;
            const float* bias = this->blobs_[1]->cpu_data();
            caffe_cpu_gemm(false, false, m_count, out_spatial_dim_, 1, 1.F,
                           bias + m_start, bias_multiplier_.cpu_data(), 1.F,
                           output + static_cast<int64_t>(m_start) * out_spatial_dim_);
          }
        }
        // implicit barrier after omp for: all threads sync before next sample
      }
    }
  }
#else
  // ── Serial fallback (OpenMP disabled) ──
  for (int n = 0; n < num_; ++n) {
    const float* input = bottom_data + static_cast<int64_t>(n) * bottom_dim_;
    float* output = top_data + static_cast<int64_t>(n) * top_dim_;
    forward_cpu_gemm(input, weight, output);
    if (bias_term_) {
      const float* bias = this->blobs_[1]->cpu_data();
      forward_cpu_bias(output, bias);
    }
  }
#endif

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  // ── Output/weight statistics (serial; only when perf logging enabled) ──
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

  CAFFE_FFI_LOG_INFO() << "[CONV-PERF] " << this->name()
                       << " Convolution forward: num=" << num_
                       << " group=" << group_
                       << " M=" << M << " N=" << conv_out_spatial_dim_ << " K=" << K
                       << " kernel=[" << kernel_h_ << "," << kernel_w_ << "]"
                       << " stride=[" << stride_h_ << "," << stride_w_ << "]"
                       << " is_1x1=" << is_1x1_
                       << " bias_term=" << bias_term_
                       << " out=[" << out_min << ", " << out_max << "]"
                       << " w=[" << w_min << ", " << w_max << "]"
                       << " w_norm=" << w_norm
                       << (bias_term_ ? " b=[" + std::to_string(b_min) + ", " + std::to_string(b_max) + "]" : "")
                       << " time=" << total_us << "us";
#endif
}

void ConvolutionLayer::Backward_cpu(const std::vector<Blob*>& top,
                                     const std::vector<bool>& propagate_down,
                                     const std::vector<Blob*>& bottom) {
  const float* weight = this->blobs_[0]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  const float* bottom_data = bottom[0]->cpu_data();
  float* weight_diff = this->param_propagate_down_[0] ? this->blobs_[0]->cpu_mutable_diff() : nullptr;
  float* bottom_diff = propagate_down[0] ? bottom[0]->cpu_mutable_diff() : nullptr;

  const int M = conv_out_channels_ / group_;
  const int N = conv_out_spatial_dim_;
  const int K = kernel_dim_;

  CAFFE_FFI_LAYER_LOG << "Convolution Backward: num=" << num_
                      << " group=" << group_
                      << " M=" << M << " N=" << N << " K=" << K
                      << " is_1x1=" << is_1x1_
                      << " bias_term=" << bias_term_
                      << " prop_down=" << (propagate_down[0] ? "true" : "false");

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  using clock = std::chrono::high_resolution_clock;
  auto t_total_start = clock::now();

  const int64_t weight_count = this->blobs_[0]->count();
  const int64_t bottom_count = bottom[0]->count();

  double t_zero_us = 0;
  {
    auto t0 = clock::now();
    if (this->param_propagate_down_[0]) {
      caffe_set_fp32(static_cast<size_t>(weight_count), 0.0f, weight_diff);
    }
    if (bias_term_ && this->param_propagate_down_[1]) {
      caffe_set_fp32(static_cast<size_t>(this->blobs_[1]->count()), 0.0f,
                     this->blobs_[1]->cpu_mutable_diff());
    }
    t_zero_us = std::chrono::duration<double, std::micro>(clock::now() - t0).count();
  }

  float top_diff_min = std::numeric_limits<float>::max();
  float top_diff_max = -std::numeric_limits<float>::max();
  float bottom_diff_min = std::numeric_limits<float>::max();
  float bottom_diff_max = -std::numeric_limits<float>::max();
  float w_diff_min = std::numeric_limits<float>::max();
  float w_diff_max = -std::numeric_limits<float>::max();
  float b_diff_min = std::numeric_limits<float>::max();
  float b_diff_max = -std::numeric_limits<float>::max();

  double t_gemm_filter_us = 0, t_gemm_data_us = 0, t_gemm_bias_us = 0;
#else
  // Zero gradients without timing
  if (this->param_propagate_down_[0]) {
    caffe_set_fp32(static_cast<size_t>(this->blobs_[0]->count()), 0.0f, weight_diff);
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    caffe_set_fp32(static_cast<size_t>(this->blobs_[1]->count()), 0.0f,
                   this->blobs_[1]->cpu_mutable_diff());
  }
#endif

  for (int n = 0; n < num_; ++n) {
    const float* input = bottom_data + static_cast<int64_t>(n) * bottom_dim_;
    const float* output = top_diff + static_cast<int64_t>(n) * top_dim_;
    float* out_diff = propagate_down[0] ? bottom_diff + static_cast<int64_t>(n) * bottom_dim_ : nullptr;

    if (this->param_propagate_down_[0]) {
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
      auto tgf = clock::now();
#endif
      weight_cpu_gemm(input, output, weight_diff);
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
      t_gemm_filter_us += std::chrono::duration<double, std::micro>(clock::now() - tgf).count();
#endif
    }

    if (propagate_down[0]) {
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
      auto tgd = clock::now();
#endif
      backward_cpu_gemm(output, weight, out_diff);
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
      t_gemm_data_us += std::chrono::duration<double, std::micro>(clock::now() - tgd).count();
#endif
    }

    if (bias_term_ && this->param_propagate_down_[1]) {
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
      auto tgb = clock::now();
#endif
      backward_cpu_bias(this->blobs_[1]->cpu_mutable_diff(), output);
#ifdef CAFFE_FFI_ENABLE_PERF_LOG
      t_gemm_bias_us += std::chrono::duration<double, std::micro>(clock::now() - tgb).count();
#endif
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

  CAFFE_FFI_LOG_INFO() << "[CONV-PERF] " << this->name()
                       << " Convolution backward: num=" << num_
                       << " group=" << group_
                       << " M=" << M << " N=" << N << " K=" << K
                       << " kernel=[" << kernel_h_ << "," << kernel_w_ << "]"
                       << " stride=[" << stride_h_ << "," << stride_w_ << "]"
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

REGISTER_LAYER_CLASS(Convolution);

}  // namespace caffe_ffi
