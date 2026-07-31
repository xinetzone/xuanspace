#ifndef CAFFE_FFI_LAYERS_BASE_CONV_LAYER_HPP_
#define CAFFE_FFI_LAYERS_BASE_CONV_LAYER_HPP_

#include <cstring>
#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

inline void im2col_cpu(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_col) {
  const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

inline void col2im_cpu(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im) {
  caffe_set_fp32(static_cast<size_t>(height) * width * channels, 0.F, data_im);
  const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w;
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

inline void caffe_cpu_gemm(bool TransA, bool TransB, int M, int N, int K,
                           float alpha, const float* A, const float* B,
                           float beta, float* C) {
#ifdef CAFFE_USE_BLAS
  ::cblas_sgemm(::CblasRowMajor,
              TransA ? ::CblasTrans : ::CblasNoTrans,
              TransB ? ::CblasTrans : ::CblasNoTrans,
              M, N, K, alpha, A, TransA ? M : K, B, TransB ? K : N, beta, C, N);
#else
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        float a = TransA ? A[k * M + i] : A[i * K + k];
        float b = TransB ? B[j * K + k] : B[k * N + j];
        sum += a * b;
      }
      C[i * N + j] = alpha * sum + beta * C[i * N + j];
    }
  }
#endif
}

inline void caffe_cpu_gemv(bool TransA, int M, int N, float alpha,
                           const float* A, const float* x, float beta, float* y) {
#ifdef CAFFE_USE_BLAS
  ::cblas_sgemv(::CblasRowMajor,
              TransA ? ::CblasTrans : ::CblasNoTrans,
              M, N, alpha, A, N, x, 1, beta, y, 1);
#else
  if (!TransA) {
    for (int i = 0; i < M; ++i) {
      float sum = 0.0f;
      for (int j = 0; j < N; ++j) {
        sum += A[i * N + j] * x[j];
      }
      y[i] = alpha * sum + beta * y[i];
    }
  } else {
    for (int j = 0; j < N; ++j) {
      y[j] *= beta;
    }
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        y[j] += alpha * A[i * N + j] * x[i];
      }
    }
  }
#endif
}

/**
 * @brief Abstract base class for Convolution and Deconvolution layers,
 *        factoring out BLAS code, im2col/col2im, and weight handling.
 *
 * Design follows native Caffe: forward_cpu_gemm/backward_cpu_gemm/weight_cpu_gemm
 * are written from the Convolution perspective. DeconvolutionLayer reverses
 * dimensions via reverse_dimensions()=true and swaps forward/backward calls.
 */
class BaseConvolutionLayer : public Layer {
 public:
  explicit BaseConvolutionLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  int ExactNumBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 1; }

  static constexpr int _type_child_slots = 4;
  TVM_FFI_DECLARE_OBJECT_INFO("caffe_ffi.BaseConvolutionLayer", BaseConvolutionLayer, Layer);

 protected:
  // Convolution-perspective GEMM helpers (used by both Conv and Deconv)
  void forward_cpu_gemm(const float* input, const float* weights,
                        float* output, bool skip_im2col = false);
  void forward_cpu_bias(float* output, const float* bias);
  void backward_cpu_gemm(const float* output, const float* weights,
                         float* input);
  void weight_cpu_gemm(const float* input, const float* output, float* weights);
  void backward_cpu_bias(float* bias, const float* input);

  /// Subclasses must implement: true for Deconv, false for Conv.
  virtual bool reverse_dimensions() = 0;
  /// Compute output_h_ and output_w_ from other parameters.
  virtual void compute_output_shape() = 0;

  /// Spatial dimensions of the im2col input (bottom for Conv, top for Deconv).
  inline int conv_input_h() const { return conv_input_shape_[1]; }
  inline int conv_input_w() const { return conv_input_shape_[2]; }

  int pad_h_, pad_w_;
  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int dilation_h_, dilation_w_;
  int group_;
  int channels_;       // bottom[0] channels (C)
  int height_, width_; // bottom[0] spatial dims (H, W)
  int conv_out_channels_;
  int conv_in_channels_;
  int output_h_, output_w_; // top spatial dims (Ho, Wo)
  int num_output_;
  bool bias_term_;
  bool is_1x1_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int weight_offset_;
  int col_offset_;
  int output_offset_;
  int bottom_dim_;  // channels per sample in bottom
  int top_dim_;     // channels per sample in top
  int out_spatial_dim_;

  // im2col input shape [C, H_im, W_im]: bottom[C,H,W] for Conv, top[C,Ho,Wo] for Deconv
  std::vector<int> conv_input_shape_;
  std::vector<int> col_buffer_shape_;

  Blob col_buffer_;
  Blob bias_multiplier_;
  int num_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_BASE_CONV_LAYER_HPP_
