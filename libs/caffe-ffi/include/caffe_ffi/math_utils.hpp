#ifndef CAFFE_FFI_MATH_UTILS_HPP_
#define CAFFE_FFI_MATH_UTILS_HPP_

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include <tvm/ffi/error.h>
#include <tvm/ffi/container/shape.h>

#if defined(USE_CBLAS) || defined(CAFFE_USE_BLAS)
  #if defined(__APPLE__)
    #include <Accelerate/Accelerate.h>
  #else
    extern "C" {
      #if defined(HAVE_CBLAS_H)
        #include <cblas.h>
      #elif defined(HAVE_OPENBLAS_CBLAS_H)
        #include <openblas/cblas.h>
      #else
        #include <cblas.h>
      #endif
    }
  #endif
  #ifndef CAFFE_USE_BLAS
    #define CAFFE_USE_BLAS
  #endif
#else
enum CBLAS_ORDER { CblasRowMajor=101, CblasColMajor=102 };
enum CBLAS_TRANSPOSE { CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113 };
#endif

namespace caffe_ffi {

using tvm::ffi::Shape;
using tvm::ffi::ShapeView;

inline int CanonicalAxisIndex(int axis_index, int num_axes) {
  TVM_FFI_ICHECK_GE(axis_index, -num_axes)
      << "axis " << axis_index << " out of range for " << num_axes << "-D blob";
  TVM_FFI_ICHECK_LT(axis_index, num_axes)
      << "axis " << axis_index << " out of range for " << num_axes << "-D blob";
  if (axis_index < 0) {
    return axis_index + num_axes;
  }
  return axis_index;
}

inline int64_t Count(ShapeView shape, int start_axis, int end_axis) {
  int n = static_cast<int>(shape.size());
  if (start_axis < 0) {
    start_axis = CanonicalAxisIndex(start_axis, n);
  } else if (start_axis > n) {
    TVM_FFI_ICHECK_LT(start_axis, n + 1)
        << "axis " << start_axis << " out of range for " << n << "-D blob";
  }
  end_axis = CanonicalAxisIndex(end_axis, n + 1);
  if (start_axis >= end_axis) return 1;
  int64_t count = 1;
  for (int i = start_axis; i < end_axis; ++i) {
    count *= shape[i];
  }
  return count;
}

inline int64_t Count(ShapeView shape, int start_axis) {
  int n = static_cast<int>(shape.size());
  if (start_axis < 0) {
    start_axis = CanonicalAxisIndex(start_axis, n);
  } else if (start_axis > n) {
    TVM_FFI_ICHECK_LT(start_axis, n + 1)
        << "axis " << start_axis << " out of range for " << n << "-D blob";
  }
  return Count(shape, start_axis, n);
}

inline int64_t Count(ShapeView shape) {
  int64_t count = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    count *= shape[i];
  }
  return count;
}

inline void im2col_fp32(const float* data_im, const int channels,
                const int height, const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w, const int stride_h,
                const int stride_w, const int dilation_h, const int dilation_w,
                float* data_col);

inline void col2im_fp32(const float* data_col, const int channels,
                const int height, const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w, const int stride_h,
                const int stride_w, const int dilation_h, const int dilation_w,
                float* data_im);

inline void caffe_cpu_gemm_fp32(const bool TransA, const bool TransB,
                                 const int M, const int N, const int K,
                                 const float alpha, const float* A, const float* B,
                                 const float beta, float* C) {
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

inline void caffe_cpu_gemv_fp32(const bool TransA,
                                 const int M, const int N,
                                 const float alpha, const float* A, const float* x,
                                 const float beta, float* y) {
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

inline float caffe_cpu_strided_dot_fp32(const int n, const float* x, const int incx,
                                        const float* y, const int incy) {
#ifdef CAFFE_USE_BLAS
  return ::cblas_sdot(n, x, incx, y, incy);
#else
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    sum += x[i * incx] * y[i * incy];
  }
  return sum;
#endif
}

inline void im2col_fp32(const float* data_im, const int channels,
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
          if (!((input_row < 0) || (input_row >= height))) {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (!((input_col < 0) || (input_col >= width))) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          } else {
            for (int output_col = output_w; output_col; output_col--) {
              *(data_col++) = 0;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

inline void col2im_fp32(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im) {
  std::memset(data_im, 0, sizeof(float) * height * width * channels);
  const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!((input_row < 0) || (input_row >= height))) {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (!((input_col < 0) || (input_col >= width))) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          } else {
            data_col += output_w;
          }
          input_row += stride_h;
        }
      }
    }
  }
}

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_MATH_UTILS_HPP_
