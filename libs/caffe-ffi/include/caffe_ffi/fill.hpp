#ifndef CAFFE_FFI_FILL_HPP_
#define CAFFE_FFI_FILL_HPP_

#include <cmath>
#include <cstddef>
#include <cstring>

#include "caffe_ffi/math_utils.hpp"

namespace caffe_ffi {

inline void caffe_axpy_fp32(const size_t N, const float alpha, const float* X, float* Y);
inline void caffe_scal_fp32(const size_t N, const float alpha, float* X);

inline void caffe_set_fp32(const size_t N, const float alpha, float* Y) {
  if (alpha == 0.0f) {
    std::memset(Y, 0, sizeof(float) * N);
    return;
  }
  for (size_t i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

inline void caffe_copy_fp32(const size_t N, const float* X, float* Y) {
  if (X != Y) {
#ifdef CAFFE_USE_BLAS
    ::cblas_scopy(static_cast<int>(N), X, 1, Y, 1);
#else
    std::memcpy(Y, X, sizeof(float) * N);
#endif
  }
}

inline void caffe_axpy_fp32(const size_t N, const float alpha, const float* X, float* Y) {
#ifdef CAFFE_USE_BLAS
  ::cblas_saxpy(static_cast<int>(N), alpha, X, 1, Y, 1);
#else
  for (size_t i = 0; i < N; ++i) {
    Y[i] += alpha * X[i];
  }
#endif
}

inline void caffe_cpu_axpby_fp32(const size_t N, const float alpha, const float* X,
                                  const float beta, float* Y) {
  if (beta == 0.0f) {
    caffe_set_fp32(N, 0.0f, Y);
  } else if (beta != 1.0f) {
    caffe_scal_fp32(N, beta, Y);
  }
  if (alpha != 0.0f) {
    caffe_axpy_fp32(N, alpha, X, Y);
  }
}

inline void caffe_scal_fp32(const size_t N, const float alpha, float* X) {
  if (alpha == 1.0f) return;
#ifdef CAFFE_USE_BLAS
  ::cblas_sscal(static_cast<int>(N), alpha, X, 1);
#else
  for (size_t i = 0; i < N; ++i) {
    X[i] *= alpha;
  }
#endif
}

inline float caffe_cpu_dot_fp32(const size_t N, const float* X, const float* Y) {
  return caffe_cpu_strided_dot_fp32(static_cast<int>(N), X, 1, Y, 1);
}

inline float caffe_cpu_asum_fp32(const size_t N, const float* X) {
#ifdef CAFFE_USE_BLAS
  return ::cblas_sasum(static_cast<int>(N), X, 1);
#else
  float sum = 0.0f;
  for (size_t i = 0; i < N; ++i) {
    sum += std::fabs(X[i]);
  }
  return sum;
#endif
}

inline void caffe_add_fp32(const size_t N, const float* a, const float* b, float* y) {
  for (size_t i = 0; i < N; ++i) {
    y[i] = a[i] + b[i];
  }
}

inline void caffe_sub_fp32(const size_t N, const float* a, const float* b, float* y) {
  for (size_t i = 0; i < N; ++i) {
    y[i] = a[i] - b[i];
  }
}

inline void caffe_mul_fp32(const size_t N, const float* a, const float* b, float* y) {
  for (size_t i = 0; i < N; ++i) {
    y[i] = a[i] * b[i];
  }
}

inline void caffe_div_fp32(const size_t N, const float* a, const float* b, float* y) {
  for (size_t i = 0; i < N; ++i) {
    y[i] = a[i] / b[i];
  }
}

inline void caffe_powx_fp32(const size_t N, const float* a, const float b, float* y) {
  for (size_t i = 0; i < N; ++i) {
    y[i] = std::pow(a[i], b);
  }
}

inline void caffe_exp_fp32(const size_t N, const float* a, float* y) {
  for (size_t i = 0; i < N; ++i) {
    y[i] = std::exp(a[i]);
  }
}

inline void caffe_sqrt_fp32(const size_t N, const float* a, float* y) {
  for (size_t i = 0; i < N; ++i) {
    y[i] = std::sqrt(a[i]);
  }
}

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_FILL_HPP_
