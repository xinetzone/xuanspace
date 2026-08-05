#include "caffe_ffi/layers/inner_product_layer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <tvm/ffi/memory.h>

#include "caffe/proto/caffe.pb.h"
#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/math_utils.hpp"

namespace caffe_ffi {

void InnerProductLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                                    const std::vector<Blob*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  K_ = static_cast<int>(bottom[0]->count(axis));
  CAFFE_FFI_LAYER_LOG << "InnerProduct LayerSetUp: num_output=" << N_
                      << " bias_term=" << bias_term_
                      << " transpose=" << transpose_
                      << " axis=" << axis << " K_=" << K_;
  if (this->blobs_.size() > 0) {
    CAFFE_FFI_LAYER_LOG << "InnerProduct: using pre-loaded weights, blobs_.size=" << this->blobs_.size();
    CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_.size(), bias_term_ ? 2U : 1U)
        << "Incorrect number of weight blobs.";
    // Legacy V1 caffemodels store InnerProduct weights as 4D blobs
    // [1, 1, N, K] or [1, 1, K, N] (num=1, channels=1, height=N/K, width=K/N).
    // Reshape to the logical 2D shape [N, K] or [K, N] instead of requiring
    // strict shape equality — total element count must match.
    const int64_t weight_count_expected = static_cast<int64_t>(N_) * static_cast<int64_t>(K_);
    CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_[0]->count(), weight_count_expected)
        << "InnerProduct '" << this->name() << "' weight blob count mismatch: "
        << "expected N*K=" << N_ << "*" << K_ << "=" << weight_count_expected
        << ", got " << this->blobs_[0]->count()
        << " (shape=" << [&]() {
             std::ostringstream oss;
             oss << "[";
             for (int i = 0; i < this->blobs_[0]->num_axes(); ++i) {
               if (i > 0) oss << ", ";
               oss << this->blobs_[0]->shape(i);
             }
             oss << "]";
             return oss.str();
           }()
        << "). The caffemodel weight tensor is incompatible with this layer's "
        << "num_output=" << N_ << " and flattened input dim K=" << K_ << ".";
    std::vector<int64_t> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0]->Reshape(weight_shape);
    if (bias_term_) {
      CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_[1]->count(), static_cast<int64_t>(N_))
          << "InnerProduct '" << this->name() << "' bias blob count mismatch: "
          << "expected N=" << N_ << ", got " << this->blobs_[1]->count();
      this->blobs_[1]->Reshape(std::vector<int64_t>{N_});
    }
    CAFFE_FFI_LAYER_LOG << "InnerProduct: reshaped pre-loaded weight to ["
                        << weight_shape[0] << ", " << weight_shape[1] << "]"
                        << (bias_term_ ? " bias to [" + std::to_string(N_) + "]" : "");
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    std::vector<int64_t> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0] = make_object<Blob>(weight_shape);
    CAFFE_FFI_LAYER_LOG << "InnerProduct: created weight blob shape=["
                        << weight_shape[0] << ", " << weight_shape[1] << "]";

    // Apply weight_filler if specified
    const auto& ip_param = this->layer_param_.inner_product_param();
    if (ip_param.has_weight_filler()) {
      const caffe::FillerParameter& filler = ip_param.weight_filler();
      const std::string filler_type = filler.type();
      float weight_value = 0.0f;
      if (filler_type == "constant") {
        weight_value = filler.value();
      } else if (filler_type == "xavier" || filler_type == "gaussian" || filler_type == "msra") {
        // For non-constant fillers, use a simple default of 1.0 for now
        // (full implementation can add proper fan-in/fan-out initialization later)
        weight_value = 1.0f;
        CAFFE_FFI_LOG_WARN() << "[IP-FILLER] filler type '" << filler_type
                             << "' not fully implemented, using constant 1.0 for weights";
      } else {
        weight_value = 0.0f;
      }
      caffe_set_fp32(static_cast<size_t>(this->blobs_[0]->count()), weight_value,
                     this->blobs_[0]->cpu_mutable_data());
      CAFFE_FFI_LAYER_LOG << "InnerProduct: applied weight_filler type='" << filler_type
                          << "' value=" << weight_value;
    } else {
      // Default: initialize weights to zero
      caffe_set_fp32(static_cast<size_t>(this->blobs_[0]->count()), 0.0f,
                     this->blobs_[0]->cpu_mutable_data());
    }

    if (bias_term_) {
      std::vector<int64_t> bias_shape = {N_};
      this->blobs_[1] = make_object<Blob>(bias_shape);
      CAFFE_FFI_LAYER_LOG << "InnerProduct: created bias blob shape=[" << N_ << "]";

      // Apply bias_filler if specified
      float bias_value = 0.0f;
      if (ip_param.has_bias_filler()) {
        const caffe::FillerParameter& filler = ip_param.bias_filler();
        const std::string filler_type = filler.type();
        if (filler_type == "constant") {
          bias_value = filler.value();
        }
        caffe_set_fp32(static_cast<size_t>(this->blobs_[1]->count()), bias_value,
                       this->blobs_[1]->cpu_mutable_data());
        CAFFE_FFI_LAYER_LOG << "InnerProduct: applied bias_filler type='" << filler_type
                            << "' value=" << bias_value;
      } else {
        caffe_set_fp32(static_cast<size_t>(this->blobs_[1]->count()), 0.0f,
                       this->blobs_[1]->cpu_mutable_data());
      }
    }
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

void InnerProductLayer::Reshape(const std::vector<Blob*>& bottom,
                                 const std::vector<Blob*>& top) {
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = static_cast<int>(bottom[0]->count(axis));

  // Helper to format shape
  auto shape_str = [](const Blob* b) -> std::string {
    std::ostringstream oss;
    oss << "(";
    for (int i = 0; i < b->num_axes(); ++i) {
      if (i > 0) oss << ", ";
      oss << b->shape(i);
    }
    oss << ")";
    return oss.str();
  };

  if (K_ != new_K) {
    CAFFE_FFI_LOG_ERROR() << "[IP-K-MISMATCH] layer='" << this->name()
                          << "' InnerProduct input dimension mismatch:"
                          << " expected K_=" << K_
                          << " (from LayerSetUp with weight shape ["
                          << this->blobs_[0]->shape(0) << ", " << this->blobs_[0]->shape(1) << "])"
                          << " but got new_K=" << new_K
                          << " from bottom[0] shape=" << shape_str(bottom[0])
                          << " (axis=" << axis << ", flattened dims from axis="
                          << axis << " to end = " << new_K << ")."
                          << " num_output(N_)=" << N_
                          << " transpose=" << (transpose_ ? "true" : "false")
                          << "\n  *** HINT: The input feature dimension (K) must match the weight matrix's inner dimension."
                          << "\n      If transpose=false: weight is [N_, K_] = [" << N_ << ", " << K_ << "],"
                          << " input last dim(s) must flatten to K_=" << K_
                          << "\n      If transpose=true: weight is [K_, N_] = [" << K_ << ", " << N_ << "],"
                          << " input last dim(s) must flatten to K_=" << K_
                          << "\n      Common cause: input shape changed (e.g. different seq_len or d_model)"
                          << " but weight was initialized for a different shape.";
  }
  CAFFE_FFI_CHECK_VALUE_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters (layer '"
      << this->name() << "'). See [IP-K-MISMATCH] above.";
  M_ = static_cast<int>(bottom[0]->count(0, axis));

  // In-place 安全守卫：InnerProduct 输出尺寸(N_)与输入尺寸(K_)通常不同，
  // 若 top 与 bottom 共享同一 Blob（in-place），top[0]->Reshape(top_shape) 会
  // 改变共享缓冲区大小，导致 Forward 时按旧尺寸（M*K）读取数据越界
  // （heap-buffer-overflow）。当缓冲区被截断（N_ < K_）时越界读最易触发。
  if (bottom[0] == top[0]) {
    const int64_t bottom_count = bottom[0]->count();
    const int64_t top_count = static_cast<int64_t>(M_) * static_cast<int64_t>(N_);
    if (top_count != bottom_count) {
      CAFFE_FFI_CHECK_VALUE_EQ(top_count, bottom_count)
          << "InnerProduct in-place operation requires input and output to have "
          << "the same total count (M*N == M*K), but got bottom_count=" << bottom_count
          << " (M*K=" << M_ << "*" << K_ << ") vs top_count=" << top_count
          << " (M*N=" << M_ << "*" << N_ << "). In-place InnerProduct with "
          << "num_output != input feature dim is unsupported (would corrupt the "
          << "shared buffer and cause out-of-bounds access).";
    }
  }

  // BVLC Caffe semantics: output shape keeps the leading dims before `axis`
  // plus N_ (num_output), i.e. exactly `axis + 1` dimensions. Dimensions after
  // `axis` are flattened into N_ and must NOT be preserved as trailing singletons.
  std::vector<int64_t> top_shape;
  for (int i = 0; i < axis; ++i) {
    top_shape.push_back(bottom[0]->shape(i));
  }
  top_shape.push_back(N_);
  CAFFE_FFI_LAYER_LOG << "InnerProduct Reshape: M_=" << M_ << " K_=" << K_
                      << " N_=" << N_ << " top_shape=[";
  for (size_t i = 0; i < top_shape.size(); ++i) {
    if (i > 0) CAFFE_FFI_LAYER_LOG << ", ";
    CAFFE_FFI_LAYER_LOG << top_shape[i];
  }
  CAFFE_FFI_LAYER_LOG << "]";
  top[0]->Reshape(top_shape);
  if (bias_term_) {
    std::vector<int64_t> bias_shape = {M_};
    bias_multiplier_ = make_object<Blob>(bias_shape);
    caffe_set_fp32(static_cast<size_t>(M_), 1.0f, bias_multiplier_->cpu_mutable_data());
    CAFFE_FFI_LAYER_LOG << "InnerProduct: created bias_multiplier_ shape=[" << M_ << "]";
  }
}

void InnerProductLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                     const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const float* weight = this->blobs_[0]->cpu_data();

  const int64_t top_count = top[0]->count();
  const int64_t weight_count = this->blobs_[0]->count();
  CAFFE_FFI_TENSOR_LOG << "InnerProduct Forward_cpu: GEMM C=" << M_ << "x" << N_
                       << " += A=" << M_ << "x" << K_ << " * B=" << K_ << "x" << N_;

  using clock = std::chrono::high_resolution_clock;
  auto t_total_start = clock::now();

  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();
  float w_min = std::numeric_limits<float>::max();
  float w_max = -std::numeric_limits<float>::max();
  float b_min = std::numeric_limits<float>::max();
  float b_max = -std::numeric_limits<float>::max();
  double w_norm_sq = 0.0;

  double t_gemm_us = 0, t_bias_us = 0;

  auto t_gemm_start = clock::now();
  caffe_cpu_gemm_fp32(false, transpose_ ? false : true,
                       M_, N_, K_, 1.0f,
                       bottom_data, weight, 0.0f, top_data);
  auto t_gemm_end = clock::now();
  t_gemm_us = std::chrono::duration<double, std::micro>(t_gemm_end - t_gemm_start).count();

  if (bias_term_) {
    CAFFE_FFI_TENSOR_LOG << "InnerProduct Forward_cpu: adding bias (GEMM C += bias_multiplier * bias)";
    auto t_bias_start = clock::now();
    caffe_cpu_gemm_fp32(false, false, M_, N_, 1, 1.0f,
                         bias_multiplier_->cpu_data(),
                         this->blobs_[1]->cpu_data(), 1.0f, top_data);
    auto t_bias_end = clock::now();
    t_bias_us = std::chrono::duration<double, std::micro>(t_bias_end - t_bias_start).count();
  }

  // GEMM后独立reduce：输出值域
  for (int64_t i = 0; i < top_count; ++i) {
    out_min = std::min(out_min, top_data[i]);
    out_max = std::max(out_max, top_data[i]);
  }
  // 权重值域+L2范数（double累加防精度丢失）
  for (int64_t i = 0; i < weight_count; ++i) {
    float w = weight[i];
    w_min = std::min(w_min, w);
    w_max = std::max(w_max, w);
    w_norm_sq += static_cast<double>(w) * static_cast<double>(w);
  }
  float w_norm = static_cast<float>(std::sqrt(w_norm_sq));
  // 偏置值域
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

  CAFFE_FFI_LOG_INFO() << "[IP-PERF] " << this->name()
                       << " InnerProduct forward: M=" << M_ << " N=" << N_ << " K=" << K_
                       << " transpose=" << (transpose_ ? "true" : "false")
                       << " bias_term=" << bias_term_
                       << " out=[" << out_min << ", " << out_max << "]"
                       << " w=[" << w_min << ", " << w_max << "]"
                       << " w_norm=" << w_norm
                       << (bias_term_ ? " b=[" + std::to_string(b_min) + ", " + std::to_string(b_max) + "]" : "")
                       << " t_gemm=" << t_gemm_us << "us"
                       << (bias_term_ ? " t_bias=" + std::to_string(t_bias_us) + "us" : "")
                       << " time=" << total_us << "us";
}

void InnerProductLayer::Backward_cpu(const std::vector<Blob*>& top,
                                      const std::vector<bool>& propagate_down,
                                      const std::vector<Blob*>& bottom) {
  const float* top_diff = top[0]->cpu_diff();
  const float* bottom_data = bottom[0]->cpu_data();
  const float* weight = this->blobs_[0]->cpu_data();
  float* weight_diff = this->param_propagate_down_[0] ? this->blobs_[0]->cpu_mutable_diff() : nullptr;
  float* bottom_diff = propagate_down[0] ? bottom[0]->cpu_mutable_diff() : nullptr;

  const int64_t weight_count = this->blobs_[0]->count();
  const int64_t top_count = top[0]->count();
  const int64_t bottom_count = bottom[0]->count();

  CAFFE_FFI_LAYER_LOG << "InnerProduct Backward: M=" << M_
                      << " N=" << N_ << " K=" << K_
                      << " transpose=" << (transpose_ ? "true" : "false")
                      << " bias_term=" << bias_term_
                      << " prop_down=" << (propagate_down[0] ? "true" : "false")
                      << " prop_w=" << this->param_propagate_down_[0]
                      << " prop_b=" << (bias_term_ ? this->param_propagate_down_[1] : false);

  using clock = std::chrono::high_resolution_clock;
  auto t_total_start = clock::now();

  // ===== 阶段0：清零梯度缓冲区（纳入计时） =====
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

  // ===== 统计变量初始化 =====
  float top_diff_min = std::numeric_limits<float>::max();
  float top_diff_max = -std::numeric_limits<float>::max();
  float bottom_diff_min = std::numeric_limits<float>::max();
  float bottom_diff_max = -std::numeric_limits<float>::max();
  float w_diff_min = std::numeric_limits<float>::max();
  float w_diff_max = -std::numeric_limits<float>::max();
  float b_diff_min = std::numeric_limits<float>::max();
  float b_diff_max = -std::numeric_limits<float>::max();

  double t_gemm_filter_us = 0, t_gemm_data_us = 0, t_gemm_bias_us = 0;

  // ===== backward_filter: dW（权重梯度，beta=0单shot，所有M样本一次完成） =====
  if (this->param_propagate_down_[0]) {
    auto tgf = clock::now();
    if (transpose_) {
      // W is [K, N], dW = X^T * dY: [K,M]*[M,N] = [K,N]
      caffe_cpu_gemm_fp32(true, false, K_, N_, M_,
                          1.0f, bottom_data, top_diff,
                          0.0f, weight_diff);
    } else {
      // W is [N, K], dW = dY^T * X: [N,M]*[M,K] = [N,K]
      caffe_cpu_gemm_fp32(true, false, N_, K_, M_,
                          1.0f, top_diff, bottom_data,
                          0.0f, weight_diff);
    }
    t_gemm_filter_us = std::chrono::duration<double, std::micro>(clock::now() - tgf).count();
  }

  // ===== backward_bias: db（偏置梯度，GEMV） =====
  if (bias_term_ && this->param_propagate_down_[1]) {
    auto tgb = clock::now();
    // db = dY^T * 1_M: [N,M]*[M] = [N] → TransA=true, M=M_, N=N_
    caffe_cpu_gemv_fp32(true, M_, N_,
                        1.0f, top_diff, bias_multiplier_->cpu_data(),
                        0.0f, this->blobs_[1]->cpu_mutable_diff());
    t_gemm_bias_us = std::chrono::duration<double, std::micro>(clock::now() - tgb).count();
  }

  // ===== backward_data: dX（bottom梯度，beta=0覆盖写） =====
  if (propagate_down[0]) {
    auto tgd = clock::now();
    if (transpose_) {
      // W is [K, N], dX = dY * W^T: [M,N]*[N,K] = [M,K]
      caffe_cpu_gemm_fp32(false, true, M_, K_, N_,
                          1.0f, top_diff, weight,
                          0.0f, bottom_diff);
    } else {
      // W is [N, K], dX = dY * W: [M,N]*[N,K] = [M,K]
      caffe_cpu_gemm_fp32(false, false, M_, K_, N_,
                          1.0f, top_diff, weight,
                          0.0f, bottom_diff);
    }
    t_gemm_data_us = std::chrono::duration<double, std::micro>(clock::now() - tgd).count();
  }

  // ===== Post-loop reduce统计（纯读，cache友好，开销<1%） =====
  // 1. bottom_diff值域（backward_data刚写完，cache-hot，最先统计）
  if (propagate_down[0]) {
    for (int64_t i = 0; i < bottom_count; ++i) {
      bottom_diff_min = std::min(bottom_diff_min, bottom_diff[i]);
      bottom_diff_max = std::max(bottom_diff_max, bottom_diff[i]);
    }
  }
  // 2. weight_diff值域+L2范数（double累加防精度丢失）
  double w_diff_norm_sq = 0.0;
  if (this->param_propagate_down_[0]) {
    for (int64_t i = 0; i < weight_count; ++i) {
      float dw = weight_diff[i];
      w_diff_min = std::min(w_diff_min, dw);
      w_diff_max = std::max(w_diff_max, dw);
      w_diff_norm_sq += static_cast<double>(dw) * static_cast<double>(dw);
    }
  }
  float w_diff_norm = static_cast<float>(std::sqrt(w_diff_norm_sq));
  // 3. top_diff值域（多次被GEMM读取，L3可能有残留）
  for (int64_t i = 0; i < top_count; ++i) {
    top_diff_min = std::min(top_diff_min, top_diff[i]);
    top_diff_max = std::max(top_diff_max, top_diff[i]);
  }
  // 4. bias_diff值域（小，N_量级，完全cache-hot）
  if (bias_term_ && this->param_propagate_down_[1]) {
    int64_t bd_count = this->blobs_[1]->count();
    const float* bd = this->blobs_[1]->cpu_diff();
    for (int64_t i = 0; i < bd_count; ++i) {
      b_diff_min = std::min(b_diff_min, bd[i]);
      b_diff_max = std::max(b_diff_max, bd[i]);
    }
  }

  double total_us = std::chrono::duration<double, std::micro>(clock::now() - t_total_start).count();

  // ===== 结构化日志输出（循环外，一次性） =====
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

  CAFFE_FFI_LOG_INFO() << "[IP-PERF] " << this->name()
                       << " InnerProduct backward: M=" << M_
                       << " N=" << N_ << " K=" << K_
                       << " transpose=" << (transpose_ ? "true" : "false")
                       << " bias_term=" << bias_term_
                       << " prop_down=" << (propagate_down[0] ? "true" : "false")
                       << " top_diff=[" << top_diff_min << ", " << top_diff_max << "]"
                       << bottom_diff_str
                       << w_diff_str
                       << b_diff_str
                       << " t_zero=" << t_zero_us << "us"
                       << " t_gemm_filter=" << t_gemm_filter_us << "us"
                       << b_bias_str
                       << " time=" << total_us << "us";
}

REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe_ffi
