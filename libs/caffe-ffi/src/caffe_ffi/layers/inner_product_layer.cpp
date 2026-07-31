#include "caffe_ffi/layers/inner_product_layer.hpp"

#include <sstream>
#include <vector>

#include <tvm/ffi/memory.h>

#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/error.hpp"
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
    if (transpose_) {
      CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_[0]->shape(0), K_);
      CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_[0]->shape(1), N_);
    } else {
      CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_[0]->shape(0), N_);
      CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_[0]->shape(1), K_);
    }
    if (bias_term_) {
      CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_[1]->count(), N_);
    }
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
    if (bias_term_) {
      std::vector<int64_t> bias_shape = {N_};
      this->blobs_[1] = make_object<Blob>(bias_shape);
      CAFFE_FFI_LAYER_LOG << "InnerProduct: created bias blob shape=[" << N_ << "]";
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
  std::vector<int64_t> top_shape;
  for (int i = 0; i < axis; ++i) {
    top_shape.push_back(bottom[0]->shape(i));
  }
  top_shape.push_back(N_);
  for (int i = axis + 1; i < bottom[0]->num_axes(); ++i) {
    top_shape.push_back(1);
  }
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
  CAFFE_FFI_TENSOR_LOG << "InnerProduct Forward_cpu: GEMM C=" << M_ << "x" << N_
                       << " += A=" << M_ << "x" << K_ << " * B=" << K_ << "x" << N_;
  caffe_cpu_gemm_fp32(false, transpose_ ? false : true,
                       M_, N_, K_, 1.0f,
                       bottom_data, weight, 0.0f, top_data);
  if (bias_term_) {
    CAFFE_FFI_TENSOR_LOG << "InnerProduct Forward_cpu: adding bias (GEMM C += bias_multiplier * bias)";
    caffe_cpu_gemm_fp32(false, false, M_, N_, 1, 1.0f,
                         bias_multiplier_->cpu_data(),
                         this->blobs_[1]->cpu_data(), 1.0f, top_data);
  }
}

REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe_ffi
