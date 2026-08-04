#include "caffe_ffi/layers/embed_layer.hpp"

#include <cstring>
#include <string>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/math_utils.hpp"

namespace caffe_ffi {

void EmbedLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                             const std::vector<Blob*>& top) {
  const caffe::EmbedParameter& param = this->layer_param_.embed_param();
  N_ = param.num_output();
  CAFFE_FFI_CHECK_VALUE_GT(N_, 0) << "EmbedLayer num_output must be positive.";
  K_ = param.input_dim();
  CAFFE_FFI_CHECK_VALUE_GT(K_, 0) << "EmbedLayer input_dim must be positive.";
  bias_term_ = param.bias_term();

  CAFFE_FFI_LAYER_LOG << "Embed LayerSetUp: num_output(N_)=" << N_
                      << " input_dim(K_)=" << K_
                      << " bias_term=" << bias_term_;

  if (this->blobs_.size() > 0) {
    CAFFE_FFI_LAYER_LOG << "Embed: using pre-loaded parameter blobs, size="
                        << this->blobs_.size();
    CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_.size(), bias_term_ ? 2U : 1U)
        << "Incorrect number of parameter blobs for EmbedLayer.";
  } else {
    this->blobs_.resize(bias_term_ ? 2 : 1);
    // Weight is transposed from InnerProductLayer for spatial locality:
    // shape [K, N] so that each index looks up a contiguous row.
    std::vector<int64_t> weight_shape = {K_, static_cast<int64_t>(N_)};
    this->blobs_[0] = make_object<Blob>(weight_shape);

    // Apply weight_filler if specified.
    float weight_value = 0.0f;
    if (param.has_weight_filler()) {
      const caffe::FillerParameter& filler = param.weight_filler();
      const std::string filler_type = filler.type();
      if (filler_type == "constant") {
        weight_value = filler.value();
      } else if (filler_type == "xavier" || filler_type == "gaussian" ||
                 filler_type == "msra") {
        weight_value = 1.0f;
        CAFFE_FFI_LOG_WARN() << "[EMBED-FILLER] weight filler type '" << filler_type
                             << "' not fully implemented, using constant 1.0";
      }
      CAFFE_FFI_LAYER_LOG << "Embed: applied weight_filler type='" << filler_type
                          << "' value=" << weight_value;
    }
    caffe_set_fp32(static_cast<size_t>(this->blobs_[0]->count()), weight_value,
                   this->blobs_[0]->cpu_mutable_data());

    if (bias_term_) {
      std::vector<int64_t> bias_shape = {N_};
      this->blobs_[1] = make_object<Blob>(bias_shape);
      float bias_value = 0.0f;
      if (param.has_bias_filler()) {
        const caffe::FillerParameter& filler = param.bias_filler();
        if (filler.type() == "constant") {
          bias_value = filler.value();
        }
        CAFFE_FFI_LAYER_LOG << "Embed: applied bias_filler type='" << filler.type()
                            << "' value=" << bias_value;
      }
      caffe_set_fp32(static_cast<size_t>(this->blobs_[1]->count()), bias_value,
                     this->blobs_[1]->cpu_mutable_data());
    }
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

void EmbedLayer::Reshape(const std::vector<Blob*>& bottom,
                          const std::vector<Blob*>& top) {
  M_ = static_cast<int>(bottom[0]->count());
  std::vector<int64_t> top_shape;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    top_shape.push_back(bottom[0]->shape(i));
  }
  top_shape.push_back(N_);
  top[0]->Reshape(top_shape);
  CAFFE_FFI_LAYER_LOG << "Embed Reshape: M_=" << M_ << " K_=" << K_
                      << " N_=" << N_ << " top_count=" << top[0]->count();
}

void EmbedLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                              const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  const float* weight = this->blobs_[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();

  for (int n = 0; n < M_; ++n) {
    const int index = static_cast<int>(bottom_data[n]);
    CAFFE_FFI_CHECK_VALUE_GE(index, 0) << "Embed index out of range (negative).";
    CAFFE_FFI_CHECK_VALUE_LT(index, K_) << "Embed index out of range: " << index
                                        << " >= K_=" << K_;
    CAFFE_FFI_CHECK_VALUE_EQ(static_cast<float>(index), bottom_data[n])
        << "Embed input must be integer-valued.";
    std::memcpy(top_data + n * N_, weight + index * N_,
                sizeof(float) * static_cast<size_t>(N_));
  }

  if (bias_term_) {
    // top += bias_multiplier(M) * bias(N), i.e. broadcast bias over rows.
    std::vector<int64_t> mult_shape = {M_};
    Blob bias_multiplier(mult_shape);
    caffe_set_fp32(static_cast<size_t>(M_), 1.0f, bias_multiplier.cpu_mutable_data());
    caffe_cpu_gemm_fp32(false, false, M_, N_, 1, 1.0f,
                        bias_multiplier.cpu_data(),
                        this->blobs_[1]->cpu_data(), 1.0f, top_data);
  }
  CAFFE_FFI_LAYER_LOG << "Embed Forward_cpu: M_=" << M_ << " N_=" << N_
                      << " bias_term=" << bias_term_;
}

void EmbedLayer::Backward_cpu(const std::vector<Blob*>& top,
                               const std::vector<bool>& propagate_down,
                               const std::vector<Blob*>& bottom) {
  CAFFE_FFI_CHECK_VALUE(!propagate_down[0])
      << "Cannot backpropagate to EmbedLayer input (indices).";
  const float* top_diff = top[0]->cpu_diff();
  const float* bottom_data = bottom[0]->cpu_data();

  if (this->param_propagate_down_[0]) {
    float* weight_diff = this->blobs_[0]->cpu_mutable_diff();
    caffe_set_fp32(static_cast<size_t>(this->blobs_[0]->count()), 0.0f, weight_diff);
    for (int n = 0; n < M_; ++n) {
      const int index = static_cast<int>(bottom_data[n]);
      CAFFE_FFI_CHECK_VALUE_GE(index, 0);
      CAFFE_FFI_CHECK_VALUE_LT(index, K_);
      caffe_axpy_fp32(static_cast<int64_t>(N_), 1.0f, top_diff + n * N_, weight_diff + index * N_);
    }
  }

  if (bias_term_ && this->param_propagate_down_[1]) {
    float* bias_diff = this->blobs_[1]->cpu_mutable_diff();
    caffe_set_fp32(static_cast<size_t>(this->blobs_[1]->count()), 0.0f, bias_diff);
    std::vector<int64_t> mult_shape = {M_};
    Blob bias_multiplier(mult_shape);
    caffe_set_fp32(static_cast<size_t>(M_), 1.0f, bias_multiplier.cpu_mutable_data());
    // db = dY^T * 1_M -> [N, M] * [M] = [N]
    caffe_cpu_gemv_fp32(true, M_, N_, 1.0f, top_diff,
                        bias_multiplier.cpu_data(), 0.0f, bias_diff);
  }
  CAFFE_FFI_LAYER_LOG << "Embed Backward_cpu: M_=" << M_ << " N_=" << N_
                      << " prop_w=" << this->param_propagate_down_[0]
                      << " prop_b=" << (bias_term_ && this->param_propagate_down_[1]);
}

REGISTER_LAYER_CLASS(Embed);

}  // namespace caffe_ffi