#include "caffe_ffi/layers/rnn_layer.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "caffe_ffi/error.hpp"
#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/math_utils.hpp"

namespace caffe_ffi {

namespace {

// Activation applied to the hidden state.
inline float RnnActivation(float z, bool tanh) {
  return tanh ? std::tanh(z) : std::max(z, 0.0f);
}

// Derivative of the activation w.r.t. its argument (used by BPTT).
// relu uses the subgradient 0 at z <= 0, 1 at z > 0 (C1 kink handled by tests).
inline float RnnActivationDeriv(float z, bool tanh) {
  if (tanh) {
    const float t = std::tanh(z);
    return 1.0f - t * t;
  }
  return z > 0.0f ? 1.0f : 0.0f;
}

}  // namespace

void RNNLayer::LayerSetUpStep(const std::vector<Blob*>& bottom) {
  const std::string act = this->layer_param_.recurrent_param().activation();
  tanh_ = (act.empty() || act == "tanh");
  CAFFE_FFI_LAYER_LOG << "RNN LayerSetUpStep: activation='" << act
                      << "' tanh=" << (tanh_ ? "true" : "false");

  if (this->blobs_.size() == 0) {
    // No RecurrentParameter field carries the hidden dim, so the layer must be
    // initialized from pre-trained weights (W_ih (H,D) determines H).
    CAFFE_FFI_THROW(ValueError)
        << "RNN layer requires pre-loaded weights [W_ih, W_hh, b_ih, b_hh]. "
           "There is no num_output parameter in RecurrentParameter to infer "
           "the hidden dimension.";
  }
  CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_.size(), 4U)
      << "RNN layer expects 4 weight blobs [W_ih, W_hh, b_ih, b_hh], got "
      << this->blobs_.size();

  D_ = static_cast<int>(bottom[0]->shape(2));
  H_ = static_cast<int>(this->blobs_[0]->shape(0));
  CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_[0]->shape(1), D_)
      << "W_ih must be (H, D) = (" << H_ << ", " << D_ << "), got ("
      << this->blobs_[0]->shape(0) << ", " << this->blobs_[0]->shape(1) << ")";
  CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_[1]->shape(0), H_)
      << "W_hh must be (H, H) = (" << H_ << ", " << H_ << ")";
  CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_[1]->shape(1), H_)
      << "W_hh must be (H, H) = (" << H_ << ", " << H_ << ")";
  CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_[2]->count(), H_)
      << "b_ih must be (H,), got " << this->blobs_[2]->count();
  CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_[3]->count(), H_)
      << "b_hh must be (H,), got " << this->blobs_[3]->count();
}

void RNNLayer::ReshapeStep() {
  const std::vector<int64_t> nh = {N_, H_};
  z_cache_.resize(T_);
  for (int t = 0; t < T_; ++t) {
    z_cache_[t] = make_object<Blob>();
    z_cache_[t]->Reshape(nh);
  }
  if (!dz_buf_) dz_buf_ = make_object<Blob>();
  dz_buf_->Reshape(nh);
}

void RNNLayer::ForwardStep(int t, const float* x_t, float* h_t) {
  const float* W_ih = this->blobs_[0]->cpu_data();
  const float* W_hh = this->blobs_[1]->cpu_data();
  const float* b_ih = this->blobs_[2]->cpu_data();
  const float* b_hh = this->blobs_[3]->cpu_data();
  const float* h_prev = h_prev_->cpu_data();
  float* z_t = z_cache_[t]->cpu_mutable_data();

  // z_t = x_t @ W_ih^T   (N, H) = (N, D) @ (D, H)
  caffe_cpu_gemm_fp32(false, true, N_, H_, D_, 1.0f, x_t, W_ih, 0.0f, z_t);
  // z_t += h_prev @ W_hh^T   (N, H) = (N, H) @ (H, H)
  caffe_cpu_gemm_fp32(false, true, N_, H_, H_, 1.0f, h_prev, W_hh, 1.0f, z_t);

  // Add biases and apply activation; cache z_t for BPTT.
  for (int i = 0; i < N_; ++i) {
    for (int j = 0; j < H_; ++j) {
      const int idx = i * H_ + j;
      const float z = z_t[idx] + b_ih[j] + b_hh[j];
      z_t[idx] = z;
      h_t[idx] = RnnActivation(z, tanh_);
    }
  }
}

void RNNLayer::BackwardStep(int t, const float* x_t, const float* h_prev,
                            const float* dy_t, float* dx_t, float* dh_next) {
  const float* W_ih = this->blobs_[0]->cpu_data();
  const float* W_hh = this->blobs_[1]->cpu_data();
  const float* z_t = z_cache_[t]->cpu_data();
  float* dz_buf = dz_buf_->cpu_mutable_data();
  const int nh = N_ * H_;

  float* w_ih_diff = this->param_propagate_down_[0] ? this->blobs_[0]->cpu_mutable_diff() : nullptr;
  float* w_hh_diff = this->param_propagate_down_[1] ? this->blobs_[1]->cpu_mutable_diff() : nullptr;
  float* b_ih_diff = this->param_propagate_down_[2] ? this->blobs_[2]->cpu_mutable_diff() : nullptr;
  float* b_hh_diff = this->param_propagate_down_[3] ? this->blobs_[3]->cpu_mutable_diff() : nullptr;

  // dh_t = dy_t + dh_next ; dz_t = dh_t * act'(z_t)
  for (int i = 0; i < nh; ++i) {
    const float dh = dy_t[i] + dh_next[i];
    dz_buf[i] = dh * RnnActivationDeriv(z_t[i], tanh_);
  }

  // dW_ih += dz^T @ x_t : (H, D) += (H, N) @ (N, D)
  if (w_ih_diff) {
    caffe_cpu_gemm_fp32(true, false, H_, D_, N_, 1.0f, dz_buf, x_t, 1.0f, w_ih_diff);
  }
  // dW_hh += dz^T @ h_prev : (H, H) += (H, N) @ (N, H)
  if (w_hh_diff) {
    caffe_cpu_gemm_fp32(true, false, H_, H_, N_, 1.0f, dz_buf, h_prev, 1.0f, w_hh_diff);
  }
  // db_ih += sum(dz) over batch ; db_hh += sum(dz) over batch
  if (b_ih_diff) {
    for (int i = 0; i < N_; ++i) {
      for (int j = 0; j < H_; ++j) b_ih_diff[j] += dz_buf[i * H_ + j];
    }
  }
  if (b_hh_diff) {
    for (int i = 0; i < N_; ++i) {
      for (int j = 0; j < H_; ++j) b_hh_diff[j] += dz_buf[i * H_ + j];
    }
  }

  // dX_t = dz @ W_ih : (N, D) = (N, H) @ (H, D)
  caffe_cpu_gemm_fp32(false, false, N_, D_, H_, 1.0f, dz_buf, W_ih, 0.0f, dx_t);
  // dh_next = dz @ W_hh : (N, H) = (N, H) @ (H, H)
  caffe_cpu_gemm_fp32(false, false, N_, H_, H_, 1.0f, dz_buf, W_hh, 0.0f, dh_next);
}

REGISTER_LAYER_CLASS(RNN);

}  // namespace caffe_ffi