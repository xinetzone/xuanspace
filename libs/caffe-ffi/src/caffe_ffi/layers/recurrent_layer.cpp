#include "caffe_ffi/layers/recurrent_layer.hpp"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "caffe_ffi/error.hpp"
#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void RecurrentLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                                const std::vector<Blob*>& top) {
  // Read the configured unroll length / flags from RecurrentParameter.
  num_steps_ = static_cast<int>(this->layer_param_.recurrent_param().num_steps());
  expose_hidden_ = this->layer_param_.recurrent_param().expose_hidden();
  CAFFE_FFI_LAYER_LOG << "RecurrentLayer LayerSetUp: num_steps=" << num_steps_
                      << " expose_hidden=" << (expose_hidden_ ? "true" : "false");

  // Let the subclass create/validate its weight blobs and set H_.
  LayerSetUpStep(bottom);

  // Every weight blob is learnable by default (BPTT accumulates into them).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

void RecurrentLayer::Reshape(const std::vector<Blob*>& bottom,
                             const std::vector<Blob*>& top) {
  // The input is a 3-D time-first sequence (T, N, D).
  CAFFE_FFI_CHECK_VALUE_EQ(bottom[0]->num_axes(), 3)
      << "RecurrentLayer input must be 3-D (T, N, D), got "
      << bottom[0]->num_axes() << " axes.";
  T_ = static_cast<int>(bottom[0]->shape(0));
  N_ = static_cast<int>(bottom[0]->shape(1));
  D_ = static_cast<int>(bottom[0]->shape(2));
  CAFFE_FFI_CHECK_VALUE_GT(T_, 0) << "RecurrentLayer timesteps must be > 0";
  CAFFE_FFI_CHECK_VALUE_GT(N_, 0) << "RecurrentLayer batch size must be > 0";
  CAFFE_FFI_CHECK_VALUE_GT(D_, 0) << "RecurrentLayer input dim must be > 0";

  // Subclass allocates per-step caches (needs T_/N_/D_ and sets/uses H_).
  ReshapeStep();

  // Reshape the carried-state working buffers.
  std::vector<int64_t> nh = {N_, H_};
  h_prev_->Reshape(nh);
  c_prev_->Reshape(nh);
  h_t_->Reshape(nh);
  dh_next_->Reshape(nh);
  dc_next_->Reshape(nh);

  // Output: every time-step hidden state (T, N, H).
  std::vector<int64_t> top_shape = {T_, N_, H_};
  top[0]->Reshape(top_shape);
  CAFFE_FFI_LAYER_LOG << "RecurrentLayer Reshape: T=" << T_ << " N=" << N_
                      << " D=" << D_ << " H=" << H_ << " top=(" << T_ << ","
                      << N_ << "," << H_ << ")";
}

void RecurrentLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                 const std::vector<Blob*>& top) {
  // Initial hidden & cell states are zero.
  const int64_t nh = static_cast<int64_t>(N_) * H_;
  caffe_set_fp32(static_cast<size_t>(nh), 0.0f, h_prev_->cpu_mutable_data());
  caffe_set_fp32(static_cast<size_t>(nh), 0.0f, c_prev_->cpu_mutable_data());

  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  for (int t = 0; t < T_; ++t) {
    const float* x_t = bottom_data + t * N_ * D_;
    float* h_t = top_data + t * N_ * H_;
    // Subclass computes h_t from x_t and the carried h_prev_/c_prev_.
    ForwardStep(t, x_t, h_t);
    // Carry h_t forward to the next step.
    caffe_copy_fp32(static_cast<size_t>(nh), h_t, h_prev_->cpu_mutable_data());
  }
}

void RecurrentLayer::Backward_cpu(const std::vector<Blob*>& top,
                                  const std::vector<bool>& propagate_down,
                                  const std::vector<Blob*>& bottom) {
  // Zero the weight gradients (gradients accumulate via += across time steps).
  for (size_t i = 0; i < this->blobs_.size(); ++i) {
    if (this->param_propagate_down_[i]) {
      caffe_set_fp32(static_cast<size_t>(this->blobs_[i]->count()), 0.0f,
                     this->blobs_[i]->cpu_mutable_diff());
    }
  }
  // Zero the bottom gradient (each time step accumulates into its own slice).
  if (propagate_down[0]) {
    caffe_set_fp32(static_cast<size_t>(bottom[0]->count()), 0.0f,
                   bottom[0]->cpu_mutable_diff());
  }
  const int64_t nh = static_cast<int64_t>(N_) * H_;
  caffe_set_fp32(static_cast<size_t>(nh), 0.0f, dh_next_->cpu_mutable_data());
  caffe_set_fp32(static_cast<size_t>(nh), 0.0f, dc_next_->cpu_mutable_data());
  // h_t_ doubles as a zero buffer for the t==0 (h_0 / c_0) state.
  caffe_set_fp32(static_cast<size_t>(nh), 0.0f, h_t_->cpu_mutable_data());

  // Subclass-specific internal gradient reset (e.g. LSTM packed W diff).
  BackwardStart();

  const float* bottom_data = bottom[0]->cpu_data();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const float* top_data = top[0]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  float* dh_next = dh_next_->cpu_mutable_data();

  // Backpropagation through time: step T-1 → 0.
  for (int t = T_ - 1; t >= 0; --t) {
    const float* x_t = bottom_data + t * N_ * D_;
    const float* h_prev =
        (t == 0) ? h_t_->cpu_data() : top_data + (t - 1) * N_ * H_;
    const float* dy_t = top_diff + t * N_ * H_;
    float* dx_t = bottom_diff + t * N_ * D_;
    BackwardStep(t, x_t, h_prev, dy_t, dx_t, dh_next);
  }

  // Subclass-specific finalization (e.g. LSTM scatter of accumulated packed
  // weight gradients into blobs_[0] diff, exactly once).
  BackwardEnd();
}

}  // namespace caffe_ffi