#include "caffe_ffi/layers/lstm_layer.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe_ffi/error.hpp"
#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/math_utils.hpp"

namespace caffe_ffi {

namespace {
// Gate offsets within the packed (4H, D+H) weight rows, in units of H.
constexpr int kI = 0;  // input gate
constexpr int kF = 1;  // forget gate
constexpr int kO = 2;  // output gate
constexpr int kG = 3;  // cell gate
// Per-step cache (N, 6H) = [i, f, o, g, c, tanh(c)].
constexpr int kCacheC = 4;
constexpr int kCacheTc = 5;

inline float Sigmoid(float x) {
  if (x >= 0.0f) {
    return 1.0f / (1.0f + std::exp(-x));
  }
  const float e = std::exp(x);
  return e / (1.0f + e);
}
}  // namespace

void LSTMLayer::LayerSetUpStep(const std::vector<Blob*>& bottom) {
  if (this->blobs_.size() == 0) {
    // No RecurrentParameter field carries the hidden dim, so the layer must be
    // initialized from pre-trained Caffe-style packed weights.
    CAFFE_FFI_THROW(ValueError)
        << "LSTM layer requires pre-loaded weights [blob[0]=(4H,D+H), "
           "blob[1]=(4H,)]. There is no num_output parameter in "
           "RecurrentParameter to infer the hidden dimension.";
  }
  CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_.size(), 2U)
      << "LSTM layer expects 2 weight blobs [W(4H,D+H), b(4H,)], got "
      << this->blobs_.size();

  D_ = static_cast<int>(bottom[0]->shape(2));
  const int64_t rows = this->blobs_[0]->shape(0);
  const int64_t cols = this->blobs_[0]->shape(1);
  CAFFE_FFI_CHECK_VALUE_EQ(rows % 4, 0)
      << "LSTM packed weight rows must be a multiple of 4, got " << rows;
  H_ = static_cast<int>(rows / 4);
  CAFFE_FFI_CHECK_VALUE_EQ(cols, D_ + H_)
      << "LSTM packed weight blob[0] must be (4H, D+H) = (" << 4 * H_ << ", "
      << D_ + H_ << "), got (" << rows << ", " << cols << ")";
  CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_[1]->count(), rows)
      << "LSTM bias blob[1] must be (4H,) with count " << rows << ", got "
      << this->blobs_[1]->count();
  CAFFE_FFI_LAYER_LOG << "LSTM LayerSetUpStep: D=" << D_ << " H=" << H_;
}

void LSTMLayer::ReshapeStep() {
  // Stage contiguous copies of the packed weight blocks.
  if (!W_ih_) W_ih_ = make_object<Blob>();
  if (!W_hh_) W_hh_ = make_object<Blob>();
  const std::vector<int64_t> wih_shape = {4 * H_, D_};
  const std::vector<int64_t> whh_shape = {4 * H_, H_};
  W_ih_->Reshape(wih_shape);
  W_hh_->Reshape(whh_shape);
  const float* W = this->blobs_[0]->cpu_data();
  float* wih = W_ih_->cpu_mutable_data();
  float* whh = W_hh_->cpu_mutable_data();
  for (int r = 0; r < 4 * H_; ++r) {
    for (int c = 0; c < D_; ++c) wih[r * D_ + c] = W[r * (D_ + H_) + c];
    for (int c = 0; c < H_; ++c) whh[r * H_ + c] = W[r * (D_ + H_) + D_ + c];
  }

  if (!gate_pre_) gate_pre_ = make_object<Blob>();
  if (!dgates_) dgates_ = make_object<Blob>();
  if (!dW_ih_) dW_ih_ = make_object<Blob>();
  if (!dW_hh_) dW_hh_ = make_object<Blob>();
  const std::vector<int64_t> gate_shape = {N_, 4 * H_};
  const std::vector<int64_t> cache_shape = {N_, 6 * H_};
  gate_pre_->Reshape(gate_shape);
  dgates_->Reshape(gate_shape);
  dW_ih_->Reshape(wih_shape);
  dW_hh_->Reshape(whh_shape);

  lstm_cache_.resize(T_);
  for (int t = 0; t < T_; ++t) {
    lstm_cache_[t] = make_object<Blob>();
    lstm_cache_[t]->Reshape(cache_shape);
  }
}

void LSTMLayer::BackwardStart() {
  // Reset the working packed-weight gradients before the BPTT accumulation loop.
  caffe_set_fp32(static_cast<size_t>(dW_ih_->count()), 0.0f, dW_ih_->cpu_mutable_data());
  caffe_set_fp32(static_cast<size_t>(dW_hh_->count()), 0.0f, dW_hh_->cpu_mutable_data());
}

void LSTMLayer::ForwardStep(int t, const float* x_t, float* h_t) {
  const float* W_ih = W_ih_->cpu_data();
  const float* W_hh = W_hh_->cpu_data();
  const float* b = this->blobs_[1]->cpu_data();
  const float* h_prev = h_prev_->cpu_data();
  const float* c_prev = c_prev_->cpu_data();
  float* gate_pre = gate_pre_->cpu_mutable_data();

  // gate_pre = x_t @ W_ih^T + h_prev @ W_hh^T   (N, 4H)
  caffe_cpu_gemm_fp32(false, true, N_, 4 * H_, D_, 1.0f, x_t, W_ih, 0.0f, gate_pre);
  caffe_cpu_gemm_fp32(false, true, N_, 4 * H_, H_, 1.0f, h_prev, W_hh, 1.0f, gate_pre);

  float* c_new = c_prev_->cpu_mutable_data();
  float* cache = lstm_cache_[t]->cpu_mutable_data();
  for (int i = 0; i < N_; ++i) {
    const float* g = gate_pre + i * 4 * H_;
    const float* cp = c_prev + i * H_;
    float* cn = c_new + i * H_;
    float* ht = h_t + i * H_;
    float* cb = cache + i * 6 * H_;
    for (int j = 0; j < H_; ++j) {
      const float iv = Sigmoid(g[kI * H_ + j] + b[kI * H_ + j]);
      const float fv = Sigmoid(g[kF * H_ + j] + b[kF * H_ + j]);
      const float ov = Sigmoid(g[kO * H_ + j] + b[kO * H_ + j]);
      const float gv = std::tanh(g[kG * H_ + j] + b[kG * H_ + j]);
      const float cprev_val = cp[j];
      const float cv = fv * cprev_val + iv * gv;
      const float tc = std::tanh(cv);
      cn[j] = cv;
      ht[j] = ov * tc;
      cb[kI * H_ + j] = iv;
      cb[kF * H_ + j] = fv;
      cb[kO * H_ + j] = ov;
      cb[kG * H_ + j] = gv;
      cb[kCacheC * H_ + j] = cv;
      cb[kCacheTc * H_ + j] = tc;
    }
  }
}

void LSTMLayer::BackwardStep(int t, const float* x_t, const float* h_prev,
                             const float* dy_t, float* dx_t, float* dh_next) {
  const float* W_ih = W_ih_->cpu_data();
  const float* W_hh = W_hh_->cpu_data();
  const float* cache = lstm_cache_[t]->cpu_data();
  // c_{t-1}: for t==0 use the zeroed h_t_ buffer (c_0 = 0); else the cached c.
  // The cache layout is (N, 6H) = [i, f, o, g, c, tanh(c)], so the cell state
  // of batch i in the previous step sits at i*6H + kCacheC*H_ (NOT i*H_).
  const float* cache_prev = (t == 0) ? h_t_->cpu_data() : lstm_cache_[t - 1]->cpu_data();
  const bool c_prev_zero = (t == 0);
  float* dgates = dgates_->cpu_mutable_data();
  float* dc_next = dc_next_->cpu_mutable_data();

  // Per-cell gate gradients (following the numpy lstm_backward formula).
  for (int i = 0; i < N_; ++i) {
    const float* cb = cache + i * 6 * H_;
    const float* cp =
        c_prev_zero ? (cache_prev + i * H_) : (cache_prev + i * 6 * H_ + kCacheC * H_);
    const float* dy = dy_t + i * H_;
    const float* dh_in = dh_next + i * H_;
    const float* dc_in = dc_next + i * H_;
    float* dg = dgates + i * 4 * H_;
    for (int j = 0; j < H_; ++j) {
      const float iv = cb[kI * H_ + j];
      const float fv = cb[kF * H_ + j];
      const float ov = cb[kO * H_ + j];
      const float gv = cb[kG * H_ + j];
      const float tc = cb[kCacheTc * H_ + j];

      const float dh = dy[j] + dh_in[j];
      const float dtanh_c = dh * ov;
      const float dc_t = dtanh_c * (1.0f - tc * tc) + dc_in[j];
      const float do_t = dh * tc;
      const float di_t = dc_t * gv;
      const float df_t = dc_t * cp[j];
      const float dg_t = dc_t * iv;
      dc_next[i * H_ + j] = dc_t * fv;  // dc_{t-1} = dc_t * f_t

      // Backprop through activations.
      const float dg_cell = dg_t * (1.0f - gv * gv);
      const float d_i = di_t * iv * (1.0f - iv);
      const float d_f = df_t * fv * (1.0f - fv);
      const float d_o = do_t * ov * (1.0f - ov);

      dg[kI * H_ + j] = d_i;
      dg[kF * H_ + j] = d_f;
      dg[kO * H_ + j] = d_o;
      dg[kG * H_ + j] = dg_cell;
    }
  }

  // dW_ih (4H, D) += dgates^T @ x_t
  caffe_cpu_gemm_fp32(true, false, 4 * H_, D_, N_, 1.0f, dgates, x_t, 1.0f,
                      dW_ih_->cpu_mutable_data());
  // dW_hh (4H, H) += dgates^T @ h_prev
  caffe_cpu_gemm_fp32(true, false, 4 * H_, H_, N_, 1.0f, dgates, h_prev, 1.0f,
                      dW_hh_->cpu_mutable_data());

  // NOTE: the packed-weight scatter (dW_ih_/dW_hh_ -> blobs_[0] diff) is done
  // once in BackwardEnd(), after the whole BPTT loop, to avoid counting the
  // accumulated dW_{ih,hh} more than once.

  // db (4H) += sum(dgates) over batch.
  float* b_diff = this->blobs_[1]->cpu_mutable_diff();
  for (int j = 0; j < 4 * H_; ++j) {
    float s = 0.0f;
    for (int i = 0; i < N_; ++i) s += dgates[i * 4 * H_ + j];
    b_diff[j] += s;
  }

  // dX_t = dgates @ W_ih : (N, D) = (N, 4H) @ (4H, D)
  caffe_cpu_gemm_fp32(false, false, N_, D_, 4 * H_, 1.0f, dgates, W_ih, 0.0f, dx_t);
  // dh_next = dgates @ W_hh : (N, H) = (N, 4H) @ (4H, H)
  caffe_cpu_gemm_fp32(false, false, N_, H_, 4 * H_, 1.0f, dgates, W_hh, 0.0f, dh_next);
}

void LSTMLayer::BackwardEnd() {
  // Scatter the fully-accumulated packed-weight gradients into blobs_[0] diff
  // exactly once (after the BPTT loop). dW_ih_/dW_hh_ hold the exact sum over
  // all time steps; scattering inside BackwardStep would add the running sum
  // T times and over-count by a factor of T.
  float* W_diff = this->blobs_[0]->cpu_mutable_diff();
  const float* dWih = dW_ih_->cpu_data();
  const float* dWhh = dW_hh_->cpu_data();
  for (int r = 0; r < 4 * H_; ++r) {
    for (int c = 0; c < D_; ++c) W_diff[r * (D_ + H_) + c] += dWih[r * D_ + c];
    for (int c = 0; c < H_; ++c) W_diff[r * (D_ + H_) + D_ + c] += dWhh[r * H_ + c];
  }
}

REGISTER_LAYER_CLASS(LSTM);

}  // namespace caffe_ffi