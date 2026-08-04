#include "caffe_ffi/layers/lstm_unit.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe_ffi/error.hpp"
#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

namespace {
// Gate offsets within the (N, 4H) input, in units of H.
constexpr int kI = 0;
constexpr int kF = 1;
constexpr int kO = 2;
constexpr int kG = 3;
// Cache layout (N, 6H) = [i, f, o, g, c, tanh(c)].
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

void LSTMUnitLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                          const std::vector<Blob*>& top) {
  // No learnable parameters.
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

void LSTMUnitLayer::Reshape(const std::vector<Blob*>& bottom,
                       const std::vector<Blob*>& top) {
  CAFFE_FFI_CHECK_VALUE_EQ(bottom[0]->num_axes(), 2)
      << "LSTMUnit input must be 2-D (N, 4H), got " << bottom[0]->num_axes()
      << " axes.";
  N_ = static_cast<int>(bottom[0]->shape(0));
  const int64_t cols = bottom[0]->shape(1);
  CAFFE_FFI_CHECK_VALUE_EQ(cols % 4, 0)
      << "LSTMUnit gate input width must be a multiple of 4, got " << cols;
  H_ = static_cast<int>(cols / 4);
  CAFFE_FFI_CHECK_VALUE_EQ(bottom[1]->shape(0), N_)
      << "LSTMUnit c_{t-1} batch must match gate input batch (" << N_ << ")";
  CAFFE_FFI_CHECK_VALUE_EQ(bottom[1]->shape(1), H_)
      << "LSTMUnit c_{t-1} width must equal H (" << H_ << ")";

  if (!cache_) cache_ = make_object<Blob>();
  const std::vector<int64_t> h_shape = {N_, H_};
  const std::vector<int64_t> cache_shape = {N_, 6 * H_};
  cache_->Reshape(cache_shape);
  top[0]->Reshape(h_shape);
  top[1]->Reshape(h_shape);
}

void LSTMUnitLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                           const std::vector<Blob*>& top) {
  const float* in = bottom[0]->cpu_data();
  const float* c_prev = bottom[1]->cpu_data();
  float* c_top = top[0]->cpu_mutable_data();
  float* h_top = top[1]->cpu_mutable_data();
  float* cache = cache_->cpu_mutable_data();

  for (int i = 0; i < N_; ++i) {
    const float* g = in + i * 4 * H_;
    const float* cp = c_prev + i * H_;
    float* ct = c_top + i * H_;
    float* ht = h_top + i * H_;
    float* cb = cache + i * 6 * H_;
    for (int j = 0; j < H_; ++j) {
      const float iv = Sigmoid(g[kI * H_ + j]);
      const float fv = Sigmoid(g[kF * H_ + j]);
      const float ov = Sigmoid(g[kO * H_ + j]);
      const float gv = std::tanh(g[kG * H_ + j]);
      const float cv = fv * cp[j] + iv * gv;
      const float tc = std::tanh(cv);
      const float hv = ov * tc;
      ct[j] = cv;
      ht[j] = hv;
      cb[kI * H_ + j] = iv;
      cb[kF * H_ + j] = fv;
      cb[kO * H_ + j] = ov;
      cb[kG * H_ + j] = gv;
      cb[kCacheC * H_ + j] = cv;
      cb[kCacheTc * H_ + j] = tc;
    }
  }
}

void LSTMUnitLayer::Backward_cpu(const std::vector<Blob*>& top,
                            const std::vector<bool>& propagate_down,
                            const std::vector<Blob*>& bottom) {
  const float* dc_t = top[0]->cpu_diff();
  const float* dh_t = top[1]->cpu_diff();
  const float* c_prev = bottom[1]->cpu_data();
  const float* cache = cache_->cpu_data();
  float* d_in = bottom[0]->cpu_mutable_diff();
  float* d_c_prev = bottom[1]->cpu_mutable_diff();

  for (int i = 0; i < N_; ++i) {
    const float* cb = cache + i * 6 * H_;
    const float* cp = c_prev + i * H_;
    const float* dct = dc_t + i * H_;
    const float* dht = dh_t + i * H_;
    float* di = d_in + i * 4 * H_;
    float* dcp = d_c_prev + i * H_;
    for (int j = 0; j < H_; ++j) {
      const float iv = cb[kI * H_ + j];
      const float fv = cb[kF * H_ + j];
      const float ov = cb[kO * H_ + j];
      const float gv = cb[kG * H_ + j];
      const float tc = cb[kCacheTc * H_ + j];

      const float dh = dht[j];
      const float dtanh_c = dh * ov;
      const float dc = dtanh_c * (1.0f - tc * tc) + dct[j];
      const float do_t = dh * tc;
      const float di_t = dc * gv;
      const float df_t = dc * cp[j];
      const float dg_t = dc * iv;
      dcp[j] = dc * fv;  // dc_{t-1} = dc_t * f_t

      // Backprop through activations.
      const float dg = dg_t * (1.0f - gv * gv);
      const float d_i = di_t * iv * (1.0f - iv);
      const float d_f = df_t * fv * (1.0f - fv);
      const float d_o = do_t * ov * (1.0f - ov);

      di[kI * H_ + j] = d_i;
      di[kF * H_ + j] = d_f;
      di[kO * H_ + j] = d_o;
      di[kG * H_ + j] = dg;
    }
  }
}

REGISTER_LAYER_CLASS(LSTMUnit);

}  // namespace caffe_ffi