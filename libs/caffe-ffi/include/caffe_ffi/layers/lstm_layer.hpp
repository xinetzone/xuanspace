#ifndef CAFFE_FFI_LAYERS_LSTM_LAYER_HPP_
#define CAFFE_FFI_LAYERS_LSTM_LAYER_HPP_

#include <vector>

#include "caffe_ffi/layers/recurrent_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief LSTM layer (time-step unrolled, BPTT).
 *
 * Uses the Caffe-style packed weight layout (see pack_lstm_weights_caffe):
 *   blobs_[0] = W (4H, D + H)  = [W_ih (4H, D) | W_hh (4H, H)], rows ordered
 *                                 input gate i, forget gate f, output gate o,
 *                                 cell gate g.
 *   blobs_[1] = b (4H,) = [b_i; b_f; b_o; b_g] (combined input+recurrent bias).
 *
 * Per-time-step cell (forward):
 *   i = sigmoid(x_t W_ii^T + h W_hi^T + b_i)
 *   f = sigmoid(x_t W_if^T + h W_hf^T + b_f)
 *   o = sigmoid(x_t W_io^T + h W_ho^T + b_o)
 *   g = tanh   (x_t W_ig^T + h W_hg^T + b_g)
 *   c_t = f * c_{t-1} + i * g
 *   h_t = o * tanh(c_t)
 *
 * H is inferred from the packed blob[0] rows (4H). Trained weights must be
 * supplied (the RecurrentParameter proto has no num_output field).
 */
class LSTMLayer : public RecurrentLayer {
 public:
  static constexpr bool _type_mutable = true;

  explicit LSTMLayer(const caffe::LayerParameter& param) : RecurrentLayer(param) {}

  const char* type() const override { return "LSTM"; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.LSTMLayer", LSTMLayer, RecurrentLayer);

 protected:
  void LayerSetUpStep(const std::vector<Blob*>& bottom) override;
  void ReshapeStep() override;
  void BackwardStart() override;
  void ForwardStep(int t, const float* x_t, float* h_t) override;
  void BackwardStep(int t, const float* x_t, const float* h_prev,
                    const float* dy_t, float* dx_t, float* dh_next) override;

  // Contiguous copies of the packed weight blocks (the packed blob has a
  // leading dimension of D+H, which GEMM cannot stride, so we stage copies).
  ObjectPtr<Blob> W_ih_;      // (4H, D)
  ObjectPtr<Blob> W_hh_;      // (4H, H)
  ObjectPtr<Blob> gate_pre_;  // (N, 4H) working pre-activation
  ObjectPtr<Blob> dgates_;    // (N, 4H) working gate gradients
  ObjectPtr<Blob> dW_ih_;     // (4H, D) working input-weight gradient
  ObjectPtr<Blob> dW_hh_;     // (4H, H) working recurrent-weight gradient
  // Per-step cache (N, 6H) = [i, f, o, g, c, tanh(c)].
  std::vector<ObjectPtr<Blob>> lstm_cache_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_LSTM_LAYER_HPP_