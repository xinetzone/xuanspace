#ifndef CAFFE_FFI_LAYERS_RNN_LAYER_HPP_
#define CAFFE_FFI_LAYERS_RNN_LAYER_HPP_

#include <vector>

#include "caffe_ffi/layers/recurrent_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Vanilla RNN layer (time-step unrolled, BPTT).
 *
 * Cell update (matching the numpy reference rnn_forward):
 *   z_t = x_t @ W_ih^T + b_ih + h_{t-1} @ W_hh^T + b_hh
 *   h_t = act(z_t)      (act = tanh or relu from recurrent_param.activation)
 *
 * Weights (pre-loaded, blobs_):
 *   blobs_[0] = W_ih (H, D)   input-to-hidden
 *   blobs_[1] = W_hh (H, H)   hidden-to-hidden
 *   blobs_[2] = b_ih (H,)
 *   blobs_[3] = b_hh (H,)
 *
 * H is inferred from the pre-loaded W_ih shape (the RecurrentParameter proto
 * has no num_output field, so trained weights must be supplied).
 */
class RNNLayer : public RecurrentLayer {
 public:
  static constexpr bool _type_mutable = true;

  explicit RNNLayer(const caffe::LayerParameter& param) : RecurrentLayer(param) {}

  const char* type() const override { return "RNN"; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.RNNLayer", RNNLayer, RecurrentLayer);

 protected:
  void LayerSetUpStep(const std::vector<Blob*>& bottom) override;
  void ReshapeStep() override;
  void ForwardStep(int t, const float* x_t, float* h_t) override;
  void BackwardStep(int t, const float* x_t, const float* h_prev,
                    const float* dy_t, float* dx_t, float* dh_next) override;

  bool tanh_ = true;                 // activation: tanh (default) or relu
  std::vector<ObjectPtr<Blob>> z_cache_;  // T blobs of (N, H) pre-activation
  ObjectPtr<Blob> dz_buf_;           // (N, H) working buffer for dz_t
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_RNN_LAYER_HPP_