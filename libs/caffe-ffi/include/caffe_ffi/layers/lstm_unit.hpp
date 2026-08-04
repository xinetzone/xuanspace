#ifndef CAFFE_FFI_LAYERS_LSTM_UNIT_HPP_
#define CAFFE_FFI_LAYERS_LSTM_UNIT_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Single-step LSTM cell (standalone layer, unrolled usage).
 *
 * This layer implements one LSTM cell update given the four gate
 * pre-activations, matching the numpy lstm_cell reference:
 *
 *   bottom[0] = gate pre-activations (N, 4H) ordered [i, f, o, g]
 *   bottom[1] = previous cell state c_{t-1} (N, H)
 *
 *   i_t = sigmoid(z_i), f_t = sigmoid(z_f), o_t = sigmoid(z_o), g_t = tanh(z_g)
 *   c_t = f_t * c_{t-1} + i_t * g_t
 *   h_t = o_t * tanh(c_t)
 *
 *   top[0] = c_t (N, H)
 *   top[1] = h_t (N, H)
 *
 * The layer has no learnable parameters; gradients flow to bottom[0] (gates)
 * and bottom[1] (c_{t-1}).
 */
class LSTMUnitLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit LSTMUnitLayer(const caffe::LayerParameter& param) : Layer(param) {}

  void LayerSetUp(const std::vector<Blob*>& bottom,
                  const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom,
               const std::vector<Blob*>& top) override;

  const char* type() const override { return "LSTMUnit"; }
  int ExactNumBottomBlobs() const override { return 2; }
  int ExactNumTopBlobs() const override { return 2; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.LSTMUnitLayer", LSTMUnitLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom,
                   const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

  int N_ = 0;   // batch size
  int H_ = 0;   // hidden dimension
  // Per-cell cache (N, 6H) = [i, f, o, g, c, tanh(c)].
  ObjectPtr<Blob> cache_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_LSTM_UNIT_HPP_