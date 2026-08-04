#ifndef CAFFE_FFI_LAYERS_SOFTSIGN_LAYER_HPP_
#define CAFFE_FFI_LAYERS_SOFTSIGN_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layers/neuron_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Softsign activation layer: y = x / (1 + |x|).
 *
 * Element-wise, parameterless, C1-smooth (no kink at x = 0; the first
 * derivative is continuous, though C2 is not). Follows the standard
 * NeuronLayer pattern: only Forward_cpu / Backward_cpu are implemented.
 */
class SoftsignLayer : public NeuronLayer {
 public:
  static constexpr bool _type_mutable = true;

  explicit SoftsignLayer(const caffe::LayerParameter& param) : NeuronLayer(param) {}

  const char* type() const override { return "Softsign"; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.SoftsignLayer", SoftsignLayer, NeuronLayer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_SOFTSIGN_LAYER_HPP_