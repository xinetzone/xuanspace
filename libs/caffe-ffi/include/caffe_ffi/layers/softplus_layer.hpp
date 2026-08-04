#ifndef CAFFE_FFI_LAYERS_SOFTPLUS_LAYER_HPP_
#define CAFFE_FFI_LAYERS_SOFTPLUS_LAYER_HPP_

#include <vector>

#include "caffe_ffi/layers/neuron_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Softplus activation layer: y = log(1 + exp(x)).
 *
 * Element-wise, parameterless, infinitely smooth (C∞, no kink anywhere).
 * Forward uses a numerically stable branch (for x > 0 compute
 * x + log1p(exp(-x)) to avoid overflow); the derivative is the logistic
 * sigmoid 1 / (1 + exp(-x)). Follows the standard NeuronLayer pattern.
 */
class SoftplusLayer : public NeuronLayer {
 public:
  static constexpr bool _type_mutable = true;

  explicit SoftplusLayer(const caffe::LayerParameter& param) : NeuronLayer(param) {}

  const char* type() const override { return "Softplus"; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.SoftplusLayer", SoftplusLayer, NeuronLayer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_SOFTPLUS_LAYER_HPP_