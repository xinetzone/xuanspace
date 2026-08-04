#ifndef CAFFE_FFI_LAYERS_LEAKY_RELU_LAYER_HPP_
#define CAFFE_FFI_LAYERS_LEAKY_RELU_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layers/neuron_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief LeakyReLU activation layer: y = max(x,0) + negative_slope * min(x,0).
 *
 * Element-wise, single-parameter (negative_slope), inheriting the standard
 * NeuronLayer Reshape logic. For negative_slope != 1 the function is C¹
 * continuous but C² discontinuous at x = 0 (a kink), so numerical-gradient
 * tests must push points away from the kink via `avoid_c1_discontinuity`.
 */
class LeakyReLULayer : public NeuronLayer {
 public:
  static constexpr bool _type_mutable = true;

  explicit LeakyReLULayer(const caffe::LayerParameter& param) : NeuronLayer(param) {}

  const char* type() const override { return "LeakyReLU"; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.LeakyReLULayer", LeakyReLULayer, NeuronLayer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_LEAKY_RELU_LAYER_HPP_