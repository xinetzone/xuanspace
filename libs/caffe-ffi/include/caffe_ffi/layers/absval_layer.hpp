#ifndef CAFFE_FFI_LAYERS_ABSVAL_LAYER_HPP_
#define CAFFE_FFI_LAYERS_ABSVAL_LAYER_HPP_

#include <vector>

#include "caffe_ffi/layers/neuron_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Absolute value activation layer: y = |x|.
 *
 * Element-wise, parameterless, inheriting the standard NeuronLayer Reshape
 * logic. The forward is continuous and piecewise-identical to x for x>0 and
 * -x for x<0. The derivative dy/dx = sign(x) is discontinuous at x = 0 (a C¹
 * kink), so numerical-gradient tests must push points away from the kink via
 * `avoid_c1_discontinuity`. Backward follows Caffe semantics: dx = dy * (x > 0
 * ? 1 : -1), i.e. at x = 0 the gradient is routed to the negative branch.
 */
class AbsValLayer : public NeuronLayer {
 public:
  static constexpr bool _type_mutable = true;

  explicit AbsValLayer(const caffe::LayerParameter& param) : NeuronLayer(param) {}

  const char* type() const override { return "AbsVal"; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.AbsValLayer", AbsValLayer, NeuronLayer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_ABSVAL_LAYER_HPP_