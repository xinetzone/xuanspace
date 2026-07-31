#ifndef CAFFE_FFI_LAYERS_TANH_LAYER_HPP_
#define CAFFE_FFI_LAYERS_TANH_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layers/neuron_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

class TanHLayer : public NeuronLayer {
 public:
  static constexpr bool _type_mutable = true;

  explicit TanHLayer(const caffe::LayerParameter& param) : NeuronLayer(param) {}

  const char* type() const override { return "TanH"; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.TanHLayer", TanHLayer, NeuronLayer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom,
                   const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_TANH_LAYER_HPP_
