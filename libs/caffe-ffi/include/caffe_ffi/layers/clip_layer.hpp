#ifndef CAFFE_FFI_LAYERS_CLIP_LAYER_HPP_
#define CAFFE_FFI_LAYERS_CLIP_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layers/neuron_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

class ClipLayer : public NeuronLayer {
 public:
  static constexpr bool _type_mutable = true;

  explicit ClipLayer(const caffe::LayerParameter& param) : NeuronLayer(param) {}

  const char* type() const override { return "Clip"; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.ClipLayer", ClipLayer, NeuronLayer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_CLIP_LAYER_HPP_