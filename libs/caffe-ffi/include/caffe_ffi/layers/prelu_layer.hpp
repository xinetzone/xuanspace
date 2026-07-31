#ifndef CAFFE_FFI_LAYERS_PRELU_LAYER_HPP_
#define CAFFE_FFI_LAYERS_PRELU_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layers/neuron_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

class PReLULayer : public NeuronLayer {
 public:
  static constexpr bool _type_mutable = true;

  explicit PReLULayer(const caffe::LayerParameter& param) : NeuronLayer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "PReLU"; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.PReLULayer", PReLULayer, NeuronLayer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

  bool channel_shared_;
  int64_t channels_;
  int64_t inner_dim_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_PRELU_LAYER_HPP_
