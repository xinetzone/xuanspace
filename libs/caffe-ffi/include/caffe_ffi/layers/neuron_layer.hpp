#ifndef CAFFE_FFI_LAYERS_NEURON_LAYER_HPP_
#define CAFFE_FFI_LAYERS_NEURON_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Abstract base class for element-wise neuron activation layers.
 *
 * NeuronLayer factors out the common Reshape logic for layers that take one blob
 * as input and produce one equally-sized blob as output, where each element of
 * the output depends only on the corresponding input element (ReLU, Sigmoid,
 * TanH, ELU, PReLU, etc.).
 *
 * Subclasses only need to implement Forward_cpu and optionally Backward_cpu;
 * Reshape is provided here (top[0]->ReshapeLike(*bottom[0])).
 */
class NeuronLayer : public Layer {
 public:
  explicit NeuronLayer(const caffe::LayerParameter& param) : Layer(param) {}

  void Reshape(const std::vector<Blob*>& bottom,
               const std::vector<Blob*>& top) override;

  int ExactNumBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 1; }

  static constexpr int _type_child_slots = 9;
  TVM_FFI_DECLARE_OBJECT_INFO("caffe_ffi.NeuronLayer", NeuronLayer, Layer);
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_NEURON_LAYER_HPP_
