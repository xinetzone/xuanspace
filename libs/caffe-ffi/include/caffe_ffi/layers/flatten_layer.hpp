#ifndef CAFFE_FFI_LAYERS_FLATTEN_LAYER_HPP_
#define CAFFE_FFI_LAYERS_FLATTEN_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

class FlattenLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit FlattenLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Flatten"; }
  int ExactNumBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.FlattenLayer", FlattenLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_FLATTEN_LAYER_HPP_
