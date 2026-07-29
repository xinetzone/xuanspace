#ifndef CAFFE_FFI_LAYERS_ELU_LAYER_HPP_
#define CAFFE_FFI_LAYERS_ELU_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

class ELULayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit ELULayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "ELU"; }
  int ExactNumBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.ELULayer", ELULayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  float alpha_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_ELU_LAYER_HPP_
