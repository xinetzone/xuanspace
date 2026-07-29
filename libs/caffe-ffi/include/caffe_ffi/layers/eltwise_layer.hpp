#ifndef CAFFE_FFI_LAYERS_ELTWISE_LAYER_HPP_
#define CAFFE_FFI_LAYERS_ELTWISE_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

class EltwiseLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit EltwiseLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Eltwise"; }
  int MinBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.EltwiseLayer", EltwiseLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  enum EltwiseOp { PROD = 0, SUM = 1, MAX = 2 };
  EltwiseOp op_;
  std::vector<float> coeffs_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_ELTWISE_LAYER_HPP_
