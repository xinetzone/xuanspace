#ifndef CAFFE_FFI_LAYERS_BIAS_LAYER_HPP_
#define CAFFE_FFI_LAYERS_BIAS_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

class BiasLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit BiasLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Bias"; }
  int MinBottomBlobs() const override { return 1; }
  int MaxBottomBlobs() const override { return 2; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.BiasLayer", BiasLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  int axis_;
  int num_axes_;
  int outer_dim_, bias_dim_, inner_dim_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_BIAS_LAYER_HPP_
