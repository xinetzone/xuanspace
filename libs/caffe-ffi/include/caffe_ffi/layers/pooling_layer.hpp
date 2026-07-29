#ifndef CAFFE_FFI_LAYERS_POOLING_LAYER_HPP_
#define CAFFE_FFI_LAYERS_POOLING_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

class PoolingLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit PoolingLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Pooling"; }
  int ExactNumBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.PoolingLayer", PoolingLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int pad_h_, pad_w_;
  int channels_;
  int height_, width_;
  int pooled_height_, pooled_width_;
  bool global_pooling_;
  caffe::PoolingParameter::PoolMethod pool_method_;
  caffe::PoolingParameter::RoundMode round_mode_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_POOLING_LAYER_HPP_
