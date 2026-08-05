#ifndef CAFFE_FFI_LAYERS_UPSAMPLE_LAYER_HPP_
#define CAFFE_FFI_LAYERS_UPSAMPLE_LAYER_HPP_

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Nearest-neighbor upsampling layer.
 *
 * Takes a bottom of shape (N, C, H, W) and produces a top of shape
 * (N, C, H*scale, W*scale) by replicating each spatial element into a
 * scale x scale block (nearest-neighbor).
 */
class UpsampleLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit UpsampleLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Upsample"; }
  int ExactNumBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.UpsampleLayer", UpsampleLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

  int scale_ = 2;  // spatial upsampling factor
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_UPSAMPLE_LAYER_HPP_