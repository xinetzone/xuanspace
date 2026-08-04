#ifndef CAFFE_FFI_LAYERS_MVN_LAYER_HPP_
#define CAFFE_FFI_LAYERS_MVN_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Mean-variance normalization layer.
 *
 * Normalizes each group of `dim` elements (either per-image or per-channel
 * plane depending on across_channels) to zero mean and optionally unit variance.
 */
class MVNLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit MVNLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "MVN"; }
  int ExactNumBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.MVNLayer", MVNLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

  bool normalize_variance_;
  bool across_channels_;
  float eps_;
  std::vector<float> std_;  // per-group std (sqrt(variance) + eps), cached during forward
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_MVN_LAYER_HPP_