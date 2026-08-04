#ifndef CAFFE_FFI_LAYERS_INSTANCE_NORM_LAYER_HPP_
#define CAFFE_FFI_LAYERS_INSTANCE_NORM_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Instance normalization layer (per-instance, per-channel mean/var).
 *
 * Normalizes each (n, c) plane independently using its own mean and variance
 * computed over the spatial dimensions. Optionally applies a learnable affine
 * transform (per-channel gamma scale + beta shift) when `affine` is true.
 *
 * Forward:  y = (x - mean) / sqrt(var + eps),  [* gamma + beta if affine]
 * Backward: standard batchnorm gradient per (n, c) group, plus dgamma/dbeta
 *           when affine is enabled.
 */
class InstanceNormLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit InstanceNormLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "InstanceNorm"; }
  int ExactNumBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.InstanceNormLayer", InstanceNormLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

  float eps_;
  bool affine_;
  bool use_global_stats_;
  int num_;
  int channels_;
  int spatial_dim_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_INSTANCE_NORM_LAYER_HPP_