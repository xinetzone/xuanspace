#ifndef CAFFE_FFI_LAYERS_L2_NORM_LAYER_HPP_
#define CAFFE_FFI_LAYERS_L2_NORM_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief L2-normalization layer: y = x / ||x||_2.
 *
 * Normalizes each group of elements (all dimensions at and after `axis`) to
 * unit L2 norm. The forward is continuous and smooth everywhere (no kink), so
 * numerical-gradient tests can use the standard tolerance (rtol=1e-3). The
 * norm is computed as sqrt(sum(x^2) + eps) for numerical stability.
 *
 * Forward:   y[i] = x[i] / norm(g),  norm(g) = sqrt(sum_{j in g} x[j]^2 + eps)
 * Backward:  dx[i] = dy[i] / norm(g)
 *                   - x[i] * (sum_{j in g} dy[j]*x[j]) / norm(g)^3
 */
class L2NormLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit L2NormLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "L2Norm"; }
  int ExactNumBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.L2NormLayer", L2NormLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

  int axis_;
  float eps_;
  int outer_dim_;
  int inner_dim_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_L2_NORM_LAYER_HPP_