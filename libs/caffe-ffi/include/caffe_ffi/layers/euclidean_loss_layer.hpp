#ifndef CAFFE_FFI_LAYERS_EUCLIDEAN_LOSS_LAYER_HPP_
#define CAFFE_FFI_LAYERS_EUCLIDEAN_LOSS_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief EuclideanLoss: L2 (mean squared, /2) regression loss.
 *
 * bottom[0]/bottom[1]: same-spatial inputs (count(1) must match).
 * loss = ||x - y||^2 / (2 * num).
 * Backward: dL/dx = diff / num, dL/dy = -diff / num.
 */
class EuclideanLossLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit EuclideanLossLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "EuclideanLoss"; }
  int ExactNumBottomBlobs() const override { return 2; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.EuclideanLossLayer", EuclideanLossLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

 private:
  ObjectPtr<Blob> diff_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_EUCLIDEAN_LOSS_LAYER_HPP_