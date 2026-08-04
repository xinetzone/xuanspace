#ifndef CAFFE_FFI_LAYERS_REDUCTION_LAYER_HPP_
#define CAFFE_FFI_LAYERS_REDUCTION_LAYER_HPP_

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Reduction layer.
 *
 * Reduces the input blob along all axes starting from a given axis down to a
 * scalar per leading group, using one of SUM / ASUM / SUMSQ / MEAN operations,
 * optionally scaled by a coefficient.
 */
class ReductionLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit ReductionLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Reduction"; }
  int ExactNumBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.ReductionLayer", ReductionLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

  int64_t num_;
  int64_t dim_;
  float coeff_;
  caffe::ReductionParameter_ReductionOp operation_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_REDUCTION_LAYER_HPP_