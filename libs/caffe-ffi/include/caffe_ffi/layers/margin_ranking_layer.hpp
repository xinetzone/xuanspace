#ifndef CAFFE_FFI_LAYERS_MARGIN_RANKING_LAYER_HPP_
#define CAFFE_FFI_LAYERS_MARGIN_RANKING_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Pairwise margin ranking loss layer.
 *
 * Three bottoms: x1 (first input), x2 (second input), label (target, values
 * in {-1, +1}). Computes the element-wise hinge-style ranking loss:
 *
 *   loss_i = max(0, -y_i * (x1_i - x2_i) + margin)
 *
 * The scalar loss output is the mean over all elements, optionally scaled by
 * `sign` (1 or -1). The label bottom never receives gradients.
 *
 * Backward (per element, with loss_weight L and N elements):
 *   dx1_i = L * sign * (-y_i * mask_i) / N
 *   dx2_i = L * sign * ( y_i * mask_i) / N
 *   where mask_i = 1 if loss_i > 0 else 0.
 */
class MarginRankingLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit MarginRankingLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "MarginRanking"; }
  int ExactNumBottomBlobs() const override { return 3; }
  int MinTopBlobs() const override { return 1; }
  int MaxTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.MarginRankingLayer", MarginRankingLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

  float margin_;
  int sign_;
  int count_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_MARGIN_RANKING_LAYER_HPP_