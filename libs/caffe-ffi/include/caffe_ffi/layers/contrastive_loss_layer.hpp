#ifndef CAFFE_FFI_LAYERS_CONTRASTIVE_LOSS_LAYER_HPP_
#define CAFFE_FFI_LAYERS_CONTRASTIVE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief ContrastiveLoss: pairwise contrastive (metric) loss.
 *
 * Three bottoms: bottom[0] = feature vector 1 (N, D), bottom[1] = feature
 * vector 2 (N, D), bottom[2] = label (N,) with values in {0, 1} (1 = similar,
 * 0 = dissimilar). Computes the scalar loss:
 *
 *   diff_i = x0_i - x1_i          (over D dims)
 *   d²_i   = ||diff_i||²
 *   loss_i = y_i * d²_i + (1 - y_i) * max(margin - d²_i, 0)²
 *   loss   = sum_i loss_i / N
 *
 * The scalar loss output is normalized by the LossParameter normalization mode.
 * Labels never receive gradients.
 *
 * Backward (per sample, with loss_weight L and dist_weight Dw = 1/normalizer):
 *   y_i = 1:            dx0_i = 2 * diff_i * L * Dw, dx1_i = -2 * diff_i * L * Dw
 *   y_i = 0, d²_i<margin: dx0_i = -2*(margin-d²_i)*diff_i*L*Dw,
 *                         dx1_i =  2*(margin-d²_i)*diff_i*L*Dw
 */
class ContrastiveLossLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit ContrastiveLossLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "ContrastiveLoss"; }
  int ExactNumBottomBlobs() const override { return 3; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.ContrastiveLossLayer", ContrastiveLossLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

  float margin_;
  bool legacy_version_;
  int normalization_;
  int num_;
  int dim_;
  float normalizer_;
  float dist_weight_;
  ObjectPtr<Blob> diff_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_CONTRASTIVE_LOSS_LAYER_HPP_