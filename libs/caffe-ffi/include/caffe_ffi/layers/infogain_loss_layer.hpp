#ifndef CAFFE_FFI_LAYERS_INFOGAIN_LOSS_LAYER_HPP_
#define CAFFE_FFI_LAYERS_INFOGAIN_LOSS_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief InfogainLoss: information-gain (weighted) classification loss.
 *
 * bottom[0] = predicted scores (N, C), bottom[1] = labels (N,), and an
 * optional bottom[2] = infogain matrix H (C, C). Softmax is applied internally
 * to bottom[0] to obtain probabilities p. The scalar loss is:
 *
 *   case 1 (no H, identity matrix): loss_i = -log(p[gt_i])
 *   case 2 (H provided):            loss_i = -sum_k H[gt_i, k] * log(p[k])
 *
 * The scalar loss output is normalized by the LossParameter normalization mode.
 * Labels never receive gradients.
 *
 * Backward per sample (softmax Jacobian), with H_row_sum = sum_k H[gt_i, k]:
 *   dL/dx_j = p_j * H_row_sum - H[gt_i, j]
 * which reduces to p_j - delta_{gt_i,j} for the identity (case 1) matrix.
 */
class InfogainLossLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit InfogainLossLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "InfogainLoss"; }
  int MinBottomBlobs() const override { return 2; }
  int MaxBottomBlobs() const override { return 3; }
  int MinTopBlobs() const override { return 1; }
  int MaxTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.InfogainLossLayer", InfogainLossLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

  int softmax_axis_;
  int outer_num_, inner_num_;
  int channels_;
  ObjectPtr<Blob> sum_multiplier_;
  ObjectPtr<Blob> scale_;
  ObjectPtr<Blob> prob_;
  bool has_ignore_label_;
  int ignore_label_;
  bool has_infogain_;
  int normalization_;
  float normalizer_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_INFOGAIN_LOSS_LAYER_HPP_