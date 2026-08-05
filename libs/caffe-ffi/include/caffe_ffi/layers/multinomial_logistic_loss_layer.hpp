#ifndef CAFFE_FFI_LAYERS_MULTINOMIAL_LOGISTIC_LOSS_LAYER_HPP_
#define CAFFE_FFI_LAYERS_MULTINOMIAL_LOGISTIC_LOSS_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief MultinomialLogisticLoss: multinomial logistic loss over probabilities.
 *
 * bottom[0] = predicted probabilities (N, C), bottom[1] = labels (N,).
 * Unlike SoftmaxWithLoss, the input is already a probability distribution
 * (no internal softmax). Computes the scalar loss:
 *
 *   loss = -sum_i log(p[i, gt_i]) / N
 *
 * The scalar loss output is normalized by the LossParameter normalization mode.
 * Labels never receive gradients.
 *
 * Backward (per sample, with loss_weight L and normalizer Z):
 *   dL/dp[i, k] = -L / (p[i, gt_i] * Z)  for k == gt_i, else 0.
 */
class MultinomialLogisticLossLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit MultinomialLogisticLossLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "MultinomialLogisticLoss"; }
  int ExactNumBottomBlobs() const override { return 2; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.MultinomialLogisticLossLayer",
                                    MultinomialLogisticLossLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

  int outer_num_, inner_num_;
  int channels_;
  bool has_ignore_label_;
  int ignore_label_;
  int normalization_;
  float normalizer_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_MULTINOMIAL_LOGISTIC_LOSS_LAYER_HPP_