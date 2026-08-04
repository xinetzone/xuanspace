#ifndef CAFFE_FFI_LAYERS_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_
#define CAFFE_FFI_LAYERS_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief SigmoidCrossEntropyLoss: multi-label binary classification loss.
 *
 * bottom[0]: logits (any shape). bottom[1]: targets (same count, values in [0,1]).
 * Loss is the numerically stable sigmoid cross-entropy, normalized per
 * LossParameter.normalization (FULL/VALID/BATCH_SIZE/NONE).
 * Only bottom[0] receives gradients (the sigmoid diff minus target).
 */
class SigmoidCrossEntropyLossLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit SigmoidCrossEntropyLossLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "SigmoidCrossEntropyLoss"; }
  int ExactNumBottomBlobs() const override { return 2; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.SigmoidCrossEntropyLossLayer",
                                    SigmoidCrossEntropyLossLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

 private:
  int outer_num_ = 0;
  int inner_num_ = 0;
  bool has_ignore_label_ = false;
  int ignore_label_ = 0;
  caffe::LossParameter_NormalizationMode normalization_ =
      caffe::LossParameter_NormalizationMode_VALID;
  float normalizer_ = 1.0f;
  ObjectPtr<Blob> sigmoid_output_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_