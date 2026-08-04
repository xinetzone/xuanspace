#ifndef CAFFE_FFI_LAYERS_HINGE_LAYER_HPP_
#define CAFFE_FFI_LAYERS_HINGE_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Hinge loss layer (multi-class margin ranking loss).
 *
 * Two bottoms: bottom[0] = class scores, bottom[1] = integer labels (sparse,
 * channel dim of label is 1 along `axis`). Computes the average hinge loss
 * over all samples:
 *
 *   For each sample (i, j) with truth label `y`:
 *     z_c = max(0, 1 + score_c - score_y)   for all c != y
 *     loss_i = sum_c z_c            (L1 norm)
 *     loss_i = sum_c z_c^2          (L2 norm)
 *
 *   scalar loss = mean_i loss_i
 *
 * The scalar loss output is the mean over all samples. Backward (with
 * loss_weight L and N = sample count):
 *   dX_c    += L * (1/N) * [ z_c > 0 ? 1 : 0 ]            (L1)
 *   dX_c    += L * (1/N) * [ z_c > 0 ? 2*z_c : 0 ]        (L2)
 *   dX_y    -=  sum_{c != y} dX_c
 * The label bottom never receives gradients.
 */
class HingeLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit HingeLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Hinge"; }
  int ExactNumBottomBlobs() const override { return 2; }
  int MinTopBlobs() const override { return 1; }
  int MaxTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.HingeLayer", HingeLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

  bool is_l2_;  // true => L2 norm, false => L1 norm
  int axis_;
  int outer_num_;
  int inner_num_;
  int count_;  // outer_num_ * inner_num_ (number of samples)
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_HINGE_LAYER_HPP_