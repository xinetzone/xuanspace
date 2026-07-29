#ifndef CAFFE_FFI_LAYERS_ACCURACY_LAYER_HPP_
#define CAFFE_FFI_LAYERS_ACCURACY_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

class AccuracyLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit AccuracyLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Accuracy"; }
  int ExactNumBottomBlobs() const override { return 2; }
  int MinTopBlobs() const override { return 1; }
  int MaxTopBlobs() const override { return 2; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.AccuracyLayer", AccuracyLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  int label_axis_, outer_num_, inner_num_;
  int top_k_;
  bool has_ignore_label_;
  int ignore_label_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_ACCURACY_LAYER_HPP_
