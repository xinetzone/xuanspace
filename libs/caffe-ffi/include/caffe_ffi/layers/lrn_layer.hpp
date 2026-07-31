#ifndef CAFFE_FFI_LAYERS_LRN_LAYER_HPP_
#define CAFFE_FFI_LAYERS_LRN_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

class LRNLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit LRNLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "LRN"; }
  int ExactNumBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.LRNLayer", LRNLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

 private:
  int size_;
  int pre_pad_;
  float alpha_;
  float beta_;
  float k_;
  int num_;
  int channels_;
  int height_;
  int width_;

  ObjectPtr<Blob> scale_;
  ObjectPtr<Blob> padded_square_;
  ObjectPtr<Blob> padded_ratio_;
  ObjectPtr<Blob> accum_ratio_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_LRN_LAYER_HPP_
