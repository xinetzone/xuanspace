#ifndef CAFFE_FFI_LAYERS_CONCAT_LAYER_HPP_
#define CAFFE_FFI_LAYERS_CONCAT_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

class ConcatLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit ConcatLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Concat"; }
  int MinBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.ConcatLayer", ConcatLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  int concat_axis_;
  int64_t outer_count_;
  int64_t inner_count_;
  std::vector<int64_t> concat_offsets_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_CONCAT_LAYER_HPP_
