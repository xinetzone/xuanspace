#ifndef CAFFE_FFI_LAYERS_SLICE_LAYER_HPP_
#define CAFFE_FFI_LAYERS_SLICE_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

class SliceLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit SliceLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Slice"; }
  int ExactNumBottomBlobs() const override { return 1; }
  int MinTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.SliceLayer", SliceLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

  int64_t count_;
  int64_t num_slices_;
  int64_t slice_size_;
  int slice_axis_;
  std::vector<int64_t> slice_point_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_SLICE_LAYER_HPP_
