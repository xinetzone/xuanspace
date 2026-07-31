#ifndef CAFFE_FFI_LAYERS_CROP_LAYER_HPP_
#define CAFFE_FFI_LAYERS_CROP_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

class CropLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit CropLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Crop"; }
  int ExactNumBottomBlobs() const override { return 2; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.CropLayer", CropLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

 private:
  std::vector<int64_t> offsets_;
  std::vector<int64_t> src_strides_;
  std::vector<int64_t> dest_strides_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_CROP_LAYER_HPP_
