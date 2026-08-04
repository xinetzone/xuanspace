#ifndef CAFFE_FFI_LAYERS_FILTER_LAYER_HPP_
#define CAFFE_FFI_LAYERS_FILTER_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Filter: drop the batch items whose selector entry is zero.
 *
 * Bottom blobs: bottom[0..k-1] are the data blobs to filter, bottom[last] is a
 * one-dimensional selector blob. Only items whose selector value is non-zero
 * are forwarded to the corresponding top blobs.
 */
class FilterLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit FilterLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Filter"; }
  int MinBottomBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.FilterLayer", FilterLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

 private:
  std::vector<int> indices_to_forward_;
  bool first_reshape_ = true;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_FILTER_LAYER_HPP_