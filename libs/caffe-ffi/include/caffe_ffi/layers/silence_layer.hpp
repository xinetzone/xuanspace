#ifndef CAFFE_FFI_LAYERS_SILENCE_LAYER_HPP_
#define CAFFE_FFI_LAYERS_SILENCE_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Silence: consume bottom blobs while producing no top blobs.
 *
 * Used to suppress the outputs of intermediate layers during testing. The
 * forward pass is a no-op; the backward pass zeroes the gradients of the
 * bottom blobs so that no gradient flows through this branch.
 */
class SilenceLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit SilenceLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Silence"; }
  int MinBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 0; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.SilenceLayer", SilenceLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_SILENCE_LAYER_HPP_