#ifndef CAFFE_FFI_LAYERS_PARAMETER_LAYER_HPP_
#define CAFFE_FFI_LAYERS_PARAMETER_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Parameter: expose a single learnable parameter blob as a top blob.
 *
 * No bottom blobs; one top blob whose shape is taken from `parameter_param.shape`.
 * The top blob shares data/diff with the internal parameter blob (zero-copy).
 */
class ParameterLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit ParameterLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Parameter"; }
  int ExactNumBottomBlobs() const override { return 0; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.ParameterLayer", ParameterLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_PARAMETER_LAYER_HPP_