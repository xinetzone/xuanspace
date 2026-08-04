#ifndef CAFFE_FFI_LAYERS_TILE_LAYER_HPP_
#define CAFFE_FFI_LAYERS_TILE_LAYER_HPP_

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Tile layer.
 *
 * Repeats the input blob `tiles` times along a given axis, replicating each
 * element group (Caffe's Tile layer).
 */
class TileLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit TileLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Tile"; }
  int ExactNumBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.TileLayer", TileLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

  int axis_;
  int tiles_;
  int64_t outer_dim_;
  int64_t inner_dim_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_TILE_LAYER_HPP_