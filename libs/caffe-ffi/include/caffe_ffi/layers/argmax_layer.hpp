#ifndef CAFFE_FFI_LAYERS_ARGMAX_LAYER_HPP_
#define CAFFE_FFI_LAYERS_ARGMAX_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief ArgMax: return the indices (and optionally values) of the top_k
 *        largest entries along a given axis (default axis=1, i.e. channel dim).
 *
 * Matches the BVLC Caffe ArgMax counter-part: with `out_max_val` it outputs
 * both indices and values; without it, only indices. When `axis` is not set,
 * the max is computed over the flattened per-instance blob (count(1)).
 */
class ArgMaxLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit ArgMaxLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "ArgMax"; }
  int ExactNumBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.ArgMaxLayer", ArgMaxLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

 private:
  bool out_max_val_ = true;
  int top_k_ = 1;
  bool has_axis_ = false;
  int axis_ = 1;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_ARGMAX_LAYER_HPP_