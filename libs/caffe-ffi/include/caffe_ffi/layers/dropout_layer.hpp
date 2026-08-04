#ifndef CAFFE_FFI_LAYERS_DROPOUT_LAYER_HPP_
#define CAFFE_FFI_LAYERS_DROPOUT_LAYER_HPP_

#include <random>
#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

class DropoutLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit DropoutLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Dropout"; }
  int ExactNumBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.DropoutLayer", DropoutLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

 private:
  float ratio_ = 0.5f;   // dropout ratio (fraction of elements dropped)
  float scale_ = 2.0f;   // inverted-dropout scale = 1 / (1 - ratio)
  ObjectPtr<Blob> mask_; // cached Bernoulli mask (0/1) reused in backward
  std::mt19937 rng_{12345};  // deterministic per-layer RNG for reproducible masks
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_DROPOUT_LAYER_HPP_
