#ifndef CAFFE_FFI_LAYERS_SPP_LAYER_HPP_
#define CAFFE_FFI_LAYERS_SPP_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief SPP: spatial pyramid pooling.
 *
 * Input: (N, C, H, W). For each pyramid level p in [0, pyramid_height),
 * the input is pooled into a (num_bins, num_bins) grid where num_bins = 2^p.
 * Each level's pooled map is flattened to C * num_bins^2 features; all levels
 * are concatenated producing an output of shape (N, C * sum(2^(2p)), 1, 1).
 *
 * Pooling semantics match BVLC Caffe's PoolingLayer: MAX takes the max over the
 * kernel window; AVE divides the sum (including padded zeros) by kernel_h*kernel_w.
 */
class SPPLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit SPPLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "SPP"; }
  int ExactNumBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.SPPLayer", SPPLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

 private:
  int pyramid_height_ = 1;
  int num_ = 0;
  int channels_ = 0;
  int bottom_h_ = 0;
  int bottom_w_ = 0;
  int64_t total_channels_ = 0;
  bool is_max_pool_ = true;
  // Per-level (kernel_h, kernel_w, pad_h, pad_w, num_bins) cache.
  std::vector<int> kernel_h_, kernel_w_, pad_h_, pad_w_, num_bins_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_SPP_LAYER_HPP_