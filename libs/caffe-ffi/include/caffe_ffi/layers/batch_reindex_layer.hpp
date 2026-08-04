#ifndef CAFFE_FFI_LAYERS_BATCH_REINDEX_LAYER_HPP_
#define CAFFE_FFI_LAYERS_BATCH_REINDEX_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief BatchReindex: reorder the batch dimension of bottom[0] according to
 *        the integer indices in bottom[1]. The output batch size equals the
 *        number of indices in bottom[1].
 *
 * bottom[0]: data blob of shape (N, D1, D2, ...).
 * bottom[1]: index blob of shape (M,) with entries in [0, N).
 * top[0]:    reordered data blob of shape (M, D1, D2, ...).
 */
class BatchReindexLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit BatchReindexLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "BatchReindex"; }
  int ExactNumBottomBlobs() const override { return 2; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.BatchReindexLayer", BatchReindexLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_BATCH_REINDEX_LAYER_HPP_