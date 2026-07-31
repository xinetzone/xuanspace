#ifndef CAFFE_FFI_LAYERS_SPLIT_LAYER_HPP_
#define CAFFE_FFI_LAYERS_SPLIT_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Split layer: fans out a single bottom blob to N top blobs.
 *
 * Split creates N views of the input blob, enabling multi-branch/skip-connection
 * topologies where one blob is consumed by multiple downstream layers (since
 * Net::AppendBottom enforces single-consumer semantics by erasing blobs from
 * available_blobs after first use).
 *
 * Zero-copy optimization:
 * - N=1: Direct ShareData/ShareDiff (Phase 1), no memcpy at all.
 * - N>=2: All tops share bottom's data/diff via intrusive refcount (Phase 2 COW).
 *   Actual copies are deferred to cpu_mutable_data()/cpu_mutable_diff() on first
 *   write, triggered by the Copy-on-Write mechanism in Blob.
 *
 * Performance logging: Forward_cpu records sharing timing, total bytes saved,
 * and COW refcount info via a dedicated [SPLIT-PERF] tag at WARN level.
 */
class SplitLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit SplitLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Split"; }
  int ExactNumBottomBlobs() const override { return 1; }
  int MinTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.SplitLayer", SplitLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_SPLIT_LAYER_HPP_
