#ifndef CAFFE_FFI_LAYERS_SPLIT_LAYER_HPP_
#define CAFFE_FFI_LAYERS_SPLIT_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Split layer: copies a single bottom blob to N top blobs.
 *
 * Split creates N copies of the input blob, enabling multi-branch/skip-connection
 * topologies where one blob is consumed by multiple downstream layers (since
 * Net::AppendBottom enforces single-consumer semantics by erasing blobs from
 * available_blobs after first use).
 *
 * When N=1 (single top), Split acts as an identity passthrough but still performs
 * a memcpy (following Caffe original behavior; zero-copy is deferred as Future Work).
 *
 * Performance logging: Forward_cpu records per-copy timing, total bytes moved,
 * and throughput via a dedicated [SPLIT-PERF] tag at WARN level (always visible
 * in Release builds) plus detailed DEBUG-level breakdown.
 */
class SplitLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit SplitLayer(const caffe::LayerParameter& param) : Layer(param) {}
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
