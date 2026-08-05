#ifndef CAFFE_FFI_LAYERS_MEMORY_DATA_LAYER_HPP_
#define CAFFE_FFI_LAYERS_MEMORY_DATA_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Memory-data input layer.
 *
 * Acts as a data source with no bottom and exactly one top. The top shape is
 * fixed by the MemoryDataParameter (batch_size / channels / height / width).
 * Data is injected externally through set_data() into an internal cache Blob
 * (data_blob_); on each Forward the cache is copied (optionally scaled) into
 * the top. If no data has been injected, the cache holds zeros and the layer
 * outputs zeros.
 *
 * \note The MemoryDataParameter proto carries no scale field, so scale_ defaults
 *       to 1.0f here.
 */
class MemoryDataLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit MemoryDataLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "MemoryData"; }
  int ExactNumBottomBlobs() const override { return 0; }
  int ExactNumTopBlobs() const override { return 1; }

  /** @brief Inject data into the internal cache; copied (and scaled) out on Forward. */
  void set_data(Tensor data);

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.MemoryDataLayer", MemoryDataLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  /** @brief Build the configured (batch, channels, height, width) shape. */
  std::vector<int64_t> DataShape() const;

  ObjectPtr<Blob> data_blob_;  // internal data cache (zeros until set_data is called)
  bool has_data_ = false;      // whether set_data() has been called
  float scale_ = 1.0f;         // optional output scaling (proto has no scale field)
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_MEMORY_DATA_LAYER_HPP_