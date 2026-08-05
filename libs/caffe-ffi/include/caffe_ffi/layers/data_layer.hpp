#ifndef CAFFE_FFI_LAYERS_DATA_LAYER_HPP_
#define CAFFE_FFI_LAYERS_DATA_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Data input layer (LMDB/LevelDB source via the Python/numpy bridge).
 *
 * No bottom; exactly two tops (data + label). Data is loaded through a
 * Python-side data-source callback registered via `caffe_ffi.data_io.register`
 * under the key "<type>.<layer_name>". On Forward the callback writes the batch
 * into the tops' mutable data tensors. Without a callback the layer degrades to
 * a no-op that outputs zeros.
 */
class DataLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit DataLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Data"; }
  int ExactNumBottomBlobs() const override { return 0; }
  int ExactNumTopBlobs() const override { return 2; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.DataLayer", DataLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

 private:
  std::string key_;  // "<type>.<name>" used to look up the data-source callback
  int batch_size_ = 0;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_DATA_LAYER_HPP_