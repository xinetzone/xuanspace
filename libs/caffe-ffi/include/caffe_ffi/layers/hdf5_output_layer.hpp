#ifndef CAFFE_FFI_LAYERS_HDF5_OUTPUT_LAYER_HPP_
#define CAFFE_FFI_LAYERS_HDF5_OUTPUT_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief HDF5Output layer (writes bottom blobs to an HDF5 file via Python/h5py).
 *
 * Two bottoms (data + label); no tops. On Forward the bottom blobs' data
 * tensors are handed to a Python-side callback registered via
 * `caffe_ffi.data_io.register` under the key "<type>.<layer_name>", which
 * persists them to the configured HDF5 file.
 */
class HDF5OutputLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit HDF5OutputLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "HDF5Output"; }
  int ExactNumBottomBlobs() const override { return 2; }
  int ExactNumTopBlobs() const override { return 0; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.HDF5OutputLayer", HDF5OutputLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

 private:
  std::string key_;        // "<type>.<name>" bridge registration key
  std::string file_name_;  // destination HDF5 file path
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_HDF5_OUTPUT_LAYER_HPP_