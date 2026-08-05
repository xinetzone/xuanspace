#ifndef CAFFE_FFI_LAYERS_HDF5_DATA_LAYER_HPP_
#define CAFFE_FFI_LAYERS_HDF5_DATA_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief HDF5Data input layer (HDF5 files via the Python/h5py bridge).
 *
 * No bottom; exactly two tops (data + label). Data is loaded through a
 * Python-side data-source callback registered via `caffe_ffi.data_io.register`
 * under the key "<type>.<layer_name>". The source field may list one or more
 * whitespace-separated HDF5 file paths. On Forward the callback writes the
 * batch into the tops' mutable data tensors.
 */
class HDF5DataLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit HDF5DataLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "HDF5Data"; }
  int ExactNumBottomBlobs() const override { return 0; }
  int ExactNumTopBlobs() const override { return 2; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.HDF5DataLayer", HDF5DataLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

 private:
  std::string key_;                      // "<type>.<name>" bridge registration key
  int batch_size_ = 0;
  std::vector<std::string> source_files_;  // one or more HDF5 file paths
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_HDF5_DATA_LAYER_HPP_