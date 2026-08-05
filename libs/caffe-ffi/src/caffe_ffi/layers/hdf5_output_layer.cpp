#include "caffe_ffi/layers/hdf5_output_layer.hpp"

#include <string>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/layers/data_io_bridge.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

void HDF5OutputLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                                 const std::vector<Blob*>& top) {
  const caffe::HDF5OutputParameter& param = this->layer_param_.hdf5_output_param();
  file_name_ = param.file_name();
  key_ = DataIOKey(type(), name());
  CAFFE_FFI_LAYER_LOG << "HDF5Output LayerSetUp: key=" << key_
                      << " file_name=" << file_name_;
}

void HDF5OutputLayer::Reshape(const std::vector<Blob*>& bottom,
                              const std::vector<Blob*>& top) {
  // No tops; output shapes are driven purely by the bottom blobs.
  CAFFE_FFI_LAYER_LOG << "HDF5Output Reshape: bottoms=" << bottom.size();
}

void HDF5OutputLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                  const std::vector<Blob*>& top) {
  // Read-only view of the bottoms is handed to the Python/h5py callback for
  // persistence. With no callback registered the write is skipped.
  if (!InvokeDataIOCallback(key_, bottom, false)) {
    CAFFE_FFI_LAYER_LOG << "HDF5Output Forward: no callback for '" << key_
                        << "', skipping write";
    return;
  }
  CAFFE_FFI_LAYER_LOG << "HDF5Output Forward: invoked callback for '" << key_ << "'";
}

REGISTER_LAYER_CLASS(HDF5Output);

}  // namespace caffe_ffi