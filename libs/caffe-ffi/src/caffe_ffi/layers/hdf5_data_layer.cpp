#include "caffe_ffi/layers/hdf5_data_layer.hpp"

#include <sstream>
#include <string>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layers/data_io_bridge.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

void HDF5DataLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                               const std::vector<Blob*>& top) {
  const caffe::HDF5DataParameter& param = this->layer_param_.hdf5_data_param();
  batch_size_ = static_cast<int>(param.batch_size());
  key_ = DataIOKey(type(), name());

  // The source may list one or more HDF5 files separated by whitespace.
  std::istringstream iss(param.source());
  std::string file;
  source_files_.clear();
  while (iss >> file) {
    source_files_.push_back(file);
  }
  CAFFE_FFI_LAYER_LOG << "HDF5Data LayerSetUp: key=" << key_
                      << " batch_size=" << batch_size_
                      << " num_source_files=" << source_files_.size()
                      << " shuffle=" << param.shuffle();
}

void HDF5DataLayer::Reshape(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  // Placeholder shapes; the Python/h5py callback fills the tops in place during
  // Forward. The batch dimension is the only known quantity at build time.
  const int batch = (batch_size_ > 0) ? batch_size_ : 1;
  if (top.size() >= 1) {
    top[0]->Reshape(std::vector<int64_t>{batch, 1, 1, 1});
  }
  if (top.size() >= 2) {
    top[1]->Reshape(std::vector<int64_t>{batch});
  }
  CAFFE_FFI_LAYER_LOG << "HDF5Data Reshape: batch_size=" << batch;
}

void HDF5DataLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                const std::vector<Blob*>& top) {
  if (!InvokeDataIOCallback(key_, top, true)) {
    for (Blob* t : top) {
      if (t->cpu_data() != nullptr) {
        caffe_set_fp32(static_cast<size_t>(t->count()), 0.0f, t->cpu_mutable_data());
      }
    }
    CAFFE_FFI_LAYER_LOG << "HDF5Data Forward: no callback for '" << key_
                        << "', outputting zeros";
    return;
  }
  CAFFE_FFI_LAYER_LOG << "HDF5Data Forward: invoked callback for '" << key_ << "'";
}

REGISTER_LAYER_CLASS(HDF5Data);

}  // namespace caffe_ffi