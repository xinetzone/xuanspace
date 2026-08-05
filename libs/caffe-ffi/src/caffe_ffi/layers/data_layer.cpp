#include "caffe_ffi/layers/data_layer.hpp"

#include <string>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layers/data_io_bridge.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

void DataLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                           const std::vector<Blob*>& top) {
  const caffe::DataParameter& param = this->layer_param_.data_param();
  batch_size_ = static_cast<int>(param.batch_size());
  key_ = DataIOKey(type(), name());
  CAFFE_FFI_LAYER_LOG << "Data LayerSetUp: key=" << key_
                      << " batch_size=" << batch_size_
                      << " source=" << param.source();
}

void DataLayer::Reshape(const std::vector<Blob*>& bottom,
                        const std::vector<Blob*>& top) {
  // Placeholder shapes: the Python data-source callback is responsible for
  // filling the pre-allocated tops in place during Forward. If the DP has no
  // explicit spatial shape, the batch dimension is the only known quantity.
  const int batch = (batch_size_ > 0) ? batch_size_ : 1;
  if (top.size() >= 1) {
    top[0]->Reshape(std::vector<int64_t>{batch, 3, 1, 1});
  }
  if (top.size() >= 2) {
    top[1]->Reshape(std::vector<int64_t>{batch});
  }
  CAFFE_FFI_LAYER_LOG << "Data Reshape: batch_size=" << batch;
}

void DataLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  if (!InvokeDataIOCallback(key_, top, true)) {
    for (Blob* t : top) {
      if (t->cpu_data() != nullptr) {
        caffe_set_fp32(static_cast<size_t>(t->count()), 0.0f, t->cpu_mutable_data());
      }
    }
    CAFFE_FFI_LAYER_LOG << "Data Forward: no callback for '" << key_
                        << "', outputting zeros";
    return;
  }
  CAFFE_FFI_LAYER_LOG << "Data Forward: invoked callback for '" << key_ << "'";
}

REGISTER_LAYER_CLASS(Data);

}  // namespace caffe_ffi