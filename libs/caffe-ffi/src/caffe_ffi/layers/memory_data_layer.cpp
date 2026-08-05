#include "caffe_ffi/layers/memory_data_layer.hpp"

#include <sstream>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

std::vector<int64_t> MemoryDataLayer::DataShape() const {
  const caffe::MemoryDataParameter& param = this->layer_param_.memory_data_param();
  return {static_cast<int64_t>(param.batch_size()),
          static_cast<int64_t>(param.channels()),
          static_cast<int64_t>(param.height()),
          static_cast<int64_t>(param.width())};
}

void MemoryDataLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                                 const std::vector<Blob*>& top) {
  const caffe::MemoryDataParameter& param = this->layer_param_.memory_data_param();
  CAFFE_FFI_CHECK_VALUE_GT(param.batch_size(), 0) << "MemoryData batch_size must be > 0";
  CAFFE_FFI_CHECK_VALUE_GT(param.channels(), 0) << "MemoryData channels must be > 0";
  CAFFE_FFI_CHECK_VALUE_GT(param.height(), 0) << "MemoryData height must be > 0";
  CAFFE_FFI_CHECK_VALUE_GT(param.width(), 0) << "MemoryData width must be > 0";

  // Create the internal data cache shaped by the configured dims. It is
  // zero-initialized until set_data() injects external data.
  this->data_blob_ = make_object<Blob>();
  this->data_blob_->Reshape(DataShape());
  this->has_data_ = false;

  CAFFE_FFI_LAYER_LOG << "MemoryData LayerSetUp: batch=" << param.batch_size()
                      << " channels=" << param.channels()
                      << " height=" << param.height()
                      << " width=" << param.width()
                      << " count=" << this->data_blob_->count();
}

void MemoryDataLayer::Reshape(const std::vector<Blob*>& bottom,
                              const std::vector<Blob*>& top) {
  top[0]->Reshape(DataShape());
  CAFFE_FFI_LAYER_LOG << "MemoryData Reshape: top=(" << top[0]->shape(0) << ","
                      << top[0]->shape(1) << "," << top[0]->shape(2) << ","
                      << top[0]->shape(3) << ")";
}

void MemoryDataLayer::set_data(Tensor data) {
  CAFFE_FFI_CHECK_TYPE(data.defined()) << "MemoryData set_data: data tensor must be defined";
  if (!this->data_blob_) {
    this->data_blob_ = make_object<Blob>();
  }
  this->data_blob_->Reshape(data.shape());
  this->data_blob_->set_data(data);
  this->has_data_ = true;
  CAFFE_FFI_LAYER_LOG << "MemoryData set_data: count=" << this->data_blob_->count()
                      << " has_data=" << this->has_data_;
}

void MemoryDataLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                  const std::vector<Blob*>& top) {
  const int64_t count = top[0]->count();
  float* top_data = top[0]->cpu_mutable_data();
  if (this->has_data_ && this->data_blob_ && this->data_blob_->cpu_data() != nullptr) {
    caffe_copy_fp32(static_cast<size_t>(count), this->data_blob_->cpu_data(), top_data);
    if (this->scale_ != 1.0f) {
      caffe_scal_fp32(static_cast<size_t>(count), this->scale_, top_data);
    }
    CAFFE_FFI_LAYER_LOG << "MemoryData Forward: copied " << count << " elements"
                        << " (scale=" << this->scale_ << ")";
  } else {
    // No data injected yet: output zeros.
    caffe_set_fp32(static_cast<size_t>(count), 0.0f, top_data);
    CAFFE_FFI_LAYER_LOG << "MemoryData Forward: no data set, outputting zeros";
  }
}

REGISTER_LAYER_CLASS(MemoryData);

}  // namespace caffe_ffi