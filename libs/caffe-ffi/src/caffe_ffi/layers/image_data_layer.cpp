#include "caffe_ffi/layers/image_data_layer.hpp"

#include <string>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layers/data_io_bridge.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

void ImageDataLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                                const std::vector<Blob*>& top) {
  const caffe::ImageDataParameter& param = this->layer_param_.image_data_param();
  batch_size_ = static_cast<int>(param.batch_size());
  channels_ = param.is_color() ? 3 : 1;
  height_ = static_cast<int>(param.new_height());
  width_ = static_cast<int>(param.new_width());
  scale_ = param.scale();
  key_ = DataIOKey(type(), name());
  CAFFE_FFI_LAYER_LOG << "ImageData LayerSetUp: key=" << key_
                      << " batch_size=" << batch_size_
                      << " channels=" << channels_
                      << " new_height=" << height_ << " new_width=" << width_
                      << " scale=" << scale_ << " source=" << param.source();
}

void ImageDataLayer::Reshape(const std::vector<Blob*>& bottom,
                             const std::vector<Blob*>& top) {
  const int batch = (batch_size_ > 0) ? batch_size_ : 1;
  const int h = (height_ > 0) ? height_ : 1;
  const int w = (width_ > 0) ? width_ : 1;
  if (top.size() >= 1) {
    top[0]->Reshape(std::vector<int64_t>{batch, channels_, h, w});
  }
  if (top.size() >= 2) {
    top[1]->Reshape(std::vector<int64_t>{batch});
  }
  CAFFE_FFI_LAYER_LOG << "ImageData Reshape: shape=(" << batch << ", " << channels_
                      << ", " << h << ", " << w << ")";
}

void ImageDataLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                 const std::vector<Blob*>& top) {
  if (!InvokeDataIOCallback(key_, top, true)) {
    for (Blob* t : top) {
      if (t->cpu_data() != nullptr) {
        caffe_set_fp32(static_cast<size_t>(t->count()), 0.0f, t->cpu_mutable_data());
      }
    }
    CAFFE_FFI_LAYER_LOG << "ImageData Forward: no callback for '" << key_
                        << "', outputting zeros";
    return;
  }
  CAFFE_FFI_LAYER_LOG << "ImageData Forward: invoked callback for '" << key_ << "'";
}

REGISTER_LAYER_CLASS(ImageData);

}  // namespace caffe_ffi