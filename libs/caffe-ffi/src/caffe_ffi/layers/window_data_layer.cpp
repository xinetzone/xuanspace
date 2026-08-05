#include "caffe_ffi/layers/window_data_layer.hpp"

#include <string>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layers/data_io_bridge.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

void WindowDataLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                                 const std::vector<Blob*>& top) {
  const caffe::WindowDataParameter& param = this->layer_param_.window_data_param();
  batch_size_ = static_cast<int>(param.batch_size());
  fg_threshold_ = param.fg_threshold();
  bg_threshold_ = param.bg_threshold();
  key_ = DataIOKey(type(), name());
  CAFFE_FFI_LAYER_LOG << "WindowData LayerSetUp: key=" << key_
                      << " batch_size=" << batch_size_
                      << " fg_threshold=" << fg_threshold_
                      << " bg_threshold=" << bg_threshold_
                      << " source=" << param.source();
}

void WindowDataLayer::Reshape(const std::vector<Blob*>& bottom,
                              const std::vector<Blob*>& top) {
  // Placeholder shapes; the Python/numpy callback fills the pre-allocated tops
  // in place during Forward. top[0] = data, top[1] = label, extras default to
  // label-shaped blobs.
  const int batch = (batch_size_ > 0) ? batch_size_ : 1;
  if (top.size() >= 1) {
    top[0]->Reshape(std::vector<int64_t>{batch, 3, 1, 1});
  }
  for (size_t i = 1; i < top.size(); ++i) {
    top[i]->Reshape(std::vector<int64_t>{batch});
  }
  CAFFE_FFI_LAYER_LOG << "WindowData Reshape: batch_size=" << batch
                      << " tops=" << top.size();
}

void WindowDataLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                  const std::vector<Blob*>& top) {
  if (!InvokeDataIOCallback(key_, top, true)) {
    for (Blob* t : top) {
      if (t->cpu_data() != nullptr) {
        caffe_set_fp32(static_cast<size_t>(t->count()), 0.0f, t->cpu_mutable_data());
      }
    }
    CAFFE_FFI_LAYER_LOG << "WindowData Forward: no callback for '" << key_
                        << "', outputting zeros";
    return;
  }
  CAFFE_FFI_LAYER_LOG << "WindowData Forward: invoked callback for '" << key_ << "'";
}

REGISTER_LAYER_CLASS(WindowData);

}  // namespace caffe_ffi