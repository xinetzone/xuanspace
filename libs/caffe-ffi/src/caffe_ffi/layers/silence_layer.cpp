#include "caffe_ffi/layers/silence_layer.hpp"

#include <vector>

#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void SilenceLayer::Reshape(const std::vector<Blob*>& bottom,
                           const std::vector<Blob*>& top) {
  // No top blobs; nothing to reshape.
  CAFFE_FFI_LAYER_LOG << "Silence Reshape: bottom.size()=" << bottom.size();
}

void SilenceLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                               const std::vector<Blob*>& top) {
  // No-op: consume the bottom blobs and produce nothing.
  CAFFE_FFI_LAYER_LOG << "Silence Forward_cpu: no-op (suppressing "
                      << bottom.size() << " bottom blobs)";
}

void SilenceLayer::Backward_cpu(const std::vector<Blob*>& top,
                                const std::vector<bool>& propagate_down,
                                const std::vector<Blob*>& bottom) {
  for (size_t i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      caffe_set_fp32(static_cast<size_t>(bottom[i]->count()), 0.0f,
                     bottom[i]->cpu_mutable_diff());
    }
  }
  CAFFE_FFI_LAYER_LOG << "Silence Backward_cpu: zeroed gradients of "
                      << bottom.size() << " bottom blobs";
}

REGISTER_LAYER_CLASS(Silence);

}  // namespace caffe_ffi