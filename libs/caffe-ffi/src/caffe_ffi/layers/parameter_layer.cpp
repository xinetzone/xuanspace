#include "caffe_ffi/layers/parameter_layer.hpp"

#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

void ParameterLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                                const std::vector<Blob*>& top) {
  const caffe::BlobShape& shape = this->layer_param_.parameter_param().shape();
  if (this->blobs_.size() > 0) {
    CAFFE_FFI_LAYER_LOG << "Parameter: using pre-loaded parameter blob";
  } else {
    this->blobs_.resize(1);
    this->blobs_[0] = make_object<Blob>();
    this->blobs_[0]->Reshape(shape);
  }
  top[0]->Reshape(shape);
  // The parameter is learnable (trainable).
  this->param_propagate_down_.resize(1, true);
  CAFFE_FFI_LAYER_LOG << "Parameter LayerSetUp: param blob count="
                      << this->blobs_[0]->count();
}

void ParameterLayer::Reshape(const std::vector<Blob*>& bottom,
                             const std::vector<Blob*>& top) {
  // Parameter layer has no bottom; shape is fixed by the parameter blob.
  top[0]->Reshape(this->blobs_[0]->shape());
}

void ParameterLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                 const std::vector<Blob*>& top) {
  top[0]->ShareData(this->blobs_[0].get());
  top[0]->ShareDiff(this->blobs_[0].get());
  CAFFE_FFI_LAYER_LOG << "Parameter Forward_cpu: shared data/diff with param blob";
}

void ParameterLayer::Backward_cpu(const std::vector<Blob*>& top,
                                  const std::vector<bool>& propagate_down,
                                  const std::vector<Blob*>& bottom) {
  // No bottom blobs; the parameter gradient flows through the shared diff.
  CAFFE_FFI_LAYER_LOG << "Parameter Backward_cpu: no-op (shared diff updated by consumers)";
}

REGISTER_LAYER_CLASS(Parameter);

}  // namespace caffe_ffi