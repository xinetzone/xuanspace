#include "caffe_ffi/layers/python_layer.hpp"

#include <string>
#include <unordered_map>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

namespace {

// Static registry of Python callbacks keyed by "<module>.<layer>".
std::unordered_map<std::string, Function>& PythonCallbackRegistry() {
  static std::unordered_map<std::string, Function> registry;
  return registry;
}

}  // namespace

void RegisterPythonLayerCallback(const std::string& name, Function callback) {
  if (name.empty()) {
    CAFFE_FFI_THROW(ValueError) << "python_layer.register: key must not be empty";
  }
  if (!callback.defined()) {
    CAFFE_FFI_THROW(ValueError) << "python_layer.register: callback must be defined";
  }
  PythonCallbackRegistry()[name] = std::move(callback);
  CAFFE_FFI_LAYER_LOG << "Registered Python layer callback for key '" << name << "'";
}

Function LookupPythonLayerCallback(const std::string& name) {
  auto it = PythonCallbackRegistry().find(name);
  if (it == PythonCallbackRegistry().end()) {
    return Function();
  }
  return it->second;
}

void ClearPythonLayerCallback() {
  PythonCallbackRegistry().clear();
  CAFFE_FFI_LAYER_LOG << "Cleared all python_layer callbacks (registry size -> 0)";
}

void PythonLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                             const std::vector<Blob*>& top) {
  const caffe::PythonParameter& param = this->layer_param_.python_param();
  const std::string module = param.module();
  const std::string layer = param.layer();
  key_ = module + "." + layer;

  callback_ = LookupPythonLayerCallback(key_);
  if (!callback_.defined()) {
    CAFFE_FFI_LOG_WARN() << "PythonLayer '" << this->name()
                         << "': no Python callback registered for '" << key_
                         << "', degrading to no-op (outputs zeros)";
  }
  CAFFE_FFI_LAYER_LOG << "Python LayerSetUp: module=" << module << " layer=" << layer
                      << " key=" << key_;
}

void PythonLayer::Reshape(const std::vector<Blob*>& bottom,
                          const std::vector<Blob*>& top) {
  // Output shapes for a Python layer are owned by the Python-side callback
  // (via DLPack). With no callback registered the layer stays a no-op; the
  // Forward no-op path guards against unallocated tops.
  CAFFE_FFI_LAYER_LOG << "Python Reshape: key=" << key_
                      << " bottoms=" << bottom.size() << " tops=" << top.size();
}

void PythonLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                              const std::vector<Blob*>& top) {
  if (!callback_.defined()) {
    // No Python callback: degrade to a no-op that writes zeros where possible.
    for (Blob* t : top) {
      if (t->cpu_data() != nullptr) {
        caffe_set_fp32(static_cast<size_t>(t->count()), 0.0f, t->cpu_mutable_data());
      }
    }
    CAFFE_FFI_LAYER_LOG << "Python Forward: no callback for '" << key_
                        << "', outputting zeros";
    return;
  }

  // Collect the top blobs' writable data tensors (DLPack) for the Python side.
  Array<Tensor> top_tensors;
  top_tensors.reserve(top.size());
  for (Blob* t : top) {
    top_tensors.push_back(t->mutable_data_tensor());
  }
  callback_(top_tensors);
  CAFFE_FFI_LAYER_LOG << "Python Forward: invoked callback for '" << key_ << "'";
}

void PythonLayer::Backward_cpu(const std::vector<Blob*>& top,
                               const std::vector<bool>& propagate_down,
                               const std::vector<Blob*>& bottom) {
  // Gradient computation for a Python layer is delegated to the Python side in
  // a full implementation; the minimal bridge treats backward as a no-op.
  CAFFE_FFI_LAYER_LOG << "Python Backward_cpu: no-op (key='" << key_ << "')";
}

REGISTER_LAYER_CLASS(Python);

}  // namespace caffe_ffi