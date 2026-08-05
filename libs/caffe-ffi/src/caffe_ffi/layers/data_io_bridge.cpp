#include "caffe_ffi/layers/data_io_bridge.hpp"

#include <string>
#include <unordered_map>
#include <vector>

#include "caffe_ffi/error.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

namespace {

// Static registry of data-source/output callbacks keyed by "<layer_type>.<name>".
std::unordered_map<std::string, Function>& DataIOCallbackRegistry() {
  static std::unordered_map<std::string, Function> registry;
  return registry;
}

}  // namespace

void RegisterDataIOCallback(const std::string& key, Function callback) {
  if (key.empty()) {
    CAFFE_FFI_THROW(ValueError) << "data_io.register: key must not be empty";
  }
  if (!callback.defined()) {
    CAFFE_FFI_THROW(ValueError) << "data_io.register: callback must be defined";
  }
  DataIOCallbackRegistry()[key] = std::move(callback);
  CAFFE_FFI_LAYER_LOG << "Registered data-io callback for key '" << key << "'";
}

Function LookupDataIOCallback(const std::string& key) {
  auto it = DataIOCallbackRegistry().find(key);
  if (it == DataIOCallbackRegistry().end()) {
    return Function();
  }
  return it->second;
}

void ClearDataIOCallback() {
  DataIOCallbackRegistry().clear();
  CAFFE_FFI_LAYER_LOG << "Cleared all data-io callbacks (registry size -> 0)";
}

bool InvokeDataIOCallback(const std::string& key, const std::vector<Blob*>& blobs,
                          bool writable) {
  Function callback = LookupDataIOCallback(key);
  if (!callback.defined()) {
    return false;
  }
  Array<Tensor> tensors;
  tensors.reserve(blobs.size());
  for (Blob* b : blobs) {
    tensors.push_back(writable ? b->mutable_data_tensor() : b->data_tensor());
  }
  callback(tensors);
  return true;
}

}  // namespace caffe_ffi