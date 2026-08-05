#ifndef CAFFE_FFI_LAYERS_PYTHON_LAYER_HPP_
#define CAFFE_FFI_LAYERS_PYTHON_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Custom Python layer bridge.
 *
 * A Python layer is driven by a Python callback registered from the Python side
 * through the FFI function `caffe_ffi.python_layer.register`. The callback is a
 * TVM FFI Function stored in a static registry keyed by "<module>.<layer>".
 * On Forward the callback is invoked with the top blobs' writable data tensors
 * (Array<Tensor>, DLPack interop) so the Python side can fill the outputs.
 *
 * If no callback is registered for the configured module/layer, the layer
 * degrades to a no-op that logs a warning and outputs zeros.
 */
class PythonLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit PythonLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Python"; }
  int MinTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.PythonLayer", PythonLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

 private:
  std::string key_;                     // "<module>.<layer>" used to look up the callback
  Function callback_;                 // registered Python callback (may be undefined)
};

/**
 * @brief Register a Python layer callback under the given key.
 * \note Called from the Python side through the FFI function
 *       `caffe_ffi.python_layer.register`. Defined in python_layer.cpp.
 */
void RegisterPythonLayerCallback(const std::string& name, Function callback);

/**
 * @brief Look up a registered Python layer callback by key.
 * \return The callback, or an undefined Function if not registered.
 */
Function LookupPythonLayerCallback(const std::string& name);

/**
 * @brief Clear all registered python_layer callbacks.
 * \note Call this from the Python side via atexit (caffe_ffi.python_layer.clear)
 *       before the Python interpreter shuts down. This releases the Python
 *       Function objects held by the static registry so they are not destroyed
 *       after Py_Finalize, which would otherwise segfault.
 */
void ClearPythonLayerCallback();

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_PYTHON_LAYER_HPP_