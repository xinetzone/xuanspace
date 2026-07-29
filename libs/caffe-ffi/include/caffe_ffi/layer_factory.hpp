#ifndef CAFFE_FFI_LAYER_FACTORY_HPP_
#define CAFFE_FFI_LAYER_FACTORY_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <tvm/ffi/error.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/memory.h>
#include "caffe_ffi/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

class Layer;

class LayerRegistry {
 public:
  using Creator = ObjectPtr<Layer> (*)(const caffe::LayerParameter&);
  using CreatorRegistry = std::unordered_map<std::string, Creator>;

  /*!
   * \brief Get the global layer registry singleton.
   * \note Defined in layer_factory.cpp to ensure a single instance
   *       across DLL boundaries on Windows.
   */
  static CreatorRegistry& Registry();

  static void AddCreator(const std::string& type, Creator creator);

  static ObjectPtr<Layer> CreateLayer(const caffe::LayerParameter& param);

  static std::vector<std::string> LayerTypeList();

 private:
  LayerRegistry() = default;

  static std::string LayerTypeListString();
};

/*!
 * \brief Register a layer class with the factory.
 * \note Uses TVM_FFI_STATIC_INIT_BLOCK for auto-registration, but the
 *       registry functions themselves are defined in layer_factory.cpp
 *       to avoid Windows DLL boundary issues with inline function-local statics.
 */
#define REGISTER_LAYER_CLASS(type)                                                         \
  namespace {                                                                              \
  ObjectPtr<Layer> Creator_##type##Layer(const caffe::LayerParameter& param) {             \
    return make_object<type##Layer>(param);                                                \
  }                                                                                        \
  TVM_FFI_STATIC_INIT_BLOCK() {                                                            \
    ::caffe_ffi::LayerRegistry::AddCreator(#type, Creator_##type##Layer);                  \
  }                                                                                        \
  }  // namespace

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYER_FACTORY_HPP_
