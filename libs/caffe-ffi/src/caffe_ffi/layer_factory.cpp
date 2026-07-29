#include "caffe_ffi/layer_factory.hpp"

#include <sstream>
#include <string>
#include <utility>

namespace caffe_ffi {

LayerRegistry::CreatorRegistry& LayerRegistry::Registry() {
  static CreatorRegistry* g_registry_ = new CreatorRegistry();
  return *g_registry_;
}

void LayerRegistry::AddCreator(const std::string& type, Creator creator) {
  CreatorRegistry& registry = Registry();
  TVM_FFI_ICHECK_EQ(registry.count(type), 0)
      << "Layer type " << type << " already registered.";
  registry[type] = std::move(creator);
}

ObjectPtr<Layer> LayerRegistry::CreateLayer(const caffe::LayerParameter& param) {
  const std::string& type = param.type();
  CreatorRegistry& registry = Registry();
  TVM_FFI_ICHECK_EQ(registry.count(type), 1)
      << "Unknown layer type: " << type << " (known types: " << LayerTypeListString() << ")";
  return registry[type](param);
}

std::vector<std::string> LayerRegistry::LayerTypeList() {
  CreatorRegistry& registry = Registry();
  std::vector<std::string> layer_types;
  layer_types.reserve(registry.size());
  for (const auto& kv : registry) {
    layer_types.push_back(kv.first);
  }
  return layer_types;
}

std::string LayerRegistry::LayerTypeListString() {
  std::vector<std::string> layer_types = LayerTypeList();
  std::string result;
  for (size_t i = 0; i < layer_types.size(); ++i) {
    if (i > 0) result += ", ";
    result += layer_types[i];
  }
  return result;
}

}  // namespace caffe_ffi
