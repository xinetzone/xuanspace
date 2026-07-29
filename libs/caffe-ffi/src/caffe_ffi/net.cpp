#include "caffe_ffi/net.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>

#include <tvm/ffi/memory.h>
#include <google/protobuf/text_format.h>

#include "caffe_ffi/error.hpp"
#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

Net::Net(const caffe::NetParameter& param) {
  CAFFE_FFI_NET_LOG << "Net(NetParameter): name='" << param.name() << "' layers=" << param.layer_size() << " inputs=" << param.input_size();
  Init(param);
}

Net::Net(const std::string& param_file) {
  CAFFE_FFI_NET_LOG << "Net(file): loading prototxt from '" << param_file << "'";
  caffe::NetParameter param = ReadNetParamsFromTextFile(param_file);
  CAFFE_FFI_NET_LOG << "Net(file): parsed prototxt, name='" << param.name() << "' layers=" << param.layer_size();
  Init(param);
}

void Net::Init(const caffe::NetParameter& param) {
  name_ = param.name();
  CAFFE_FFI_NET_LOG << "Init: starting network '" << name_ << "' initialization";
  std::set<std::string> available_blobs;
  std::map<std::string, int> blob_name_to_idx;
  int num_layers = param.layer_size();
  layers_.resize(num_layers);
  layer_names_.resize(num_layers);
  bottom_vecs_.resize(num_layers);
  top_vecs_.resize(num_layers);
  blob_names_.clear();
  blobs_.clear();
  blob_names_index_.clear();
  net_input_blobs_.clear();
  net_output_blobs_.clear();
  net_input_blob_indices_.clear();
  net_output_blob_indices_.clear();
  net_input_blob_names_.clear();
  net_output_blob_names_.clear();

  int num_inputs = param.input_size();
  for (int i = 0; i < num_inputs; ++i) {
    const std::string& blob_name = param.input(i);
    blob_name_to_idx[blob_name] = static_cast<int>(blobs_.size());
    blob_names_.push_back(blob_name);
    blob_names_index_[blob_name] = static_cast<int>(blobs_.size());
    blobs_.push_back(make_object<Blob>());
    blobs_.back()->set_name(blob_name);
    available_blobs.insert(blob_name);
    net_input_blobs_.push_back(blobs_.back().get());
    net_input_blob_indices_.push_back(static_cast<int>(blobs_.size()) - 1);
    net_input_blob_names_.push_back(blob_name);
    if (param.input_shape_size() > i) {
      blobs_.back()->Reshape(param.input_shape(i));
    } else if (param.input_dim_size() > 0) {
      int num_dims = param.input_dim_size() / num_inputs;
      std::vector<int64_t> dims(num_dims);
      for (int d = 0; d < num_dims; ++d) {
        dims[d] = param.input_dim(i * num_dims + d);
      }
      blobs_.back()->Reshape(dims);
    }
  }

  for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
    const caffe::LayerParameter& layer_param = param.layer(layer_id);
    layers_[layer_id] = LayerRegistry::CreateLayer(layer_param);
    layer_names_[layer_id] = layer_param.name();
    layer_names_index_[layer_param.name()] = layer_id;

    int num_bottom = layer_param.bottom_size();
    bottom_vecs_[layer_id].resize(num_bottom);
    for (int bottom_id = 0; bottom_id < num_bottom; ++bottom_id) {
      AppendBottom(param, layer_id, bottom_id, &available_blobs, &blob_name_to_idx);
    }

    int num_top = layer_param.top_size();
    top_vecs_[layer_id].resize(num_top);
    for (int top_id = 0; top_id < num_top; ++top_id) {
      AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
      if (num_inputs == 0 && layer_id == 0) {
        net_input_blobs_.push_back(blobs_[blob_name_to_idx[layer_param.top(top_id)]].get());
        net_input_blob_indices_.push_back(blob_name_to_idx[layer_param.top(top_id)]);
        net_input_blob_names_.push_back(layer_param.top(top_id));
      }
    }

    layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);

    auto& layer_blobs = layers_[layer_id]->blobs();
    for (int param_id = 0; param_id < layer_param.param_size(); ++param_id) {
      if (param_id < static_cast<int>(layer_blobs.size())) {
        layer_blobs[param_id]->set_name(layer_param.param(param_id).name());
      }
    }
  }

  for (const auto& blob_name : available_blobs) {
    int blob_idx = blob_name_to_idx[blob_name];
    net_output_blobs_.push_back(blobs_[blob_idx].get());
    net_output_blob_indices_.push_back(blob_idx);
    net_output_blob_names_.push_back(blob_name);
  }
  CAFFE_FFI_NET_LOG << "Init: network '" << name_ << "' initialized: "
                    << layers_.size() << " layers, " << blobs_.size() << " blobs, "
                    << net_input_blobs_.size() << " inputs, " << net_output_blobs_.size() << " outputs";
}

void Net::AppendTop(const caffe::NetParameter& param, int layer_id,
                    int top_id, std::set<std::string>* available_blobs,
                    std::map<std::string, int>* blob_name_to_idx) {
  const caffe::LayerParameter& layer_param = param.layer(layer_id);
  const std::string& blob_name = layer_param.top(top_id);
  CAFFE_FFI_NET_LOG << "AppendTop: layer[" << layer_id << "]='" << layer_param.name()
                    << "', top[" << top_id << "]='" << blob_name << "'";
  CAFFE_FFI_CHECK_RUNTIME(available_blobs->find(blob_name) == available_blobs->end())
      << "Top blob '" << blob_name << "' produced by multiple sources.";
  if (blob_name_to_idx->find(blob_name) == blob_name_to_idx->end()) {
    CAFFE_FFI_NET_LOG << "AppendTop: creating new blob '" << blob_name << "' at index " << blobs_.size();
    (*blob_name_to_idx)[blob_name] = static_cast<int>(blobs_.size());
    blob_names_.push_back(blob_name);
    blob_names_index_[blob_name] = static_cast<int>(blobs_.size());
    blobs_.push_back(make_object<Blob>());
    blobs_.back()->set_name(blob_name);
  }
  int blob_idx = (*blob_name_to_idx)[blob_name];
  top_vecs_[layer_id][top_id] = blobs_[blob_idx].get();
  available_blobs->insert(blob_name);
  CAFFE_FFI_NET_LOG << "AppendTop: blob '" << blob_name << "' (idx=" << blob_idx << ") now available (in-place)";
}

int Net::AppendBottom(const caffe::NetParameter& param, int layer_id,
                      int bottom_id, std::set<std::string>* available_blobs,
                      std::map<std::string, int>* blob_name_to_idx) {
  const caffe::LayerParameter& layer_param = param.layer(layer_id);
  const std::string& blob_name = layer_param.bottom(bottom_id);
  CAFFE_FFI_NET_LOG << "AppendBottom: layer[" << layer_id << "]='" << layer_param.name()
                    << "', bottom[" << bottom_id << "]='" << blob_name << "'";
  CAFFE_FFI_CHECK_KEY(available_blobs->find(blob_name) != available_blobs->end())
      << "Unknown bottom blob '" << blob_name << "' (layer '" << layer_param.name()
      << "', bottom index " << bottom_id << ")";
  int blob_idx = (*blob_name_to_idx)[blob_name];
  bottom_vecs_[layer_id][bottom_id] = blobs_[blob_idx].get();
  available_blobs->erase(blob_name);
  CAFFE_FFI_NET_LOG << "AppendBottom: blob '" << blob_name << "' (idx=" << blob_idx << ") consumed by layer";
  return blob_idx;
}

Map<String, ObjectPtr<Blob>> Net::Forward(const Map<String, Tensor>& inputs) {
  CAFFE_FFI_NET_LOG << "Forward: input map size=" << inputs.size();
  for (const auto& kv : inputs) {
    const std::string& name = kv.first;
    const Tensor& data = kv.second;
    CAFFE_FFI_NET_LOG << "Forward: feeding input '" << name << "' with "
                      << data.numel() << " elements (ndim=" << data.ndim() << ")";
    if (has_blob(name)) {
      ObjectPtr<Blob> blob = blob_by_name(name);
      const int64_t data_size = data.numel();

      CAFFE_FFI_CHECK_TYPE(data.defined()) << "Input tensor '" << name << "' is undefined";
      CAFFE_FFI_CHECK_TYPE(data.dtype().code == kDLFloat && data.dtype().bits == 32)
          << "Forward input '" << name << "' expects float32 Tensor, got dtype code="
          << static_cast<int>(data.dtype().code) << " bits=" << data.dtype().bits;

      // Reshape blob if needed (uninitialized or shape mismatch)
      bool need_reshape = (blob->count() == 0);
      if (!need_reshape && blob->count() != data_size) {
        need_reshape = true;
      }
      // Also check ndim matches
      if (!need_reshape && blob->num_axes() != data.ndim()) {
        need_reshape = true;
      }
      for (int i = 0; !need_reshape && i < data.ndim(); ++i) {
        if (blob->shape(i) != data.size(i)) {
          need_reshape = true;
          break;
        }
      }

      if (need_reshape) {
        std::vector<int64_t> shape(data.ndim());
        for (int i = 0; i < data.ndim(); ++i) {
          shape[i] = data.size(i);
        }
        CAFFE_FFI_NET_LOG << "Forward: reshaping input blob '" << name << "' to "
                          << [&]() {
                               std::ostringstream oss;
                               oss << "(";
                               for (size_t i = 0; i < shape.size(); ++i) {
                                 if (i > 0) oss << ",";
                                 oss << shape[i];
                               }
                               oss << ")";
                               return oss.str();
                             }();
        blob->Reshape(shape);
      }

      float* dst = blob->cpu_data();
      const float* src = static_cast<const float*>(data.data_ptr());
      int64_t nbytes = data_size * sizeof(float);
      CAFFE_FFI_TENSOR_LOG << "Forward: memcpy " << data_size << " input elements ("
                           << nbytes << "B) to blob " << name << " at " << dst;
      std::memcpy(dst, src, nbytes);
    } else {
      CAFFE_FFI_CHECK_KEY(has_blob(name))
          << "Forward: input blob '" << name << "' not found in network '" << name_ << "'. "
          << "Available blobs: " << [&]() {
               std::ostringstream oss;
               for (size_t i = 0; i < blob_names_.size(); ++i) {
                 if (i > 0) oss << ", ";
                 oss << "'" << blob_names_[i] << "'";
               }
               return oss.str();
             }();
    }
  }
  CAFFE_FFI_NET_LOG << "Forward: starting ForwardFromTo(0, " << (layers_.size() - 1) << ")";
  float loss = ForwardFromTo(0, static_cast<int>(layers_.size()) - 1);
  CAFFE_FFI_NET_LOG << "Forward: completed, total_loss=" << loss;
  Map<String, ObjectPtr<Blob>> outputs;
  for (size_t i = 0; i < net_output_blobs_.size(); ++i) {
    int blob_idx = net_output_blob_indices_[i];
    outputs.Set(String(net_output_blob_names_[i]), blobs_[blob_idx]);
  }
  return outputs;
}

float Net::ForwardFromTo(int start, int end) {
  CAFFE_FFI_CHECK_INDEX_GE(start, 0);
  CAFFE_FFI_CHECK_INDEX_LT(end, static_cast<int>(layers_.size()));
  CAFFE_FFI_NET_LOG << "ForwardFromTo: layers[" << start << ".." << end << "]";
  float total_loss = 0.0f;
  for (int i = start; i <= end; ++i) {
    CAFFE_FFI_LAYER_LOG << "ForwardFromTo: >>> layer[" << i << "] '" << layer_names_[i] << "' forward";
    float layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    CAFFE_FFI_LAYER_LOG << "ForwardFromTo: <<< layer[" << i << "] '" << layer_names_[i] << "' loss=" << layer_loss;
    total_loss += layer_loss;
  }
  CAFFE_FFI_NET_LOG << "ForwardFromTo: completed, total_loss=" << total_loss;
  return total_loss;
}

Array<String> Net::layer_names_array() const {
  Array<String> result;
  for (const auto& name : layer_names_) {
    result.push_back(String(name));
  }
  return result;
}

Array<String> Net::blob_names_array() const {
  Array<String> result;
  for (const auto& name : blob_names_) {
    result.push_back(String(name));
  }
  return result;
}

Array<String> Net::input_blob_names_array() const {
  Array<String> result;
  for (const auto& name : net_input_blob_names_) {
    result.push_back(String(name));
  }
  return result;
}

Array<String> Net::output_blob_names_array() const {
  Array<String> result;
  for (const auto& name : net_output_blob_names_) {
    result.push_back(String(name));
  }
  return result;
}

Array<ObjectPtr<Blob>> Net::blobs_array() const {
  Array<ObjectPtr<Blob>> result;
  for (const auto& blob : blobs_) {
    result.push_back(blob);
  }
  return result;
}

Array<ObjectPtr<Layer>> Net::layers_array() const {
  Array<ObjectPtr<Layer>> result;
  for (const auto& layer : layers_) {
    result.push_back(layer);
  }
  return result;
}

Array<ObjectPtr<Blob>> Net::input_blobs_array() const {
  Array<ObjectPtr<Blob>> result;
  for (int blob_idx : net_input_blob_indices_) {
    result.push_back(blobs_[blob_idx]);
  }
  return result;
}

Array<ObjectPtr<Blob>> Net::output_blobs_array() const {
  Array<ObjectPtr<Blob>> result;
  for (int blob_idx : net_output_blob_indices_) {
    result.push_back(blobs_[blob_idx]);
  }
  return result;
}

ObjectPtr<Blob> Net::blob_by_name(const std::string& blob_name) const {
  auto it = blob_names_index_.find(blob_name);
  CAFFE_FFI_CHECK_KEY(it != blob_names_index_.end()) << "Unknown blob: " << blob_name;
  return blobs_[it->second];
}

ObjectPtr<Layer> Net::layer_by_name(const std::string& layer_name) const {
  auto it = layer_names_index_.find(layer_name);
  CAFFE_FFI_CHECK_KEY(it != layer_names_index_.end()) << "Unknown layer: " << layer_name;
  return layers_[it->second];
}

bool Net::has_blob(const std::string& blob_name) const {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

bool Net::has_layer(const std::string& layer_name) const {
  return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

void Net::CopyTrainedLayersFrom(const std::string& trained_filename) {
  caffe::NetParameter trained_net_param = ReadNetParamsFromBinaryFile(trained_filename);
  CopyTrainedLayersFrom(trained_net_param);
}

void Net::CopyTrainedLayersFrom(const caffe::NetParameter& trained_net_param) {
  std::map<std::string, int> trained_layer_names_index;
  for (int i = 0; i < trained_net_param.layer_size(); ++i) {
    trained_layer_names_index[trained_net_param.layer(i).name()] = i;
  }

  for (int i = 0; i < static_cast<int>(layers_.size()); ++i) {
    const std::string& layer_name = layer_names_[i];
    auto it = trained_layer_names_index.find(layer_name);
    if (it == trained_layer_names_index.end()) {
      continue;
    }
    const caffe::LayerParameter& source_layer = trained_net_param.layer(it->second);
    Layer* target_layer = layers_[i].get();
    auto& target_blobs = target_layer->blobs();
    int num_target_blobs = static_cast<int>(target_blobs.size());
    int num_source_blobs = source_layer.blobs_size();
    int num_blobs_to_copy = std::min(num_target_blobs, num_source_blobs);
    for (int j = 0; j < num_blobs_to_copy; ++j) {
      target_blobs[j]->FromProto(source_layer.blobs(j), true);
    }
  }
}

caffe::NetParameter ReadNetParamsFromTextFile(const std::string& filename) {
  std::ifstream ifs(filename);
  CAFFE_FFI_CHECK_RUNTIME(ifs.is_open()) << "Could not open file: " << filename;
  std::stringstream ss;
  ss << ifs.rdbuf();
  return ReadNetParamsFromTextString(ss.str());
}

caffe::NetParameter ReadNetParamsFromTextString(const std::string& text) {
  caffe::NetParameter param;
  bool success = google::protobuf::TextFormat::ParseFromString(text, &param);
  CAFFE_FFI_CHECK_RUNTIME(success) << "Failed to parse NetParameter from text string";
  return param;
}

caffe::NetParameter ReadNetParamsFromBinaryFile(const std::string& filename) {
  std::ifstream ifs(filename, std::ios::binary);
  CAFFE_FFI_CHECK_RUNTIME(ifs.is_open()) << "Failed to open binary file: " << filename;
  caffe::NetParameter param;
  CAFFE_FFI_CHECK_RUNTIME(param.ParseFromIstream(&ifs)) << "Failed to parse binary file: " << filename;
  return param;
}

}  // namespace caffe_ffi
