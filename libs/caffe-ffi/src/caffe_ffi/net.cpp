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

// ======================================================================
// InsertSplits: Graph transformation pass that inserts explicit Split
// layers for blobs consumed by multiple layers (fan-out > 1).
//
// This is a port of native Caffe's InsertSplits() (caffe/util/insert_splits.cpp),
// adapted for caffe-ffi's NetParameter format (which supports both
// legacy param.input() inputs and Input-type layers).
//
// Without this pass, a blob consumed by multiple layers causes
// "Unknown bottom blob" errors because AppendBottom erases blobs from
// available_blobs after first consumption.
//
// Algorithm (two-pass):
//   Pass 1: Build data structures mapping bottom references to their
//           producing (layer, top_idx), and count how many times each
//           top is consumed. External inputs (param.input()) are tracked
//           with a virtual producer index (-1, input_idx).
//   Pass 2: For each layer in order:
//           - Rewrite bottoms that come from multi-consumer tops to point
//             to the corresponding split output name.
//           - After processing each layer's tops, if a top is consumed
//             by >1 downstream layers/losses, insert a Split layer.
//
// Split layer naming: <blob_name>_<producer_name>_<top_idx>_split
// Split output naming: <blob_name>_<producer_name>_<top_idx>_split_<k>
// ======================================================================

namespace {

// Generate split layer name matching native Caffe convention.
std::string SplitLayerName(const std::string& producer_name,
                           const std::string& blob_name,
                           int blob_idx) {
  std::ostringstream oss;
  oss << blob_name << "_" << producer_name << "_" << blob_idx << "_split";
  return oss.str();
}

// Generate split blob (top) name matching native Caffe convention.
std::string SplitBlobName(const std::string& producer_name,
                          const std::string& blob_name,
                          int blob_idx,
                          int split_idx) {
  std::ostringstream oss;
  oss << blob_name << "_" << producer_name << "_" << blob_idx
      << "_split_" << split_idx;
  return oss.str();
}

void ConfigureSplitLayer(const std::string& producer_name,
                         const std::string& blob_name,
                         int blob_idx, int split_count, float loss_weight,
                         caffe::LayerParameter* split_layer_param,
                         int log_level = 0) {
  split_layer_param->Clear();
  split_layer_param->add_bottom(blob_name);
  split_layer_param->set_name(SplitLayerName(producer_name, blob_name, blob_idx));
  split_layer_param->set_type("Split");
  for (int k = 0; k < split_count; ++k) {
    split_layer_param->add_top(
        SplitBlobName(producer_name, blob_name, blob_idx, k));
    if (loss_weight != 0.0f) {
      split_layer_param->add_loss_weight(k == 0 ? loss_weight : 0.0f);
    }
  }
  if (log_level >= 2) {
    CAFFE_FFI_SPLIT_LOG << "  ConfigureSplitLayer: name='"
                       << split_layer_param->name()
                       << "' type=Split bottom='" << blob_name
                       << "' tops=" << split_count;
    for (int k = 0; k < split_count; ++k) {
      CAFFE_FFI_SPLIT_LOG << "    top[" << k << "] = '"
                         << split_layer_param->top(k) << "'";
    }
  }
}

void InsertSplits(const caffe::NetParameter& in_param, caffe::NetParameter* out_param) {
  using std::make_pair;
  CAFFE_FFI_SPLIT_LOG << "=== InsertSplits BEGIN ===";
  CAFFE_FFI_SPLIT_LOG << "Input: name='" << in_param.name()
                     << "' layers=" << in_param.layer_size()
                     << " inputs=" << in_param.input_size();

  // Initialize output by copying metadata from input
  out_param->CopyFrom(in_param);
  out_param->clear_layer();

  // === Data structures for Pass 1 ===
  std::map<std::string, std::pair<int, int>> blob_name_to_last_top_idx;
  std::map<std::pair<int, int>, std::pair<int, int>> bottom_idx_to_source_top_idx;
  std::map<std::pair<int, int>, int> top_idx_to_bottom_count;
  std::map<std::pair<int, int>, float> top_idx_to_loss_weight;
  std::map<std::pair<int, int>, int> top_idx_to_bottom_split_idx;
  std::map<int, std::string> layer_idx_to_layer_name;

  // Register external inputs (param.input()) as virtual producers at (-1, i)
  CAFFE_FFI_SPLIT_LOG << "--- Pass 1a: registering external inputs ---";
  for (int i = 0; i < in_param.input_size(); ++i) {
    const std::string& blob_name = in_param.input(i);
    blob_name_to_last_top_idx[blob_name] = make_pair(-1, i);
    top_idx_to_bottom_count[make_pair(-1, i)] = 0;
    CAFFE_FFI_SPLIT_LOG << "  External input['" << blob_name
                       << "'] -> virtual producer (-1," << i << ")";
  }

  // === Pass 1: Count references and build mappings ===
  CAFFE_FFI_SPLIT_LOG << "--- Pass 1b: counting bottom references across layers ---";
  for (int i = 0; i < in_param.layer_size(); ++i) {
    const caffe::LayerParameter& layer_param = in_param.layer(i);
    layer_idx_to_layer_name[i] = layer_param.name();

    CAFFE_FFI_SPLIT_LOG << "  Pass1 layer[" << i << "] '" << layer_param.name()
                       << "'(" << layer_param.type() << ")"
                       << " bottoms=" << layer_param.bottom_size()
                       << " tops=" << layer_param.top_size();

    // Record bottom -> source top mapping
    for (int j = 0; j < layer_param.bottom_size(); ++j) {
      const std::string& blob_name = layer_param.bottom(j);
      CAFFE_FFI_CHECK_RUNTIME(blob_name_to_last_top_idx.count(blob_name) > 0)
          << "InsertSplits: Unknown bottom blob '" << blob_name
          << "' (layer '" << layer_param.name()
          << "', bottom index " << j << ")";
      auto bottom_idx = make_pair(i, j);
      auto source_top_idx = blob_name_to_last_top_idx[blob_name];
      bottom_idx_to_source_top_idx[bottom_idx] = source_top_idx;
      ++top_idx_to_bottom_count[source_top_idx];
      CAFFE_FFI_SPLIT_LOG << "    bottom[" << j << "]='" << blob_name
                         << "' <- producer layer[" << source_top_idx.first
                         << "] top[" << source_top_idx.second << "]"
                         << " (consumer_count=" << top_idx_to_bottom_count[source_top_idx]
                         << ")";
    }

    // Register this layer's tops
    for (int j = 0; j < layer_param.top_size(); ++j) {
      const std::string& blob_name = layer_param.top(j);
      blob_name_to_last_top_idx[blob_name] = make_pair(i, j);
      if (top_idx_to_bottom_count.find(make_pair(i, j)) == top_idx_to_bottom_count.end()) {
        top_idx_to_bottom_count[make_pair(i, j)] = 0;
      }
    }

    // Loss-weighted tops also count as consumers
    int last_loss = std::min(layer_param.loss_weight_size(), layer_param.top_size());
    for (int j = 0; j < last_loss; ++j) {
      const std::string& blob_name = layer_param.top(j);
      auto top_idx = blob_name_to_last_top_idx[blob_name];
      top_idx_to_loss_weight[top_idx] = layer_param.loss_weight(j);
      if (top_idx_to_loss_weight[top_idx] != 0.0f) {
        ++top_idx_to_bottom_count[top_idx];
        CAFFE_FFI_SPLIT_LOG << "    top[" << j << "]='" << blob_name
                           << "' has loss_weight=" << layer_param.loss_weight(j)
                           << " -> consumer_count=" << top_idx_to_bottom_count[top_idx]
                           << " (loss counts as consumer)";
      }
    }
  }

  // Pass 1 summary: list all tops that need splits
  CAFFE_FFI_SPLIT_LOG << "--- Pass 1 fan-out summary ---";
  int split_needed_count = 0;
  for (const auto& kv : top_idx_to_bottom_count) {
    int layer_id = kv.first.first;
    int top_id = kv.first.second;
    int count = kv.second;
    bool needs_split = (count > 1);
    std::string producer_desc;
    if (layer_id == -1) {
      producer_desc = "external_input[" + std::to_string(top_id) + "]='"
                    + in_param.input(top_id) + "'";
    } else {
      producer_desc = "layer[" + std::to_string(layer_id) + "]='"
                    + layer_idx_to_layer_name[layer_id] + "' top["
                    + std::to_string(top_id) + "]";
    }
    CAFFE_FFI_SPLIT_LOG << "  " << producer_desc
                       << " consumers=" << count
                       << (needs_split ? " *** NEEDS SPLIT ***" : "");
    if (needs_split) split_needed_count++;
  }
  CAFFE_FFI_SPLIT_LOG << "Total blobs needing split: " << split_needed_count
                     << " (out of " << top_idx_to_bottom_count.size() << " total tops)";

  if (split_needed_count == 0) {
    CAFFE_FFI_SPLIT_LOG << "No splits needed, copying all layers directly";
    for (int i = 0; i < in_param.layer_size(); ++i) {
      out_param->add_layer()->CopyFrom(in_param.layer(i));
    }
    CAFFE_FFI_SPLIT_LOG << "=== InsertSplits END (no splits inserted) ===";
    return;
  }

  // === Pass 2: Rewrite bottoms and insert Split layers ===
  CAFFE_FFI_SPLIT_LOG << "--- Pass 2: rewriting layers and inserting splits ---";
  for (int i = 0; i < in_param.layer_size(); ++i) {
    const caffe::LayerParameter& layer_param = in_param.layer(i);
    CAFFE_FFI_SPLIT_LOG << "  Pass2 layer[" << i << "] '" << layer_param.name()
                       << "'(" << layer_param.type() << ")";

    caffe::LayerParameter* layer_param_ptr = out_param->add_layer();
    layer_param_ptr->CopyFrom(layer_param);

    // Step 2a: Rewrite bottom references for multi-consumer blobs
    for (int j = 0; j < layer_param_ptr->bottom_size(); ++j) {
      auto source_top_idx = bottom_idx_to_source_top_idx[make_pair(i, j)];
      int split_count = top_idx_to_bottom_count[source_top_idx];
      if (split_count > 1) {
        int producer_layer = source_top_idx.first;
        int producer_top = source_top_idx.second;
        std::string producer_name;
        std::string blob_name = layer_param_ptr->bottom(j);
        if (producer_layer == -1) {
          producer_name = "input";
          blob_name = in_param.input(producer_top);
        } else {
          producer_name = layer_idx_to_layer_name[producer_layer];
        }
        int& split_idx = top_idx_to_bottom_split_idx[source_top_idx];
        std::string new_blob_name = SplitBlobName(producer_name, blob_name,
                                                  producer_top, split_idx);
        CAFFE_FFI_SPLIT_LOG << "    Rewriting bottom[" << j << "] '"
                           << layer_param_ptr->bottom(j)
                           << "' -> '" << new_blob_name << "'"
                           << " (split output " << (split_idx + 1) << "/" << split_count
                           << " from producer '" << producer_name << "')";
        layer_param_ptr->set_bottom(j, new_blob_name);
        split_idx++;
      } else {
        CAFFE_FFI_SPLIT_LOG << "    bottom[" << j << "]='"
                           << layer_param_ptr->bottom(j)
                           << "' (single consumer, no rewrite)";
      }
    }

    // Step 2b: After this layer, insert Split layers for any tops that need them
    for (int j = 0; j < layer_param_ptr->top_size(); ++j) {
      auto top_idx = make_pair(i, j);
      int split_count = top_idx_to_bottom_count[top_idx];
      if (split_count > 1) {
        const std::string& producer_name = layer_idx_to_layer_name[i];
        const std::string& blob_name = layer_param_ptr->top(j);
        float loss_weight = 0.0f;
        if (top_idx_to_loss_weight.count(top_idx)) {
          loss_weight = top_idx_to_loss_weight[top_idx];
        }
        CAFFE_FFI_SPLIT_LOG << "    *** Inserting Split after '"
                           << layer_param.name() << "' for top[" << j << "]='"
                           << blob_name << "' (consumers=" << split_count
                           << ", loss_weight=" << loss_weight << ")";
        caffe::LayerParameter* split_layer_param = out_param->add_layer();
        ConfigureSplitLayer(producer_name, blob_name, j, split_count,
                            loss_weight, split_layer_param, 2);
        if (loss_weight != 0.0f) {
          layer_param_ptr->clear_loss_weight();
          top_idx_to_bottom_split_idx[top_idx]++;
        }
      }
    }
  }

  // Handle external inputs that need splits (insert at the very beginning)
  // Collect splits in input order, then insert them all at position 0 in order.
  CAFFE_FFI_SPLIT_LOG << "--- Pass 2b: handling external input splits ---";
  std::vector<caffe::LayerParameter> input_splits;
  for (int i = 0; i < in_param.input_size(); ++i) {
    auto top_idx = make_pair(-1, i);
    int split_count = top_idx_to_bottom_count[top_idx];
    if (split_count > 1) {
      const std::string& blob_name = in_param.input(i);
      CAFFE_FFI_SPLIT_LOG << "  External input '" << blob_name
                         << "' has " << split_count << " consumers, inserting Split";
      caffe::LayerParameter split_layer;
      ConfigureSplitLayer("input", blob_name, i, split_count, 0.0f, &split_layer, 2);
      input_splits.push_back(split_layer);
    }
  }
  if (!input_splits.empty()) {
    // Make room at the beginning: shift all existing layers to the right by
    // input_splits.size() positions, then insert splits at positions 0..n-1.
    int existing = out_param->layer_size();
    int n_ext = static_cast<int>(input_splits.size());

    // --- Log: layer order BEFORE moving external input splits to front ---
    CAFFE_FFI_SPLIT_LOG << "  Pass2b MOVE: shifting " << n_ext
                       << " external-input splits to front"
                       << " (existing layers=" << existing << ")";
    {
      std::ostringstream before_ss;
      before_ss << "  Pass2b BEFORE: [";
      for (int i = 0; i < existing; ++i) {
        if (i > 0) before_ss << ", ";
        before_ss << i << ":'" << out_param->layer(i).name() << "'";
      }
      before_ss << "]";
      CAFFE_FFI_SPLIT_LOG << before_ss.str();
    }
    {
      std::ostringstream splits_ss;
      splits_ss << "  Pass2b SPLITS to insert (in order): [";
      for (int k = 0; k < n_ext; ++k) {
        if (k > 0) splits_ss << ", ";
        splits_ss << k << ":'" << input_splits[k].name() << "'";
      }
      splits_ss << "]";
      CAFFE_FFI_SPLIT_LOG << splits_ss.str();
    }

    // Step 1: append empty slots to make room
    CAFFE_FFI_SPLIT_LOG << "  Pass2b Step1: add " << n_ext << " empty slots (size "
                       << existing << " -> " << (existing + n_ext) << ")";
    for (int k = 0; k < n_ext; ++k) {
      out_param->add_layer();
    }

    // Step 2: shift existing layers right by n_ext positions (copy from back to front)
    CAFFE_FFI_SPLIT_LOG << "  Pass2b Step2: shift " << existing
                       << " existing layers right by " << n_ext << " positions";
    for (int i = existing - 1; i >= 0; --i) {
      out_param->mutable_layer(i + n_ext)->CopyFrom(out_param->layer(i));
    }

    // Step 3: write external input splits at positions 0..n_ext-1
    CAFFE_FFI_SPLIT_LOG << "  Pass2b Step3: write " << n_ext
                       << " splits at positions 0.." << (n_ext - 1);
    for (int k = 0; k < n_ext; ++k) {
      out_param->mutable_layer(k)->CopyFrom(input_splits[k]);
    }

    // --- Log: layer order AFTER moving ---
    {
      std::ostringstream after_ss;
      after_ss << "  Pass2b AFTER: [";
      int total = out_param->layer_size();
      for (int i = 0; i < total; ++i) {
        if (i > 0) after_ss << ", ";
        after_ss << i << ":'" << out_param->layer(i).name() << "'";
      }
      after_ss << "]";
      CAFFE_FFI_SPLIT_LOG << after_ss.str();
    }
    CAFFE_FFI_SPLIT_LOG << "  Pass2b MOVE DONE: splits now at head, total layers="
                       << out_param->layer_size();
  }

  // === Final verification and summary ===
  int num_split_layers = 0;
  for (int i = 0; i < out_param->layer_size(); ++i) {
    if (out_param->layer(i).type() == "Split") num_split_layers++;
  }
  CAFFE_FFI_SPLIT_LOG << "=== InsertSplits END ===";
  CAFFE_FFI_SPLIT_LOG << "Output: name='" << out_param->name()
                     << "' layers=" << out_param->layer_size()
                     << " (original " << in_param.layer_size()
                     << " + " << num_split_layers << " auto-inserted Split layers)";

  // Log full transformed layer list for traceability
  CAFFE_FFI_SPLIT_LOG << "--- Transformed layer list ---";
  for (int i = 0; i < out_param->layer_size(); ++i) {
    const caffe::LayerParameter& lp = out_param->layer(i);
    std::ostringstream bs, ts;
    for (int b = 0; b < lp.bottom_size(); ++b) {
      if (b > 0) bs << ", ";
      bs << "'" << lp.bottom(b) << "'";
    }
    for (int t = 0; t < lp.top_size(); ++t) {
      if (t > 0) ts << ", ";
      ts << "'" << lp.top(t) << "'";
    }
    const char* auto_split = (lp.type() == "Split" &&
                              lp.name().find("_split") != std::string::npos)
                              ? " [AUTO-INSERTED]" : "";
    CAFFE_FFI_SPLIT_LOG << "  layer[" << i << "] '" << lp.name()
                       << "'(" << lp.type() << ")" << auto_split
                       << " bottoms=[" << bs.str() << "]"
                       << " tops=[" << ts.str() << "]";
  }
}
}  // namespace

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

void Net::Init(const caffe::NetParameter& in_param) {
  name_ = in_param.name();
  CAFFE_FFI_NET_LOG << "Init: starting network '" << name_ << "' initialization";

  // Step 0: Run InsertSplits graph transformation pass to automatically insert
  // Split layers for blobs consumed by multiple layers (fan-out > 1).
  // This mirrors native Caffe's behavior where implicit splits are handled
  // transparently so users don't need to manually add Split layers.
  caffe::NetParameter param;
  InsertSplits(in_param, &param);
  int num_layers = param.layer_size();
  CAFFE_FFI_NET_LOG << "Init: after InsertSplits, network has " << num_layers
                    << " layers (original: " << in_param.layer_size() << ")";

  std::set<std::string> available_blobs;
  std::map<std::string, int> blob_name_to_idx;
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
  if (available_blobs->find(blob_name) == available_blobs->end()) {
    // Build detailed diagnostic: list all available blobs, all consumed blobs,
    // and suggest fixes for common errors.
    std::ostringstream available_ss;
    bool first = true;
    for (const auto& name : *available_blobs) {
      if (!first) available_ss << ", ";
      first = false;
      available_ss << "'" << name << "'";
    }
    // List all known blob names (including consumed ones)
    std::ostringstream all_blobs_ss;
    first = true;
    for (const auto& kv : *blob_name_to_idx) {
      if (!first) all_blobs_ss << ", ";
      first = false;
      bool is_avail = available_blobs->count(kv.first) > 0;
      all_blobs_ss << "'" << kv.first << "'"
                   << (is_avail ? " [available]" : " [consumed]");
    }
    // List previously processed layers (so user can see ordering)
    std::ostringstream prev_layers_ss;
    for (int prev = 0; prev < layer_id; ++prev) {
      if (prev > 0) prev_layers_ss << " -> ";
      prev_layers_ss << param.layer(prev).name() << "(" << param.layer(prev).type() << ")";
    }
    // Check if this blob name appears as a top of a future layer (wrong order)
    std::ostringstream future_producers_ss;
    for (int future = layer_id + 1; future < param.layer_size(); ++future) {
      for (int t = 0; t < param.layer(future).top_size(); ++t) {
        if (param.layer(future).top(t) == blob_name) {
          if (future_producers_ss.tellp() > 0) future_producers_ss << ", ";
          future_producers_ss << "layer[" << future << "]='" << param.layer(future).name()
                              << "'(" << param.layer(future).type() << ")";
        }
      }
    }
    // Check if this blob is consumed by a previous layer (needs Split)
    std::ostringstream previous_consumers_ss;
    for (int prev = 0; prev < layer_id; ++prev) {
      for (int b = 0; b < param.layer(prev).bottom_size(); ++b) {
        if (param.layer(prev).bottom(b) == blob_name) {
          if (previous_consumers_ss.tellp() > 0) previous_consumers_ss << ", ";
          previous_consumers_ss << "layer[" << prev << "]='" << param.layer(prev).name()
                                << "'(" << param.layer(prev).type() << ")";
        }
      }
    }

    CAFFE_FFI_LOG_ERROR() << "[BLOB-NOT-FOUND] layer[" << layer_id << "]='" << layer_param.name()
                          << "'(" << layer_param.type() << ") bottom[" << bottom_id << "]='"
                          << blob_name << "' not available."
                          << "\n  *** Currently available blobs (" << available_blobs->size() << "): "
                          << available_ss.str()
                          << "\n  *** All known blobs: " << all_blobs_ss.str()
                          << "\n  *** Layer processing order so far: " << prev_layers_ss.str()
                          << "\n  *** Previous consumers of '" << blob_name << "': "
                          << (previous_consumers_ss.tellp() > 0 ? previous_consumers_ss.str() : "none")
                          << "\n  *** Future producers of '" << blob_name << "': "
                          << (future_producers_ss.tellp() > 0 ? future_producers_ss.str() : "none")
                          << "\n  *** Note: caffe-ffi automatically inserts Split layers for multi-consumer blobs"
                          << "\n      (see [SPLIT-INSERT] logs above). If you see this error, common causes are:"
                          << "\n      1) LAYER ORDERING: The layer producing '" << blob_name << "' comes after this layer."
                          << "\n         Move the producer layer before this layer in the prototxt."
                          << "\n      2) TYPO: Check for misspelled blob name."
                          << "\n      3) IN-PLACE CYCLE: An in-place layer (top==same bottom) combined with skip"
                          << "\n         connection may confuse the split inserter. Use explicit Split layer."
                          << "\n      4) Check [SPLIT-INSERT] logs above to verify the graph transformation.";
  }
  CAFFE_FFI_CHECK_KEY(available_blobs->find(blob_name) != available_blobs->end())
      << "Unknown bottom blob '" << blob_name << "' (layer '" << layer_param.name()
      << "', bottom index " << bottom_id << "). See [BLOB-NOT-FOUND] above.";
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

      float* dst = blob->cpu_mutable_data();
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

void Net::Backward() {
  CAFFE_FFI_NET_LOG << "Backward: starting BackwardFromTo(" << (layers_.size() - 1) << ", 0)";
  BackwardFromTo(static_cast<int>(layers_.size()) - 1, 0);
  CAFFE_FFI_NET_LOG << "Backward: completed";
}

void Net::BackwardFromTo(int start, int end) {
  CAFFE_FFI_CHECK_INDEX_GE(end, 0);
  CAFFE_FFI_CHECK_INDEX_LT(start, static_cast<int>(layers_.size()));
  CAFFE_FFI_NET_LOG << "BackwardFromTo: layers[" << start << ".." << end << "] (reverse)";
  // Backward traversal: from last layer to first
  for (int i = start; i >= end; --i) {
    // For MVP backward support: propagate gradients down to all bottom blobs.
    // Layers with no bottoms (e.g., Input/Data layers) will have empty bottom_vecs,
    // and their Backward will be a no-op. Learnable parameter blobs (layer.blobs_)
    // gradients are handled separately (not yet implemented).
    const std::vector<Blob*>& bottom = bottom_vecs_[i];
    const std::vector<Blob*>& top = top_vecs_[i];
    std::vector<bool> propagate_down(bottom.size(), true);
    CAFFE_FFI_LAYER_LOG << "BackwardFromTo: <<< layer[" << i << "] '" << layer_names_[i] << "' backward";
    layers_[i]->Backward(top, propagate_down, bottom);
    CAFFE_FFI_LAYER_LOG << "BackwardFromTo: >>> layer[" << i << "] '" << layer_names_[i] << "' backward done";
  }
  CAFFE_FFI_NET_LOG << "BackwardFromTo: completed";
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
