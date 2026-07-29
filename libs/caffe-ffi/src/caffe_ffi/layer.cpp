#include "caffe_ffi/layer.hpp"

#include <string>
#include <vector>

#include <tvm/ffi/memory.h>

#include "caffe_ffi/error.hpp"
#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

Layer::Layer(const caffe::LayerParameter& param) : layer_param_(param) {
  CAFFE_FFI_LAYER_LOG << "Layer('" << param.name() << "', type='" << param.type() << "'): constructor, blobs_size=" << param.blobs_size();
  if (layer_param_.blobs_size() > 0) {
    blobs_.resize(layer_param_.blobs_size());
    for (int i = 0; i < layer_param_.blobs_size(); ++i) {
      CAFFE_FFI_LAYER_LOG << "Layer('" << param.name() << "'): loading blob[" << i << "] from proto";
      blobs_[i] = make_object<Blob>();
      blobs_[i]->FromProto(layer_param_.blobs(i));
    }
  }
}

void Layer::SetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) {
  CAFFE_FFI_LAYER_LOG << "SetUp('" << layer_param_.name() << "', " << type() << "): bottom=" << bottom.size() << " top=" << top.size();
  CheckBlobCounts(bottom, top);
  CAFFE_FFI_LAYER_LOG << "SetUp('" << layer_param_.name() << "'): LayerSetUp";
  LayerSetUp(bottom, top);
  CAFFE_FFI_LAYER_LOG << "SetUp('" << layer_param_.name() << "'): Reshape";
  Reshape(bottom, top);
  CAFFE_FFI_LAYER_LOG << "SetUp('" << layer_param_.name() << "'): SetLossWeights";
  SetLossWeights(top);
  CAFFE_FFI_LAYER_LOG << "SetUp('" << layer_param_.name() << "'): complete, internal blobs=" << blobs_.size();
}

float Layer::Forward(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) {
  float loss = 0.0f;
  CAFFE_FFI_LAYER_LOG << "Forward('" << layer_param_.name() << "', " << type() << "): Reshape + Forward_cpu";
  Reshape(bottom, top);
  Forward_cpu(bottom, top);
  for (size_t top_id = 0; top_id < top.size(); ++top_id) {
    if (!this->loss(static_cast<int>(top_id))) continue;
    const int64_t count = top[top_id]->count();
    const float* data = top[top_id]->cpu_data();
    const float* loss_weights = top[top_id]->cpu_diff();
    float blob_loss = caffe_cpu_dot_fp32(static_cast<size_t>(count), data, loss_weights);
    CAFFE_FFI_LAYER_LOG << "Forward('" << layer_param_.name() << "'): top[" << top_id << "] loss=" << blob_loss << " (count=" << count << ")";
    loss += blob_loss;
  }
  CAFFE_FFI_LAYER_LOG << "Forward('" << layer_param_.name() << "'): total_loss=" << loss;
  return loss;
}

Array<ObjectPtr<Blob>> Layer::blobs_array() const {
  Array<ObjectPtr<Blob>> result;
  for (const auto& blob : blobs_) {
    result.push_back(blob);
  }
  return result;
}

void Layer::ToProto(caffe::LayerParameter* param, bool write_diff) {
  param->Clear();
  param->CopyFrom(layer_param_);
  param->clear_blobs();
  for (size_t i = 0; i < blobs_.size(); ++i) {
    blobs_[i]->ToProto(param->add_blobs());
  }
}

void Layer::CheckBlobCounts(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  const std::string& layer_name = layer_param_.name();
  const char* type_str = type();
  if (ExactNumBottomBlobs() >= 0) {
    CAFFE_FFI_CHECK_VALUE_EQ(ExactNumBottomBlobs(), static_cast<int>(bottom.size()))
        << type_str << " layer '" << layer_name << "' takes " << ExactNumBottomBlobs()
        << " bottom blob(s) as input, got " << bottom.size();
  }
  if (MinBottomBlobs() >= 0) {
    CAFFE_FFI_CHECK_VALUE_LE(MinBottomBlobs(), static_cast<int>(bottom.size()))
        << type_str << " layer '" << layer_name << "' takes at least " << MinBottomBlobs()
        << " bottom blob(s), got " << bottom.size();
  }
  if (MaxBottomBlobs() >= 0) {
    CAFFE_FFI_CHECK_VALUE_GE(MaxBottomBlobs(), static_cast<int>(bottom.size()))
        << type_str << " layer '" << layer_name << "' takes at most " << MaxBottomBlobs()
        << " bottom blob(s), got " << bottom.size();
  }
  if (ExactNumTopBlobs() >= 0) {
    CAFFE_FFI_CHECK_VALUE_EQ(ExactNumTopBlobs(), static_cast<int>(top.size()))
        << type_str << " layer '" << layer_name << "' produces " << ExactNumTopBlobs()
        << " top blob(s), got " << top.size();
  }
  if (MinTopBlobs() >= 0) {
    CAFFE_FFI_CHECK_VALUE_LE(MinTopBlobs(), static_cast<int>(top.size()))
        << type_str << " layer '" << layer_name << "' produces at least " << MinTopBlobs()
        << " top blob(s), got " << top.size();
  }
  if (MaxTopBlobs() >= 0) {
    CAFFE_FFI_CHECK_VALUE_GE(MaxTopBlobs(), static_cast<int>(top.size()))
        << type_str << " layer '" << layer_name << "' produces at most " << MaxTopBlobs()
        << " top blob(s), got " << top.size();
  }
  if (EqualNumBottomTopBlobs()) {
    CAFFE_FFI_CHECK_VALUE_EQ(bottom.size(), top.size())
        << type_str << " layer '" << layer_name << "' produces one top blob per bottom blob, "
        << "got " << bottom.size() << " bottom and " << top.size() << " top";
  }
}

void Layer::SetLossWeights(const std::vector<Blob*>& top) {
  const int num_loss_weights = layer_param_.loss_weight_size();
  if (num_loss_weights) {
    CAFFE_FFI_CHECK_VALUE_EQ(top.size(), static_cast<size_t>(num_loss_weights))
        << "loss_weight must be unspecified or specified once per top blob "
        << "in layer '" << layer_param_.name() << "' (" << type() << "), "
        << "got " << num_loss_weights << " loss_weight for " << top.size() << " top blobs.";
    for (size_t top_id = 0; top_id < top.size(); ++top_id) {
      const float loss_weight = layer_param_.loss_weight(static_cast<int>(top_id));
      if (loss_weight == 0.0f) continue;
      this->set_loss(static_cast<int>(top_id), loss_weight);
      const int64_t count = top[top_id]->count();
      float* loss_multiplier = top[top_id]->cpu_diff();
      caffe_set_fp32(static_cast<size_t>(count), loss_weight, loss_multiplier);
    }
  }
}

}  // namespace caffe_ffi
