#include "caffe_ffi/layers/dummy_data_layer.hpp"

#include <random>
#include <sstream>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

namespace {

// File-local RNG so the generated placeholder data is deterministic per process.
std::mt19937& Rng() {
  static std::mt19937 rng(12345);
  return rng;
}

// Fill a blob from a FillerParameter. Supported types: constant, uniform,
// gaussian. Any other type falls back to constant 0 with a warning.
void FillBlobWithFiller(Blob* blob, const caffe::FillerParameter& filler) {
  const std::string type = filler.type();
  const int64_t count = blob->count();
  float* data = blob->cpu_mutable_data();
  if (type == "constant") {
    caffe_set_fp32(static_cast<size_t>(count), filler.value(), data);
  } else if (type == "uniform") {
    std::uniform_real_distribution<float> dist(filler.min(), filler.max());
    for (int64_t i = 0; i < count; ++i) {
      data[i] = dist(Rng());
    }
  } else if (type == "gaussian") {
    std::normal_distribution<float> dist(filler.mean(), filler.std());
    for (int64_t i = 0; i < count; ++i) {
      data[i] = dist(Rng());
    }
  } else {
    CAFFE_FFI_LOG_WARN() << "[DUMMY-FILLER] filler type '" << type
                         << "' not implemented, using constant 0.0";
    caffe_set_fp32(static_cast<size_t>(count), 0.0f, data);
  }
}

}  // namespace

void DummyDataLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                                const std::vector<Blob*>& top) {
  const caffe::DummyDataParameter& param = this->layer_param_.dummy_data_param();
  const int num_top = static_cast<int>(top.size());
  const int num_shape = param.shape_size();
  CAFFE_FFI_CHECK_VALUE(num_shape == 0 || num_shape == 1 || num_shape == num_top)
      << "Must specify 'shape' once, once per top blob, or not at all: "
      << num_top << " tops vs. " << num_shape << " shapes.";
  CAFFE_FFI_LAYER_LOG << "DummyData LayerSetUp: num_top=" << num_top
                      << " num_shape=" << num_shape
                      << " num_fillers=" << param.data_filler_size();
}

void DummyDataLayer::Reshape(const std::vector<Blob*>& bottom,
                             const std::vector<Blob*>& top) {
  const caffe::DummyDataParameter& param = this->layer_param_.dummy_data_param();
  const int num_top = static_cast<int>(top.size());
  const int num_shape = param.shape_size();
  for (int i = 0; i < num_top; ++i) {
    if (num_shape > 0) {
      const int shape_index = (num_shape == 1) ? 0 : i;
      top[i]->Reshape(param.shape(shape_index));
    } else if (param.num_size() > 0) {
      // Deprecated 4-D shape fields.
      const int num_index = (param.num_size() == 1) ? 0 : i;
      const int channels_index = (param.channels_size() == 1) ? 0 : i;
      const int height_index = (param.height_size() == 1) ? 0 : i;
      const int width_index = (param.width_size() == 1) ? 0 : i;
      std::vector<int64_t> shape = {static_cast<int64_t>(param.num(num_index)),
                                    static_cast<int64_t>(param.channels(channels_index)),
                                    static_cast<int64_t>(param.height(height_index)),
                                    static_cast<int64_t>(param.width(width_index))};
      top[i]->Reshape(shape);
    } else {
      CAFFE_FFI_THROW(ValueError) << "DummyData top[" << i
                                  << "] has no shape specified (neither 'shape' nor "
                                     "'num/channels/height/width' fields).";
    }
    CAFFE_FFI_LAYER_LOG << "DummyData Reshape: top[" << i << "] count=" << top[i]->count();
  }
}

void DummyDataLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                 const std::vector<Blob*>& top) {
  const caffe::DummyDataParameter& param = this->layer_param_.dummy_data_param();
  const int num_top = static_cast<int>(top.size());
  const int num_fillers = param.data_filler_size();
  for (int i = 0; i < num_top; ++i) {
    if (num_fillers > 0) {
      const int filler_index = (num_fillers == 1) ? 0 : i;
      FillBlobWithFiller(top[i], param.data_filler(filler_index));
    } else {
      // No filler specified: default to constant 0.
      caffe_set_fp32(static_cast<size_t>(top[i]->count()), 0.0f, top[i]->cpu_mutable_data());
    }
    CAFFE_FFI_LAYER_LOG << "DummyData Forward: top[" << i << "] count=" << top[i]->count();
  }
}

REGISTER_LAYER_CLASS(DummyData);

}  // namespace caffe_ffi