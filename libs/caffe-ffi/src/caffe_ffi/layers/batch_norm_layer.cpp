#include "caffe_ffi/layers/batch_norm_layer.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

#include <tvm/ffi/memory.h>

#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/math_utils.hpp"

namespace caffe_ffi {

void BatchNormLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                                 const std::vector<Blob*>& top) {
  const caffe::BatchNormParameter& param = this->layer_param_.batch_norm_param();
  use_global_stats_ = param.use_global_stats();
  moving_average_fraction_ = param.moving_average_fraction();
  eps_ = param.eps();

  CAFFE_FFI_LAYER_LOG << "BatchNorm LayerSetUp: use_global_stats_=" << use_global_stats_
                      << " moving_average_fraction_=" << moving_average_fraction_
                      << " eps_=" << eps_;

  if (this->blobs_.size() > 0) {
    CAFFE_FFI_LAYER_LOG << "BatchNorm: using pre-loaded weights, blobs_.size=" << this->blobs_.size();
    CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_.size(), 3U)
        << "Incorrect number of batch norm blobs.";
  } else {
    this->blobs_.resize(3);
    channels_ = static_cast<int>(bottom[0]->shape(1));
    std::vector<int64_t> sz = {channels_};
    this->blobs_[0] = make_object<Blob>(sz);
    this->blobs_[1] = make_object<Blob>(sz);
    std::vector<int64_t> one = {1};
    this->blobs_[2] = make_object<Blob>(one);
    caffe_set_fp32(static_cast<size_t>(this->blobs_[2]->count()), 1.0f, this->blobs_[2]->cpu_data());
    CAFFE_FFI_TENSOR_LOG << "BatchNorm: created mean blob shape=[" << channels_ << "]";
    CAFFE_FFI_TENSOR_LOG << "BatchNorm: created variance blob shape=[" << channels_ << "]";
    CAFFE_FFI_TENSOR_LOG << "BatchNorm: created scale factor blob shape=[1] (initialized to 1.0)";
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

void BatchNormLayer::Reshape(const std::vector<Blob*>& bottom,
                              const std::vector<Blob*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  channels_ = static_cast<int>(bottom[0]->shape(1));
  if (bottom[0]->num_axes() == 1) {
    channels_ = 1;
  }

  std::ostringstream input_shape_ss;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) input_shape_ss << ", ";
    input_shape_ss << bottom[0]->shape(i);
  }
  std::ostringstream output_shape_ss;
  for (int i = 0; i < top[0]->num_axes(); ++i) {
    if (i > 0) output_shape_ss << ", ";
    output_shape_ss << top[0]->shape(i);
  }

  std::vector<int64_t> sz = {channels_};
  if (!this->blobs_[0] || this->blobs_[0]->count() != channels_) {
    this->blobs_[0] = make_object<Blob>(sz);
    this->blobs_[1] = make_object<Blob>(sz);
    CAFFE_FFI_TENSOR_LOG << "BatchNorm Reshape: recreated mean/variance blobs shape=[" << channels_ << "]";
  }

  int spatial_dim = static_cast<int>(bottom[0]->count(2));
  if (bottom[0]->num_axes() == 1) {
    spatial_dim = 1;
  }

  CAFFE_FFI_LAYER_LOG << "BatchNorm Reshape: input=[" << input_shape_ss.str()
                      << "] output=[" << output_shape_ss.str()
                      << "] channels_=" << channels_
                      << " spatial_dim=" << spatial_dim;
}

void BatchNormLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                  const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_data();
  const int num = static_cast<int>(bottom[0]->shape(0));
  const int channels = channels_;
  int spatial_dim = static_cast<int>(bottom[0]->count(2));
  if (bottom[0]->num_axes() == 1) {
    spatial_dim = 1;
  }

  const float* mean = this->blobs_[0]->cpu_data();
  const float* variance = this->blobs_[1]->cpu_data();
  const float scale_factor = this->blobs_[2]->cpu_data()[0] == 0.0f
      ? 0.0f
      : 1.0f / this->blobs_[2]->cpu_data()[0];

  CAFFE_FFI_LAYER_LOG << "BatchNorm Forward: num=" << num
                      << " channels=" << channels
                      << " spatial_dim=" << spatial_dim
                      << " use_global_stats_=" << use_global_stats_
                      << " scale_factor=" << scale_factor;

  const float scale_factor_use = scale_factor == 0.0f ? 1.0f : scale_factor;
  const int count = static_cast<int>(bottom[0]->count());
  for (int i = 0; i < count; ++i) {
    int c = (i / spatial_dim) % channels;
    top_data[i] = (bottom_data[i] - mean[c] * scale_factor_use)
        / std::sqrt(std::max(variance[c] * scale_factor_use, 0.0f) + eps_);
  }
}

REGISTER_LAYER_CLASS(BatchNorm);

}  // namespace caffe_ffi
