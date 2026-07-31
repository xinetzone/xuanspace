#include "caffe_ffi/layers/lrn_layer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

#include <tvm/ffi/memory.h>

#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/math_utils.hpp"

namespace caffe_ffi {

void LRNLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                           const std::vector<Blob*>& top) {
  size_ = static_cast<int>(this->layer_param_.lrn_param().local_size());
  CAFFE_FFI_CHECK_VALUE_EQ(size_ % 2, 1) << "LRN only supports odd values for local_size";
  pre_pad_ = (size_ - 1) / 2;
  alpha_ = this->layer_param_.lrn_param().alpha();
  beta_ = this->layer_param_.lrn_param().beta();
  k_ = this->layer_param_.lrn_param().k();

  CAFFE_FFI_CHECK_VALUE(this->layer_param_.lrn_param().norm_region() == caffe::LRNParameter_NormRegion_ACROSS_CHANNELS)
      << "LRN currently only supports ACROSS_CHANNELS mode";

  CAFFE_FFI_LAYER_LOG << "LRN LayerSetUp: size=" << size_
                      << " pre_pad=" << pre_pad_
                      << " alpha=" << alpha_
                      << " beta=" << beta_
                      << " k=" << k_;
}

void LRNLayer::Reshape(const std::vector<Blob*>& bottom,
                        const std::vector<Blob*>& top) {
  CAFFE_FFI_CHECK_VALUE_EQ(bottom[0]->num_axes(), 4)
      << "Input must have 4 axes (N, C, H, W)";

  num_ = static_cast<int>(bottom[0]->shape(0));
  channels_ = static_cast<int>(bottom[0]->shape(1));
  height_ = static_cast<int>(bottom[0]->shape(2));
  width_ = static_cast<int>(bottom[0]->shape(3));

  std::vector<int64_t> top_shape = {num_, channels_, height_, width_};
  top[0]->Reshape(top_shape);

  scale_ = make_object<Blob>(top_shape);
  padded_square_ = make_object<Blob>(std::vector<int64_t>{1, channels_ + size_ - 1, height_, width_});
  padded_ratio_ = make_object<Blob>(std::vector<int64_t>{1, channels_ + size_ - 1, height_, width_});
  accum_ratio_ = make_object<Blob>(std::vector<int64_t>{1, 1, height_, width_});

  CAFFE_FFI_LAYER_LOG << "LRN Reshape: input=[" << num_ << "," << channels_
                      << "," << height_ << "," << width_ << "]"
                      << " scale_size=" << scale_->count()
                      << " padded_size=" << padded_square_->count();
}

void LRNLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  float* scale_data = scale_->cpu_mutable_data();
  float* padded_square_data = padded_square_->cpu_mutable_data();

  const int64_t scale_count = scale_->count();
  const int64_t spatial_count = height_ * width_;
  const float alpha_over_size = alpha_ / size_;

  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  caffe_set_fp32(static_cast<size_t>(scale_count), k_, scale_data);
  caffe_set_fp32(static_cast<size_t>(padded_square_->count()), 0.0f, padded_square_data);

  for (int n = 0; n < num_; ++n) {
    const float* bottom_slice = bottom_data + static_cast<int64_t>(n) * channels_ * spatial_count;
    float* padded_slice = padded_square_data + static_cast<int64_t>(pre_pad_) * spatial_count;
    float* scale_slice = scale_data + static_cast<int64_t>(n) * channels_ * spatial_count;

    caffe_sqr_fp32(channels_ * spatial_count, bottom_slice, padded_slice);

    for (int c = 0; c < size_; ++c) {
      caffe_axpy_fp32(spatial_count, alpha_over_size,
                      padded_square_data + static_cast<int64_t>(c) * spatial_count,
                      scale_slice);
    }

    for (int c = 1; c < channels_; ++c) {
      float* curr_scale = scale_slice + static_cast<int64_t>(c) * spatial_count;
      float* prev_scale = scale_slice + static_cast<int64_t>(c - 1) * spatial_count;
      caffe_copy_fp32(static_cast<size_t>(spatial_count), prev_scale, curr_scale);

      caffe_axpy_fp32(spatial_count, alpha_over_size,
                      padded_square_data + static_cast<int64_t>(c + size_ - 1) * spatial_count,
                      curr_scale);
      caffe_axpy_fp32(spatial_count, -alpha_over_size,
                      padded_square_data + static_cast<int64_t>(c - 1) * spatial_count,
                      curr_scale);
    }
  }

  caffe_powx_fp32(scale_count, scale_data, -beta_, top_data);
  caffe_mul_fp32(scale_count, top_data, bottom_data, top_data);

  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();
  for (int64_t i = 0; i < top[0]->count(); ++i) {
    out_min = std::min(out_min, top_data[i]);
    out_max = std::max(out_max, top_data[i]);
  }

  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[LRN-PERF] " << this->name()
                       << " LRN forward: num=" << num_ << " channels=" << channels_
                       << " size=" << size_ << " out=[" << out_min << ", " << out_max << "]"
                       << " time=" << elapsed_us << "us";
}

void LRNLayer::Backward_cpu(const std::vector<Blob*>& top,
                             const std::vector<bool>& propagate_down,
                             const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    CAFFE_FFI_LAYER_LOG << "LRN Backward: propagate_down[0]=false, skipping";
    return;
  }

  const float* top_diff = top[0]->cpu_diff();
  const float* top_data = top[0]->cpu_data();
  const float* bottom_data = bottom[0]->cpu_data();
  const float* scale_data = scale_->cpu_data();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  float* padded_ratio_data = padded_ratio_->cpu_mutable_data();
  float* accum_ratio_data = accum_ratio_->cpu_mutable_data();
  float* accum_ratio_times_bottom = accum_ratio_->cpu_mutable_diff();

  const int64_t scale_count = scale_->count();
  const int64_t spatial_count = height_ * width_;
  const float cache_ratio_value = 2.0f * alpha_ * beta_ / size_;
  const int inverse_pre_pad = size_ - (size_ + 1) / 2;

  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  caffe_powx_fp32(scale_count, scale_data, -beta_, bottom_diff);
  caffe_mul_fp32(scale_count, top_diff, bottom_diff, bottom_diff);

  caffe_set_fp32(static_cast<size_t>(padded_ratio_->count()), 0.0f, padded_ratio_data);

  for (int n = 0; n < num_; ++n) {
    const int64_t block_offset = static_cast<int64_t>(n) * channels_ * spatial_count;
    float* padded_offset = padded_ratio_data + static_cast<int64_t>(inverse_pre_pad) * spatial_count;

    caffe_mul_fp32(channels_ * spatial_count,
                   top_diff + block_offset, top_data + block_offset,
                   padded_offset);
    caffe_div_fp32(channels_ * spatial_count,
                   padded_offset, scale_data + block_offset,
                   padded_offset);

    caffe_set_fp32(static_cast<size_t>(spatial_count), 0.0f, accum_ratio_data);
    for (int c = 0; c < size_ - 1; ++c) {
      caffe_axpy_fp32(spatial_count, 1.0f,
                      padded_ratio_data + static_cast<int64_t>(c) * spatial_count,
                      accum_ratio_data);
    }

    for (int c = 0; c < channels_; ++c) {
      const int64_t channel_offset = block_offset + static_cast<int64_t>(c) * spatial_count;

      caffe_axpy_fp32(spatial_count, 1.0f,
                      padded_ratio_data + static_cast<int64_t>(c + size_ - 1) * spatial_count,
                      accum_ratio_data);

      caffe_mul_fp32(spatial_count,
                     bottom_data + channel_offset,
                     accum_ratio_data, accum_ratio_times_bottom);
      caffe_axpy_fp32(spatial_count, -cache_ratio_value,
                      accum_ratio_times_bottom, bottom_diff + channel_offset);
      caffe_axpy_fp32(spatial_count, -1.0f,
                      padded_ratio_data + static_cast<int64_t>(c) * spatial_count,
                      accum_ratio_data);
    }
  }

  float diff_min = std::numeric_limits<float>::max();
  float diff_max = -std::numeric_limits<float>::max();
  for (int64_t i = 0; i < bottom[0]->count(); ++i) {
    diff_min = std::min(diff_min, bottom_diff[i]);
    diff_max = std::max(diff_max, bottom_diff[i]);
  }

  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[LRN-PERF] " << this->name()
                       << " LRN backward: num=" << num_ << " channels=" << channels_
                       << " size=" << size_
                       << " diff=[" << diff_min << ", " << diff_max << "]"
                       << " time=" << elapsed_us << "us";
}

REGISTER_LAYER_CLASS(LRN);

}  // namespace caffe_ffi
