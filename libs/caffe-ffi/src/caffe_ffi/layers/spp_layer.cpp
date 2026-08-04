#include "caffe_ffi/layers/spp_layer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"

namespace caffe_ffi {

void SPPLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                          const std::vector<Blob*>& top) {
  const caffe::SPPParameter& param = this->layer_param_.spp_param();
  pyramid_height_ = param.pyramid_height();
  is_max_pool_ = (param.pool() == caffe::SPPParameter_PoolMethod_MAX);
  CAFFE_FFI_CHECK_VALUE_GT(pyramid_height_, 0)
      << "pyramid_height must be positive.";
  CAFFE_FFI_LAYER_LOG << "SPP LayerSetUp: pyramid_height=" << pyramid_height_
                      << " is_max_pool=" << is_max_pool_;
}

void SPPLayer::Reshape(const std::vector<Blob*>& bottom,
                       const std::vector<Blob*>& top) {
  CAFFE_FFI_CHECK_VALUE_EQ(bottom[0]->num_axes(), 4)
      << "Input must have 4 axes (num, channels, height, width).";
  num_ = static_cast<int>(bottom[0]->shape(0));
  channels_ = static_cast<int>(bottom[0]->shape(1));
  bottom_h_ = static_cast<int>(bottom[0]->shape(2));
  bottom_w_ = static_cast<int>(bottom[0]->shape(3));
  CAFFE_FFI_CHECK_VALUE_GT(bottom_h_, 0) << "Input height cannot be zero.";
  CAFFE_FFI_CHECK_VALUE_GT(bottom_w_, 0) << "Input width cannot be zero.";

  // Compute per-level pooling parameters.
  kernel_h_.resize(pyramid_height_);
  kernel_w_.resize(pyramid_height_);
  pad_h_.resize(pyramid_height_);
  pad_w_.resize(pyramid_height_);
  num_bins_.resize(pyramid_height_);
  int64_t total_channels = 0;
  for (int p = 0; p < pyramid_height_; ++p) {
    const int num_bins = 1 << p;
    num_bins_[p] = num_bins;
    const int kh = static_cast<int>(std::ceil(bottom_h_ / static_cast<double>(num_bins)));
    const int remainder_h = kh * num_bins - bottom_h_;
    const int ph = (remainder_h + 1) / 2;
    const int kw = static_cast<int>(std::ceil(bottom_w_ / static_cast<double>(num_bins)));
    const int remainder_w = kw * num_bins - bottom_w_;
    const int pw = (remainder_w + 1) / 2;
    kernel_h_[p] = kh;
    kernel_w_[p] = kw;
    pad_h_[p] = ph;
    pad_w_[p] = pw;
    total_channels += static_cast<int64_t>(channels_) * num_bins * num_bins;
  }
  total_channels_ = total_channels;

  std::vector<int64_t> top_shape = {num_, total_channels, 1, 1};
  top[0]->Reshape(top_shape);
  CAFFE_FFI_LAYER_LOG << "SPP Reshape: num=" << num_ << " channels=" << channels_
                      << " H=" << bottom_h_ << " W=" << bottom_w_
                      << " total_channels=" << total_channels;
}

void SPPLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                           const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int H = bottom_h_, W = bottom_w_, C = channels_;
  const int spatial = H * W;

  int64_t level_base = 0;
  for (int p = 0; p < pyramid_height_; ++p) {
    const int nb = num_bins_[p];
    const int kh = kernel_h_[p], kw = kernel_w_[p];
    const int ph = pad_h_[p], pw = pad_w_[p];
    const int out_area = nb * nb;
    for (int n = 0; n < num_; ++n) {
      const float* img = bottom_data + n * C * spatial;
      for (int c = 0; c < C; ++c) {
        const float* data = img + c * spatial;
        for (int ib = 0; ib < nb; ++ib) {
          const int h_start = ib * kh - ph;
          for (int jb = 0; jb < nb; ++jb) {
            const int w_start = jb * kw - pw;
            float out_val;
            if (is_max_pool_) {
              float maxval = -std::numeric_limits<float>::max();
              for (int h = h_start; h < h_start + kh; ++h) {
                if (h < 0 || h >= H) continue;
                for (int w = w_start; w < w_start + kw; ++w) {
                  if (w < 0 || w >= W) continue;
                  maxval = std::max(maxval, data[h * W + w]);
                }
              }
              out_val = maxval;
            } else {
              float sum = 0.0f;
              int count = 0;
              for (int h = h_start; h < h_start + kh; ++h) {
                if (h < 0 || h >= H) continue;
                for (int w = w_start; w < w_start + kw; ++w) {
                  if (w < 0 || w >= W) continue;
                  sum += data[h * W + w];
                  ++count;
                }
              }
              // AVE pooling divides by the full kernel area (caffe semantics).
              out_val = sum / (static_cast<float>(kh) * static_cast<float>(kw));
            }
            const int64_t out_index =
                static_cast<int64_t>(n) * total_channels_ + level_base +
                c * out_area + ib * nb + jb;
            top_data[out_index] = out_val;
          }
        }
      }
    }
    level_base += static_cast<int64_t>(C) * out_area;
  }
  CAFFE_FFI_LAYER_LOG << "SPP Forward_cpu: pyramid_height=" << pyramid_height_
                      << " is_max_pool=" << is_max_pool_;
}

void SPPLayer::Backward_cpu(const std::vector<Blob*>& top,
                            const std::vector<bool>& propagate_down,
                            const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const float* bottom_data = bottom[0]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const int H = bottom_h_, W = bottom_w_, C = channels_;
  const int spatial = H * W;

  // Zero the bottom gradient first.
  for (int64_t i = 0; i < bottom[0]->count(); ++i) {
    bottom_diff[i] = 0.0f;
  }

  int64_t level_base = 0;
  for (int p = 0; p < pyramid_height_; ++p) {
    const int nb = num_bins_[p];
    const int kh = kernel_h_[p], kw = kernel_w_[p];
    const int ph = pad_h_[p], pw = pad_w_[p];
    const int out_area = nb * nb;
    for (int n = 0; n < num_; ++n) {
      float* img_diff = bottom_diff + n * C * spatial;
      const float* img_data = bottom_data + n * C * spatial;
      for (int c = 0; c < C; ++c) {
        float* diff = img_diff + c * spatial;
        const float* data = img_data + c * spatial;
        for (int ib = 0; ib < nb; ++ib) {
          const int h_start = ib * kh - ph;
          for (int jb = 0; jb < nb; ++jb) {
            const int w_start = jb * kw - pw;
            const int64_t out_index =
                static_cast<int64_t>(n) * total_channels_ + level_base +
                c * out_area + ib * nb + jb;
            const float grad = top_diff[out_index];
            if (is_max_pool_) {
              // Winner-takes-all: route gradient to the max position.
              float maxval = -std::numeric_limits<float>::max();
              int max_h = -1, max_w = -1;
              for (int h = h_start; h < h_start + kh; ++h) {
                if (h < 0 || h >= H) continue;
                for (int w = w_start; w < w_start + kw; ++w) {
                  if (w < 0 || w >= W) continue;
                  const float v = data[h * W + w];
                  if (v > maxval) {
                    maxval = v;
                    max_h = h;
                    max_w = w;
                  }
                }
              }
              if (max_h >= 0) {
                diff[max_h * W + max_w] += grad;
              }
            } else {
              const float inv_area = 1.0f / (static_cast<float>(kh) * static_cast<float>(kw));
              for (int h = h_start; h < h_start + kh; ++h) {
                if (h < 0 || h >= H) continue;
                for (int w = w_start; w < w_start + kw; ++w) {
                  if (w < 0 || w >= W) continue;
                  diff[h * W + w] += grad * inv_area;
                }
              }
            }
          }
        }
      }
    }
    level_base += static_cast<int64_t>(C) * out_area;
  }
  CAFFE_FFI_LAYER_LOG << "SPP Backward_cpu: completed";
}

REGISTER_LAYER_CLASS(SPP);

}  // namespace caffe_ffi