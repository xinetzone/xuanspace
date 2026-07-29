#include "caffe_ffi/layers/pooling_layer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/math_utils.hpp"

namespace caffe_ffi {

void PoolingLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                               const std::vector<Blob*>& top) {
  const caffe::PoolingParameter& pool_param = this->layer_param_.pooling_param();
  pool_method_ = pool_param.pool();
  global_pooling_ = pool_param.global_pooling();
  round_mode_ = pool_param.round_mode();

  if (global_pooling_) {
    kernel_h_ = kernel_w_ = 0;
  } else {
    if (pool_param.has_kernel_h() || pool_param.has_kernel_w()) {
      kernel_h_ = static_cast<int>(pool_param.kernel_h());
      kernel_w_ = static_cast<int>(pool_param.kernel_w());
    } else {
      kernel_h_ = kernel_w_ = static_cast<int>(pool_param.kernel_size());
    }
  }

  if (pool_param.has_stride_h() || pool_param.has_stride_w()) {
    stride_h_ = static_cast<int>(pool_param.stride_h());
    stride_w_ = static_cast<int>(pool_param.stride_w());
  } else {
    stride_h_ = stride_w_ = static_cast<int>(pool_param.stride());
  }

  if (pool_param.has_pad_h() || pool_param.has_pad_w()) {
    pad_h_ = static_cast<int>(pool_param.pad_h());
    pad_w_ = static_cast<int>(pool_param.pad_w());
  } else {
    pad_h_ = pad_w_ = static_cast<int>(pool_param.pad());
  }

  const char* pool_method_str = "UNKNOWN";
  if (pool_method_ == caffe::PoolingParameter::MAX) {
    pool_method_str = "MAX";
  } else if (pool_method_ == caffe::PoolingParameter::AVE) {
    pool_method_str = "AVE";
  } else if (pool_method_ == caffe::PoolingParameter::STOCHASTIC) {
    pool_method_str = "STOCHASTIC";
  }
  const char* round_mode_str = (round_mode_ == caffe::PoolingParameter::CEIL) ? "CEIL" : "FLOOR";

  CAFFE_FFI_LAYER_LOG << "Pooling LayerSetUp: pool_method=" << pool_method_str
                      << " global_pooling=" << global_pooling_
                      << " round_mode=" << round_mode_str
                      << " kernel=[" << kernel_h_ << "," << kernel_w_ << "]"
                      << " stride=[" << stride_h_ << "," << stride_w_ << "]"
                      << " pad=[" << pad_h_ << "," << pad_w_ << "]";
}

void PoolingLayer::Reshape(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  channels_ = static_cast<int>(bottom[0]->shape(1));
  height_ = static_cast<int>(bottom[0]->shape(2));
  width_ = static_cast<int>(bottom[0]->shape(3));

  if (global_pooling_) {
    kernel_h_ = height_;
    kernel_w_ = width_;
    stride_h_ = 1;
    stride_w_ = 1;
    pad_h_ = 0;
    pad_w_ = 0;
  }

  if (round_mode_ == caffe::PoolingParameter::CEIL) {
    pooled_height_ = static_cast<int>(std::ceil(
        static_cast<float>(height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
    pooled_width_ = static_cast<int>(std::ceil(
        static_cast<float>(width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  } else {
    pooled_height_ = static_cast<int>(std::floor(
        static_cast<float>(height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
    pooled_width_ = static_cast<int>(std::floor(
        static_cast<float>(width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  }

  if (pad_h_ > 0 || pad_w_ > 0) {
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
  }

  CAFFE_FFI_CHECK_VALUE_GT(pooled_height_, 0) << "Pooled height should be positive.";
  CAFFE_FFI_CHECK_VALUE_GT(pooled_width_, 0) << "Pooled width should be positive.";

  std::vector<int64_t> top_shape = {bottom[0]->shape(0), channels_, pooled_height_, pooled_width_};
  top[0]->Reshape(top_shape);

  std::ostringstream top_shape_ss;
  for (int i = 0; i < static_cast<int>(top_shape.size()); ++i) {
    if (i > 0) top_shape_ss << ", ";
    top_shape_ss << top_shape[i];
  }
  CAFFE_FFI_LAYER_LOG << "Pooling Reshape: input=[" << bottom[0]->shape(0)
                      << "," << channels_ << "," << height_ << "," << width_
                      << "] output=[" << top_shape_ss.str() << "]"
                      << " pooled=[" << pooled_height_ << "," << pooled_width_ << "]";
}

void PoolingLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_data();
  const int num = static_cast<int>(bottom[0]->shape(0));
  const int top_count = static_cast<int>(top[0]->count());

  const char* pool_method_str = "UNKNOWN";
  if (pool_method_ == caffe::PoolingParameter::MAX) {
    pool_method_str = "MAX";
  } else if (pool_method_ == caffe::PoolingParameter::AVE) {
    pool_method_str = "AVE";
  }
  CAFFE_FFI_LAYER_LOG << "Pooling Forward: num=" << num
                      << " pool_method=" << pool_method_str
                      << " channels=" << channels_
                      << " kernel=[" << kernel_h_ << "," << kernel_w_ << "]"
                      << " pooled=[" << pooled_height_ << "," << pooled_width_ << "]"
                      << " top_count=" << top_count;

  caffe_set_fp32(static_cast<size_t>(top_count), (pool_method_ == caffe::PoolingParameter::AVE) ? 0.0f : -std::numeric_limits<float>::max(), top_data);

  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          int hstart = ph * stride_h_ - pad_h_;
          int wstart = pw * stride_w_ - pad_w_;
          int hend = std::min(hstart + kernel_h_, height_);
          int wend = std::min(wstart + kernel_w_, width_);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          const int pool_index = ph * pooled_width_ + pw;

          if (pool_method_ == caffe::PoolingParameter::MAX) {
            float max_val = -std::numeric_limits<float>::max();
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
                if (bottom_data[index] > max_val) {
                  max_val = bottom_data[index];
                }
              }
            }
            top_data[pool_index] = max_val;
          } else if (pool_method_ == caffe::PoolingParameter::AVE) {
            float sum = 0.0f;
            int count = 0;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
                sum += bottom_data[index];
                ++count;
              }
            }
            top_data[pool_index] = (count > 0) ? sum / count : 0.0f;
          }
        }
      }
      bottom_data += height_ * width_;
      top_data += pooled_height_ * pooled_width_;
    }
  }
}

REGISTER_LAYER_CLASS(Pooling);

}  // namespace caffe_ffi
