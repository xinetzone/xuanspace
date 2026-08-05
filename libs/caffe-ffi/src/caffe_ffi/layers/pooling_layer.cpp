#include "caffe_ffi/layers/pooling_layer.hpp"

#include <algorithm>
#include <chrono>
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

  // Allocate max_idx_ for MAX pooling: stores flat index (within channel plane) of the max value
  // for each pooling window, used by Backward_cpu to route gradients.
  if (pool_method_ == caffe::PoolingParameter::MAX) {
    max_idx_ = make_object<Blob>(top_shape);
    CAFFE_FFI_TENSOR_LOG << "Pooling: created max_idx_ blob shape=[" << top_shape[0]
                         << "," << channels_ << "," << pooled_height_ << "," << pooled_width_ << "]";
  }

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
  float* top_data = top[0]->cpu_mutable_data();
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

  auto t_start = std::chrono::high_resolution_clock::now();

  float in_min = std::numeric_limits<float>::max();
  float in_max = -std::numeric_limits<float>::max();
  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();

  // Initialize top_data and max_idx_
  caffe_set_fp32(static_cast<size_t>(top_count),
                 (pool_method_ == caffe::PoolingParameter::AVE) ? 0.0f : -std::numeric_limits<float>::max(),
                 top_data);
  float* mask_data = nullptr;
  if (pool_method_ == caffe::PoolingParameter::MAX) {
    mask_data = max_idx_->cpu_mutable_data();
    caffe_set_fp32(static_cast<size_t>(top_count), -1.0f, mask_data);
  }

  // Main pooling loop. Flatten (n, c) into a single nc index so that
  // parallelism = num * channels_ (typically >> num when batch=1 inference).
  // Each nc writes a disjoint region of top_data/mask_data — no cross-thread
  // write race. Do NOT use OpenMP min/max reductions here: they require
  // `/openmp:llvm` and break MSVC's default `/openmp`. in_min/in_max are
  // computed in a separate serial pass below (outside the timed region).
  const int nc_total = num * channels_;
  const int hw = height_ * width_;
  const int pooled_hw = pooled_height_ * pooled_width_;
  #ifdef CAFFE_USE_OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int nc = 0; nc < nc_total; ++nc) {
    const int base = nc * hw;
    const int pool_base = nc * pooled_hw;
    for (int ph = 0; ph < pooled_height_; ++ph) {
      for (int pw = 0; pw < pooled_width_; ++pw) {
        int hstart = ph * stride_h_ - pad_h_;
        int wstart = pw * stride_w_ - pad_w_;
        int hend = std::min(hstart + kernel_h_, height_);
        int wend = std::min(wstart + kernel_w_, width_);
        hstart = std::max(hstart, 0);
        wstart = std::max(wstart, 0);
        const int pool_index = pool_base + ph * pooled_width_ + pw;

        if (pool_method_ == caffe::PoolingParameter::MAX) {
          float max_val = -std::numeric_limits<float>::max();
          int max_idx = -1;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = base + h * width_ + w;
              float val = bottom_data[index];
              if (val > max_val) {
                max_val = val;
                max_idx = h * width_ + w;  // flat index within channel plane
              }
            }
          }
          top_data[pool_index] = max_val;
          mask_data[pool_index] = static_cast<float>(max_idx);
        } else if (pool_method_ == caffe::PoolingParameter::AVE) {
          float sum = 0.0f;
          int count = 0;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              sum += bottom_data[base + h * width_ + w];
              ++count;
            }
          }
          top_data[pool_index] = (count > 0) ? sum / count : 0.0f;
        }
      }
    }
  }

  // out值域统计
  for (int i = 0; i < top_count; ++i) {
    out_min = std::min(out_min, top_data[i]);
    out_max = std::max(out_max, top_data[i]);
  }

  // in值域统计（单独串行遍历，避免并行循环内对诊断变量的数据竞争）
  {
    const int bottom_count = static_cast<int>(bottom[0]->count());
    for (int i = 0; i < bottom_count; ++i) {
      in_min = std::min(in_min, bottom_data[i]);
      in_max = std::max(in_max, bottom_data[i]);
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[POOL-PERF] " << this->name()
                       << " Pooling forward: num=" << num
                       << " channels=" << channels_
                       << " pool=" << pool_method_str
                       << " kernel=[" << kernel_h_ << "," << kernel_w_ << "]"
                       << " stride=[" << stride_h_ << "," << stride_w_ << "]"
                       << " pad=[" << pad_h_ << "," << pad_w_ << "]"
                       << " in=[" << in_min << ", " << in_max << "]"
                       << " out=[" << out_min << ", " << out_max << "]"
                       << " time=" << elapsed_us << "us";
}

void PoolingLayer::Backward_cpu(const std::vector<Blob*>& top,
                                 const std::vector<bool>& propagate_down,
                                 const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    CAFFE_FFI_LAYER_LOG << "Pooling Backward_cpu: propagate_down[0]=false, skipping";
    return;
  }

  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const int num = static_cast<int>(bottom[0]->shape(0));
  const int bottom_count = static_cast<int>(bottom[0]->count());

  const char* pool_method_str = "UNKNOWN";
  if (pool_method_ == caffe::PoolingParameter::MAX) {
    pool_method_str = "MAX";
  } else if (pool_method_ == caffe::PoolingParameter::AVE) {
    pool_method_str = "AVE";
  }
  CAFFE_FFI_LAYER_LOG << "Pooling Backward: num=" << num
                      << " pool_method=" << pool_method_str
                      << " channels=" << channels_
                      << " bottom_count=" << bottom_count;

  auto t_start = std::chrono::high_resolution_clock::now();

  // Zero out bottom diff
  caffe_set_fp32(static_cast<size_t>(bottom_count), 0.0f, bottom_diff);

  float diff_in_min = std::numeric_limits<float>::max();
  float diff_in_max = -std::numeric_limits<float>::max();
  float diff_out_min = std::numeric_limits<float>::max();
  float diff_out_max = -std::numeric_limits<float>::max();

  const float* mask_data = (pool_method_ == caffe::PoolingParameter::MAX) ? max_idx_->cpu_data() : nullptr;

  // Flatten (n, c) into nc: parallelism = num * channels_, safe because each
  // nc plane is a disjoint memory region in [N,C,H,W] layout and the inner
  // ph/pw loops run serially within one thread — no cross-thread write race
  // on bottom_diff even when pooling windows overlap (stride < kernel).
  const int nc_total = num * channels_;
  const int hw = height_ * width_;
  const int pooled_hw = pooled_height_ * pooled_width_;
  #ifdef CAFFE_USE_OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int nc = 0; nc < nc_total; ++nc) {
    const int base = nc * hw;
    const int pool_base = nc * pooled_hw;
    for (int ph = 0; ph < pooled_height_; ++ph) {
      for (int pw = 0; pw < pooled_width_; ++pw) {
        int hstart = ph * stride_h_ - pad_h_;
        int wstart = pw * stride_w_ - pad_w_;
        int hend = std::min(hstart + kernel_h_, height_);
        int wend = std::min(wstart + kernel_w_, width_);
        hstart = std::max(hstart, 0);
        wstart = std::max(wstart, 0);
        const int pool_index = pool_base + ph * pooled_width_ + pw;
        const float dy = top_diff[pool_index];

        if (pool_method_ == caffe::PoolingParameter::MAX) {
          // Route gradient to the max-pooling winner
          const int winner = static_cast<int>(mask_data[pool_index]);
          if (winner >= 0) {
            const int bottom_idx = base + winner;
            bottom_diff[bottom_idx] += dy;
          }
        } else if (pool_method_ == caffe::PoolingParameter::AVE) {
          // Distribute gradient equally across the pooling window
          const int pool_size = (hend - hstart) * (wend - wstart);
          const float scale = (pool_size > 0) ? dy / pool_size : 0.0f;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              bottom_diff[base + h * width_ + w] += scale;
            }
          }
        }
      }
    }
  }

  // diff值域统计（串行遍历，避免并行循环内的数据竞争）
  {
    const int top_count_diff = static_cast<int>(top[0]->count());
    for (int i = 0; i < top_count_diff; ++i) {
      diff_in_min = std::min(diff_in_min, top_diff[i]);
      diff_in_max = std::max(diff_in_max, top_diff[i]);
    }
    for (int i = 0; i < bottom_count; ++i) {
      diff_out_min = std::min(diff_out_min, bottom_diff[i]);
      diff_out_max = std::max(diff_out_max, bottom_diff[i]);
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[POOL-PERF] " << this->name()
                       << " Pooling backward: num=" << num
                       << " channels=" << channels_
                       << " pool=" << pool_method_str
                       << " kernel=[" << kernel_h_ << "," << kernel_w_ << "]"
                       << " stride=[" << stride_h_ << "," << stride_w_ << "]"
                       << " pad=[" << pad_h_ << "," << pad_w_ << "]"
                       << " diff_in=[" << diff_in_min << ", " << diff_in_max << "]"
                       << " diff_out=[" << diff_out_min << ", " << diff_out_max << "]"
                       << " time=" << elapsed_us << "us";
}

REGISTER_LAYER_CLASS(Pooling);

}  // namespace caffe_ffi
