#include "caffe_ffi/layers/softmax_layer.hpp"

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

void SoftmaxLayer::Reshape(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  softmax_axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  std::vector<int64_t> mult_dims = {bottom[0]->shape(softmax_axis_)};
  sum_multiplier_ = make_object<Blob>(mult_dims);
  float* multiplier_data = sum_multiplier_->cpu_mutable_data();
  caffe_set_fp32(static_cast<size_t>(sum_multiplier_->count()), 1.0f, multiplier_data);
  outer_num_ = static_cast<int>(bottom[0]->count(0, softmax_axis_));
  inner_num_ = static_cast<int>(bottom[0]->count(softmax_axis_ + 1));
  std::vector<int64_t> scale_dims;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i == softmax_axis_) {
      scale_dims.push_back(1);
    } else {
      scale_dims.push_back(bottom[0]->shape(i));
    }
  }
  scale_ = make_object<Blob>(scale_dims);

  std::ostringstream bottom_shape_ss;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) bottom_shape_ss << ", ";
    bottom_shape_ss << bottom[0]->shape(i);
  }
  std::ostringstream scale_shape_ss;
  for (int i = 0; i < scale_->num_axes(); ++i) {
    if (i > 0) scale_shape_ss << ", ";
    scale_shape_ss << scale_->shape(i);
  }
  CAFFE_FFI_LAYER_LOG << "Softmax Reshape: softmax_axis=" << softmax_axis_
                      << " outer_num=" << outer_num_ << " inner_num=" << inner_num_
                      << " bottom_shape=[" << bottom_shape_ss.str() << "]"
                      << " sum_multiplier shape=[" << mult_dims[0] << "]"
                      << " scale shape=[" << scale_shape_ss.str() << "]";
}

void SoftmaxLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  float* scale_data = scale_->cpu_mutable_data();
  int channels = static_cast<int>(bottom[0]->shape(softmax_axis_));
  int dim = channels * inner_num_;
  const int64_t count = bottom[0]->count();

  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  caffe_copy_fp32(static_cast<size_t>(count), bottom_data, top_data);
  for (int i = 0; i < outer_num_; ++i) {
    float* top_data_i = top_data + i * dim;
    float* scale_data_i = scale_data + i * inner_num_;
    for (int k = 0; k < inner_num_; ++k) {
      scale_data_i[k] = top_data_i[k];
      for (int j = 1; j < channels; ++j) {
        scale_data_i[k] = std::max(scale_data_i[k], top_data_i[j * inner_num_ + k]);
      }
    }
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < inner_num_; ++k) {
        top_data_i[j * inner_num_ + k] -= scale_data_i[k];
      }
    }
    caffe_exp_fp32(static_cast<size_t>(dim), top_data_i, top_data_i);
    for (int k = 0; k < inner_num_; ++k) {
      scale_data_i[k] = 0;
      for (int j = 0; j < channels; ++j) {
        scale_data_i[k] += top_data_i[j * inner_num_ + k];
      }
    }
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < inner_num_; ++k) {
        top_data_i[j * inner_num_ + k] /= scale_data_i[k];
      }
    }
  }

  // 后处理独立reduce：概率分布统计（out值域/最大概率/熵/sum校验）
  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();
  float sum_max_prob = 0.0f;
  float sum_entropy = 0.0f;
  int n_samples = outer_num_ * inner_num_;
  for (int i = 0; i < outer_num_; ++i) {
    const float* top_data_i = top_data + i * dim;
    for (int k = 0; k < inner_num_; ++k) {
      float sample_max = 0.0f;
      float sample_entropy = 0.0f;
      for (int j = 0; j < channels; ++j) {
        float p = top_data_i[j * inner_num_ + k];
        out_min = std::min(out_min, p);
        out_max = std::max(out_max, p);
        sample_max = std::max(sample_max, p);
        if (p > 0.0f) {
          sample_entropy -= p * std::log(p);
        }
      }
      sum_max_prob += sample_max;
      sum_entropy += sample_entropy;
    }
  }
  float avg_max_prob = sum_max_prob / static_cast<float>(n_samples);
  float avg_entropy = sum_entropy / static_cast<float>(n_samples);
  float max_entropy = std::log(static_cast<float>(channels));

  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[SOFTMAX-PERF] " << this->name()
                       << " Softmax forward: outer_num=" << outer_num_
                       << " channels=" << channels
                       << " inner_num=" << inner_num_
                       << " axis=" << softmax_axis_
                       << " out=[" << out_min << ", " << out_max << "]"
                       << " avg_max_prob=" << avg_max_prob
                       << " avg_entropy=" << avg_entropy
                       << " max_entropy=" << max_entropy
                       << " time=" << elapsed_us << "us";
}

void SoftmaxLayer::Backward_cpu(const std::vector<Blob*>& top,
                                 const std::vector<bool>& propagate_down,
                                 const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    CAFFE_FFI_LAYER_LOG << "Softmax Backward_cpu: propagate_down[0]=false, skipping";
    return;
  }

  const float* top_diff = top[0]->cpu_diff();
  const float* top_data = top[0]->cpu_data();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const int channels = static_cast<int>(bottom[0]->shape(softmax_axis_));
  const int dim = channels * inner_num_;
  const int64_t count = bottom[0]->count();

  CAFFE_FFI_LAYER_LOG << "Softmax Backward_cpu: outer_num=" << outer_num_
                      << " channels=" << channels
                      << " inner_num=" << inner_num_
                      << " axis=" << softmax_axis_
                      << " count=" << count;

  auto t_start = std::chrono::high_resolution_clock::now();

  // Softmax Jacobian-vector product:
  //   y_i = exp(x_i) / sum_j exp(x_j)
  //   J_{ij} = dy_i/dx_j = y_i * (delta_{ij} - y_j)
  //   dx_i = sum_j (dy_j * J_{ji}) = y_i * (dy_i - sum_j(dy_j * y_j))
  //         = y_i * (dy_i - dot),  where dot = dy · y per (outer, inner) position
  //
  // Data layout: index = i*dim + j*inner_num_ + k
  //   i: outer index (batch), j: channel (softmax axis), k: inner (spatial)

  float dx_min = std::numeric_limits<float>::max();
  float dx_max = -std::numeric_limits<float>::max();
  double sum_sq = 0.0;

  for (int i = 0; i < outer_num_; ++i) {
    const float* top_diff_i = top_diff + i * dim;
    const float* top_data_i = top_data + i * dim;
    float* bottom_diff_i = bottom_diff + i * dim;

    for (int k = 0; k < inner_num_; ++k) {
      // Compute dot = sum_j(dy_j * y_j) for this (i, k) position
      float dot = 0.0f;
      for (int j = 0; j < channels; ++j) {
        dot += top_diff_i[j * inner_num_ + k] * top_data_i[j * inner_num_ + k];
      }
      // Compute dx_j = y_j * (dy_j - dot)
      for (int j = 0; j < channels; ++j) {
        float yj = top_data_i[j * inner_num_ + k];
        float dyj = top_diff_i[j * inner_num_ + k];
        float val = yj * (dyj - dot);
        bottom_diff_i[j * inner_num_ + k] = val;
        dx_min = std::min(dx_min, val);
        dx_max = std::max(dx_max, val);
        sum_sq += static_cast<double>(val) * static_cast<double>(val);
      }
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
  float grad_norm = static_cast<float>(std::sqrt(sum_sq));

  CAFFE_FFI_LOG_INFO() << "[SOFTMAX-PERF] " << this->name()
                       << " Softmax backward: outer_num=" << outer_num_
                       << " channels=" << channels
                       << " inner_num=" << inner_num_
                       << " dx=[" << dx_min << ", " << dx_max << "]"
                       << " grad_l2norm=" << grad_norm
                       << " time=" << elapsed_us << "us";
}

REGISTER_LAYER_CLASS(Softmax);

}  // namespace caffe_ffi
