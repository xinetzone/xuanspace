#include "caffe_ffi/layers/hinge_layer.hpp"

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

void HingeLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  const caffe::HingeParameter& param = this->layer_param_.hinge_param();
  is_l2_ = (param.norm() == caffe::HingeParameter::L2);
  axis_ = bottom[0]->CanonicalAxisIndex(param.axis());

  CAFFE_FFI_LAYER_LOG << "Hinge LayerSetUp: is_l2_=" << is_l2_
                      << " axis_=" << axis_;
}

void HingeLayer::Reshape(const std::vector<Blob*>& bottom,
                         const std::vector<Blob*>& top) {
  // Validate label shape: same as data except channel dim must be 1 on axis.
  CAFFE_FFI_CHECK_VALUE_EQ(bottom[0]->num_axes(), bottom[1]->num_axes())
      << "Hinge data and label must have the same number of axes.";
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i == axis_) {
      CAFFE_FFI_CHECK_VALUE_EQ(bottom[1]->shape(i), 1)
          << "Hinge label channel dimension must be 1.";
    } else {
      CAFFE_FFI_CHECK_VALUE_EQ(bottom[0]->shape(i), bottom[1]->shape(i))
          << "Hinge data and label dimensions mismatch at axis " << i;
    }
  }

  outer_num_ = static_cast<int>(bottom[0]->count(0, axis_));
  inner_num_ = static_cast<int>(bottom[0]->count(axis_ + 1));
  count_ = outer_num_ * inner_num_;

  std::vector<int64_t> loss_shape = {1};
  top[0]->Reshape(loss_shape);

  std::ostringstream data_shape_ss;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) data_shape_ss << ", ";
    data_shape_ss << bottom[0]->shape(i);
  }

  CAFFE_FFI_LAYER_LOG << "Hinge Reshape: data=[" << data_shape_ss.str()
                      << "] axis_=" << axis_
                      << " outer_num_=" << outer_num_
                      << " inner_num_=" << inner_num_
                      << " count_=" << count_
                      << " is_l2_=" << is_l2_;
}

void HingeLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                             const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  const float* label = bottom[1]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int channels = static_cast<int>(bottom[0]->shape(axis_));
  const int dim = channels * inner_num_;
  const int outer_num = outer_num_;
  const int inner_num = inner_num_;
  const bool is_l2 = is_l2_;

  CAFFE_FFI_LAYER_LOG << "Hinge Forward: outer_num=" << outer_num
                      << " inner_num=" << inner_num
                      << " channels=" << channels
                      << " is_l2=" << is_l2;

  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  double total = 0.0;
  long active = 0;  // number of non-zero violation terms
  float loss_min = std::numeric_limits<float>::max();
  float loss_max = -std::numeric_limits<float>::max();

  for (int i = 0; i < outer_num; ++i) {
    const float* data_i = bottom_data + i * dim;
    for (int j = 0; j < inner_num; ++j) {
      const int label_value = static_cast<int>(label[i * inner_num + j]);
      CAFFE_FFI_CHECK_VALUE_GE(label_value, 0);
      CAFFE_FFI_CHECK_VALUE_LT(label_value, channels);

      const float score_truth = data_i[label_value * inner_num + j];
      double sample_loss = 0.0;
      for (int c = 0; c < channels; ++c) {
        if (c == label_value) {
          continue;
        }
        const float z = 1.0f + data_i[c * inner_num + j] - score_truth;
        if (z > 0.0f) {
          sample_loss += is_l2 ? static_cast<double>(z) * static_cast<double>(z)
                               : static_cast<double>(z);
          ++active;
        }
      }
      total += sample_loss;
      const float l = static_cast<float>(sample_loss);
      loss_min = std::min(loss_min, l);
      loss_max = std::max(loss_max, l);
    }
  }

  const float mean_loss = static_cast<float>(total / static_cast<double>(count_));
  top_data[0] = mean_loss;

  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[HINGE-PERF] " << this->name()
                       << " Hinge forward: samples=" << count_
                       << " channels=" << channels
                       << " norm=" << (is_l2 ? "L2" : "L1")
                       << " active_terms=" << active
                       << " per-sample loss=[" << loss_min << ", " << loss_max << "]"
                       << " mean_loss=" << mean_loss
                       << " top0=" << top_data[0]
                       << " time=" << elapsed_us << "us";
}

void HingeLayer::Backward_cpu(const std::vector<Blob*>& top,
                              const std::vector<bool>& propagate_down,
                              const std::vector<Blob*>& bottom) {
  // The label bottom (index 1) never receives gradients.
  if (propagate_down.size() > 1 && propagate_down[1]) {
    CAFFE_FFI_LOG_WARN() << "Hinge Backward: cannot backpropagate to label inputs.";
  }

  if (!propagate_down[0]) {
    CAFFE_FFI_LAYER_LOG << "Hinge Backward_cpu: propagate_down[0]=false, skipping";
    return;
  }

  const float* bottom_data = bottom[0]->cpu_data();
  const float* label = bottom[1]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const int channels = static_cast<int>(bottom[0]->shape(axis_));
  const int dim = channels * inner_num_;
  const int outer_num = outer_num_;
  const int inner_num = inner_num_;
  const bool is_l2 = is_l2_;
  const float loss_weight = top_diff[0];

  CAFFE_FFI_LAYER_LOG << "Hinge Backward: outer_num=" << outer_num
                      << " inner_num=" << inner_num
                      << " channels=" << channels
                      << " is_l2=" << is_l2
                      << " loss_weight=" << loss_weight;

  // loss = mean_i ( sum_{c != y} z_c )  with z_c = max(0, 1 + score_c - score_y)
  // d(loss_weight * loss)/dscore_c (c != y): += loss_weight * d(z_c)/count_
  //   L1: d(z_c)/dscore_c = 1 if z_c > 0 else 0
  //   L2: d(z_c)/dscore_c = 2*z_c if z_c > 0 else 0
  // dscore_y -= sum_{c != y} of the above.
  const float scale = loss_weight / static_cast<float>(count_);

  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  caffe_set_fp32(static_cast<size_t>(bottom[0]->count()), 0.0f, bottom_diff);

  for (int i = 0; i < outer_num; ++i) {
    const float* data_i = bottom_data + i * dim;
    float* diff_i = bottom_diff + i * dim;
    for (int j = 0; j < inner_num; ++j) {
      const int label_value = static_cast<int>(label[i * inner_num + j]);
      const float score_truth = data_i[label_value * inner_num + j];

      float acc = 0.0f;
      for (int c = 0; c < channels; ++c) {
        if (c == label_value) {
          continue;
        }
        const float z = 1.0f + data_i[c * inner_num + j] - score_truth;
        if (z > 0.0f) {
          const float dz = is_l2 ? (2.0f * z) : 1.0f;
          const float g = scale * dz;
          diff_i[c * inner_num + j] += g;
          acc += g;
        }
      }
      diff_i[label_value * inner_num + j] -= acc;
    }
  }

  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[HINGE-PERF] " << this->name()
                       << " Hinge backward: samples=" << count_
                       << " channels=" << channels
                       << " norm=" << (is_l2 ? "L2" : "L1")
                       << " loss_weight=" << loss_weight
                       << " scale=" << scale
                       << " time=" << elapsed_us << "us";
}

REGISTER_LAYER_CLASS(Hinge);

}  // namespace caffe_ffi