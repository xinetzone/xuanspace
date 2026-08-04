#include "caffe_ffi/layers/margin_ranking_layer.hpp"

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

void MarginRankingLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                                    const std::vector<Blob*>& top) {
  const caffe::MarginRankingParameter& param = this->layer_param_.margin_ranking_param();
  margin_ = param.margin();
  sign_ = param.sign();

  CAFFE_FFI_LAYER_LOG << "MarginRanking LayerSetUp: margin_=" << margin_
                      << " sign_=" << sign_;
}

void MarginRankingLayer::Reshape(const std::vector<Blob*>& bottom,
                                 const std::vector<Blob*>& top) {
  // x1 and x2 must have identical shapes.
  CAFFE_FFI_CHECK_VALUE_EQ(bottom[0]->count(), bottom[1]->count())
      << "MarginRanking x1 and x2 must have the same number of elements.";
  // Label must be element-wise (same element count as x1/x2).
  CAFFE_FFI_CHECK_VALUE_EQ(bottom[2]->count(), bottom[0]->count())
      << "MarginRanking label must have the same number of elements as x1/x2.";

  std::vector<int64_t> loss_shape = {1};
  top[0]->Reshape(loss_shape);
  count_ = static_cast<int>(bottom[0]->count());

  std::ostringstream x1_shape_ss;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) x1_shape_ss << ", ";
    x1_shape_ss << bottom[0]->shape(i);
  }

  CAFFE_FFI_LAYER_LOG << "MarginRanking Reshape: x1=[" << x1_shape_ss.str()
                      << "] count_=" << count_
                      << " margin_=" << margin_
                      << " sign_=" << sign_;
}

void MarginRankingLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                     const std::vector<Blob*>& top) {
  const float* x1 = bottom[0]->cpu_data();
  const float* x2 = bottom[1]->cpu_data();
  const float* label = bottom[2]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int count = count_;
  const float margin = margin_;
  const int sign = sign_;

  CAFFE_FFI_LAYER_LOG << "MarginRanking Forward: count=" << count
                      << " margin=" << margin << " sign=" << sign;

  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  // loss_i = max(0, -y_i * (x1_i - x2_i) + margin)
  double total = 0.0;
  int active = 0;
  float loss_min = std::numeric_limits<float>::max();
  float loss_max = -std::numeric_limits<float>::max();
  for (int i = 0; i < count; ++i) {
    const float y = label[i];
    const float v = -y * (x1[i] - x2[i]) + margin;
    const float l = v > 0.0f ? v : 0.0f;
    total += static_cast<double>(l);
    if (l > 0.0f) ++active;
    loss_min = std::min(loss_min, l);
    loss_max = std::max(loss_max, l);
  }

  const float mean_loss = static_cast<float>(total / static_cast<double>(count));
  top_data[0] = static_cast<float>(sign) * mean_loss;

  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[MARGINRANKING-PERF] " << this->name()
                       << " MarginRanking forward: count=" << count
                       << " margin=" << margin
                       << " sign=" << sign
                       << " active=" << active
                       << " per-element loss=[" << loss_min << ", " << loss_max << "]"
                       << " mean_loss=" << mean_loss
                       << " top0=" << top_data[0]
                       << " time=" << elapsed_us << "us";
}

void MarginRankingLayer::Backward_cpu(const std::vector<Blob*>& top,
                                      const std::vector<bool>& propagate_down,
                                      const std::vector<Blob*>& bottom) {
  // The label bottom (index 2) never receives gradients.
  if (propagate_down.size() > 2 && propagate_down[2]) {
    CAFFE_FFI_LOG_WARN() << "MarginRanking Backward: cannot backpropagate to label inputs.";
  }

  const bool need_dx1 = propagate_down[0];
  const bool need_dx2 = propagate_down[1];
  if (!need_dx1 && !need_dx2) {
    CAFFE_FFI_LAYER_LOG << "MarginRanking Backward_cpu: no data gradients needed, skipping";
    return;
  }

  const float* x1 = bottom[0]->cpu_data();
  const float* x2 = bottom[1]->cpu_data();
  const float* label = bottom[2]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  float* x1_diff = need_dx1 ? bottom[0]->cpu_mutable_diff() : nullptr;
  float* x2_diff = need_dx2 ? bottom[1]->cpu_mutable_diff() : nullptr;
  const int count = count_;
  const float margin = margin_;
  const int sign = sign_;
  const float loss_weight = top_diff[0];

  CAFFE_FFI_LAYER_LOG << "MarginRanking Backward: count=" << count
                      << " margin=" << margin << " sign=" << sign
                      << " loss_weight=" << loss_weight
                      << " need_dx1=" << need_dx1 << " need_dx2=" << need_dx2;

  // Forward: loss = sign * mean_i( max(0, -y_i*(x1_i - x2_i) + margin) )
  // Backward (per element, d(loss_weight * mean_loss)/d·):
  //   dx1_i = loss_weight * sign * (-y_i * mask_i) / count
  //   dx2_i = loss_weight * sign * ( y_i * mask_i) / count
  //   mask_i = 1 if loss_i > 0 else 0
  const float scale = loss_weight * static_cast<float>(sign) / static_cast<float>(count);

  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  for (int i = 0; i < count; ++i) {
    const float y = label[i];
    const float v = -y * (x1[i] - x2[i]) + margin;
    const float mask = v > 0.0f ? 1.0f : 0.0f;
    const float g = scale * mask * y;
    if (need_dx1) {
      x1_diff[i] = -g;
    }
    if (need_dx2) {
      x2_diff[i] = g;
    }
  }

  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[MARGINRANKING-PERF] " << this->name()
                       << " MarginRanking backward: count=" << count
                       << " loss_weight=" << loss_weight
                       << " scale=" << scale
                       << " time=" << elapsed_us << "us";
}

REGISTER_LAYER_CLASS(MarginRanking);

}  // namespace caffe_ffi