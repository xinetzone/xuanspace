#include "caffe_ffi/layers/concat_layer.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"

namespace caffe_ffi {

void ConcatLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                              const std::vector<Blob*>& top) {
  const caffe::ConcatParameter& param = this->layer_param_.concat_param();
  if (param.has_axis()) {
    concat_axis_ = bottom[0]->CanonicalAxisIndex(param.axis());
  } else {
    concat_axis_ = bottom[0]->CanonicalAxisIndex(param.concat_dim());
  }
  CAFFE_FFI_LAYER_LOG << "Concat LayerSetUp: concat_axis=" << concat_axis_;
}

void ConcatLayer::Reshape(const std::vector<Blob*>& bottom,
                           const std::vector<Blob*>& top) {
  const int num_axes = bottom[0]->num_axes();
  CAFFE_FFI_CHECK_VALUE_GE(concat_axis_, 0);
  CAFFE_FFI_CHECK_VALUE_LT(concat_axis_, num_axes);

  std::vector<int64_t> top_shape(num_axes);
  for (int i = 0; i < num_axes; ++i) {
    top_shape[i] = bottom[0]->shape(i);
  }
  top_shape[concat_axis_] = 0;

  const int num_bottoms = static_cast<int>(bottom.size());
  concat_offsets_.resize(num_bottoms + 1);
  concat_offsets_[0] = 0;

  for (int i = 0; i < num_bottoms; ++i) {
    CAFFE_FFI_CHECK_VALUE_EQ(bottom[i]->num_axes(), num_axes)
        << "All bottom blobs must have the same number of axes.";
    for (int j = 0; j < num_axes; ++j) {
      if (j == concat_axis_) continue;
      CAFFE_FFI_CHECK_VALUE_EQ(bottom[i]->shape(j), top_shape[j])
          << "All bottom blobs must have matching dimensions except along concat axis.";
    }
    top_shape[concat_axis_] += bottom[i]->shape(concat_axis_);
    concat_offsets_[i + 1] = top_shape[concat_axis_];
  }

  top[0]->Reshape(top_shape);

  outer_count_ = bottom[0]->count(0, concat_axis_);
  inner_count_ = bottom[0]->count(concat_axis_ + 1);

  std::ostringstream top_shape_ss;
  for (int i = 0; i < num_axes; ++i) {
    if (i > 0) top_shape_ss << ", ";
    top_shape_ss << top_shape[i];
  }
  CAFFE_FFI_LAYER_LOG << "Concat Reshape: num_bottoms=" << num_bottoms
                      << " concat_axis=" << concat_axis_
                      << " outer_count=" << outer_count_
                      << " inner_count=" << inner_count_
                      << " output shape=[" << top_shape_ss.str() << "]";

  std::ostringstream offsets_ss;
  for (int i = 0; i <= num_bottoms; ++i) {
    if (i > 0) offsets_ss << ", ";
    offsets_ss << concat_offsets_[i];
  }
  CAFFE_FFI_LAYER_LOG << "Concat: concat_offsets=[" << offsets_ss.str() << "]";
}

void ConcatLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                               const std::vector<Blob*>& top) {
  float* top_data = top[0]->cpu_mutable_data();
  const int num_bottoms = static_cast<int>(bottom.size());
  const int64_t total_concat = top[0]->shape(concat_axis_);

  CAFFE_FFI_LAYER_LOG << "Concat Forward: num_bottoms=" << num_bottoms
                      << " concat_axis=" << concat_axis_
                      << " outer_count=" << outer_count_
                      << " inner_count=" << inner_count_
                      << " total_concat=" << total_concat;

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();
  int64_t top_count = top[0]->count();
#endif

  for (int i = 0; i < num_bottoms; ++i) {
    const float* bottom_data = bottom[i]->cpu_data();
    const int64_t concat_dim = bottom[i]->shape(concat_axis_);
    const int64_t copy_size = concat_dim * inner_count_;
    const int64_t offset_concat = concat_offsets_[i];

    for (int64_t n = 0; n < outer_count_; ++n) {
      const int64_t src_offset = n * copy_size;
      const int64_t dst_offset = (n * total_concat + offset_concat) * inner_count_;
      std::memcpy(top_data + dst_offset, bottom_data + src_offset,
                  sizeof(float) * copy_size);
    }
  }

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  for (int64_t i = 0; i < top_count; ++i) {
    out_min = std::min(out_min, top_data[i]);
    out_max = std::max(out_max, top_data[i]);
  }

  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[CONCAT-PERF] " << this->name()
                       << " Concat forward: num_bottoms=" << num_bottoms
                       << " concat_axis=" << concat_axis_
                       << " out=[" << out_min << ", " << out_max << "]"
                       << " time=" << elapsed_us << "us";
#endif
}

void ConcatLayer::Backward_cpu(const std::vector<Blob*>& top,
                                const std::vector<bool>& propagate_down,
                                const std::vector<Blob*>& bottom) {
  const float* top_diff = top[0]->cpu_diff();
  const int num_bottoms = static_cast<int>(bottom.size());
  const int64_t total_concat = top[0]->shape(concat_axis_);

  CAFFE_FFI_LAYER_LOG << "Concat Backward_cpu: num_bottoms=" << num_bottoms
                      << " concat_axis=" << concat_axis_
                      << " outer_count=" << outer_count_
                      << " inner_count=" << inner_count_
                      << " total_concat=" << total_concat;

  bool any_propagate = false;
  for (int j = 0; j < num_bottoms; ++j) {
    if (propagate_down[j]) { any_propagate = true; break; }
  }
  if (!any_propagate) {
    CAFFE_FFI_LAYER_LOG << "Concat Backward_cpu: no gradients needed, skipping";
    return;
  }

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  float dx_min = std::numeric_limits<float>::max();
  float dx_max = -std::numeric_limits<float>::max();
#endif

  for (int i = 0; i < num_bottoms; ++i) {
    if (!propagate_down[i]) continue;
    float* bottom_diff = bottom[i]->cpu_mutable_diff();
    const int64_t concat_dim = bottom[i]->shape(concat_axis_);
    const int64_t copy_size = concat_dim * inner_count_;
    const int64_t offset_concat = concat_offsets_[i];

    for (int64_t n = 0; n < outer_count_; ++n) {
      const int64_t src_offset = (n * total_concat + offset_concat) * inner_count_;
      const int64_t dst_offset = n * copy_size;
      std::memcpy(bottom_diff + dst_offset, top_diff + src_offset,
                  sizeof(float) * copy_size);
    }

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
    const int64_t bottom_count = bottom[i]->count();
    for (int64_t j = 0; j < bottom_count; ++j) {
      dx_min = std::min(dx_min, bottom_diff[j]);
      dx_max = std::max(dx_max, bottom_diff[j]);
    }
#endif
  }

#ifdef CAFFE_FFI_ENABLE_PERF_LOG
  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[CONCAT-PERF] " << this->name()
                       << " Concat backward: num_bottoms=" << num_bottoms
                       << " concat_axis=" << concat_axis_
                       << " dx=[" << dx_min << ", " << dx_max << "]"
                       << " time=" << elapsed_us << "us";
#endif
}

REGISTER_LAYER_CLASS(Concat);

}  // namespace caffe_ffi
