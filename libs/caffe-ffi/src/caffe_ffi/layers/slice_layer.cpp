#include "caffe_ffi/layers/slice_layer.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <sstream>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/math_utils.hpp"

namespace caffe_ffi {

void SliceLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                             const std::vector<Blob*>& top) {
  const caffe::SliceParameter& slice_param = this->layer_param_.slice_param();
  CAFFE_FFI_CHECK_VALUE(!(slice_param.has_axis() && slice_param.has_slice_dim()))
      << "Either axis or slice_dim should be specified; not both.";

  slice_point_.clear();
  for (int i = 0; i < slice_param.slice_point_size(); ++i) {
    slice_point_.push_back(static_cast<int64_t>(slice_param.slice_point(i)));
  }
}

void SliceLayer::Reshape(const std::vector<Blob*>& bottom,
                          const std::vector<Blob*>& top) {
  const int num_axes = static_cast<int>(bottom[0]->num_axes());
  const caffe::SliceParameter& slice_param = this->layer_param_.slice_param();

  if (slice_param.has_slice_dim()) {
    slice_axis_ = static_cast<int>(slice_param.slice_dim());
    CAFFE_FFI_CHECK_VALUE_GE(slice_axis_, 0) << "slice_dim must be >= 0";
    CAFFE_FFI_CHECK_VALUE_LT(slice_axis_, num_axes) << "slice_dim out of range.";
  } else {
    int axis = slice_param.axis();
    if (axis < 0) axis += num_axes;
    slice_axis_ = axis;
  }

  std::vector<int64_t> top_shape;
  for (int i = 0; i < num_axes; ++i) {
    top_shape.push_back(bottom[0]->shape(i));
  }
  const int64_t bottom_slice_axis = bottom[0]->shape(slice_axis_);
  num_slices_ = bottom[0]->count(0, slice_axis_);
  slice_size_ = bottom[0]->count(slice_axis_ + 1);
  int64_t count = 0;

  if (slice_point_.size() != 0) {
    CAFFE_FFI_CHECK_VALUE_EQ(static_cast<int>(slice_point_.size()), static_cast<int>(top.size()) - 1);
    CAFFE_FFI_CHECK_VALUE_LE(static_cast<int>(top.size()), static_cast<int>(bottom_slice_axis));
    int64_t prev = 0;
    std::vector<int64_t> slices;
    for (int i = 0; i < static_cast<int>(slice_point_.size()); ++i) {
      CAFFE_FFI_CHECK_VALUE_GT(slice_point_[i], prev);
      slices.push_back(slice_point_[i] - prev);
      prev = slice_point_[i];
    }
    slices.push_back(bottom_slice_axis - prev);
    for (int i = 0; i < static_cast<int>(top.size()); ++i) {
      top_shape[slice_axis_] = slices[i];
      top[i]->Reshape(top_shape);
      count += top[i]->count();
    }
  } else {
    CAFFE_FFI_CHECK_VALUE_EQ(bottom_slice_axis % static_cast<int64_t>(top.size()), 0)
        << "Number of top blobs (" << top.size() << ") should evenly "
        << "divide input slice axis (" << bottom_slice_axis << ")";
    top_shape[slice_axis_] = bottom_slice_axis / static_cast<int64_t>(top.size());
    for (int i = 0; i < static_cast<int>(top.size()); ++i) {
      top[i]->Reshape(top_shape);
      count += top[i]->count();
    }
  }

  CAFFE_FFI_CHECK_VALUE_EQ(count, bottom[0]->count());
  count_ = count;

  std::ostringstream ss;
  ss << "Slice Reshape: slice_axis=" << slice_axis_
     << " num_slices=" << num_slices_
     << " slice_size=" << slice_size_
     << " bottom_slice_axis=" << bottom_slice_axis
     << " top count=" << top.size();
  if (slice_point_.size() > 0) {
    ss << " slice_points=[";
    for (int i = 0; i < static_cast<int>(slice_point_.size()); ++i) {
      if (i > 0) ss << ", ";
      ss << slice_point_[i];
    }
    ss << "]";
  }
  CAFFE_FFI_LAYER_LOG << ss.str();

  if (top.size() == 1) {
    // N=1 Slice is identity/passthrough (same as N=1 Split).
    // Use Identity share so mutable access on top[0] does NOT trigger COW --
    // in-place writes propagate directly to bottom[0].
    top[0]->ShareDataIdentity(bottom[0]);
    top[0]->ShareDiffIdentity(bottom[0]);
  }
}

void SliceLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                              const std::vector<Blob*>& top) {
  if (top.size() == 1) {
    CAFFE_FFI_LAYER_LOG << "Slice Forward: single top, shared data, no copy needed";
    return;
  }

  const float* bottom_data = bottom[0]->cpu_data();
  const int64_t bottom_slice_axis = bottom[0]->shape(slice_axis_);
  int64_t offset_slice_axis = 0;

  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  float in_min = std::numeric_limits<float>::max();
  float in_max = -std::numeric_limits<float>::max();
  int64_t total_copied = 0;

  for (int i = 0; i < static_cast<int>(top.size()); ++i) {
    float* top_data = top[i]->cpu_mutable_data();
    const int64_t top_slice_axis = top[i]->shape(slice_axis_);
    for (int64_t n = 0; n < num_slices_; ++n) {
      const int64_t top_offset = n * top_slice_axis * slice_size_;
      const int64_t bottom_offset =
          (n * bottom_slice_axis + offset_slice_axis) * slice_size_;
      caffe_copy_fp32(top_slice_axis * slice_size_,
                      bottom_data + bottom_offset, top_data + top_offset);
      total_copied += top_slice_axis * slice_size_;
    }
    offset_slice_axis += top_slice_axis;
  }

  for (int64_t i = 0; i < bottom[0]->count(); ++i) {
    in_min = std::min(in_min, bottom_data[i]);
    in_max = std::max(in_max, bottom_data[i]);
  }

  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[SLICE-PERF] " << this->name()
                       << " Slice forward: top_count=" << top.size()
                       << " slice_axis=" << slice_axis_
                       << " num_slices=" << num_slices_
                       << " slice_size=" << slice_size_
                       << " elements_copied=" << total_copied
                       << " in=[" << in_min << ", " << in_max << "]"
                       << " time=" << elapsed_us << "us";
}

void SliceLayer::Backward_cpu(const std::vector<Blob*>& top,
                               const std::vector<bool>& propagate_down,
                               const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    CAFFE_FFI_LAYER_LOG << "Slice Backward: propagate_down[0]=false, skipping";
    return;
  }
  if (top.size() == 1) {
    // N=1 identity backward: if diff is still shared (identity alias), gradients
    // written to top[0] are already visible in bottom[0] -- zero-copy passthrough.
    if (top[0]->SharesDiffWith(bottom[0])) {
      CAFFE_FFI_LAYER_LOG << "Slice Backward(N=1 IDENTITY): zero-copy passthrough, no copy needed";
      return;
    }
    // Fallback: COW already happened, copy diff down.
    float* bottom_diff = bottom[0]->cpu_mutable_diff();
    const float* top_diff = top[0]->cpu_diff();
    if (top_diff != bottom_diff) {
      caffe_copy_fp32(static_cast<size_t>(bottom[0]->count()), top_diff, bottom_diff);
    }
    CAFFE_FFI_LAYER_LOG << "Slice Backward(N=1): "
                        << (top_diff != bottom_diff ? "copied diff (COW detected)" : "zero-copy passthrough");
    return;
  }

  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const int64_t bottom_slice_axis = bottom[0]->shape(slice_axis_);
  int64_t offset_slice_axis = 0;

  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  float diff_min = std::numeric_limits<float>::max();
  float diff_max = -std::numeric_limits<float>::max();
  int64_t total_copied = 0;

  for (int i = 0; i < static_cast<int>(top.size()); ++i) {
    const float* top_diff = top[i]->cpu_diff();
    const int64_t top_slice_axis = top[i]->shape(slice_axis_);
    for (int64_t n = 0; n < num_slices_; ++n) {
      const int64_t top_offset = n * top_slice_axis * slice_size_;
      const int64_t bottom_offset =
          (n * bottom_slice_axis + offset_slice_axis) * slice_size_;
      caffe_copy_fp32(top_slice_axis * slice_size_,
                      top_diff + top_offset, bottom_diff + bottom_offset);
      total_copied += top_slice_axis * slice_size_;
    }
    offset_slice_axis += top_slice_axis;
  }

  for (int64_t i = 0; i < bottom[0]->count(); ++i) {
    diff_min = std::min(diff_min, bottom_diff[i]);
    diff_max = std::max(diff_max, bottom_diff[i]);
  }

  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[SLICE-PERF] " << this->name()
                       << " Slice backward: top_count=" << top.size()
                       << " slice_axis=" << slice_axis_
                       << " num_slices=" << num_slices_
                       << " slice_size=" << slice_size_
                       << " elements_copied=" << total_copied
                       << " diff=[" << diff_min << ", " << diff_max << "]"
                       << " time=" << elapsed_us << "us";
}

REGISTER_LAYER_CLASS(Slice);

}  // namespace caffe_ffi
