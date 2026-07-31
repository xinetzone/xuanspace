#include "caffe_ffi/layers/crop_layer.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <sstream>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/fill.hpp"

namespace caffe_ffi {

void CropLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  const caffe::CropParameter& param = this->layer_param_.crop_param();
  CAFFE_FFI_CHECK_VALUE_EQ(bottom.size(), 2U) << "Wrong number of bottom blobs.";
  int input_dim = static_cast<int>(bottom[0]->num_axes());
  int axis = param.axis();
  if (axis < 0) axis += input_dim;
  const int start_axis = axis;
  CAFFE_FFI_CHECK_VALUE_LT(start_axis, input_dim) << "crop axis bigger than input dim";
  if (param.offset_size() > 1) {
    CAFFE_FFI_CHECK_VALUE_EQ(start_axis + param.offset_size(), input_dim)
        << "number of offset values specified must be equal to the number of "
        << "dimensions following axis.";
  }
}

void CropLayer::Reshape(const std::vector<Blob*>& bottom,
                         const std::vector<Blob*>& top) {
  const caffe::CropParameter& param = this->layer_param_.crop_param();
  int input_dim = static_cast<int>(bottom[0]->num_axes());
  int axis = param.axis();
  if (axis < 0) axis += input_dim;
  const int start_axis = axis;

  std::vector<int64_t> new_shape;
  offsets_.resize(input_dim);
  src_strides_.resize(input_dim);
  dest_strides_.resize(input_dim);

  for (int i = 0; i < input_dim; ++i) {
    int64_t crop_offset = 0;
    int64_t new_size = bottom[0]->shape(i);
    if (i >= start_axis) {
      new_size = bottom[1]->shape(i);
      if (param.offset_size() == 1) {
        crop_offset = static_cast<int64_t>(param.offset(0));
      } else if (param.offset_size() > 1) {
        crop_offset = static_cast<int64_t>(param.offset(i - start_axis));
      }
      CAFFE_FFI_CHECK_VALUE_GE(bottom[0]->shape(i) - crop_offset, bottom[1]->shape(i))
          << "the crop for dimension " << i << " is out-of-bounds with "
          << "size " << bottom[1]->shape(i) << " and offset " << crop_offset;
    }
    new_shape.push_back(new_size);
    offsets_[i] = crop_offset;
  }
  top[0]->Reshape(new_shape);

  for (int i = 0; i < input_dim; ++i) {
    src_strides_[i] = bottom[0]->count(i + 1);
    dest_strides_[i] = top[0]->count(i + 1);
  }

  std::ostringstream ss;
  ss << "Crop Reshape: start_axis=" << start_axis
     << " offsets=[";
  for (int i = 0; i < input_dim; ++i) {
    if (i > 0) ss << ", ";
    ss << offsets_[i];
  }
  ss << "] top_shape=[";
  for (int i = 0; i < input_dim; ++i) {
    if (i > 0) ss << ", ";
    ss << new_shape[i];
  }
  ss << "]";
  CAFFE_FFI_LAYER_LOG << ss.str();
}

void CropLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                             const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int num_axes = static_cast<int>(top[0]->num_axes());
  const int64_t last_dim = top[0]->shape(num_axes - 1);

  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  std::vector<int64_t> idx(num_axes, 0);
  int64_t total_copied = 0;

  while (true) {
    int64_t src_off = 0, dst_off = 0;
    for (int d = 0; d < num_axes; ++d) {
      src_off += (idx[d] + offsets_[d]) * src_strides_[d];
      dst_off += idx[d] * dest_strides_[d];
    }
    caffe_copy_fp32(static_cast<size_t>(last_dim), bottom_data + src_off, top_data + dst_off);
    total_copied += last_dim;

    int d;
    for (d = num_axes - 2; d >= 0; --d) {
      idx[d]++;
      if (idx[d] < top[0]->shape(d)) break;
      idx[d] = 0;
    }
    if (d < 0) break;
  }

  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();
  int64_t top_count = top[0]->count();
  for (int64_t i = 0; i < top_count; ++i) {
    out_min = std::min(out_min, top_data[i]);
    out_max = std::max(out_max, top_data[i]);
  }

  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[CROP-PERF] " << this->name()
                       << " Crop forward: elements_copied=" << total_copied
                       << " out=[" << out_min << ", " << out_max << "]"
                       << " time=" << elapsed_us << "us";
}

void CropLayer::Backward_cpu(const std::vector<Blob*>& top,
                              const std::vector<bool>& propagate_down,
                              const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    CAFFE_FFI_LAYER_LOG << "Crop Backward: propagate_down[0]=false, skipping";
    return;
  }

  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const float* top_diff = top[0]->cpu_diff();
  const int num_axes = static_cast<int>(top[0]->num_axes());
  const int64_t last_dim = top[0]->shape(num_axes - 1);

  caffe_set_fp32(static_cast<size_t>(bottom[0]->count()), 0.0f, bottom_diff);

  using clock = std::chrono::high_resolution_clock;
  auto t_start = clock::now();

  std::vector<int64_t> idx(num_axes, 0);
  int64_t total_copied = 0;

  while (true) {
    int64_t src_off = 0, dst_off = 0;
    for (int d = 0; d < num_axes; ++d) {
      dst_off += (idx[d] + offsets_[d]) * src_strides_[d];
      src_off += idx[d] * dest_strides_[d];
    }
    caffe_copy_fp32(static_cast<size_t>(last_dim), top_diff + src_off, bottom_diff + dst_off);
    total_copied += last_dim;

    int d;
    for (d = num_axes - 2; d >= 0; --d) {
      idx[d]++;
      if (idx[d] < top[0]->shape(d)) break;
      idx[d] = 0;
    }
    if (d < 0) break;
  }

  float diff_min = std::numeric_limits<float>::max();
  float diff_max = -std::numeric_limits<float>::max();
  int64_t bottom_count = bottom[0]->count();
  for (int64_t i = 0; i < bottom_count; ++i) {
    diff_min = std::min(diff_min, bottom_diff[i]);
    diff_max = std::max(diff_max, bottom_diff[i]);
  }

  auto t_end = clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[CROP-PERF] " << this->name()
                       << " Crop backward: elements_copied=" << total_copied
                       << " diff=[" << diff_min << ", " << diff_max << "]"
                       << " time=" << elapsed_us << "us";
}

REGISTER_LAYER_CLASS(Crop);

}  // namespace caffe_ffi
