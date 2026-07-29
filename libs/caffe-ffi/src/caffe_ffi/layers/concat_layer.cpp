#include "caffe_ffi/layers/concat_layer.hpp"

#include <algorithm>
#include <cstring>
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
  float* top_data = top[0]->cpu_data();
  const int num_bottoms = static_cast<int>(bottom.size());
  const int64_t total_concat = top[0]->shape(concat_axis_);

  CAFFE_FFI_LAYER_LOG << "Concat Forward: num_bottoms=" << num_bottoms
                      << " concat_axis=" << concat_axis_
                      << " outer_count=" << outer_count_
                      << " inner_count=" << inner_count_
                      << " total_concat=" << total_concat;

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
}

REGISTER_LAYER_CLASS(Concat);

}  // namespace caffe_ffi
