#include "caffe_ffi/layers/reshape_layer.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/math_utils.hpp"

namespace caffe_ffi {

void ReshapeLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                               const std::vector<Blob*>& top) {
  const caffe::ReshapeParameter& param = this->layer_param_.reshape_param();
  axis_ = param.axis();
  num_axes_ = param.num_axes();

  std::ostringstream shape_ss;
  const caffe::BlobShape& shape = param.shape();
  for (int i = 0; i < shape.dim_size(); ++i) {
    if (i > 0) shape_ss << ", ";
    shape_ss << shape.dim(i);
  }
  CAFFE_FFI_LAYER_LOG << "Reshape LayerSetUp: axis=" << axis_
                      << " num_axes=" << num_axes_
                      << " shape=[" << shape_ss.str() << "]";
}

void ReshapeLayer::Reshape(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  const Blob* input_blob = bottom[0];
  const int input_num_axes = input_blob->num_axes();
  const caffe::ReshapeParameter& param = this->layer_param_.reshape_param();
  const caffe::BlobShape& shape = param.shape();
  const int shape_dim_size = shape.dim_size();

  int start_axis = CanonicalAxisIndex(axis_, input_num_axes + 1);
  CAFFE_FFI_CHECK_VALUE_GE(start_axis, 0);
  CAFFE_FFI_CHECK_VALUE_LE(start_axis, input_num_axes);

  int end_axis;
  if (num_axes_ == -1) {
    end_axis = input_num_axes;
  } else {
    end_axis = start_axis + num_axes_;
    end_axis = std::min(end_axis, input_num_axes);
  }
  CAFFE_FFI_CHECK_VALUE_GE(end_axis, start_axis);

  std::vector<int64_t> top_shape;
  for (int i = 0; i < start_axis; ++i) {
    top_shape.push_back(input_blob->shape(i));
  }

  int inferred_axis = -1;
  int64_t constant_count = 1;
  for (int i = 0; i < shape_dim_size; ++i) {
    int dim = static_cast<int>(shape.dim(i));
    if (dim == 0) {
      CAFFE_FFI_CHECK_VALUE_LT(start_axis + i, input_num_axes)
          << "dim=0 (copy axis) out of input bounds";
      top_shape.push_back(input_blob->shape(start_axis + i));
      constant_count *= top_shape.back();
    } else if (dim == -1) {
      CAFFE_FFI_CHECK_VALUE_EQ(inferred_axis, -1)
          << "Reshape shape contains multiple -1 dims";
      inferred_axis = top_shape.size();
      top_shape.push_back(0);
    } else {
      CAFFE_FFI_CHECK_VALUE_GT(dim, 0) << "Reshape dim must be positive, -1, or 0";
      top_shape.push_back(dim);
      constant_count *= dim;
    }
  }

  for (int i = end_axis; i < input_num_axes; ++i) {
    top_shape.push_back(input_blob->shape(i));
  }

  int64_t input_region_count = 1;
  for (int i = start_axis; i < end_axis; ++i) {
    input_region_count *= input_blob->shape(i);
  }

  if (inferred_axis >= 0) {
    CAFFE_FFI_CHECK_VALUE_GT(constant_count, 0);
    CAFFE_FFI_CHECK_VALUE_EQ(input_region_count % constant_count, 0)
        << "Cannot infer reshape dim: input count not divisible by constant count";
    top_shape[inferred_axis] = input_region_count / constant_count;
  } else {
    CAFFE_FFI_CHECK_VALUE_EQ(input_region_count, constant_count)
        << "Reshape count mismatch";
  }

  top[0]->Reshape(top_shape);

  std::ostringstream bottom_shape_ss;
  for (int i = 0; i < input_num_axes; ++i) {
    if (i > 0) bottom_shape_ss << ", ";
    bottom_shape_ss << input_blob->shape(i);
  }
  std::ostringstream top_shape_ss;
  for (int i = 0; i < static_cast<int>(top_shape.size()); ++i) {
    if (i > 0) top_shape_ss << ", ";
    top_shape_ss << top_shape[i];
  }
  CAFFE_FFI_LAYER_LOG << "Reshape Reshape: start_axis=" << start_axis
                      << " end_axis=" << end_axis
                      << " inferred_axis=" << inferred_axis
                      << " input=[" << bottom_shape_ss.str() << "]"
                      << " output=[" << top_shape_ss.str() << "]"
                      << " count=" << top[0]->count();
}

void ReshapeLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_data();
  const int64_t count = bottom[0]->count();
  CAFFE_FFI_LAYER_LOG << "Reshape Forward: count=" << count
                      << " inplace=" << (bottom[0] == top[0] ? "true" : "false");
  if (bottom[0] != top[0]) {
    std::memcpy(top_data, bottom_data, sizeof(float) * count);
  }
}

REGISTER_LAYER_CLASS(Reshape);

}  // namespace caffe_ffi
