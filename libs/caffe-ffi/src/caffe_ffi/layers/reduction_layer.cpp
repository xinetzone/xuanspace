#include "caffe_ffi/layers/reduction_layer.hpp"

#include <cmath>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void ReductionLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                                const std::vector<Blob*>& top) {
  const caffe::ReductionParameter& param = this->layer_param_.reduction_param();
  operation_ = param.operation();

  CAFFE_FFI_LAYER_LOG << "Reduction LayerSetUp: operation=" << operation_;
}

void ReductionLayer::Reshape(const std::vector<Blob*>& bottom,
                             const std::vector<Blob*>& top) {
  const int axis = bottom[0]->CanonicalAxisIndex(this->layer_param_.reduction_param().axis());

  std::vector<int64_t> top_shape;
  for (int i = 0; i < axis; ++i) {
    top_shape.push_back(bottom[0]->shape(i));
  }
  top[0]->Reshape(top_shape);

  num_ = bottom[0]->count(0, axis);
  dim_ = bottom[0]->count(axis);
  coeff_ = this->layer_param_.reduction_param().coeff();
  if (operation_ == caffe::ReductionParameter_ReductionOp_MEAN) {
    coeff_ /= dim_;
  }

  CAFFE_FFI_LAYER_LOG << "Reduction Reshape: axis=" << axis
                      << " num_=" << num_ << " dim_=" << dim_
                      << " coeff_=" << coeff_;
}

void ReductionLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                 const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();

  for (int64_t i = 0; i < num_; ++i) {
    const float* group = bottom_data + i * dim_;
    float sum = 0.0f;
    if (operation_ == caffe::ReductionParameter_ReductionOp_SUM ||
        operation_ == caffe::ReductionParameter_ReductionOp_MEAN) {
      for (int64_t j = 0; j < dim_; ++j) {
        sum += group[j];
      }
    } else if (operation_ == caffe::ReductionParameter_ReductionOp_ASUM) {
      for (int64_t j = 0; j < dim_; ++j) {
        sum += std::fabs(group[j]);
      }
    } else {  // SUMSQ
      for (int64_t j = 0; j < dim_; ++j) {
        sum += group[j] * group[j];
      }
    }
    top_data[i] = coeff_ * sum;
  }

  CAFFE_FFI_LAYER_LOG << "Reduction Forward: num_=" << num_ << " dim_=" << dim_
                      << " operation=" << operation_ << " coeff_=" << coeff_;
}

void ReductionLayer::Backward_cpu(const std::vector<Blob*>& top,
                                  const std::vector<bool>& propagate_down,
                                  const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    CAFFE_FFI_LAYER_LOG << "Reduction Backward_cpu: propagate_down[0]=false, skipping";
    return;
  }

  const float* bottom_data = bottom[0]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();

  for (int64_t i = 0; i < num_; ++i) {
    const float c = top_diff[i] * coeff_;
    for (int64_t j = 0; j < dim_; ++j) {
      if (operation_ == caffe::ReductionParameter_ReductionOp_SUM ||
          operation_ == caffe::ReductionParameter_ReductionOp_MEAN) {
        bottom_diff[i * dim_ + j] = c;
      } else if (operation_ == caffe::ReductionParameter_ReductionOp_ASUM) {
        const float v = bottom_data[i * dim_ + j];
        bottom_diff[i * dim_ + j] = (v > 0.0f) ? c : ((v < 0.0f) ? -c : 0.0f);
      } else {  // SUMSQ
        bottom_diff[i * dim_ + j] = 2.0f * c * bottom_data[i * dim_ + j];
      }
    }
  }

  CAFFE_FFI_LAYER_LOG << "Reduction Backward: num_=" << num_ << " dim_=" << dim_
                      << " operation=" << operation_ << " coeff_=" << coeff_;
}

REGISTER_LAYER_CLASS(Reduction);

}  // namespace caffe_ffi