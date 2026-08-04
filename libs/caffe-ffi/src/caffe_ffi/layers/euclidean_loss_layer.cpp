#include "caffe_ffi/layers/euclidean_loss_layer.hpp"

#include <vector>

#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/math_utils.hpp"

namespace caffe_ffi {

void EuclideanLossLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                                    const std::vector<Blob*>& top) {
  CAFFE_FFI_LAYER_LOG << "EuclideanLoss LayerSetUp";
}

void EuclideanLossLayer::Reshape(const std::vector<Blob*>& bottom,
                                 const std::vector<Blob*>& top) {
  CAFFE_FFI_CHECK_VALUE_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "EuclideanLoss inputs must have the same spatial dimension.";
  diff_ = make_object<Blob>();
  diff_->ReshapeLike(*bottom[0]);
  std::vector<int64_t> loss_shape = {1};
  top[0]->Reshape(loss_shape);
  CAFFE_FFI_LAYER_LOG << "EuclideanLoss Reshape: count=" << bottom[0]->count()
                      << " num=" << bottom[0]->shape(0);
}

void EuclideanLossLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                      const std::vector<Blob*>& top) {
  const int count = bottom[0]->count();
  caffe_sub_fp32(static_cast<int64_t>(count), bottom[0]->cpu_data(), bottom[1]->cpu_data(),
                 diff_->cpu_mutable_data());
  const float dot = caffe_cpu_dot_fp32(static_cast<size_t>(count),
                                       diff_->cpu_data(), diff_->cpu_data());
  const float loss = dot / static_cast<float>(bottom[0]->shape(0)) / 2.0f;
  top[0]->cpu_mutable_data()[0] = loss;
  CAFFE_FFI_LAYER_LOG << "EuclideanLoss Forward: dot=" << dot
                      << " loss=" << loss;
}

void EuclideanLossLayer::Backward_cpu(const std::vector<Blob*>& top,
                                      const std::vector<bool>& propagate_down,
                                      const std::vector<Blob*>& bottom) {
  const float loss_weight = top[0]->cpu_diff()[0];
  for (int i = 0; i < 2; ++i) {
    if (!propagate_down[i]) {
      continue;
    }
    const float sign = (i == 0) ? 1.0f : -1.0f;
    const float alpha = sign * loss_weight / static_cast<float>(bottom[i]->shape(0));
    const int count = bottom[i]->count();
    float* bottom_diff = bottom[i]->cpu_mutable_diff();
    for (int j = 0; j < count; ++j) {
      bottom_diff[j] = alpha * diff_->cpu_data()[j];
    }
    CAFFE_FFI_LAYER_LOG << "EuclideanLoss Backward: bottom[" << i << "]"
                        << " alpha=" << alpha;
  }
}

REGISTER_LAYER_CLASS(EuclideanLoss);

}  // namespace caffe_ffi