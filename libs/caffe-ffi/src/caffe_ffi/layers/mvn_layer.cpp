#include "caffe_ffi/layers/mvn_layer.hpp"

#include <cmath>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

void MVNLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                          const std::vector<Blob*>& top) {
  const caffe::MVNParameter& param = this->layer_param_.mvn_param();
  normalize_variance_ = param.normalize_variance();
  across_channels_ = param.across_channels();
  eps_ = param.eps();

  CAFFE_FFI_LAYER_LOG << "MVN LayerSetUp: normalize_variance=" << normalize_variance_
                      << " across_channels=" << across_channels_
                      << " eps=" << eps_;
}

void MVNLayer::Reshape(const std::vector<Blob*>& bottom,
                       const std::vector<Blob*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  CAFFE_FFI_LAYER_LOG << "MVN Reshape: input count=" << bottom[0]->count()
                      << " output count=" << top[0]->count();
}

void MVNLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                           const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int64_t num = across_channels_ ? bottom[0]->num()
                                       : bottom[0]->num() * bottom[0]->channels();
  const int64_t dim = bottom[0]->count() / num;
  std_.resize(num);

  for (int64_t g = 0; g < num; ++g) {
    const float* group = bottom_data + g * dim;
    float* top_group = top_data + g * dim;

    double sum = 0.0;
    for (int64_t j = 0; j < dim; ++j) {
      sum += static_cast<double>(group[j]);
    }
    const float mean = static_cast<float>(sum / static_cast<double>(dim));

    double sum_sq = 0.0;
    for (int64_t j = 0; j < dim; ++j) {
      const float d = group[j] - mean;
      top_group[j] = d;
      sum_sq += static_cast<double>(d) * static_cast<double>(d);
    }

    if (normalize_variance_) {
      const float std = std::sqrt(static_cast<float>(sum_sq / static_cast<double>(dim))) + eps_;
      std_[g] = std;
      for (int64_t j = 0; j < dim; ++j) {
        top_group[j] /= std;
      }
    } else {
      std_[g] = 1.0f;
    }
  }

  CAFFE_FFI_LAYER_LOG << "MVN Forward: num=" << num << " dim=" << dim
                      << " normalize_variance=" << normalize_variance_;
}

void MVNLayer::Backward_cpu(const std::vector<Blob*>& top,
                            const std::vector<bool>& propagate_down,
                            const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    CAFFE_FFI_LAYER_LOG << "MVN Backward_cpu: propagate_down[0]=false, skipping";
    return;
  }

  const float* top_data = top[0]->cpu_data();
  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const int64_t num = across_channels_ ? bottom[0]->num()
                                       : bottom[0]->num() * bottom[0]->channels();
  const int64_t dim = bottom[0]->count() / num;

  for (int64_t g = 0; g < num; ++g) {
    const float* y = top_data + g * dim;
    const float* dy = top_diff + g * dim;
    float* dx = bottom_diff + g * dim;

    if (normalize_variance_) {
      // Recompute the input mean (from bottom_data) for the y-term is not needed:
      // y is the normalized output already. Use the cached per-group std.
      double sum_dy = 0.0;
      double sum_dy_y = 0.0;
      for (int64_t j = 0; j < dim; ++j) {
        sum_dy += static_cast<double>(dy[j]);
        sum_dy_y += static_cast<double>(dy[j]) * static_cast<double>(y[j]);
      }
      const float mean_dy = static_cast<float>(sum_dy / static_cast<double>(dim));
      const float mean_dy_y = static_cast<float>(sum_dy_y / static_cast<double>(dim));
      const float std = std_[g];
      for (int64_t j = 0; j < dim; ++j) {
        dx[j] = (dy[j] - mean_dy - y[j] * mean_dy_y) / std;
      }
    } else {
      double sum_dy = 0.0;
      for (int64_t j = 0; j < dim; ++j) {
        sum_dy += static_cast<double>(dy[j]);
      }
      const float mean_dy = static_cast<float>(sum_dy / static_cast<double>(dim));
      for (int64_t j = 0; j < dim; ++j) {
        dx[j] = dy[j] - mean_dy;
      }
    }
  }

  CAFFE_FFI_LAYER_LOG << "MVN Backward: num=" << num << " dim=" << dim
                      << " normalize_variance=" << normalize_variance_;
}

REGISTER_LAYER_CLASS(MVN);

}  // namespace caffe_ffi