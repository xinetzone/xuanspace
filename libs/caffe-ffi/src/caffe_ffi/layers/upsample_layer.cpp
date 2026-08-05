#include "caffe_ffi/layers/upsample_layer.hpp"

#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"

namespace caffe_ffi {

void UpsampleLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                               const std::vector<Blob*>& top) {
  scale_ = this->layer_param_.upsample_param().scale();
  CAFFE_FFI_CHECK_VALUE_GT(scale_, 0) << "Upsample scale must be > 0, got " << scale_;
  CAFFE_FFI_LAYER_LOG << "Upsample LayerSetUp: scale=" << scale_;
}

void UpsampleLayer::Reshape(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  scale_ = this->layer_param_.upsample_param().scale();
  CAFFE_FFI_CHECK_VALUE_GT(scale_, 0) << "Upsample scale must be > 0, got " << scale_;
  CAFFE_FFI_CHECK_VALUE_EQ(bottom[0]->num_axes(), 4)
      << "Upsample expects a 4-D (N, C, H, W) input, got " << bottom[0]->num_axes()
      << " axes.";

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  std::vector<int64_t> top_shape = {num, channels, height * scale_, width * scale_};
  top[0]->Reshape(top_shape);
  CAFFE_FFI_LAYER_LOG << "Upsample Reshape: bottom=(" << num << "," << channels << ","
                      << height << "," << width << ") top=(" << num << "," << channels
                      << "," << height * scale_ << "," << width * scale_
                      << ") scale=" << scale_;
}

void UpsampleLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int top_h = height * scale_;
  const int top_w = width * scale_;

  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ih = 0; ih < height; ++ih) {
        for (int iw = 0; iw < width; ++iw) {
          const float v =
              bottom_data[((n * channels + c) * height + ih) * width + iw];
          for (int sh = 0; sh < scale_; ++sh) {
            for (int sw = 0; sw < scale_; ++sw) {
              top_data[((n * channels + c) * top_h + ih * scale_ + sh) * top_w +
                       iw * scale_ + sw] = v;
            }
          }
        }
      }
    }
  }
  CAFFE_FFI_LAYER_LOG << "Upsample Forward: scale=" << scale_;
}

void UpsampleLayer::Backward_cpu(const std::vector<Blob*>& top,
                                 const std::vector<bool>& propagate_down,
                                 const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    CAFFE_FFI_LAYER_LOG << "Upsample Backward_cpu: propagate_down[0]=false, skipping";
    return;
  }
  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int top_h = height * scale_;
  const int top_w = width * scale_;

  // Zero the accumulated bottom gradient first.
  caffe_set_fp32(static_cast<size_t>(bottom[0]->count()), 0.0f, bottom_diff);

  // Each output pixel copies the source pixel, so the source gradient is the
  // sum of the upstream gradient over its scale x scale block.
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ih = 0; ih < height; ++ih) {
        for (int iw = 0; iw < width; ++iw) {
          float acc = 0.0f;
          for (int sh = 0; sh < scale_; ++sh) {
            for (int sw = 0; sw < scale_; ++sw) {
              acc += top_diff[((n * channels + c) * top_h + ih * scale_ + sh) * top_w +
                              iw * scale_ + sw];
            }
          }
          bottom_diff[((n * channels + c) * height + ih) * width + iw] = acc;
        }
      }
    }
  }
  CAFFE_FFI_LAYER_LOG << "Upsample Backward: scale=" << scale_;
}

REGISTER_LAYER_CLASS(Upsample);

}  // namespace caffe_ffi