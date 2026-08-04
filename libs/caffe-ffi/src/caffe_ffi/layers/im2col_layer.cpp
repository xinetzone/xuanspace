#include "caffe_ffi/layers/im2col_layer.hpp"

#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/math_utils.hpp"

namespace caffe_ffi {

void Im2colLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                             const std::vector<Blob*>& top) {
  const caffe::Im2colParameter& param = this->layer_param_.im2col_param();

  if (param.has_kernel_h() || param.has_kernel_w()) {
    kernel_h_ = param.has_kernel_h() ? static_cast<int>(param.kernel_h()) : 1;
    kernel_w_ = param.has_kernel_w() ? static_cast<int>(param.kernel_w()) : 1;
  } else {
    kernel_h_ = param.kernel_size().size() > 0 ? static_cast<int>(param.kernel_size(0)) : 1;
    kernel_w_ = param.kernel_size().size() > 1 ? static_cast<int>(param.kernel_size(1)) : kernel_h_;
  }

  if (param.has_stride_h() || param.has_stride_w()) {
    stride_h_ = param.has_stride_h() ? static_cast<int>(param.stride_h()) : 1;
    stride_w_ = param.has_stride_w() ? static_cast<int>(param.stride_w()) : 1;
  } else if (param.stride_size() > 0) {
    stride_h_ = param.stride_size() > 0 ? static_cast<int>(param.stride(0)) : 1;
    stride_w_ = param.stride_size() > 1 ? static_cast<int>(param.stride(1)) : stride_h_;
  } else {
    stride_h_ = 1;
    stride_w_ = 1;
  }

  if (param.has_pad_h() || param.has_pad_w()) {
    pad_h_ = param.has_pad_h() ? static_cast<int>(param.pad_h()) : 0;
    pad_w_ = param.has_pad_w() ? static_cast<int>(param.pad_w()) : 0;
  } else if (param.pad_size() > 0) {
    pad_h_ = param.pad_size() > 0 ? static_cast<int>(param.pad(0)) : 0;
    pad_w_ = param.pad_size() > 1 ? static_cast<int>(param.pad(1)) : pad_h_;
  } else {
    pad_h_ = 0;
    pad_w_ = 0;
  }

  if (param.has_dilation_h() || param.has_dilation_w()) {
    dilation_h_ = param.has_dilation_h() ? static_cast<int>(param.dilation_h()) : 1;
    dilation_w_ = param.has_dilation_w() ? static_cast<int>(param.dilation_w()) : 1;
  } else if (param.dilation_size() > 0) {
    dilation_h_ = param.dilation_size() > 0 ? static_cast<int>(param.dilation(0)) : 1;
    dilation_w_ = param.dilation_size() > 1 ? static_cast<int>(param.dilation(1)) : dilation_h_;
  } else {
    dilation_h_ = 1;
    dilation_w_ = 1;
  }

  CAFFE_FFI_LAYER_LOG << "Im2col LayerSetUp: kernel=" << kernel_h_ << "x" << kernel_w_
                      << " pad=" << pad_h_ << "x" << pad_w_
                      << " stride=" << stride_h_ << "x" << stride_w_
                      << " dilation=" << dilation_h_ << "x" << dilation_w_;
}

void Im2colLayer::Reshape(const std::vector<Blob*>& bottom,
                          const std::vector<Blob*>& top) {
  channels_ = static_cast<int>(bottom[0]->shape(1));
  height_ = static_cast<int>(bottom[0]->shape(2));
  width_ = static_cast<int>(bottom[0]->shape(3));
  num_ = bottom[0]->count(0, 1);

  const int out_h = (height_ + 2 * pad_h_ - (dilation_h_ * (kernel_h_ - 1) + 1)) / stride_h_ + 1;
  const int out_w = (width_ + 2 * pad_w_ - (dilation_w_ * (kernel_w_ - 1) + 1)) / stride_w_ + 1;

  std::vector<int64_t> top_shape = {
      num_, static_cast<int64_t>(channels_) * kernel_h_ * kernel_w_, out_h, out_w};
  top[0]->Reshape(top_shape);

  bottom_dim_ = static_cast<int64_t>(channels_) * height_ * width_;
  top_dim_ = static_cast<int64_t>(channels_) * kernel_h_ * kernel_w_ * out_h * out_w;

  CAFFE_FFI_LAYER_LOG << "Im2col Reshape: num_=" << num_ << " channels_=" << channels_
                      << " in=" << height_ << "x" << width_
                      << " out=" << out_h << "x" << out_w
                      << " bottom_dim_=" << bottom_dim_ << " top_dim_=" << top_dim_;
}

void Im2colLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                              const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();

  for (int64_t n = 0; n < num_; ++n) {
    im2col_fp32(bottom_data + n * bottom_dim_, channels_, height_, width_,
                kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
                dilation_h_, dilation_w_, top_data + n * top_dim_);
  }

  CAFFE_FFI_LAYER_LOG << "Im2col Forward: num_=" << num_ << " bottom_dim_=" << bottom_dim_
                      << " top_dim_=" << top_dim_;
}

void Im2colLayer::Backward_cpu(const std::vector<Blob*>& top,
                               const std::vector<bool>& propagate_down,
                               const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    CAFFE_FFI_LAYER_LOG << "Im2col Backward_cpu: propagate_down[0]=false, skipping";
    return;
  }

  const float* top_diff = top[0]->cpu_diff();
  float* bottom_diff = bottom[0]->cpu_mutable_diff();

  for (int64_t n = 0; n < num_; ++n) {
    col2im_fp32(top_diff + n * top_dim_, channels_, height_, width_,
                kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
                dilation_h_, dilation_w_, bottom_diff + n * bottom_dim_);
  }

  CAFFE_FFI_LAYER_LOG << "Im2col Backward: num_=" << num_ << " bottom_dim_=" << bottom_dim_
                      << " top_dim_=" << top_dim_;
}

REGISTER_LAYER_CLASS(Im2col);

}  // namespace caffe_ffi