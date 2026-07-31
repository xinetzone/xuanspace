#include "caffe_ffi/layers/base_conv_layer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>
#include <string>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/common.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

void BaseConvolutionLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                                      const std::vector<Blob*>& top) {
  const caffe::ConvolutionParameter& conv_param =
      layer_param_.convolution_param();
  CAFFE_FFI_CHECK_VALUE(conv_param.has_kernel_h() == conv_param.has_kernel_w())
      << "Filter size must be defined for both H and W dimensions.";
  if (conv_param.has_kernel_h() && conv_param.has_kernel_w()) {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  } else {
    if (conv_param.kernel_size_size() > 0) {
      kernel_h_ = kernel_w_ = conv_param.kernel_size(0);
    } else {
      kernel_h_ = kernel_w_ = 1;
    }
  }
  if (conv_param.has_stride_h() && conv_param.has_stride_w()) {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  } else {
    if (conv_param.stride_size() > 0) {
      stride_h_ = stride_w_ = conv_param.stride(0);
    } else {
      stride_h_ = stride_w_ = 1;
    }
  }
  if (conv_param.has_pad_h() && conv_param.has_pad_w()) {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  } else {
    if (conv_param.pad_size() > 0) {
      pad_h_ = pad_w_ = conv_param.pad(0);
    } else {
      pad_h_ = pad_w_ = 0;
    }
  }
  if (conv_param.has_dilation_h() && conv_param.has_dilation_w()) {
    dilation_h_ = conv_param.dilation_h();
    dilation_w_ = conv_param.dilation_w();
  } else {
    if (conv_param.dilation_size() > 0) {
      dilation_h_ = dilation_w_ = conv_param.dilation(0);
    } else {
      dilation_h_ = dilation_w_ = 1;
    }
  }

  num_output_ = conv_param.num_output();
  bias_term_ = conv_param.bias_term();
  group_ = conv_param.group();
  channels_ = static_cast<int>(bottom[0]->shape(1));
  height_ = static_cast<int>(bottom[0]->shape(2));
  width_ = static_cast<int>(bottom[0]->shape(3));
  num_ = static_cast<int>(bottom[0]->shape(0));

  CAFFE_FFI_CHECK_VALUE_EQ(channels_ % group_, 0)
      << "Number of input channels should be divided by group.";
  CAFFE_FFI_CHECK_VALUE_EQ(num_output_ % group_, 0)
      << "Number of output channels should be divided by group.";

  // Swap channels for deconvolution
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }

  // 1x1 detection (stride=1, pad=0, dilation=1, kernel=1)
  is_1x1_ = (kernel_h_ == 1 && kernel_w_ == 1 && stride_h_ == 1 && stride_w_ == 1 &&
             pad_h_ == 0 && pad_w_ == 0 && dilation_h_ == 1 && dilation_w_ == 1);

  // Weight shape: [conv_out_channels_, conv_in_channels_/group_, Kh, Kw]
  // This is the same layout as Caffe: for Conv it's [Co, Ci/g, Kh, Kw];
  // for Deconv it's [Ci, Co/g, Kh, Kw] (same memory layout, interpreted differently)
  std::vector<int64_t> weight_shape{
      conv_out_channels_, conv_in_channels_ / group_, kernel_h_, kernel_w_};
  kernel_dim_ = static_cast<int>(weight_shape[1] * weight_shape[2] * weight_shape[3]);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;

  if (bias_term_) {
    this->blobs_.resize(2);
  } else {
    this->blobs_.resize(1);
  }
  this->blobs_[0] = make_object<Blob>(weight_shape);
  if (bias_term_) {
    std::vector<int64_t> bias_shape{num_output_};
    this->blobs_[1] = make_object<Blob>(bias_shape);
    caffe_set_fp32(static_cast<size_t>(blobs_[1]->count()), 0.F, blobs_[1]->cpu_mutable_data());
  }
}

void BaseConvolutionLayer::Reshape(const std::vector<Blob*>& bottom,
                                   const std::vector<Blob*>& top) {
  height_ = static_cast<int>(bottom[0]->shape(2));
  width_ = static_cast<int>(bottom[0]->shape(3));
  num_ = static_cast<int>(bottom[0]->shape(0));
  channels_ = static_cast<int>(bottom[0]->shape(1));

  compute_output_shape();

  std::vector<int64_t> top_shape{num_, num_output_, output_h_, output_w_};
  top[0]->Reshape(top_shape);

  bottom_dim_ = channels_ * height_ * width_;
  top_dim_ = num_output_ * output_h_ * output_w_;
  out_spatial_dim_ = output_h_ * output_w_;

  // The critical difference between Conv and Deconv:
  // - Conv: conv_out_spatial_dim = Ho*Wo (top spatial), im2col from bottom[Ci,H,W]
  // - Deconv: conv_out_spatial_dim = H*W (bottom spatial), col2im writes to top[Co,Ho,Wo]
  //
  // conv_input_shape_ = [C_im, H_im, W_im] describes the image that im2col reads from
  // and col2im writes to. For Conv it is the bottom; for Deconv (which swaps fwd/bwd)
  // it is the top. C_im = conv_in_channels_ = Ci for Conv, = Co = num_output_ for Deconv.
  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = height_ * width_;
    conv_input_shape_ = {num_output_, output_h_, output_w_};
    col_buffer_shape_ = {kernel_dim_ * group_, height_, width_};
  } else {
    conv_out_spatial_dim_ = output_h_ * output_w_;
    conv_input_shape_ = {channels_, height_, width_};
    col_buffer_shape_ = {kernel_dim_ * group_, output_h_, output_w_};
  }
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;

  std::vector<int64_t> col_buf_shape{col_buffer_shape_[0], col_buffer_shape_[1], col_buffer_shape_[2]};
  col_buffer_.Reshape(col_buf_shape);

  if (bias_term_) {
    std::vector<int64_t> bias_multiplier_shape{out_spatial_dim_};
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set_fp32(static_cast<size_t>(bias_multiplier_.count()), 1.F, bias_multiplier_.cpu_mutable_data());
  }
}

// ---------------------------------------------------------------------------
// Convolution-perspective GEMM helpers (identical for Conv and Deconv;
// Deconv reverses direction by calling backward_cpu_gemm from Forward, etc.)
// ---------------------------------------------------------------------------

void BaseConvolutionLayer::forward_cpu_gemm(const float* input,
                                            const float* weights,
                                            float* output,
                                            bool skip_im2col) {
  const float* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      im2col_cpu(input, conv_in_channels_, conv_input_h(), conv_input_w(),
                 kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
                 dilation_h_, dilation_w_, col_buffer_.cpu_mutable_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm(false, false, conv_out_channels_ / group_,
                   conv_out_spatial_dim_, kernel_dim_, 1.F,
                   weights + weight_offset_ * g, col_buff + col_offset_ * g,
                   0.F, output + output_offset_ * g);
  }
}

void BaseConvolutionLayer::forward_cpu_bias(float* output,
                                            const float* bias) {
  caffe_cpu_gemm(false, false, num_output_, out_spatial_dim_,
                 1, 1.F, bias, bias_multiplier_.cpu_data(), 1.F, output);
}

void BaseConvolutionLayer::backward_cpu_gemm(const float* output,
                                             const float* weights,
                                             float* input) {
  float* col_buff = col_buffer_.cpu_mutable_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm(true, false, kernel_dim_, conv_out_spatial_dim_,
                   conv_out_channels_ / group_, 1.F,
                   weights + weight_offset_ * g, output + output_offset_ * g,
                   0.F, col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    col2im_cpu(col_buff, conv_in_channels_, conv_input_h(), conv_input_w(),
               kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
               dilation_h_, dilation_w_, input);
  }
}

void BaseConvolutionLayer::weight_cpu_gemm(const float* input,
                                           const float* output,
                                           float* weights) {
  const float* col_buff = input;
  if (!is_1x1_) {
    im2col_cpu(input, conv_in_channels_, conv_input_h(), conv_input_w(),
               kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
               dilation_h_, dilation_w_, col_buffer_.cpu_mutable_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm(false, true, conv_out_channels_ / group_,
                   kernel_dim_, conv_out_spatial_dim_, 1.F,
                   output + output_offset_ * g, col_buff + col_offset_ * g,
                   1.F, weights + weight_offset_ * g);
  }
}

void BaseConvolutionLayer::backward_cpu_bias(float* bias, const float* input) {
  caffe_cpu_gemv(false, num_output_, out_spatial_dim_, 1.F, input,
                 bias_multiplier_.cpu_data(), 1.F, bias);
}

}  // namespace caffe_ffi
