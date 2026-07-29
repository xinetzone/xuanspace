#include "caffe_ffi/layers/conv_layer.hpp"

#include <sstream>
#include <vector>

#include <tvm/ffi/memory.h>

#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/error.hpp"
#include "caffe_ffi/math_utils.hpp"

namespace caffe_ffi {

void ConvolutionLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                                   const std::vector<Blob*>& top) {
  const caffe::ConvolutionParameter& conv_param = this->layer_param_.convolution_param();
  num_output_ = static_cast<int>(conv_param.num_output());
  bias_term_ = conv_param.bias_term();
  group_ = static_cast<int>(conv_param.group());

  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    kernel_h_ = static_cast<int>(conv_param.kernel_h());
    kernel_w_ = static_cast<int>(conv_param.kernel_w());
  } else {
    kernel_h_ = kernel_w_ = static_cast<int>(conv_param.kernel_size(0));
  }

  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    stride_h_ = static_cast<int>(conv_param.stride_h());
    stride_w_ = static_cast<int>(conv_param.stride_w());
  } else if (conv_param.stride_size() > 0) {
    stride_h_ = stride_w_ = static_cast<int>(conv_param.stride(0));
  } else {
    stride_h_ = stride_w_ = 1;
  }

  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    pad_h_ = static_cast<int>(conv_param.pad_h());
    pad_w_ = static_cast<int>(conv_param.pad_w());
  } else if (conv_param.pad_size() > 0) {
    pad_h_ = pad_w_ = static_cast<int>(conv_param.pad(0));
  } else {
    pad_h_ = pad_w_ = 0;
  }

  if (conv_param.has_dilation_h() || conv_param.has_dilation_w()) {
    dilation_h_ = static_cast<int>(conv_param.dilation_h());
    dilation_w_ = static_cast<int>(conv_param.dilation_w());
  } else if (conv_param.dilation_size() > 0) {
    dilation_h_ = dilation_w_ = static_cast<int>(conv_param.dilation(0));
  } else {
    dilation_h_ = dilation_w_ = 1;
  }

  channels_ = static_cast<int>(bottom[0]->shape(1));
  CAFFE_FFI_CHECK_VALUE_EQ(channels_ % group_, 0)
      << "Input channels should be divided by group.";
  conv_out_channels_ = num_output_;
  conv_in_channels_ = channels_ / group_;

  kernel_dim_ = conv_in_channels_ * kernel_h_ * kernel_w_;
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  is_1x1_ = kernel_h_ == 1 && kernel_w_ == 1 && pad_h_ == 0 && pad_w_ == 0
            && stride_h_ == 1 && stride_w_ == 1 && dilation_h_ == 1 && dilation_w_ == 1;

  CAFFE_FFI_LAYER_LOG << "Convolution LayerSetUp: num_output=" << num_output_
                      << " kernel=[" << kernel_h_ << "," << kernel_w_ << "]"
                      << " stride=[" << stride_h_ << "," << stride_w_ << "]"
                      << " pad=[" << pad_h_ << "," << pad_w_ << "]"
                      << " dilation=[" << dilation_h_ << "," << dilation_w_ << "]"
                      << " group=" << group_
                      << " bias_term=" << bias_term_
                      << " is_1x1=" << is_1x1_
                      << " channels=" << channels_
                      << " conv_in_channels=" << conv_in_channels_
                      << " kernel_dim=" << kernel_dim_;

  if (this->blobs_.size() > 0) {
    CAFFE_FFI_LAYER_LOG << "Convolution: using pre-loaded weights, blobs_.size=" << this->blobs_.size();
    CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_.size(), bias_term_ ? 2U : 1U)
        << "Incorrect number of weight blobs.";
    CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_[0]->shape(0), num_output_);
    CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_[0]->shape(1), kernel_dim_);
    if (bias_term_) {
      CAFFE_FFI_CHECK_VALUE_EQ(this->blobs_[1]->count(), num_output_);
    }
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    std::vector<int64_t> weight_shape = {num_output_, kernel_dim_};
    this->blobs_[0] = make_object<Blob>(weight_shape);
    CAFFE_FFI_TENSOR_LOG << "Convolution: created weight blob shape=["
                         << weight_shape[0] << ", " << weight_shape[1] << "]";
    if (bias_term_) {
      std::vector<int64_t> bias_shape = {num_output_};
      this->blobs_[1] = make_object<Blob>(bias_shape);
      CAFFE_FFI_TENSOR_LOG << "Convolution: created bias blob shape=[" << bias_shape[0] << "]";
    }
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

void ConvolutionLayer::Reshape(const std::vector<Blob*>& bottom,
                                const std::vector<Blob*>& top) {
  height_ = static_cast<int>(bottom[0]->shape(2));
  width_ = static_cast<int>(bottom[0]->shape(3));

  output_h_ = (height_ + 2 * pad_h_ - dilation_h_ * (kernel_h_ - 1) - 1) / stride_h_ + 1;
  output_w_ = (width_ + 2 * pad_w_ - dilation_w_ * (kernel_w_ - 1) - 1) / stride_w_ + 1;

  CAFFE_FFI_CHECK_VALUE_GT(output_h_, 0) << "Output height should be positive.";
  CAFFE_FFI_CHECK_VALUE_GT(output_w_, 0) << "Output width should be positive.";

  conv_out_spatial_dim_ = output_h_ * output_w_;

  std::vector<int64_t> top_shape = {bottom[0]->shape(0), num_output_, output_h_, output_w_};
  top[0]->Reshape(top_shape);

  std::ostringstream top_shape_ss;
  for (int i = 0; i < static_cast<int>(top_shape.size()); ++i) {
    if (i > 0) top_shape_ss << ", ";
    top_shape_ss << top_shape[i];
  }
  CAFFE_FFI_LAYER_LOG << "Convolution Reshape: input=[" << bottom[0]->shape(0)
                      << "," << channels_ << "," << height_ << "," << width_
                      << "] output=[" << top_shape_ss.str() << "]"
                      << " conv_out_spatial_dim=" << conv_out_spatial_dim_
                      << " (output_h=" << output_h_ << ", output_w=" << output_w_ << ")";

  if (bias_term_) {
    std::vector<int64_t> bias_shape = {1, num_output_, 1, 1};
    bias_multiplier_ = make_object<Blob>(std::vector<int64_t>{conv_out_spatial_dim_});
    caffe_set_fp32(static_cast<size_t>(conv_out_spatial_dim_), 1.0f, bias_multiplier_->cpu_data());
    CAFFE_FFI_TENSOR_LOG << "Convolution: created bias_multiplier shape=["
                         << conv_out_spatial_dim_ << "]";
  }

  if (!is_1x1_) {
    std::vector<int64_t> col_buffer_shape = {kernel_dim_, conv_out_spatial_dim_};
    col_buffer_ = make_object<Blob>(col_buffer_shape);
    CAFFE_FFI_TENSOR_LOG << "Convolution: created col_buffer shape=["
                         << col_buffer_shape[0] << ", " << col_buffer_shape[1] << "]";
  }
}

void ConvolutionLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                                    const std::vector<Blob*>& top) {
  const float* weight = this->blobs_[0]->cpu_data();
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_data();
  float* col_buff = nullptr;
  if (!is_1x1_) {
    col_buff = col_buffer_->cpu_data();
  }

  const int num = static_cast<int>(bottom[0]->shape(0));
  CAFFE_FFI_LAYER_LOG << "Convolution Forward: num=" << num
                      << " group=" << group_
                      << " M=" << conv_out_channels_ / group_
                      << " N=" << conv_out_spatial_dim_
                      << " K=" << kernel_dim_
                      << " is_1x1=" << is_1x1_
                      << " bias_term=" << bias_term_;
  for (int n = 0; n < num; ++n) {
    for (int g = 0; g < group_; ++g) {
      const float* bottom_slice = bottom_data + n * channels_ * height_ * width_
                                  + g * conv_in_channels_ * height_ * width_;
      float* top_slice = top_data + n * num_output_ * output_h_ * output_w_
                         + g * (conv_out_channels_ / group_) * conv_out_spatial_dim_;
      const float* weight_slice = weight + g * weight_offset_;

      if (is_1x1_) {
        col_buff = const_cast<float*>(bottom_slice);
      } else {
        im2col_fp32(bottom_slice, conv_in_channels_, height_, width_,
                   kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
                   dilation_h_, dilation_w_, col_buff);
      }

      caffe_cpu_gemm_fp32(false, false,
                          conv_out_channels_ / group_, conv_out_spatial_dim_, kernel_dim_,
                          1.0f, weight_slice, col_buff, 0.0f, top_slice);
    }

    if (bias_term_) {
      const float* bias = this->blobs_[1]->cpu_data();
      caffe_cpu_gemm_fp32(false, false, num_output_, conv_out_spatial_dim_, 1,
                          1.0f, bias, bias_multiplier_->cpu_data(), 1.0f,
                          top_data + n * num_output_ * conv_out_spatial_dim_);
    }
  }
}

REGISTER_LAYER_CLASS(Convolution);

}  // namespace caffe_ffi
