#ifndef CAFFE_FFI_LAYERS_CONV_LAYER_HPP_
#define CAFFE_FFI_LAYERS_CONV_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

class ConvolutionLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit ConvolutionLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Convolution"; }
  int ExactNumBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.ConvolutionLayer", ConvolutionLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  int num_output_;
  bool bias_term_;
  int pad_h_, pad_w_;
  int kernel_h_, kernel_w_;
  int stride_h_, stride_w_;
  int dilation_h_, dilation_w_;
  int group_;
  int channels_;
  int height_, width_;
  int conv_out_channels_;
  int conv_in_channels_;
  int output_h_, output_w_;
  ObjectPtr<Blob> col_buffer_;
  ObjectPtr<Blob> bias_multiplier_;
  bool is_1x1_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int weight_offset_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_CONV_LAYER_HPP_
