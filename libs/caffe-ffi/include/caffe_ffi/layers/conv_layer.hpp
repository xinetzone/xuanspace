#ifndef CAFFE_FFI_LAYERS_CONV_LAYER_HPP_
#define CAFFE_FFI_LAYERS_CONV_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layers/base_conv_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

class ConvolutionLayer : public BaseConvolutionLayer {
 public:
  static constexpr bool _type_mutable = true;

  explicit ConvolutionLayer(const caffe::LayerParameter& param) : BaseConvolutionLayer(param) {}

  const char* type() const override { return "Convolution"; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.ConvolutionLayer", ConvolutionLayer, BaseConvolutionLayer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

  bool reverse_dimensions() override { return false; }
  void compute_output_shape() override;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_CONV_LAYER_HPP_
