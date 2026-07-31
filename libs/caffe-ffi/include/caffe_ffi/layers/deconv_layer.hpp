#ifndef CAFFE_FFI_LAYERS_DECONV_LAYER_HPP_
#define CAFFE_FFI_LAYERS_DECONV_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layers/base_conv_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Convolve the input with a bank of learned filters, treating filters
 *        and convolution parameters in the opposite sense as ConvolutionLayer.
 *
 * DeconvolutionLayer computes the transpose of ConvolutionLayer: it multiplies
 * each input value by a filter elementwise and sums over the resulting output
 * windows (upsampling). Forward reuses backward_cpu_gemm from the base class
 * (transposed convolution), and Backward reuses forward_cpu_gemm.
 */
class DeconvolutionLayer : public BaseConvolutionLayer {
 public:
  static constexpr bool _type_mutable = true;

  explicit DeconvolutionLayer(const caffe::LayerParameter& param) : BaseConvolutionLayer(param) {}

  const char* type() const override { return "Deconvolution"; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.DeconvolutionLayer", DeconvolutionLayer, BaseConvolutionLayer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

  bool reverse_dimensions() override { return true; }
  void compute_output_shape() override;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_DECONV_LAYER_HPP_
