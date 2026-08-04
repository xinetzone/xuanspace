#ifndef CAFFE_FFI_LAYERS_IM2COL_LAYER_HPP_
#define CAFFE_FFI_LAYERS_IM2COL_LAYER_HPP_

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Im2col layer.
 *
 * Rearranges a 4D NCHW input into columns suitable for convolution matrix
 * multiplication, producing a [num, channels*kh*kw, out_h, out_w] output.
 */
class Im2colLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit Im2colLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Im2col"; }
  int ExactNumBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.Im2colLayer", Im2colLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

  int kernel_h_;
  int kernel_w_;
  int pad_h_;
  int pad_w_;
  int stride_h_;
  int stride_w_;
  int dilation_h_;
  int dilation_w_;

  int channels_;
  int height_;
  int width_;
  int64_t num_;
  int64_t bottom_dim_;
  int64_t top_dim_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_IM2COL_LAYER_HPP_