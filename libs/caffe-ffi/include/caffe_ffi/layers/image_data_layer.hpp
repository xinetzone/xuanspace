#ifndef CAFFE_FFI_LAYERS_IMAGE_DATA_LAYER_HPP_
#define CAFFE_FFI_LAYERS_IMAGE_DATA_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief ImageData input layer (image list via the Python/numpy bridge).
 *
 * No bottom; exactly two tops (data + label). Images are decoded and loaded
 * through a Python-side data-source callback registered via
 * `caffe_ffi.data_io.register` under the key "<type>.<layer_name>". The output
 * data shape is (N, C, H, W) where C=3 if is_color else 1, and H/W follow
 * new_height/new_width (or a placeholder when unspecified). On Forward the
 * callback writes the batch into the tops' mutable data tensors.
 */
class ImageDataLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit ImageDataLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "ImageData"; }
  int ExactNumBottomBlobs() const override { return 0; }
  int ExactNumTopBlobs() const override { return 2; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.ImageDataLayer", ImageDataLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

 private:
  std::string key_;  // "<type>.<name>" used to look up the data-source callback
  int batch_size_ = 1;
  int channels_ = 3;       // 3 if is_color, else 1
  int height_ = 1;         // new_height, or placeholder 1 when unspecified
  int width_ = 1;          // new_width, or placeholder 1 when unspecified
  float scale_ = 1.0f;     // simple scaling applied by the Python bridge
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_IMAGE_DATA_LAYER_HPP_