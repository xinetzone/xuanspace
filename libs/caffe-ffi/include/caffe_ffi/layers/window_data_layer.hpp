#ifndef CAFFE_FFI_LAYERS_WINDOW_DATA_LAYER_HPP_
#define CAFFE_FFI_LAYERS_WINDOW_DATA_LAYER_HPP_

#include <string>
#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief WindowData input layer (detection windows via the Python/numpy bridge).
 *
 * No bottom; two or more tops (data + label + optional extras). Detection
 * windows are loaded through a Python-side data-source callback registered via
 * `caffe_ffi.data_io.register` under the key "<type>.<layer_name>". On Forward
 * the callback writes the batch into the tops' mutable data tensors.
 */
class WindowDataLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit WindowDataLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "WindowData"; }
  int ExactNumBottomBlobs() const override { return 0; }
  int MinTopBlobs() const override { return 2; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.WindowDataLayer", WindowDataLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

 private:
  std::string key_;  // "<type>.<name>" used to look up the data-source callback
  int batch_size_ = 0;
  float fg_threshold_ = 0.5f;  // foreground (object) overlap threshold
  float bg_threshold_ = 0.5f;  // background (non-object) overlap threshold
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_WINDOW_DATA_LAYER_HPP_