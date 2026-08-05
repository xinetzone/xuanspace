#ifndef CAFFE_FFI_LAYERS_DUMMY_DATA_LAYER_HPP_
#define CAFFE_FFI_LAYERS_DUMMY_DATA_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Placeholder data layer.
 *
 * Produces one or more top blobs with arbitrary shapes, each filled by an
 * optional FillerParameter (data_filler). Shapes come from
 * DummyDataParameter::shape() (or the deprecated 4-D num/channels/height/width
 * fields). Has no bottom and auto-creates its tops.
 */
class DummyDataLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit DummyDataLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "DummyData"; }
  int ExactNumBottomBlobs() const override { return 0; }
  bool AutoTopBlobs() const override { return true; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.DummyDataLayer", DummyDataLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_DUMMY_DATA_LAYER_HPP_