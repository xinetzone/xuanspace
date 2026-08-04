#ifndef CAFFE_FFI_LAYERS_EMBED_LAYER_HPP_
#define CAFFE_FFI_LAYERS_EMBED_LAYER_HPP_

#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Embed: embedding / lookup layer.
 *
 * bottom[0]: integer indices of shape (M, ...) (flattened to M entries).
 * The weight blob is [K, N] = [input_dim, num_output]; each index looks up a
 * row of the weight. An optional bias of shape [N] is added. Output shape is
 * the input shape plus a trailing N dimension.
 */
class EmbedLayer : public Layer {
 public:
  static constexpr bool _type_mutable = true;

  explicit EmbedLayer(const caffe::LayerParameter& param) : Layer(param) {}
  void LayerSetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Reshape(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;

  const char* type() const override { return "Embed"; }
  int ExactNumBottomBlobs() const override { return 1; }
  int ExactNumTopBlobs() const override { return 1; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("caffe_ffi.EmbedLayer", EmbedLayer, Layer);

 protected:
  void Forward_cpu(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top) override;
  void Backward_cpu(const std::vector<Blob*>& top,
                    const std::vector<bool>& propagate_down,
                    const std::vector<Blob*>& bottom) override;

 private:
  int N_ = 0;  // num_output
  int K_ = 0;  // input_dim
  int M_ = 0;  // number of indices (flattened input count)
  bool bias_term_ = true;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYERS_EMBED_LAYER_HPP_