#ifndef CAFFE_FFI_LAYER_HPP_
#define CAFFE_FFI_LAYER_HPP_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "caffe_ffi/blob.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

/**
 * @brief Abstract base class for all neural network layers.
 *
 * Layer defines the interface that all concrete layer implementations must follow:
 * LayerSetUp (one-time parameter initialization), Reshape (infer output shapes from inputs),
 * and Forward_cpu (compute forward pass). Layers own parameter Blobs (weights/biases) and
 * produce loss values for loss layers.
 *
 * Concrete layers inherit from Layer and override LayerSetUp(), Reshape(), Forward_cpu(),
 * and type(). The layer factory (layer_factory.hpp) creates layers by type name string.
 */
class Layer : public Object {
 public:
  static constexpr bool _type_mutable = true;
  static constexpr int _type_child_slots = 32;
  static constexpr bool _type_child_slots_can_overflow = true;

  /** @brief Construct a layer from a LayerParameter protobuf message. */
  explicit Layer(const caffe::LayerParameter& param);
  virtual ~Layer() = default;

  /**
   * @brief Initialize the layer: call LayerSetUp and Reshape in sequence.
   * @param bottom Input blobs (read-only).
   * @param top Output blobs (to be allocated and shaped).
   */
  void SetUp(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top);

  /**
   * @brief Perform forward pass and return the scalar loss (0 for non-loss layers).
   * @param bottom Input blobs.
   * @param top Output blobs.
   * @return Total loss contributed by this layer.
   */
  float Forward(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top);

  /**
   * @brief One-time layer-specific initialization (read layer parameters, create weight blobs).
   * Called once during network construction after blob shapes are known.
   */
  virtual void LayerSetUp(const std::vector<Blob*>& bottom,
                          const std::vector<Blob*>& top) {}

  /**
   * @brief Adjust output blob shapes based on input shapes. Must be overridden by subclasses.
   * Called after LayerSetUp and before Forward.
   */
  virtual void Reshape(const std::vector<Blob*>& bottom,
                       const std::vector<Blob*>& top) = 0;

  /** @brief Get mutable parameter blobs (weights/biases). */
  std::vector<ObjectPtr<Blob>>& blobs() { return blobs_; }
  /** @brief Get const parameter blobs (weights/biases). */
  const std::vector<ObjectPtr<Blob>>& blobs() const { return blobs_; }

  /** @brief Get parameter blobs as a TVM FFI Array for Python interop. */
  Array<ObjectPtr<Blob>> blobs_array() const;

  /** @brief Get the layer parameter protobuf. */
  const caffe::LayerParameter& layer_param() const { return layer_param_; }

  /** @brief Serialize layer parameters to a LayerParameter protobuf message. */
  void ToProto(caffe::LayerParameter* param, bool write_diff = false);

  /** @brief Get the loss weight for a given top blob index. */
  float loss(int top_index) const {
    return (loss_.size() > static_cast<size_t>(top_index)) ? loss_[top_index] : 0.0f;
  }

  /** @brief Set the loss weight for a given top blob index. */
  void set_loss(int top_index, float value) {
    if (loss_.size() <= static_cast<size_t>(top_index)) {
      loss_.resize(top_index + 1, 0.0f);
    }
    loss_[top_index] = value;
  }

  /** @brief Get the layer type name string (e.g., "InnerProduct", "ReLU"). Must be overridden. */
  virtual const char* type() const { return ""; }

  /** @brief Get the layer name. */
  std::string name() const { return layer_param_.name(); }

  /** @brief Return the required number of bottom blobs, or -1 if not constrained. */
  virtual int ExactNumBottomBlobs() const { return -1; }
  /** @brief Return the minimum number of bottom blobs, or -1 if not constrained. */
  virtual int MinBottomBlobs() const { return -1; }
  /** @brief Return the maximum number of bottom blobs, or -1 if not constrained. */
  virtual int MaxBottomBlobs() const { return -1; }
  /** @brief Return the required number of top blobs, or -1 if not constrained. */
  virtual int ExactNumTopBlobs() const { return -1; }
  /** @brief Return the minimum number of top blobs, or -1 if not constrained. */
  virtual int MinTopBlobs() const { return -1; }
  /** @brief Return the maximum number of top blobs, or -1 if not constrained. */
  virtual int MaxTopBlobs() const { return -1; }
  /** @brief Return true if the layer requires equal numbers of bottom and top blobs. */
  virtual bool EqualNumBottomTopBlobs() const { return false; }
  /** @brief Return true if top blobs are automatically created (e.g., Input layer). */
  virtual bool AutoTopBlobs() const { return false; }

  /** @brief Check if gradients should be propagated down for a given parameter blob. */
  bool param_propagate_down(int param_id) const {
    return (param_propagate_down_.size() > static_cast<size_t>(param_id))
               ? param_propagate_down_[param_id]
               : false;
  }
  /** @brief Set whether to propagate gradients down for a given parameter blob. */
  void set_param_propagate_down(int param_id, bool value) {
    if (param_propagate_down_.size() <= static_cast<size_t>(param_id)) {
      param_propagate_down_.resize(param_id + 1, true);
    }
    param_propagate_down_[param_id] = value;
  }

  TVM_FFI_DECLARE_OBJECT_INFO("caffe_ffi.Layer", Layer, Object);

 protected:
  caffe::LayerParameter layer_param_;
  std::vector<ObjectPtr<Blob>> blobs_;
  std::vector<bool> param_propagate_down_;
  std::vector<float> loss_;

  virtual void Forward_cpu(const std::vector<Blob*>& bottom,
                           const std::vector<Blob*>& top) = 0;

  void CheckBlobCounts(const std::vector<Blob*>& bottom,
                       const std::vector<Blob*>& top);

  void SetLossWeights(const std::vector<Blob*>& top);

 private:
  Layer(const Layer&) = delete;
  Layer& operator=(const Layer&) = delete;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_LAYER_HPP_
