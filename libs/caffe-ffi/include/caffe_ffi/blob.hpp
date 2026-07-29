#ifndef CAFFE_FFI_BLOB_HPP_
#define CAFFE_FFI_BLOB_HPP_

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "caffe_ffi/common.hpp"
#include "caffe_ffi/math_utils.hpp"
#include "caffe_ffi/backtrace.hpp"

namespace caffe {
class BlobProto;
class BlobShape;
}

namespace caffe_ffi {

int64_t TotalAllocatedBytes();

int64_t LiveBlobCount();

/**
 * @brief Tensor storage wrapper for network parameters and intermediate activations.
 *
 * Blob is the fundamental data container in caffe-ffi, wrapping a pair of TVM FFI
 * Tensors (data_tensor_ and diff_tensor_) for forward activations and backward gradients.
 * It uses intrusive reference counting via TVM FFI Object system and supports zero-copy
 * DLPack interop with Python/numpy.
 *
 * Memory is allocated lazily on Reshape() and freed in the destructor, with global
 * counters tracking total allocation and live blob count for leak detection.
 */
class Blob : public Object {
 public:
  static constexpr bool _type_mutable = true;

  /** @brief Construct an empty Blob with no shape allocated. */
  Blob();
  /** @brief Construct a Blob with the given shape, allocating CPU memory. */
  explicit Blob(ShapeView shape);
  /** @brief Construct a Blob from a vector of dimension sizes. */
  explicit Blob(const std::vector<int64_t>& shape);
  ~Blob();

  /**
   * @brief Change the shape of the Blob, reallocating memory if needed.
   * @param shape New shape as a ShapeView. Negative dimensions are not allowed.
   */
  void Reshape(ShapeView shape);
  /** @brief Reshape from a vector of dimension sizes. */
  void Reshape(const std::vector<int64_t>& shape);
  /** @brief Reshape from a TVM FFI Shape (zero-copy). */
  void Reshape(Shape shape);
  /** @brief Reshape from a caffe::BlobShape protobuf message. */
  void Reshape(const caffe::BlobShape& shape);
  /** @brief Reshape to match the shape of another Blob. */
  void ReshapeLike(const Blob& other);

  /** @brief Get the current shape as a TVM FFI Shape. */
  Shape shape() const { return Shape(data_tensor_.shape()); }
  /** @brief Get the number of dimensions (axes). */
  int num_axes() const { return data_tensor_.ndim(); }
  /** @brief Get the total number of elements (product of all dimensions). */
  int64_t count() const { return data_tensor_.numel(); }
  /** @brief Count elements from start_axis to the last axis. */
  int64_t count(int start_axis) const { return Count(data_tensor_.shape(), start_axis); }
  /** @brief Count elements in the range [start_axis, end_axis). */
  int64_t count(int start_axis, int end_axis) const {
    return Count(data_tensor_.shape(), start_axis, end_axis);
  }

  /**
   * @brief Convert a possibly-negative axis index to a canonical non-negative index.
   * @param axis_index Axis index (negative values count from the end).
   * @return Canonical axis index in [0, num_axes()).
   */
  int CanonicalAxisIndex(int axis_index) const {
    return caffe_ffi::CanonicalAxisIndex(axis_index, num_axes());
  }

  /** @brief Get the dimension size at the given axis. */
  int64_t shape(int index) const {
    return data_tensor_.size(this->CanonicalAxisIndex(index));
  }

  int64_t LegacyShape(int index) const;
  /** @brief Legacy Caffe API: get batch size (dimension 0). */
  int num() const { return LegacyShape(0); }
  /** @brief Legacy Caffe API: get channel count (dimension 1). */
  int channels() const { return LegacyShape(1); }
  /** @brief Legacy Caffe API: get spatial height (dimension 2). */
  int height() const { return LegacyShape(2); }
  /** @brief Legacy Caffe API: get spatial width (dimension 3). */
  int width() const { return LegacyShape(3); }

  /** @brief Get mutable pointer to CPU data buffer. */
  float* cpu_data() { return static_cast<float*>(data_tensor_.data_ptr()); }
  /** @brief Get const pointer to CPU data buffer. */
  const float* cpu_data() const { return static_cast<const float*>(data_tensor_.data_ptr()); }
  /** @brief Get mutable pointer to CPU diff (gradient) buffer. */
  float* cpu_diff() { return static_cast<float*>(diff_tensor_.data_ptr()); }
  /** @brief Get const pointer to CPU diff (gradient) buffer. */
  const float* cpu_diff() const { return static_cast<const float*>(diff_tensor_.data_ptr()); }

  /** @brief Get the data tensor (forward activations) for DLPack zero-copy interop. */
  Tensor data_tensor() const;
  /** @brief Get the diff tensor (backward gradients) for DLPack zero-copy interop. */
  Tensor diff_tensor() const;

  /** @brief Load blob data from a BlobProto protobuf message. */
  void FromProto(const caffe::BlobProto& proto, bool reshape = true);
  /** @brief Serialize blob data to a BlobProto protobuf message. */
  void ToProto(caffe::BlobProto* proto) const;
  /** @brief Update data by subtracting diff (gradient descent step: data -= diff). */
  void Update();

  /** @brief Get data as an Array<float> (FFI convenience, copies data). */
  Array<float> get_data() const;
  /** @brief Set data from a TVM FFI Tensor (supports numpy zero-copy interop via DLPack, direct memcpy). */
  void set_data(Tensor data);
  /** @brief Get diff as an Array<float> (FFI convenience, copies data). */
  Array<float> get_diff() const;
  /** @brief Set diff from a TVM FFI Tensor (supports numpy zero-copy interop via DLPack, direct memcpy). */
  void set_diff(Tensor diff);

  /** @brief Set the blob name. */
  void set_name(const std::string& name) { name_ = name; }
  /** @brief Get the blob name. */
  std::string name() const { return name_; }

  /** @brief Get unique blob ID (for debugging). */
  int64_t id() const { return id_; }

  /** @brief Get the construction backtrace string (for debugging memory issues). */
  std::string construction_backtrace() const { return construct_bt_; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      "caffe_ffi.Blob", Blob, Object);

 private:
  int64_t id_;
  std::string name_;
  std::string construct_bt_;
  Tensor data_tensor_;
  Tensor diff_tensor_;
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_BLOB_HPP_
