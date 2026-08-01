#ifndef CAFFE_FFI_BLOB_HPP_
#define CAFFE_FFI_BLOB_HPP_

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "caffe_ffi/common.hpp"
#include "caffe_ffi/math_utils.hpp"
#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/backtrace.hpp"

namespace caffe {
class BlobProto;
class BlobShape;
}

namespace caffe_ffi {

int64_t TotalAllocatedBytes();

int64_t LiveBlobCount();

/**
 * @brief Runtime COW enable/disable switch (Phase 2).
 *
 * When COW is disabled at runtime, cpu_mutable_data()/cpu_mutable_diff()
 * still trigger COW if the data is shared -- this is the safety default.
 * When disabled, the COW logic is bypassed and the raw pointer is returned
 * (same as Phase 1 behavior). This is useful for emergency rollback or
 * performance comparison.
 *
 * The compile-time switch CAFFE_FFI_ENABLE_COW (CMake option) controls
 * whether the COW code is compiled at all. When OFF, all COW methods
 * become no-ops at compile time.
 */
void SetCOWEnabled(bool enabled);
bool IsCOWEnabled();

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
  Shape shape() const {
    if (is_lazy_allocated_) {
      return Shape(shape_only_.begin(), shape_only_.end());
    }
    return Shape(data_tensor_.shape());
  }
  /** @brief Get the number of dimensions (axes). */
  int num_axes() const {
    return is_lazy_allocated_ ? static_cast<int>(shape_only_.size()) : data_tensor_.ndim();
  }
  /** @brief Get the total number of elements (product of all dimensions). */
  int64_t count() const {
    if (is_lazy_allocated_) {
      if (shape_only_.empty()) return 0;
      int64_t prod = 1;
      for (int64_t d : shape_only_) prod *= d;
      return prod;
    }
    return data_tensor_.numel();
  }
  /** @brief Count elements from start_axis to the last axis. */
  int64_t count(int start_axis) const {
    if (is_lazy_allocated_) {
      int canonical = CanonicalAxisIndex(start_axis);
      int64_t prod = 1;
      for (size_t i = canonical; i < shape_only_.size(); ++i) prod *= shape_only_[i];
      return prod;
    }
    return Count(data_tensor_.shape(), start_axis);
  }
  /** @brief Count elements in the range [start_axis, end_axis). */
  int64_t count(int start_axis, int end_axis) const {
    if (is_lazy_allocated_) {
      int canonical_start = CanonicalAxisIndex(start_axis);
      int canonical_end = CanonicalAxisIndex(end_axis);
      int64_t prod = 1;
      for (int i = canonical_start; i < canonical_end; ++i) prod *= shape_only_[i];
      return prod;
    }
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
    if (is_lazy_allocated_) {
      return shape_only_[this->CanonicalAxisIndex(index)];
    }
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

  /** @brief Get const pointer to CPU data buffer (read-only, zero-overhead). */
  const float* cpu_data() const {
    if (is_lazy_allocated_ || !data_tensor_.defined()) return nullptr;
    return static_cast<const float*>(data_tensor_.data_ptr());
  }
  /** @brief Get const pointer to CPU diff buffer (read-only, zero-overhead). */
  const float* cpu_diff() const {
    if (is_lazy_allocated_ || !diff_tensor_.defined()) return nullptr;
    return static_cast<const float*>(diff_tensor_.data_ptr());
  }

  /**
   * @brief Get mutable pointer to CPU data buffer with Copy-on-Write semantics.
   *
   * If the data tensor is shared (use_count > 1, e.g. after ShareData for N>=2
   * Split fan-out), this call triggers Copy-on-Write: the shared tensor is cloned
   * into a private copy before returning the mutable pointer. This follows the
   * PAT-001 "explicit break semantics" pattern -- calling cpu_mutable_data()
   * explicitly signals write intent and breaks sharing.
   *
   * This method guarantees the returned pointer points to private (unshared)
   * memory. Use this when you intend to mutate the data. For read-only access,
   * use cpu_data() const instead.
   */
  float* cpu_mutable_data() {
#ifdef CAFFE_FFI_ENABLE_COW
    if (is_lazy_allocated_) {
      // Phase 3.1: Lazy blob -- allocate both data and diff tensors now (first write).
      auto sv = ShapeView(shape_only_.data(), shape_only_.size());
      data_tensor_ = NewCPUTensor(sv);
      diff_tensor_ = NewCPUTensor(sv);
      caffe_set_fp32(static_cast<size_t>(diff_tensor_.numel()), 0.0f,
                     static_cast<float*>(diff_tensor_.data_ptr()));
      is_lazy_allocated_ = false;
      shape_only_.clear();
      data_shared_ = false;
      diff_shared_ = false;
      CAFFE_FFI_MEM_LOG << "[LAZY] Blob#" << id_
                        << " cpu_mutable_data() allocated data+diff for lazy blob"
                        << " nbytes=" << (data_tensor_.numel() * static_cast<int64_t>(sizeof(float)));
      return static_cast<float*>(data_tensor_.data_ptr());
    }
    if (IsCOWEnabled() && data_tensor_.defined() && data_tensor_.use_count() > 1) {
      int64_t nbytes = data_tensor_.numel() * static_cast<int64_t>(sizeof(float));
      int refcount = data_tensor_.use_count();
      const void* old_ptr = data_tensor_.data_ptr();
      Tensor new_tensor = NewCPUTensor(
          ShapeView(data_tensor_.shape().data(),
                    static_cast<size_t>(data_tensor_.ndim())));
      std::memcpy(new_tensor.data_ptr(), old_ptr, static_cast<size_t>(nbytes));
      data_tensor_ = new_tensor;
      data_shared_ = false;  // COW broke sharing, now private owner
      CAFFE_FFI_MEM_LOG << "[COW] Blob#" << id_
                        << " cpu_mutable_data() unshared data"
                        << " refcount=" << refcount
                        << " old_ptr=" << old_ptr
                        << " new_ptr=" << data_tensor_.data_ptr()
                        << " nbytes=" << nbytes;
    }
#endif
    return static_cast<float*>(data_tensor_.data_ptr());
  }
  /**
   * @brief Get mutable pointer to CPU diff buffer with Copy-on-Write semantics.
   *
   * If the diff tensor is shared (use_count > 1), this call triggers
   * Copy-on-Write: the shared tensor is cloned into a private copy before
   * returning the mutable pointer. Use this when you intend to mutate the diff.
   */
  float* cpu_mutable_diff() {
#ifdef CAFFE_FFI_ENABLE_COW
    if (is_lazy_allocated_) {
      // Phase 3.1: Lazy blob -- allocate both data and diff tensors now.
      auto sv = ShapeView(shape_only_.data(), shape_only_.size());
      data_tensor_ = NewCPUTensor(sv);
      diff_tensor_ = NewCPUTensor(sv);
      caffe_set_fp32(static_cast<size_t>(diff_tensor_.numel()), 0.0f,
                     static_cast<float*>(diff_tensor_.data_ptr()));
      is_lazy_allocated_ = false;
      shape_only_.clear();
      data_shared_ = false;
      diff_shared_ = false;
      CAFFE_FFI_MEM_LOG << "[LAZY] Blob#" << id_
                        << " cpu_mutable_diff() allocated data+diff for lazy blob"
                        << " nbytes=" << (diff_tensor_.numel() * static_cast<int64_t>(sizeof(float)));
      return static_cast<float*>(diff_tensor_.data_ptr());
    }
#endif
    if (!diff_tensor_.defined()) {
      // Diff tensor not yet allocated (e.g. after cpu_mutable_data allocated only data)
      // -- allocate diff matching data shape.
      if (data_tensor_.defined()) {
        diff_tensor_ = NewCPUTensor(
            ShapeView(data_tensor_.shape().data(),
                      static_cast<size_t>(data_tensor_.ndim())));
        caffe_set_fp32(static_cast<size_t>(data_tensor_.numel()), 0.0f,
                       static_cast<float*>(diff_tensor_.data_ptr()));
        diff_shared_ = false;  // newly allocated, private
        CAFFE_FFI_MEM_LOG << "[MEM] Blob#" << id_
                          << " cpu_mutable_diff() allocated diff to match data shape"
                          << " nbytes=" << (diff_tensor_.numel() * static_cast<int64_t>(sizeof(float)));
      }
      return static_cast<float*>(diff_tensor_.data_ptr());
    }
    if (diff_tensor_.defined() && diff_tensor_.use_count() > 1) {
      int64_t nbytes = diff_tensor_.numel() * static_cast<int64_t>(sizeof(float));
      int refcount = diff_tensor_.use_count();
      const void* old_ptr = diff_tensor_.data_ptr();
      Tensor new_tensor = NewCPUTensor(
          ShapeView(diff_tensor_.shape().data(),
                    static_cast<size_t>(diff_tensor_.ndim())));
      std::memcpy(new_tensor.data_ptr(), old_ptr, static_cast<size_t>(nbytes));
      diff_tensor_ = new_tensor;
      diff_shared_ = false;  // COW broke sharing, now private owner
      CAFFE_FFI_MEM_LOG << "[COW] Blob#" << id_
                        << " cpu_mutable_diff() unshared diff"
                        << " refcount=" << refcount
                        << " old_ptr=" << old_ptr
                        << " new_ptr=" << diff_tensor_.data_ptr()
                        << " nbytes=" << nbytes;
    }
    return static_cast<float*>(diff_tensor_.data_ptr());
  }

  /**
   * @brief Get mutable pointer to GPU data buffer with Copy-on-Write semantics.
   *
   * @note GPU support is not yet implemented. This is a placeholder for Phase 2
   * GPU COW support. Currently delegates to cpu_mutable_data().
   */
  float* gpu_mutable_data() { return cpu_mutable_data(); }
  /**
   * @brief Get mutable pointer to GPU diff buffer with Copy-on-Write semantics.
   * @note Placeholder for Phase 2 GPU COW support.
   */
  float* gpu_mutable_diff() { return cpu_mutable_diff(); }

  /** @brief Get the data tensor (forward activations) for DLPack zero-copy interop. */
  Tensor data_tensor() const;
  /** @brief Get the diff tensor (backward gradients) for DLPack zero-copy interop. */
  Tensor diff_tensor() const;

  /**
   * @brief Get mutable data tensor with COW trigger for DLPack write interop.
   *
   * Unlike data_tensor() which is read-only, this method triggers COW
   * (calls UnshareData()) before returning the tensor. Use this when
   * the caller intends to modify the tensor data through DLPack.
   *
   * @note The returned tensor has exclusive ownership (refcount=1).
   */
  Tensor mutable_data_tensor();
  /**
   * @brief Get mutable diff tensor with COW trigger for DLPack write interop.
   */
  Tensor mutable_diff_tensor();

  /**
   * @brief Zero-copy share data tensor from another Blob (Phase 1 N=1 split shortcut).
   *
   * Instead of allocating new memory and memcpy-ing, directly shares the underlying
   * TVM FFI Tensor via intrusive reference counting. After this call, both Blobs
   * point to the same data buffer. Safe for N=1 fan-out (identity passthrough).
   *
   * For N>=2 fan-out, use CopyFrom() or the traditional memcpy path to avoid
   * accidental cross-contamination from in-place writes (Phase 2 COW will address this).
   *
   * A subsequent Reshape() on this Blob will break the share (allocates new private memory).
   *
   * @param other Source Blob whose data tensor will be shared.
   */
  void ShareData(const Blob* other);
  void ShareDiff(const Blob* other);
  bool SharesDataWith(const Blob* other) const;
  bool SharesDiffWith(const Blob* other) const;

#ifdef CAFFE_FFI_ENABLE_COW_PHASE3
  /**
   * @brief Phase 3 prototype: Batch zero-copy data sharing for large-N Split.
   *
   * Shares data tensor from @p source to all blobs in @p targets using a single
   * atomic refcount increment (O(1) atomics instead of O(N)), followed by raw
   * pointer assignment without per-target IncRef. This reduces Forward latency
   * from O(N) atomic ops to O(1) atomic op + O(N) raw pointer writes for
   * large fan-out scenarios (N >= BATCH_SHARE_THRESHOLD).
   *
   * @note Prototype API -- guarded by CAFFE_FFI_ENABLE_COW_PHASE3 compile-time flag.
   *       Requires TVM FFI Object header layout knowledge (single Object* at offset 0
   *       in ObjectPtr). Safe on MSVC/GCC/Clang where ObjectPtr is standard-layout
   *       with a single pointer member.
   *
   * @param source  Source Blob whose data tensor is shared (must outlive all targets).
   * @param targets Vector of target Blobs that will share the source data tensor.
   */
  static void BatchShareData(const Blob* source, const std::vector<Blob*>& targets);

  /**
   * @brief Phase 3 prototype: Batch zero-copy diff sharing for large-N Split.
   * @see BatchShareData for semantics.
   */
  static void BatchShareDiff(const Blob* source, const std::vector<Blob*>& targets);
#endif  // CAFFE_FFI_ENABLE_COW_PHASE3

  /** @brief Check if data tensor is shared (borrowed via ShareData and still has multiple refs). */
  bool IsDataShared() const {
    return data_shared_ && data_tensor_.defined() && data_tensor_.use_count() > 1;
  }
  /** @brief Check if diff tensor is shared (borrowed via ShareDiff and still has multiple refs). */
  bool IsDiffShared() const {
    return diff_shared_ && diff_tensor_.defined() && diff_tensor_.use_count() > 1;
  }
  /** @brief Get data tensor refcount (0 if undefined). */
  int DataRefCount() const { return (data_tensor_.defined() && data_tensor_.numel() > 0) ? data_tensor_.use_count() : 0; }
  /** @brief Get diff tensor refcount (0 if undefined or empty). */
  int DiffRefCount() const { return (diff_tensor_.defined() && diff_tensor_.numel() > 0) ? diff_tensor_.use_count() : 0; }

  /**
   * @brief Explicitly force Copy-on-Write for data tensor.
   *
   * If data is shared (refcount > 1), clones it into a private copy.
   * Returns the data pointer. No-op if already private or undefined.
   */
  void* UnshareData();
  /**
   * @brief Explicitly force Copy-on-Write for diff tensor.
   *
   * If diff is shared (refcount > 1), clones it into a private copy.
   * Returns the diff pointer. No-op if already private or undefined.
   */
  void* UnshareDiff();

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

  // ── Phase 3.1: Lazy Allocation (SetShapeOnly) ──────────────────────

  /**
   * @brief Set shape metadata only, without allocating data memory.
   *
   * Stores the shape in a separate vector (shape_only_) and marks the Blob
   * as lazy-allocated. After this call, shape()/num_axes()/count() work
   * correctly, but cpu_data()/cpu_diff() return nullptr and data_tensor()
   * returns undefined. Designed for large-N Split layer Reshape where
   * per-top allocation is wasteful (will be replaced by ShareData).
   *
   * @param shape The target shape. Dimensions must be > 0.
   * @throws std::invalid_argument if any dimension <= 0.
   */
  void SetShapeOnly(ShapeView shape);

  /** @brief Check if this Blob is in lazy-allocation mode. */
  bool IsLazyAllocated() const { return is_lazy_allocated_; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL(
      "caffe_ffi.Blob", Blob, Object);

 private:
  int64_t id_;
  std::string name_;
  std::string construct_bt_;
  Tensor data_tensor_;
  Tensor diff_tensor_;

  // COW sharing state: true if tensor was borrowed via ShareData/ShareDiff
  // and hasn't been privatized by COW/Reshape yet.
  bool data_shared_ = false;
  bool diff_shared_ = false;

  // Phase 3.1: lazy allocation support
  std::vector<int64_t> shape_only_;       // stored shape for lazy allocation
  bool is_lazy_allocated_ = false;         // whether in lazy-allocation mode
};

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_BLOB_HPP_
