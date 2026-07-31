#include "caffe_ffi/blob.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>

#include "caffe_ffi/error.hpp"
#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/log.hpp"
#include "caffe_ffi/backtrace.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe_ffi {

std::atomic<int64_t> g_total_allocated_bytes{0};

namespace {

std::atomic<int64_t> g_live_blob_count{0};
std::atomic<int64_t> g_next_blob_id{1};

// Runtime COW switch (default: enabled)
// Guarded by compile-time CAFFE_FFI_ENABLE_COW; when the CMake option is OFF,
// all COW logic is elided at compile time.
std::atomic<bool> g_cow_enabled{true};

std::string ShapeToString(ShapeView shape) {
  std::ostringstream oss;
  oss << "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) oss << ",";
    oss << shape[i];
  }
  oss << ")";
  return oss.str();
}

int64_t TensorNBytes(const Tensor& t) {
  if (!t.defined()) return 0;
  return t.numel() * static_cast<int64_t>(t.dtype().bits / 8);
}

std::string PtrToString(const void* p) {
  std::ostringstream oss;
  oss << p;
  return oss.str();
}

std::string FormatBytes(int64_t bytes) {
  std::ostringstream oss;
  if (bytes < 0) {
    oss << "-";
    bytes = -bytes;
  }
  if (bytes >= 1024 * 1024) {
    oss << (bytes / (1024.0 * 1024.0)) << " MB";
  } else if (bytes >= 1024) {
    oss << (bytes / 1024.0) << " KB";
  } else {
    oss << bytes << " B";
  }
  return oss.str();
}

/**
 * @brief Clone a tensor by allocating a new CPU tensor and copying data.
 *
 * This is the single memcpy point for COW — all unshare operations
 * go through this function to ensure consistent logging and auditing.
 */
Tensor CloneTensor(const Tensor& src) {
  CAFFE_FFI_CHECK_TYPE(src.defined()) << "CloneTensor: source tensor is undefined";

  // 1. Allocate new private CPU tensor (same shape, same dtype)
  Tensor dst = NewCPUTensor(
      ShapeView(src.shape().data(), static_cast<size_t>(src.ndim())));

  // 2. Perform memcpy (the single copy point for COW)
  int64_t nbytes = src.numel() * static_cast<int64_t>(src.dtype().bits / 8);
  std::memcpy(dst.data_ptr(), src.data_ptr(), static_cast<size_t>(nbytes));

  CAFFE_FFI_MEM_LOG << "[COW] CloneTensor: " << nbytes << "B ("
                    << FormatBytes(nbytes) << ")"
                    << " shape=" << ShapeToString(ShapeView(src.shape().data(),
                                                            static_cast<size_t>(src.ndim())))
                    << " src_ptr=" << PtrToString(src.data_ptr())
                    << " dst_ptr=" << PtrToString(dst.data_ptr());

  return dst;
}

}  // namespace

Blob::Blob() : id_(g_next_blob_id.fetch_add(1, std::memory_order_relaxed)) {
  g_live_blob_count.fetch_add(1, std::memory_order_relaxed);
  construct_bt_ = backtrace::GetBacktrace(3);
  CAFFE_FFI_BLOB_LOG << "[MEM-LIFECYCLE] Blob#" << id_ << " constructed (default) this=" << this
                     << " live_blobs=" << g_live_blob_count.load(std::memory_order_relaxed);
  Reshape(std::vector<int64_t>{0});
}

Blob::Blob(ShapeView shape) : id_(g_next_blob_id.fetch_add(1, std::memory_order_relaxed)) {
  g_live_blob_count.fetch_add(1, std::memory_order_relaxed);
  construct_bt_ = backtrace::GetBacktrace(3);
  CAFFE_FFI_BLOB_LOG << "[MEM-LIFECYCLE] Blob#" << id_ << " constructed (ShapeView) this=" << this
                     << " shape=" << ShapeToString(shape)
                     << " live_blobs=" << g_live_blob_count.load(std::memory_order_relaxed);
  Reshape(shape);
}

Blob::Blob(const std::vector<int64_t>& shape)
    : id_(g_next_blob_id.fetch_add(1, std::memory_order_relaxed)) {
  g_live_blob_count.fetch_add(1, std::memory_order_relaxed);
  construct_bt_ = backtrace::GetBacktrace(3);
  CAFFE_FFI_BLOB_LOG << "[MEM-LIFECYCLE] Blob#" << id_ << " constructed (vector) this=" << this
                     << " shape=" << ShapeToString(ShapeView(shape.data(), shape.size()))
                     << " live_blobs=" << g_live_blob_count.load(std::memory_order_relaxed);
  Reshape(ShapeView(shape.data(), shape.size()));
}

Blob::~Blob() {
  int64_t data_nbytes = TensorNBytes(data_tensor_);
  int64_t diff_nbytes = TensorNBytes(diff_tensor_);
  int64_t total_freed = data_nbytes + diff_nbytes;

  int64_t live_before = g_live_blob_count.fetch_sub(1, std::memory_order_relaxed);
  int64_t live_after = live_before - 1;

  CAFFE_FFI_MEM_LOG << "[MEM-FREE] Blob#" << id_ << " this=" << this
                    << " shape=" << ShapeToString(ShapeView(data_tensor_.shape().data(),
                                                            static_cast<size_t>(data_tensor_.ndim())))
                    << " data_ptr=" << PtrToString(data_tensor_.data_ptr())
                    << " diff_ptr=" << PtrToString(diff_tensor_.data_ptr())
                    << " freed=" << total_freed << "B (" << FormatBytes(total_freed) << ")"
                    << " live_blobs=" << live_after;

  data_tensor_ = Tensor();
  diff_tensor_ = Tensor();

  CAFFE_FFI_LOG_TRACE() << "[MEM-LIFECYCLE] Blob#" << id_
                        << " construction backtrace:\n" << construct_bt_;

  CAFFE_FFI_MEM_LOG << "[MEM-LIFECYCLE] Blob#" << id_
                    << " destroyed, global_total=" << g_total_allocated_bytes.load(std::memory_order_relaxed) << "B"
                    << " live_blobs=" << live_after << " (was " << live_before << ")";
}

Tensor Blob::data_tensor() const {
  CAFFE_FFI_TENSOR_LOG << "data_tensor() Blob#" << id_ << " this=" << this
                       << " ptr=" << PtrToString(data_tensor_.data_ptr())
                       << " shape=" << ShapeToString(ShapeView(data_tensor_.shape().data(),
                                                               static_cast<size_t>(data_tensor_.ndim())))
                       << " numel=" << data_tensor_.numel()
                       << " nbytes=" << TensorNBytes(data_tensor_)
                       << " dtype=" << DTypeCodeToString(data_tensor_.dtype().code)
                       << static_cast<int>(data_tensor_.dtype().bits)
                       << " device_type=" << static_cast<int>(data_tensor_.device().device_type);
  return data_tensor_;
}

Tensor Blob::diff_tensor() const {
  CAFFE_FFI_TENSOR_LOG << "diff_tensor() Blob#" << id_ << " this=" << this
                       << " ptr=" << PtrToString(diff_tensor_.data_ptr())
                       << " shape=" << ShapeToString(ShapeView(diff_tensor_.shape().data(),
                                                               static_cast<size_t>(diff_tensor_.ndim())))
                       << " numel=" << diff_tensor_.numel()
                       << " nbytes=" << TensorNBytes(diff_tensor_)
                       << " dtype=" << DTypeCodeToString(diff_tensor_.dtype().code)
                       << static_cast<int>(diff_tensor_.dtype().bits)
                       << " device_type=" << static_cast<int>(diff_tensor_.device().device_type);
  return diff_tensor_;
}

Tensor Blob::mutable_data_tensor() {
  if (data_tensor_.defined() && data_tensor_.use_count() > 1) {
    int refcount = data_tensor_.use_count();
    const void* old_ptr = data_tensor_.data_ptr();
    int64_t nbytes = data_tensor_.numel() * static_cast<int64_t>(sizeof(float));
    data_tensor_ = CloneTensor(data_tensor_);
    CAFFE_FFI_MEM_LOG << "[COW] Blob#" << id_
                      << " mutable_data_tensor() COW"
                      << " refcount=" << refcount
                      << " old_ptr=" << old_ptr
                      << " new_ptr=" << data_tensor_.data_ptr()
                      << " nbytes=" << nbytes;
  }
  CAFFE_FFI_TENSOR_LOG << "mutable_data_tensor() Blob#" << id_ << " this=" << this
                       << " ptr=" << PtrToString(data_tensor_.data_ptr())
                       << " refcount=" << DataRefCount();
  return data_tensor_;
}

Tensor Blob::mutable_diff_tensor() {
  if (diff_tensor_.defined() && diff_tensor_.use_count() > 1) {
    int refcount = diff_tensor_.use_count();
    const void* old_ptr = diff_tensor_.data_ptr();
    int64_t nbytes = diff_tensor_.numel() * static_cast<int64_t>(sizeof(float));
    diff_tensor_ = CloneTensor(diff_tensor_);
    CAFFE_FFI_MEM_LOG << "[COW] Blob#" << id_
                      << " mutable_diff_tensor() COW"
                      << " refcount=" << refcount
                      << " old_ptr=" << old_ptr
                      << " new_ptr=" << diff_tensor_.data_ptr()
                      << " nbytes=" << nbytes;
  }
  CAFFE_FFI_TENSOR_LOG << "mutable_diff_tensor() Blob#" << id_ << " this=" << this
                       << " ptr=" << PtrToString(diff_tensor_.data_ptr())
                       << " refcount=" << DiffRefCount();
  return diff_tensor_;
}

void Blob::ShareData(const Blob* other) {
  CAFFE_FFI_CHECK_TYPE(other != nullptr)
      << "ShareData: source Blob must not be null";
  CAFFE_FFI_CHECK_TYPE(other->data_tensor_.defined())
      << "ShareData: source Blob#" << other->id_ << " has undefined data tensor";

  // Phase 3.1: Clear lazy allocation flag when ShareData replaces the tensor
  is_lazy_allocated_ = false;
  shape_only_.clear();

  CAFFE_FFI_MEM_LOG << "[ZEROCOPY] Blob#" << id_ << " ShareData from Blob#" << other->id_
                    << " this=" << this
                    << " old_data_ptr=" << PtrToString(data_tensor_.data_ptr())
                    << " new_data_ptr=" << PtrToString(other->data_tensor_.data_ptr())
                    << " shape=" << ShapeToString(ShapeView(other->data_tensor_.shape().data(),
                                                             static_cast<size_t>(other->data_tensor_.ndim())))
                    << " nbytes=" << TensorNBytes(other->data_tensor_)
                    << " (zero-copy: refcount shared, no memcpy)";
  data_tensor_ = other->data_tensor_;
}

void Blob::ShareDiff(const Blob* other) {
  CAFFE_FFI_CHECK_TYPE(other != nullptr)
      << "ShareDiff: source Blob must not be null";
  CAFFE_FFI_CHECK_TYPE(other->diff_tensor_.defined())
      << "ShareDiff: source Blob#" << other->id_ << " has undefined diff tensor";

  // Phase 3.1: Clear lazy allocation flag
  is_lazy_allocated_ = false;
  shape_only_.clear();

  CAFFE_FFI_MEM_LOG << "[ZEROCOPY] Blob#" << id_ << " ShareDiff from Blob#" << other->id_
                    << " this=" << this
                    << " old_diff_ptr=" << PtrToString(diff_tensor_.data_ptr())
                    << " new_diff_ptr=" << PtrToString(other->diff_tensor_.data_ptr())
                    << " nbytes=" << TensorNBytes(other->diff_tensor_)
                    << " (zero-copy: refcount shared, no memcpy)";
  diff_tensor_ = other->diff_tensor_;
}

bool Blob::SharesDataWith(const Blob* other) const {
  return other != nullptr && data_tensor_.data_ptr() == other->data_tensor_.data_ptr() && data_tensor_.defined();
}

bool Blob::SharesDiffWith(const Blob* other) const {
  return other != nullptr && diff_tensor_.data_ptr() == other->diff_tensor_.data_ptr() && diff_tensor_.defined();
}

#ifdef CAFFE_FFI_ENABLE_COW_PHASE3
namespace {

// ─── Phase 3 Batch Refcount Internal Helpers ─────────────────────────────────
//
// These helpers implement O(1) batch IncRef by directly atomically adding N to
// the TVMFFIObject::combined_ref_count (lower 32 bits = strong refcount).
// The normal per-copy ObjectPtr constructor calls Object::IncRef() which does
// a fetch_add(1) — for N targets this means N atomic operations. For N=100
// that's ~10μs of pure atomic overhead. The batch approach reduces this to a
// single fetch_add(N) (~10ns) plus N raw pointer writes.
//
// ═══ RISK ANALYSIS (memory leak / refcount imbalance) ═══
//
// Risk #1: Self-reference (source ∈ targets)
//   - If source Blob's own tensor is released in Phase 1, src_obj becomes dangling.
//   - Mitigation: Filter out targets[i] == source before Phase 1 (defensive guard).
//
// Risk #2: Exception between Phase 1 and Phase 3
//   - If Phase 1 completes (old refs released) but Phase 2/3 throws, some targets
//     have nullptr tensors (valid but data-less) and source refcount is not yet
//     incremented by n. This is recoverable (targets are reset to null, source
//     is still alive). BatchStrongIncRef is a compiler intrinsic that cannot throw;
//     AssignRawTensorNoIncRef is a memcpy that cannot throw. The only throwing
//     points are CAFFE_FFI_CHECK_* macros which fire on programmer errors —
//     in that case we abort loudly rather than silently corrupt memory.
//
// Risk #3: ObjectPtr memory layout assumption violation
//   - We rely on Tensor being pointer-sized with Object* at offset 0. This is
//     guaranteed by TVM FFI's design (ObjectRef → ObjectPtr<Object> → Object*).
//   - Mitigation: static_assert(sizeof(Tensor) == sizeof(Object*)) catches any
//     layout change at compile time.
//
// Risk #4: Weak refcount bit corruption
//   - Adding n to combined_ref_count adds to lower 32 bits (strong). As long as
//     strong refcount < 2^32, no carry into upper 32 bits (weak). With n ≤ 10000
//     and typical refcounts < 1000, overflow is impossible. Normal IncRef() does
//     the exact same add-1 to combined_ref_count, so we are consistent with TVM FFI.
//
// Risk #5: Double-IncRef after AssignRawTensorNoIncRef
//   - After memcpy-planting the raw pointer, the Tensor "owns" one reference
//     (balanced by the n added in Phase 2). The Tensor's destructor/reset() will
//     DecRef exactly once. If ShareData() or operator= is later called on the
//     target, ObjectPtr's copy assignment IncRefs the new object first, then
//     DecRefs the old (our planted) pointer — net zero for self-assignment, net
//     +1 for new object — correct refcount semantics.
//
// Risk #6: Reshape/COW interaction
//   - After batch share, Reshape() allocates a new tensor (DecRefs source, IncRefs
//     new private tensor) — correct. cpu_mutable_data() triggers UnshareData()
//     which checks use_count() > 1 and clones — correct, because use_count()
//     includes the n batch-added references.

inline void BatchStrongIncRef(const Object* obj, int n) {
  TVMFFIObject* header = details::ObjectUnsafe::GetHeader(obj);
#ifdef _MSC_VER
  _InlineInterlockedAdd64(
      reinterpret_cast<volatile __int64*>(&header->combined_ref_count),
      static_cast<__int64>(n));
#else
  __atomic_fetch_add(&header->combined_ref_count, static_cast<uint64_t>(n), __ATOMIC_RELAXED);
#endif
}

inline const Object* ReleaseTensorRef(Tensor& t) {
  const Object* old = t.defined() ? t.get() : nullptr;
  t = Tensor();  // reset: DecRef old object, data_ becomes nullptr
  return old;
}

inline void AssignRawTensorNoIncRef(Tensor& t, Object* raw) {
  static_assert(sizeof(Tensor) == sizeof(Object*),
                "Tensor must be pointer-sized for Phase 3 batch share");
  Object* p = raw;
  std::memcpy(&t, &p, sizeof(p));
}

/**
 * @brief Core implementation shared by BatchShareData and BatchShareDiff.
 * @param TensorField  Member pointer to Tensor (e.g., &Blob::data_tensor_)
 * @param kind         "Data" or "Diff" for log messages
 */
void BatchShareImpl(const Blob* source, const std::vector<Blob*>& targets,
                    Tensor Blob::*TensorField, const char* kind) {
  CAFFE_FFI_CHECK_TYPE(source != nullptr)
      << "BatchShare" << kind << ": source must not be null";
  CAFFE_FFI_CHECK_TYPE((source->*TensorField).defined())
      << "BatchShare" << kind << ": source Blob#" << source->id_
      << " has undefined " << kind << " tensor";

  int n_raw = static_cast<int>(targets.size());

  // ── Defensive: filter out self-references (source must not be in targets) ──
  std::vector<Blob*> safe_targets;
  safe_targets.reserve(n_raw);
  for (int i = 0; i < n_raw; ++i) {
    Blob* tgt = targets[i];
    CAFFE_FFI_CHECK_TYPE(tgt != nullptr)
        << "BatchShare" << kind << ": targets[" << i << "] is null";
    if (tgt == source) {
      CAFFE_FFI_LOG_WARN() << "[BATCH-SHARE] WARNING: targets[" << i
                           << "] == source (Blob#" << source->id_
                           << "), skipping self-reference to prevent double-free";
      continue;
    }
    safe_targets.push_back(tgt);
  }

  int n = static_cast<int>(safe_targets.size());
  if (n == 0) {
    CAFFE_FFI_LOG_WARN() << "[BATCH-SHARE] BatchShare" << kind
                         << ": no valid targets after filtering, nothing to do";
    return;
  }

  const Tensor& src_tensor = source->*TensorField;
  const Object* src_obj = src_tensor.get();
  const int64_t nbytes = TensorNBytes(src_tensor);
  const void* src_dptr = src_tensor.data_ptr();

  // ── ENTRY LOG: state before any mutations ──
  uint64_t rc_entry = src_obj->use_count();
  CAFFE_FFI_LOG_WARN() << "[BATCH-SHARE] ═══ ENTRY BatchShare" << kind
                       << " ═══ src=Blob#" << source->id_
                       << " n=" << n << " (raw_input=" << n_raw << ")"
                       << " src_dptr=" << PtrToString(src_dptr)
                       << " nbytes=" << nbytes
                       << " rc_entry=" << rc_entry
                       << " (expected: 1 atomic add of " << n << ")";

  // Per-target pre-snapshot: log which targets already share with source
  int already_shared = 0;
  for (int i = 0; i < n; ++i) {
    Blob* tgt = safe_targets[i];
    const Tensor& tgt_tensor = tgt->*TensorField;
    bool already = tgt_tensor.defined() && tgt_tensor.get() == src_obj;
    if (already) ++already_shared;
    CAFFE_FFI_TENSOR_LOG << "[BATCH-SHARE]   pre[" << i << "] Blob#" << tgt->id_
                         << " defined=" << (tgt_tensor.defined() ? "yes" : "no")
                         << " dptr=" << PtrToString(tgt_tensor.defined() ? tgt_tensor.data_ptr() : nullptr)
                         << " already_sharing_src=" << (already ? "yes" : "no");
  }
  if (already_shared > 0) {
    CAFFE_FFI_LOG_WARN() << "[BATCH-SHARE] " << already_shared << "/" << n
                         << " targets already share source " << kind
                         << " — they will be re-linked (DecRef+IncRef cancels out)";
  }

  // ── Phase 1: Release all target old references ──
  auto t_p1_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n; ++i) {
    Blob* tgt = safe_targets[i];
    const Object* old = ReleaseTensorRef(tgt->*TensorField);
    CAFFE_FFI_TENSOR_LOG << "[BATCH-SHARE]   p1_release[" << i << "] Blob#" << tgt->id_
                         << " released old_obj=" << old;
  }
  auto t_p1_end = std::chrono::high_resolution_clock::now();
  uint64_t rc_after_p1 = src_obj->use_count();
  double p1_us = std::chrono::duration<double, std::micro>(t_p1_end - t_p1_start).count();
  CAFFE_FFI_LOG_WARN() << "[BATCH-SHARE]   Phase 1 (release) done: rc "
                       << rc_entry << " → " << rc_after_p1
                       << " (Δ=" << static_cast<int64_t>(rc_after_p1) - static_cast<int64_t>(rc_entry) << ")"
                       << " time=" << p1_us << "us";

  // ── Phase 2: Single atomic add of N to source refcount ──
  auto t_p2_start = std::chrono::high_resolution_clock::now();
  BatchStrongIncRef(src_obj, n);
  auto t_p2_end = std::chrono::high_resolution_clock::now();
  uint64_t rc_after_p2 = src_obj->use_count();
  double p2_us = std::chrono::duration<double, std::micro>(t_p2_end - t_p2_start).count();
  CAFFE_FFI_LOG_WARN() << "[BATCH-SHARE]   Phase 2 (atomic+" << n << ") done: rc "
                       << rc_after_p1 << " → " << rc_after_p2
                       << " (Δ=" << static_cast<int64_t>(rc_after_p2) - static_cast<int64_t>(rc_after_p1) << ")"
                       << " time=" << p2_us << "us"
                       << " (vs " << n << "× ~10ns = " << n * 10 << "ns per-atomics path)";

  // Sanity: refcount should be rc_after_p1 + n
  CAFFE_FFI_CHECK_RUNTIME_EQ(rc_after_p2, rc_after_p1 + static_cast<uint64_t>(n))
      << "BatchShare" << kind << ": refcount mismatch after atomic add: expected "
      << (rc_after_p1 + n) << ", got " << rc_after_p2;

  // ── Phase 3: Raw pointer assignment without IncRef ──
  auto t_p3_start = std::chrono::high_resolution_clock::now();
  Object* raw = const_cast<Object*>(src_obj);
  for (int i = 0; i < n; ++i) {
    AssignRawTensorNoIncRef(safe_targets[i]->*TensorField, raw);
  }
  auto t_p3_end = std::chrono::high_resolution_clock::now();
  uint64_t rc_after_p3 = src_obj->use_count();
  double p3_us = std::chrono::duration<double, std::micro>(t_p3_end - t_p3_start).count();

  // ── Post-assignment verification ──
  int verify_ok = 0, verify_fail = 0;
  for (int i = 0; i < n; ++i) {
    Blob* tgt = safe_targets[i];
    const Tensor& tgt_tensor = tgt->*TensorField;
    bool points_to_src = tgt_tensor.defined() && tgt_tensor.get() == src_obj;
    bool dptr_matches = tgt_tensor.defined() && tgt_tensor.data_ptr() == src_dptr;
    if (points_to_src && dptr_matches) { ++verify_ok; }
    else { ++verify_fail; }
    CAFFE_FFI_TENSOR_LOG << "[BATCH-SHARE]   p3_assign[" << i << "] Blob#" << tgt->id_
                         << " points_to_src=" << (points_to_src ? "yes" : "NO")
                         << " dptr_matches=" << (dptr_matches ? "yes" : "NO")
                         << " rc_now=" << src_obj->use_count();
  }

  // Final sanity check
  CAFFE_FFI_CHECK_RUNTIME_EQ(rc_after_p3, rc_after_p1 + static_cast<uint64_t>(n))
      << "BatchShare" << kind << ": refcount mismatch after assignment: expected "
      << (rc_after_p1 + n) << ", got " << rc_after_p3;
  CAFFE_FFI_CHECK_RUNTIME_EQ(verify_fail, 0u)
      << "BatchShare" << kind << ": " << verify_fail << "/" << n
      << " targets failed verification (not pointing to source)";

  double total_us = p1_us + p2_us + p3_us;
  double per_target_atomic_savings_ns = static_cast<double>(n - 1) * 10.0;
  CAFFE_FFI_LOG_WARN() << "[BATCH-SHARE] ═══ EXIT BatchShare" << kind
                       << " ═══ rc: " << rc_entry << " → " << rc_after_p3
                       << " (net Δ=" << static_cast<int64_t>(rc_after_p3) - static_cast<int64_t>(rc_entry) << ")"
                       << " verify=" << verify_ok << "/" << n << " OK"
                       << " time: p1=" << p1_us << "us + p2=" << p2_us
                       << "us + p3=" << p3_us << "us = " << total_us << "us"
                       << " atomic_savings≈" << per_target_atomic_savings_ns << "ns"
                       << " (replaced " << n << " atomics with 1)";
}

}  // namespace

void Blob::BatchShareData(const Blob* source, const std::vector<Blob*>& targets) {
  BatchShareImpl(source, targets, &Blob::data_tensor_, "Data");
}

void Blob::BatchShareDiff(const Blob* source, const std::vector<Blob*>& targets) {
  BatchShareImpl(source, targets, &Blob::diff_tensor_, "Diff");
}
#endif  // CAFFE_FFI_ENABLE_COW_PHASE3

void* Blob::UnshareData() {
  if (data_tensor_.defined() && data_tensor_.use_count() > 1) {
    int refcount = data_tensor_.use_count();
    const void* old_ptr = data_tensor_.data_ptr();
    int64_t nbytes = data_tensor_.numel() * static_cast<int64_t>(sizeof(float));
    data_tensor_ = CloneTensor(data_tensor_);
    CAFFE_FFI_MEM_LOG << "[COW] Blob#" << id_
                      << " UnshareData() explicit COW"
                      << " refcount=" << refcount
                      << " old_ptr=" << old_ptr
                      << " new_ptr=" << data_tensor_.data_ptr()
                      << " nbytes=" << nbytes;
  }
  return data_tensor_.defined() ? data_tensor_.data_ptr() : nullptr;
}

void* Blob::UnshareDiff() {
  if (diff_tensor_.defined() && diff_tensor_.use_count() > 1) {
    int refcount = diff_tensor_.use_count();
    const void* old_ptr = diff_tensor_.data_ptr();
    int64_t nbytes = diff_tensor_.numel() * static_cast<int64_t>(sizeof(float));
    diff_tensor_ = CloneTensor(diff_tensor_);
    CAFFE_FFI_MEM_LOG << "[COW] Blob#" << id_
                      << " UnshareDiff() explicit COW"
                      << " refcount=" << refcount
                      << " old_ptr=" << old_ptr
                      << " new_ptr=" << diff_tensor_.data_ptr()
                      << " nbytes=" << nbytes;
  }
  return diff_tensor_.defined() ? diff_tensor_.data_ptr() : nullptr;
}

void Blob::Reshape(ShapeView shape) {
  // Phase 3.1: Clear lazy allocation flag on any Reshape call
  is_lazy_allocated_ = false;
  shape_only_.clear();

  for (size_t i = 0; i < shape.size(); ++i) {
    CAFFE_FFI_CHECK_VALUE_GE(shape[i], 0)
        << "Blob#" << id_ << " Reshape: dimension " << i << " is negative (" << shape[i] << ")";
  }
  bool shape_changed = !data_tensor_.defined() || (shape.size() != static_cast<size_t>(data_tensor_.ndim()));
  if (!shape_changed) {
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] != data_tensor_.size(static_cast<int>(i))) {
        shape_changed = true;
        break;
      }
    }
  }
  int64_t new_count = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    new_count *= shape[i];
  }
  int64_t old_count = data_tensor_.defined() ? data_tensor_.numel() : 0;
  const void* old_data_ptr = data_tensor_.defined() ? data_tensor_.data_ptr() : nullptr;
  const void* old_diff_ptr = diff_tensor_.defined() ? diff_tensor_.data_ptr() : nullptr;
  int64_t old_nbytes = TensorNBytes(data_tensor_) + TensorNBytes(diff_tensor_);

  if (shape_changed || !data_tensor_.defined()) {
    int64_t new_total_nbytes = new_count * sizeof(float) * 2;
    int64_t net_delta = new_total_nbytes - old_nbytes;

    CAFFE_FFI_MEM_LOG << "[MEM-RESIZE] Blob#" << id_ << " this=" << this
                      << " shape=" << ShapeToString(shape)
                      << " old_count=" << old_count << " new_count=" << new_count
                      << " old_data_ptr=" << PtrToString(old_data_ptr)
                      << " old_diff_ptr=" << PtrToString(old_diff_ptr)
                      << " old_nbytes=" << old_nbytes << "B (" << FormatBytes(old_nbytes) << ")"
                      << " new_nbytes=" << new_total_nbytes << "B (" << FormatBytes(new_total_nbytes) << ")"
                      << " net_delta=" << (net_delta >= 0 ? "+" : "") << net_delta << "B";

    int64_t global_before = g_total_allocated_bytes.load(std::memory_order_relaxed);

    data_tensor_ = NewCPUTensor(shape);
    diff_tensor_ = NewCPUTensor(shape);

    int64_t global_after = g_total_allocated_bytes.load(std::memory_order_relaxed);

    CAFFE_FFI_MEM_LOG << "[MEM-RESIZE] Blob#" << id_
                      << " new_data_ptr=" << PtrToString(data_tensor_.data_ptr())
                      << " new_diff_ptr=" << PtrToString(diff_tensor_.data_ptr())
                      << " global_delta=" << (net_delta >= 0 ? "+" : "") << net_delta << "B"
                      << " global_before=" << global_before << "B (" << FormatBytes(global_before) << ")"
                      << " global_after=" << global_after << "B (" << FormatBytes(global_after) << ")"
                      << " live_blobs=" << g_live_blob_count.load(std::memory_order_relaxed);
  } else {
    CAFFE_FFI_TENSOR_LOG << "Reshape: Blob#" << id_ << " shape unchanged " << ShapeToString(shape)
                         << " (count=" << new_count << "), skipping reallocation"
                         << " data_ptr=" << PtrToString(data_tensor_.data_ptr())
                         << " diff_ptr=" << PtrToString(diff_tensor_.data_ptr());
  }
}

void Blob::Reshape(const std::vector<int64_t>& shape) {
  Reshape(ShapeView(shape.data(), shape.size()));
}

void Blob::Reshape(Shape shape) {
  Reshape(ShapeView(shape.data(), shape.size()));
}

void Blob::Reshape(const caffe::BlobShape& shape) {
  std::vector<int64_t> dims;
  for (int i = 0; i < shape.dim_size(); ++i) {
    dims.push_back(shape.dim(i));
  }
  Reshape(dims);
}

void Blob::ReshapeLike(const Blob& other) {
  std::vector<int64_t> dims;
  for (int i = 0; i < other.num_axes(); ++i) {
    dims.push_back(other.shape(i));
  }
  Reshape(dims);
}

int64_t Blob::LegacyShape(int index) const {
  if (index >= num_axes()) {
    return 1;
  }
  return shape(index);
}

// ── Phase 3.1: Lazy Allocation (SetShapeOnly) ────────────────────────

void Blob::SetShapeOnly(ShapeView shape) {
  // Validate: all dimensions must be positive
  for (size_t i = 0; i < shape.size(); ++i) {
    CAFFE_FFI_CHECK_VALUE_GT(shape[i], 0)
        << "Blob#" << id_ << " SetShapeOnly: dimension " << i
        << " is " << shape[i] << " (must be positive)";
  }

  // Store shape metadata without allocating data tensor
  shape_only_.assign(shape.data(), shape.data() + shape.size());
  is_lazy_allocated_ = true;

  // Compute count for log
  int64_t total_count = 1;
  for (size_t i = 0; i < shape.size(); ++i) total_count *= shape[i];

  CAFFE_FFI_MEM_LOG << "[LAZY] Blob#" << id_
                    << " SetShapeOnly: shape=" << ShapeToString(shape)
                    << " count=" << total_count
                    << " (no data allocated, data_tensor_ remains undefined)";
}

void Blob::FromProto(const caffe::BlobProto& proto, bool reshape) {
  CAFFE_FFI_CONTAINER_LOG << "FromProto: Blob#" << id_ << " reshape=" << reshape
                           << " proto.data_size=" << proto.data_size()
                           << " proto.double_data_size=" << proto.double_data_size()
                           << " proto.diff_size=" << proto.diff_size();
  if (reshape) {
    std::vector<int64_t> shape;
    if (proto.has_shape()) {
      for (int i = 0; i < proto.shape().dim_size(); ++i) {
        shape.push_back(proto.shape().dim(i));
      }
    } else {
      if (proto.num() > 0) shape.push_back(proto.num());
      if (proto.channels() > 0) shape.push_back(proto.channels());
      if (proto.height() > 0) shape.push_back(proto.height());
      if (proto.width() > 0) shape.push_back(proto.width());
      if (shape.empty()) shape.push_back(0);
    }
    CAFFE_FFI_TENSOR_LOG << "FromProto: Blob#" << id_ << " reshaping to " << ShapeToString(ShapeView(shape.data(), shape.size()));
    Reshape(shape);
  }
  float* data_ptr = cpu_mutable_data();
  const int data_count = proto.data_size();
  const int double_data_count = proto.double_data_size();
  if (data_count > 0) {
    CAFFE_FFI_CHECK_RUNTIME_EQ(data_count, count())
        << "Incorrect data size for Blob: expected " << count() << ", got " << data_count;
    CAFFE_FFI_CONTAINER_LOG << "FromProto: Blob#" << id_ << " copying " << data_count << " float data elements to " << data_ptr;
    std::copy(proto.data().begin(), proto.data().end(), data_ptr);
  } else if (double_data_count > 0) {
    CAFFE_FFI_CHECK_RUNTIME_EQ(double_data_count, count())
        << "Incorrect double_data size for Blob: expected " << count() << ", got " << double_data_count;
    CAFFE_FFI_CONTAINER_LOG << "FromProto: Blob#" << id_ << " converting " << double_data_count << " double→float data elements";
    for (int i = 0; i < double_data_count; ++i) {
      data_ptr[i] = static_cast<float>(proto.double_data(i));
    }
  }
  float* diff_ptr = cpu_mutable_diff();
  const int diff_count = proto.diff_size();
  const int double_diff_count = proto.double_diff_size();
  if (diff_count > 0) {
    CAFFE_FFI_CHECK_RUNTIME_EQ(diff_count, count())
        << "Incorrect diff size for Blob: expected " << count() << ", got " << diff_count;
    CAFFE_FFI_CONTAINER_LOG << "FromProto: Blob#" << id_ << " copying " << diff_count << " float diff elements to " << diff_ptr;
    std::copy(proto.diff().begin(), proto.diff().end(), diff_ptr);
  } else if (double_diff_count > 0) {
    CAFFE_FFI_CHECK_RUNTIME_EQ(double_diff_count, count())
        << "Incorrect double_diff size for Blob: expected " << count() << ", got " << double_diff_count;
    CAFFE_FFI_CONTAINER_LOG << "FromProto: Blob#" << id_ << " converting " << double_diff_count << " double→float diff elements";
    for (int i = 0; i < double_diff_count; ++i) {
      diff_ptr[i] = static_cast<float>(proto.double_diff(i));
    }
  } else {
    CAFFE_FFI_CONTAINER_LOG << "FromProto: Blob#" << id_ << " zeroing diff tensor (" << count() << " elements at " << diff_ptr << ")";
    caffe_set_fp32(static_cast<size_t>(count()), 0.0f, diff_ptr);
  }
  CAFFE_FFI_BLOB_LOG << "FromProto: Blob#" << id_ << " completed, data_ptr=" << data_ptr << " diff_ptr=" << diff_ptr;
}

void Blob::ToProto(caffe::BlobProto* proto) const {
  proto->Clear();
  auto* shape_proto = proto->mutable_shape();
  shape_proto->clear_dim();
  for (int i = 0; i < num_axes(); ++i) {
    shape_proto->add_dim(shape(i));
  }
  proto->clear_data();
  const float* data_ptr = cpu_data();
  for (int64_t i = 0; i < count(); ++i) {
    proto->add_data(data_ptr[i]);
  }
  proto->clear_diff();
  const float* diff_ptr = cpu_diff();
  for (int64_t i = 0; i < count(); ++i) {
    proto->add_diff(diff_ptr[i]);
  }
}

void Blob::Update() {
  CAFFE_FFI_TENSOR_LOG << "Update() Blob#" << id_ << " this=" << this
                       << " data_ptr=" << PtrToString(cpu_data())
                       << " diff_ptr=" << PtrToString(cpu_diff())
                       << " count=" << count()
                       << " operation: data -= diff";
  caffe_cpu_axpby_fp32(static_cast<size_t>(count()), -1.0f, cpu_diff(), 1.0f, cpu_mutable_data());
}

Array<float> Blob::get_data() const {
  Array<float> result;
  result.reserve(count());
  const float* ptr = cpu_data();
  CAFFE_FFI_CONTAINER_LOG << "get_data: Blob#" << id_ << " copying " << count() << " elements from " << ptr << " to Array<float>";
  for (int64_t i = 0; i < count(); ++i) {
    result.push_back(ptr[i]);
  }
  CAFFE_FFI_CONTAINER_LOG << "get_data: Blob#" << id_ << " Array<float> size=" << result.size() << " created";
  return result;
}

void Blob::set_data(Tensor data) {
  // Support DLPack zero-copy interop with numpy/PyTorch/etc.
  CAFFE_FFI_CHECK_TYPE(data.defined()) << "Cannot set_data from undefined Tensor (Blob#" << id_ << ")";
  CAFFE_FFI_CHECK_VALUE_EQ(data.ndim(), num_axes())
      << "Tensor ndim mismatch for Blob#" << id_ << ": expected " << num_axes() << ", got " << data.ndim();
  for (int i = 0; i < data.ndim(); ++i) {
    CAFFE_FFI_CHECK_VALUE_EQ(data.size(i), shape(i))
        << "Tensor shape mismatch at axis " << i << " for Blob#" << id_
        << ": expected " << shape(i) << ", got " << data.size(i);
  }
  CAFFE_FFI_CHECK_TYPE(data.dtype().code == kDLFloat && data.dtype().bits == 32)
      << "set_data expects float32 Tensor for Blob#" << id_
      << ", got dtype code=" << static_cast<int>(data.dtype().code) << " bits=" << data.dtype().bits;

  float* dst = cpu_mutable_data();
  const float* src = static_cast<const float*>(data.data_ptr());
  int64_t nbytes = count() * sizeof(float);
  CAFFE_FFI_CONTAINER_LOG << "set_data(Tensor): Blob#" << id_ << " memcpy " << count()
                          << " elements (" << nbytes << "B) from " << src << " to " << dst;
  std::memcpy(dst, src, nbytes);
}

Array<float> Blob::get_diff() const {
  Array<float> result;
  result.reserve(count());
  const float* ptr = cpu_diff();
  CAFFE_FFI_CONTAINER_LOG << "get_diff: Blob#" << id_ << " copying " << count() << " elements from " << ptr << " to Array<float>";
  for (int64_t i = 0; i < count(); ++i) {
    result.push_back(ptr[i]);
  }
  return result;
}

void Blob::set_diff(Tensor diff) {
  CAFFE_FFI_CHECK_TYPE(diff.defined()) << "Cannot set_diff from undefined Tensor (Blob#" << id_ << ")";
  CAFFE_FFI_CHECK_VALUE_EQ(diff.ndim(), num_axes())
      << "Tensor ndim mismatch for Blob#" << id_ << " diff: expected " << num_axes() << ", got " << diff.ndim();
  for (int i = 0; i < diff.ndim(); ++i) {
    CAFFE_FFI_CHECK_VALUE_EQ(diff.size(i), shape(i))
        << "Tensor shape mismatch at axis " << i << " for Blob#" << id_ << " diff"
        << ": expected " << shape(i) << ", got " << diff.size(i);
  }
  CAFFE_FFI_CHECK_TYPE(diff.dtype().code == kDLFloat && diff.dtype().bits == 32)
      << "set_diff expects float32 Tensor for Blob#" << id_
      << ", got dtype code=" << static_cast<int>(diff.dtype().code) << " bits=" << diff.dtype().bits;

  float* dst = cpu_mutable_diff();
  const float* src = static_cast<const float*>(diff.data_ptr());
  int64_t nbytes = count() * sizeof(float);
  CAFFE_FFI_CONTAINER_LOG << "set_diff(Tensor): Blob#" << id_ << " memcpy " << count()
                          << " elements (" << nbytes << "B) from " << src << " to " << dst;
  std::memcpy(dst, src, nbytes);
}

int64_t TotalAllocatedBytes() {
  int64_t val = g_total_allocated_bytes.load(std::memory_order_relaxed);
  CAFFE_FFI_MEM_LOG << "[MEM-QUERY] TotalAllocatedBytes() = " << val << "B (" << FormatBytes(val) << ")"
                    << " live_blobs=" << g_live_blob_count.load(std::memory_order_relaxed);
  return val;
}

int64_t LiveBlobCount() {
  int64_t val = g_live_blob_count.load(std::memory_order_relaxed);
  CAFFE_FFI_MEM_LOG << "[MEM-QUERY] LiveBlobCount() = " << val
                    << " total_allocated=" << g_total_allocated_bytes.load(std::memory_order_relaxed) << "B";
  return val;
}

void SetCOWEnabled(bool enabled) {
  bool old = g_cow_enabled.exchange(enabled, std::memory_order_relaxed);
  CAFFE_FFI_MEM_LOG << "[COW] Runtime switch: " << (old ? "ENABLED→" : "DISABLED→")
                    << (enabled ? "ENABLED" : "DISABLED");
}

bool IsCOWEnabled() {
  return g_cow_enabled.load(std::memory_order_relaxed);
}

}  // namespace caffe_ffi
