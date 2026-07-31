#include "caffe_ffi/layers/split_layer.hpp"

#include <chrono>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/fill.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

/// Phase 3.0: Log aggregation threshold for Split layer.
/// When N >= this threshold, per-top detailed logs (CAFFE_FFI_LAYER_LOG
/// level) are skipped in both Reshape() and Forward_cpu() to prevent log
/// flooding. The summary [SPLIT-PERF] log (CAFFE_FFI_LOG_WARN level) is
/// always emitted regardless of N.
///
/// Design rationale: N=32 is chosen as the default because:
///   - N=32 produces ~64 per-top log lines (data+diff), already noisy
///   - N=32 is above typical CNN fan-out (1-16) but below extreme cases
///   - Atomic op overhead for N=32 is ~640ns, still negligible
constexpr int kLogAggregateThreshold = 32;

/// Phase 3.1: Lazy Reshape threshold for Split layer.
/// When N >= this threshold, Reshape() uses SetShapeOnly() to store only
/// shape metadata without allocating data memory. The actual allocation is
/// deferred to Forward() where ShareData() replaces the lazy tensor with
/// a shared reference.
///
/// N=16 is chosen because:
///   - N=16 produces 16 × 1KB = 16KB allocation overhead — negligible
///   - N=16 ≈ 2× typical CNN fan-out (1-8), above which lazy pays off
///   - Aligns with kLogAggregateThreshold (32) as a "medium" tier
///
/// Three-tier layering (kLazyReshapeThreshold=16, kLogAggregateThreshold=32):
///   N<16:  Phase 2 — per-top ReshapeLike with timing and per-top log
///   16≤N<32: Phase 3.1 — SetShapeOnly (no allocation), no per-top log
///   N≥32: Phase 3.0+3.1 — SetShapeOnly (no allocation), [SPLIT-PERF] summary
///          shows lazy_reshape=yes, total_alloc_bytes=0, log_aggregated=yes
constexpr int kLazyReshapeThreshold = 16;

void SplitLayer::LayerSetUp(const std::vector<Blob*>& bottom,
                             const std::vector<Blob*>& top) {
  int num_top = static_cast<int>(top.size());
  int num_bottom = static_cast<int>(bottom.size());

  CAFFE_FFI_LAYER_LOG << "Split LayerSetUp: name='" << this->name()
                      << "' num_bottom=" << num_bottom
                      << " num_top=" << num_top;

  // N=1 single-top Split is a rename/identity operation.
  // This is useful for in-place computation but can be confusing because
  // the original bottom blob name becomes unavailable (consumed by Split).
  // For residual connections, you typically need N=2 (one top for the
  // identity/residual path, one for the sublayer path).
  if (num_top == 1) {
    CAFFE_FFI_LOG_WARN() << "[SPLIT-N1] Split '" << this->name()
                         << "' has num_top=1 (identity/rename path)."
                         << " The bottom blob is consumed and ONLY the single top name"
                         << " will be available downstream. If you need the original blob"
                         << " for a residual/skip connection, use num_top=2:"
                         << " one top for the sublayer input and one top for the"
                         << " residual identity path."
                         << " COW sharing: N=1 uses direct zero-copy ShareData/ShareDiff.";
  } else {
    CAFFE_FFI_LOG_WARN() << "[SPLIT-FANOUT] Split '" << this->name()
                         << "' fans out bottom to " << num_top << " tops."
                         << " COW (Copy-on-Write) is enabled: all tops initially share"
                         << " bottom's memory; actual copies are deferred to first"
                         << " mutable access (cpu_mutable_data/cpu_mutable_diff)."
                         << " Top names will all be available as independent blobs downstream.";
  }
}

void SplitLayer::Reshape(const std::vector<Blob*>& bottom,
                          const std::vector<Blob*>& top) {
  int count = bottom[0]->count();
  int num_top = static_cast<int>(top.size());

  std::ostringstream shape_ss;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i > 0) shape_ss << ", ";
    shape_ss << bottom[0]->shape(i);
  }

  CAFFE_FFI_LAYER_LOG << "Split Reshape: bottom shape=[" << shape_ss.str() << "]"
                      << " count=" << count
                      << " num_top=" << num_top
                      << " copy_bytes=" << (count * static_cast<int64_t>(sizeof(float)));

  // Measure total reshape (memory allocation) time
  auto t_reshape_start = std::chrono::high_resolution_clock::now();
  int64_t total_alloc_bytes = 0;
  bool used_lazy_reshape = false;

  // Note: For N=1 zero-copy path, we still reshape top[0] to the correct shape
  // here so downstream layers see valid shapes during their own Reshape().
  // The zero-copy ShareData() in Forward() will replace the tensor reference,
  // freeing the Reshape-allocated buffer (one alloc+free overhead but avoids
  // breaking the layer setup contract).
  for (int i = 0; i < num_top; ++i) {
#ifdef CAFFE_FFI_ENABLE_COW_PHASE3
    if (num_top >= kLazyReshapeThreshold) {
      // Phase 3.1: Lazy allocation — store shape only, no memory allocation.
      // Forward() will replace the lazy tensor with ShareData().
      auto bottom_shape = bottom[0]->shape();
      top[i]->SetShapeOnly(ShapeView(bottom_shape.data(), bottom_shape.size()));
      used_lazy_reshape = true;
      continue;
    }
#endif
    if (num_top < kLogAggregateThreshold) {
      auto t_top_start = std::chrono::high_resolution_clock::now();
      int64_t bytes_before = top[i]->count() * static_cast<int64_t>(sizeof(float));
      top[i]->ReshapeLike(*bottom[0]);
      int64_t bytes_after = top[i]->count() * static_cast<int64_t>(sizeof(float));
      auto t_top_end = std::chrono::high_resolution_clock::now();
      double top_reshape_us = std::chrono::duration<double, std::micro>(
          t_top_end - t_top_start).count();
      total_alloc_bytes += (bytes_after - bytes_before);
      CAFFE_FFI_LAYER_LOG << "Split Reshape: top[" << i << "] reshape done"
                          << " bytes_before=" << bytes_before
                          << " bytes_after=" << bytes_after
                          << " reshape_time=" << top_reshape_us << "us";
    } else {
      // N >= kLogAggregateThreshold: skip per-top timing and log,
      // compute total_alloc_bytes once from batch size.
      top[i]->ReshapeLike(*bottom[0]);
    }
  }
  // For large N (Phase 2, no lazy), compute total_alloc_bytes once.
  // When Phase 3 lazy reshape is active, total_alloc_bytes stays 0 (no allocation).
  if (num_top >= kLogAggregateThreshold && !used_lazy_reshape) {
    total_alloc_bytes = num_top * count * static_cast<int64_t>(sizeof(float));
  }

  auto t_reshape_end = std::chrono::high_resolution_clock::now();
  double reshape_ms = std::chrono::duration<double, std::milli>(
      t_reshape_end - t_reshape_start).count();

  // For N=1, Forward will zero-copy share (no memcpy).
  // For N>=2, Forward will COW-share via refcount (no memcpy). Actual copies
  // are deferred to cpu_mutable_data()/cpu_mutable_diff() on first write.
  int64_t bytes_copied_per_fwd = 0;

  CAFFE_FFI_LOG_WARN() << "[SPLIT-PERF] " << this->name()
                       << " Reshape: num_top=" << num_top
                       << " count=" << count
                       << " elem_size=" << sizeof(float) << "B"
                       << " bytes_copied_per_fwd=" << bytes_copied_per_fwd << "B"
                       << " reshape_time=" << reshape_ms << "ms"
                       << " net_alloc=" << total_alloc_bytes << "B"
                       << " zerocopy_n1=" << ((num_top == 1) ? "yes" : "no")
#ifdef CAFFE_FFI_ENABLE_COW_PHASE3
                       << " lazy_reshape=" << ((num_top >= kLazyReshapeThreshold) ? "yes" : "no")
#endif
                       << " log_aggregated=" << ((num_top >= kLogAggregateThreshold) ? "yes" : "no");
}

void SplitLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                              const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  int count = bottom[0]->count();
  int num_top = static_cast<int>(top.size());
  int64_t copy_bytes_per_top = count * static_cast<int64_t>(sizeof(float));
  int64_t total_copy_bytes = num_top * copy_bytes_per_top;

  CAFFE_FFI_LAYER_LOG << "Split Forward: count=" << count
                      << " num_top=" << num_top
                      << " copy_bytes_per_top=" << copy_bytes_per_top
                      << " total_copy_bytes=" << total_copy_bytes
                      << " bottom_ptr=" << static_cast<const void*>(bottom_data);

  if (num_top == 1) {
    // Phase 1 N=1 zero-copy shortcut: share data/diff tensors directly (refcount)
    // instead of allocating + memcpy. Safe because N=1 means no fan-out —
    // the single top is semantically an identity view of the bottom.
    // Subsequent Reshape() on top[0] will break the share (allocate private copy).
    auto t0 = std::chrono::high_resolution_clock::now();
    bool was_shared = top[0]->SharesDataWith(bottom[0]);
    top[0]->ShareData(bottom[0]);
    top[0]->ShareDiff(bottom[0]);
    auto t1 = std::chrono::high_resolution_clock::now();
    double share_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    bool now_shared = top[0]->SharesDataWith(bottom[0]);
    CAFFE_FFI_LOG_WARN() << "[SPLIT-PERF] " << this->name()
                         << " Forward(N=1 ZEROCOPY): count=" << count
                         << " shared_bytes=" << copy_bytes_per_top << "B"
                         << " share_time=" << share_us << "us"
                         << " data_ptr_equal=" << (now_shared ? "yes" : "no")
                         << " was_already_shared=" << (was_shared ? "yes" : "no")
                         << " memcpy_saved=" << copy_bytes_per_top << "B (zero-copy path)";
    CAFFE_FFI_LAYER_LOG << "Split Forward(N=1 ZEROCOPY): top[0] now shares bottom data,"
                        << " data_ptr=" << static_cast<const void*>(top[0]->cpu_data())
                        << " bottom_ptr=" << static_cast<const void*>(bottom_data);
    return;
  }

  // N >= 2: COW zero-copy sharing (Phase 2)
  // Share data and diff tensors with bottom via intrusive refcount.
  // All tops initially share the same memory as bottom; the first
  // cpu_mutable_data() / cpu_mutable_diff() call on any top triggers
  // COW, cloning the shared tensor into a private copy.
  auto t_share_start = std::chrono::high_resolution_clock::now();

#ifdef CAFFE_FFI_ENABLE_COW_PHASE3
  // Phase 3 batch path: use BatchShareData/BatchShareDiff for large N
  // to reduce atomic refcount operations from O(N) to O(1).
  // Threshold chosen based on atomic op latency (~10-20ns per op):
  //   N >= 16 → batch overhead amortized, ~10× speedup for N=100.
  constexpr int kBATCH_SHARE_THRESHOLD = 16;
  if (num_top >= kBATCH_SHARE_THRESHOLD) {
    Blob::BatchShareData(bottom[0], top);
    Blob::BatchShareDiff(bottom[0], top);

    auto t_share_end = std::chrono::high_resolution_clock::now();
    double share_ms = std::chrono::duration<double, std::milli>(
        t_share_end - t_share_start).count();

    bool all_shared = true;
    for (int i = 0; i < num_top; ++i) {
      if (!top[i]->SharesDataWith(bottom[0])) { all_shared = false; break; }
    }

    CAFFE_FFI_LOG_WARN() << "[SPLIT-PERF] " << this->name()
                         << " Forward(N=" << num_top << " COW-BATCH): count=" << count
                         << " shared_bytes=" << total_copy_bytes << "B"
                         << " share_time=" << share_ms << "ms"
                         << " all_shared=" << (all_shared ? "yes" : "no")
                         << " threshold=" << kBATCH_SHARE_THRESHOLD
                         << " memcpy_saved=" << total_copy_bytes << "B (batch refcount: 1 atomic add of " << num_top << ")";
    CAFFE_FFI_LAYER_LOG << "Split Forward(N=" << num_top << " COW-BATCH): batch share complete,"
                        << " data_ptr=" << static_cast<const void*>(top[0]->cpu_data())
                        << " bottom_ptr=" << static_cast<const void*>(bottom_data)
                        << " refcount=" << bottom[0]->DataRefCount();
    return;
  }
#endif  // CAFFE_FFI_ENABLE_COW_PHASE3

  // Phase 2 per-top path (N < threshold or Phase 3 disabled)
  bool all_shared = true;
  int not_shared_count = 0;

  for (int i = 0; i < num_top; ++i) {
    bool data_was_shared = top[i]->SharesDataWith(bottom[0]);
    bool diff_was_shared = top[i]->SharesDiffWith(bottom[0]);

    top[i]->ShareData(bottom[0]);
    top[i]->ShareDiff(bottom[0]);

    bool data_now_shared = top[i]->SharesDataWith(bottom[0]);
    bool diff_now_shared = top[i]->SharesDiffWith(bottom[0]);

    if (!data_now_shared) { all_shared = false; ++not_shared_count; }

    // Phase 3.0: per-top log aggregation — skip when N >= threshold
    if (num_top < kLogAggregateThreshold) {
      CAFFE_FFI_LAYER_LOG << "Split Forward(N=" << num_top << " COW): top[" << i
                          << "] data_shared=" << (data_now_shared ? "yes" : "no")
                          << " diff_shared=" << (diff_now_shared ? "yes" : "no")
                          << " was_data_shared=" << (data_was_shared ? "yes" : "no")
                          << " was_diff_shared=" << (diff_was_shared ? "yes" : "no")
                          << " data_ptr=" << static_cast<const void*>(top[i]->cpu_data())
                          << " bottom_ptr=" << static_cast<const void*>(bottom_data);
    }
  }

  auto t_share_end = std::chrono::high_resolution_clock::now();
  double share_ms = std::chrono::duration<double, std::milli>(
      t_share_end - t_share_start).count();

  CAFFE_FFI_LOG_WARN() << "[SPLIT-PERF] " << this->name()
                       << " Forward(N=" << num_top << " COW): count=" << count
                       << " shared_bytes=" << total_copy_bytes << "B"
                       << " share_time=" << share_ms << "ms"
                       << " all_shared=" << (all_shared ? "yes" : "no")
                       << " not_shared=" << not_shared_count
                       << " memcpy_saved=" << total_copy_bytes << "B (COW zero-copy)";
}

void SplitLayer::Backward_cpu(const std::vector<Blob*>& top,
                               const std::vector<bool>& propagate_down,
                               const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) {
    CAFFE_FFI_LAYER_LOG << "Split Backward_cpu: propagate_down[0]=false, skipping gradient accumulation";
    return;
  }

  int num_top = static_cast<int>(top.size());
  int64_t count = bottom[0]->count();
  size_t nbytes = static_cast<size_t>(count) * sizeof(float);

  CAFFE_FFI_LAYER_LOG << "Split Backward_cpu: count=" << count
                      << " num_top=" << num_top
                      << " nbytes=" << nbytes;

  // Get a writable pointer to bottom diff. COW triggers automatically if bottom's
  // diff is still shared with any top (non-COW'd borrowers), ensuring bottom gets
  // a private accumulation buffer that cannot alias with any top's diff pointer.
  float* bottom_diff = bottom[0]->cpu_mutable_diff();

  if (num_top == 1) {
    // N=1 zero-copy backward: copy the single top's diff to bottom.
    // After downstream ReLU/Conv backward calls cpu_mutable_diff() on top[0],
    // top[0] has COW'd to a private buffer with valid gradients. Copy them down.
    const float* top_diff = top[0]->cpu_diff();
    if (top_diff != bottom_diff) {
      caffe_copy_fp32(static_cast<size_t>(count), top_diff, bottom_diff);
    }
    CAFFE_FFI_LOG_WARN() << "[SPLIT-PERF] " << this->name()
                         << " Backward(N=1): count=" << count
                         << " memcpy_bytes=" << nbytes << "B"
                         << " top_ptr=" << static_cast<const void*>(top_diff)
                         << " bottom_ptr=" << static_cast<const void*>(bottom_diff);
    return;
  }

  // N≥2 gradient accumulation: initialize bottom diff with first top's gradients,
  // then axpy remaining tops' gradients into bottom. This implements the standard
  // split backward: d_bottom = sum_i(d_top_i), which is mathematically correct
  // because Split is the identity operation on each branch.
  auto t_acc_start = std::chrono::high_resolution_clock::now();

  const float* first_top_diff = top[0]->cpu_diff();
  if (first_top_diff != bottom_diff) {
    caffe_copy_fp32(static_cast<size_t>(count), first_top_diff, bottom_diff);
  } else {
    // First top still shares buffer with bottom: zero it to start fresh
    // (this case should be rare after the cpu_mutable_diff() COW above,
    // but guard against it for safety).
    caffe_set_fp32(static_cast<size_t>(count), 0.0f, bottom_diff);
  }

  for (int i = 1; i < num_top; ++i) {
    const float* top_diff = top[i]->cpu_diff();
    if (top_diff == bottom_diff) {
      // Skip self-referential accumulation (shouldn't happen after COW,
      // but guard against it to avoid 2x scaling).
      CAFFE_FFI_LAYER_LOG << "Split Backward_cpu: top[" << i
                          << "] diff aliases bottom diff, skipping";
      continue;
    }
    caffe_axpy_fp32(static_cast<size_t>(count), 1.0f, top_diff, bottom_diff);
  }

  auto t_acc_end = std::chrono::high_resolution_clock::now();
  double acc_ms = std::chrono::duration<double, std::milli>(
      t_acc_end - t_acc_start).count();

  CAFFE_FFI_LOG_WARN() << "[SPLIT-PERF] " << this->name()
                       << " Backward(N=" << num_top << " ACCUMULATE): count=" << count
                       << " num_tops_accumulated=" << num_top
                       << " accum_bytes=" << nbytes << "B"
                       << " accum_time=" << acc_ms << "ms"
                       << " bottom_ptr=" << static_cast<const void*>(bottom_diff);
}

REGISTER_LAYER_CLASS(Split);

}  // namespace caffe_ffi
