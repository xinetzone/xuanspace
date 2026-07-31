#include "caffe_ffi/layers/split_layer.hpp"

#include <chrono>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "caffe_ffi/layer_factory.hpp"
#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

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

  // Note: For N=1 zero-copy path, we still reshape top[0] to the correct shape
  // here so downstream layers see valid shapes during their own Reshape().
  // The zero-copy ShareData() in Forward() will replace the tensor reference,
  // freeing the Reshape-allocated buffer (one alloc+free overhead but avoids
  // breaking the layer setup contract).
  for (int i = 0; i < num_top; ++i) {
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
                       << " zerocopy_n1=" << ((num_top == 1) ? "yes" : "no");
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

    CAFFE_FFI_LAYER_LOG << "Split Forward(N=" << num_top << " COW): top[" << i
                        << "] data_shared=" << (data_now_shared ? "yes" : "no")
                        << " diff_shared=" << (diff_now_shared ? "yes" : "no")
                        << " was_data_shared=" << (data_was_shared ? "yes" : "no")
                        << " was_diff_shared=" << (diff_was_shared ? "yes" : "no")
                        << " data_ptr=" << static_cast<const void*>(top[i]->cpu_data())
                        << " bottom_ptr=" << static_cast<const void*>(bottom_data);
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

REGISTER_LAYER_CLASS(Split);

}  // namespace caffe_ffi
