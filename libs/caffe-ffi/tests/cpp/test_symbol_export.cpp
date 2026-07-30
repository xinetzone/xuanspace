#include "test_harness.hpp"

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/common.hpp"

#include <atomic>
#include <cstdint>
#include <vector>

using namespace caffe_ffi;

// ──────────────────────────────────────────────────────────────
// Symbol Export tests: verify that g_total_allocated_bytes
// (a global data symbol in _caffe_ffi.dll) is properly exported
// and accessible from the test executable.
//
// WINDOWS_EXPORT_ALL_SYMBOLS only exports function symbols, not
// data symbols. The CAFFE_FFI_API macro (__declspec(dllexport))
// is required on MSVC to export global variables. These tests
// prevent regression of the LNK2019 linker error.
// ──────────────────────────────────────────────────────────────

// ── Direct access: read/write g_total_allocated_bytes ──

TEST(SymbolExport, ReadGlobalCounter) {
  // Verify the counter is accessible (no linker error = pass)
  int64_t val = g_total_allocated_bytes.load(std::memory_order_relaxed);
  // The counter tracks all live Blob allocations; it should be >= 0
  EXPECT_GE(val, 0);
}

TEST(SymbolExport, AtomicStoreAndLoad) {
  int64_t saved = g_total_allocated_bytes.load(std::memory_order_relaxed);

  g_total_allocated_bytes.store(42, std::memory_order_relaxed);
  EXPECT_EQ(g_total_allocated_bytes.load(std::memory_order_relaxed), 42);

  g_total_allocated_bytes.store(0, std::memory_order_relaxed);
  EXPECT_EQ(g_total_allocated_bytes.load(std::memory_order_relaxed), 0);

  // Restore
  g_total_allocated_bytes.store(saved, std::memory_order_relaxed);
}

TEST(SymbolExport, AtomicFetchAdd) {
  int64_t saved = g_total_allocated_bytes.load(std::memory_order_relaxed);

  g_total_allocated_bytes.store(100, std::memory_order_relaxed);
  int64_t prev = g_total_allocated_bytes.fetch_add(50, std::memory_order_relaxed);
  EXPECT_EQ(prev, 100);
  EXPECT_EQ(g_total_allocated_bytes.load(std::memory_order_relaxed), 150);

  // Restore
  g_total_allocated_bytes.store(saved, std::memory_order_relaxed);
}

TEST(SymbolExport, AtomicFetchSub) {
  int64_t saved = g_total_allocated_bytes.load(std::memory_order_relaxed);

  g_total_allocated_bytes.store(200, std::memory_order_relaxed);
  int64_t prev = g_total_allocated_bytes.fetch_sub(30, std::memory_order_relaxed);
  EXPECT_EQ(prev, 200);
  EXPECT_EQ(g_total_allocated_bytes.load(std::memory_order_relaxed), 170);

  // Restore
  g_total_allocated_bytes.store(saved, std::memory_order_relaxed);
}

// ── Cross-DLL boundary: Blob allocation (inside DLL) increments counter ──

TEST(SymbolExport, BlobAllocationIncrementsCounter) {
  int64_t before = g_total_allocated_bytes.load(std::memory_order_relaxed);

  // Allocate a Blob (inside _caffe_ffi.dll); CPUMemAlloc::AllocData
  // calls g_total_allocated_bytes.fetch_add from within the DLL.
  std::vector<int64_t> shape = {2, 3, 4};
  auto blob = make_object<Blob>(shape);
  // Force data allocation
  (void)blob->cpu_data();

  int64_t after = g_total_allocated_bytes.load(std::memory_order_relaxed);
  // The counter should have increased (at least data + diff tensors)
  EXPECT_GT(after, before);
}

TEST(SymbolExport, BlobDestructionDecrementsCounter) {
  int64_t before = g_total_allocated_bytes.load(std::memory_order_relaxed);

  {
    std::vector<int64_t> shape = {1, 2, 3};
    auto blob = make_object<Blob>(shape);
    (void)blob->cpu_data();
  }
  // After blob goes out of scope and is destroyed, the counter should
  // return to the original value (no leak).
  int64_t after = g_total_allocated_bytes.load(std::memory_order_relaxed);
  EXPECT_EQ(after, before);
}

TEST(SymbolExport, MultipleBlobsCounterConsistency) {
  int64_t before = g_total_allocated_bytes.load(std::memory_order_relaxed);

  std::vector<ObjectPtr<Blob>> blobs;
  std::vector<int64_t> shape = {4, 4};
  for (int i = 0; i < 5; ++i) {
    blobs.push_back(make_object<Blob>(shape));
    (void)blobs.back()->cpu_data();
  }

  int64_t during = g_total_allocated_bytes.load(std::memory_order_relaxed);
  EXPECT_GT(during, before);

  blobs.clear();

  int64_t after = g_total_allocated_bytes.load(std::memory_order_relaxed);
  EXPECT_EQ(after, before);
}

TEST(SymbolExport, CounterIsSameInstanceAcrossDllBoundary) {
  // Verify that g_total_allocated_bytes refers to the same atomic
  // instance in the DLL, not a local copy. We do this by:
  // 1. Setting a known value
  // 2. Allocating a Blob (which modifies the counter inside the DLL)
  // 3. Reading the counter from the test exe
  // If it's the same instance, the value will reflect the DLL's update.

  int64_t saved = g_total_allocated_bytes.load(std::memory_order_relaxed);
  g_total_allocated_bytes.store(1000, std::memory_order_relaxed);

  // Allocate a Blob (DLL adds allocation size to counter)
  std::vector<int64_t> shape = {1, 1};
  auto blob = make_object<Blob>(shape);
  (void)blob->cpu_data();

  // The counter should be > 1000 (DLL added allocation bytes)
  int64_t after_alloc = g_total_allocated_bytes.load(std::memory_order_relaxed);
  EXPECT_GT(after_alloc, 1000);

  // Restore
  g_total_allocated_bytes.store(saved, std::memory_order_relaxed);
}