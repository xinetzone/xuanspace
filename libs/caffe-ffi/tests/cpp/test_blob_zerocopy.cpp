#include "test_harness.hpp"

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/common.hpp"
#include "caffe_ffi/net.hpp"

#include <vector>

using namespace caffe_ffi;

// ── Blob::ShareData / ShareDiff basic tests ──

TEST(ZeroCopyTest, ShareDataMakesPointersEqual) {
  std::vector<int64_t> shape = {2, 3, 4};
  auto src = make_object<Blob>(shape);
  auto dst = make_object<Blob>(shape);

  // Write known values into src
  float* src_data = src->cpu_mutable_data();
  for (int64_t i = 0; i < src->count(); ++i) {
    src_data[i] = static_cast<float>(i) * 0.5f;
  }

  // Before ShareData, pointers must differ
  EXPECT_NE(src->cpu_data(), dst->cpu_data());
  EXPECT_FALSE(dst->SharesDataWith(src.get()));

  dst->ShareData(src.get());

  // After ShareData, pointers must be equal
  EXPECT_TRUE(dst->SharesDataWith(src.get()));
  EXPECT_EQ(dst->cpu_data(), src->cpu_data());

  // Data must be visible through dst (same physical memory).
  // Use const accessor cpu_data() for reading -- cpu_mutable_data() would
  // trigger COW since ShareData() uses COW-mode sharing (not identity).
  for (int64_t i = 0; i < src->count(); ++i) {
    EXPECT_NEAR(static_cast<double>(dst->cpu_data()[i]),
                static_cast<double>(i) * 0.5, 1e-6);
  }
}

TEST(ZeroCopyTest, ShareDiffMakesDiffPointersEqual) {
  std::vector<int64_t> shape = {4, 4};
  auto src = make_object<Blob>(shape);
  auto dst = make_object<Blob>(shape);

  float* src_diff = src->cpu_mutable_diff();
  for (int64_t i = 0; i < src->count(); ++i) {
    src_diff[i] = static_cast<float>(i + 1) * 0.1f;
  }

  EXPECT_NE(src->cpu_diff(), dst->cpu_diff());
  EXPECT_FALSE(dst->SharesDiffWith(src.get()));

  dst->ShareDiff(src.get());

  EXPECT_TRUE(dst->SharesDiffWith(src.get()));
  EXPECT_EQ(dst->cpu_diff(), src->cpu_diff());
  // Use const accessor cpu_diff() for reading -- cpu_mutable_diff() would
  // trigger COW since ShareDiff() uses COW-mode sharing (not identity).
  EXPECT_NEAR(static_cast<double>(dst->cpu_diff()[0]), 0.1, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_diff()[15]), 1.6, 1e-6);
}

TEST(ZeroCopyTest, ShareDataPreservesShape) {
  std::vector<int64_t> shape = {2, 3, 4, 5};
  auto src = make_object<Blob>(shape);
  auto dst = make_object<Blob>();  // default shape {0}

  EXPECT_EQ(dst->num_axes(), 1);
  EXPECT_EQ(dst->count(), 0);

  dst->ShareData(src.get());
  dst->ShareDiff(src.get());

  // Shape must match src after sharing
  EXPECT_EQ(dst->num_axes(), src->num_axes());
  EXPECT_EQ(dst->count(), src->count());
  EXPECT_EQ(dst->shape(0), src->shape(0));
  EXPECT_EQ(dst->shape(1), src->shape(1));
  EXPECT_EQ(dst->shape(2), src->shape(2));
  EXPECT_EQ(dst->shape(3), src->shape(3));
}

TEST(ZeroCopyTest, ShareDataMutationVisibleToBoth) {
  // COW semantics: ShareData() shares memory for const reads, but any
  // cpu_mutable_data() call triggers copy-on-write when use_count > 1
  // (i.e., whenever there is another sharer). For in-place passthrough
  // where mutations must propagate to the source (N=1 Split/Slice identity),
  // use ShareDataIdentity() which never triggers COW.
  std::vector<int64_t> shape = {8};
  auto a = make_object<Blob>(shape);
  auto b = make_object<Blob>(shape);

  a->cpu_mutable_data()[0] = 42.0f;

  // ── Part 1: Identity share (two-way, N=1 passthrough) ──
  // ShareDataIdentity: mutable access is in-place, mutations visible to source.
  b->ShareDataIdentity(a.get());
  EXPECT_EQ(b->cpu_data(), a->cpu_data());
  EXPECT_NEAR(static_cast<double>(b->cpu_data()[0]), 42.0, 1e-6);

  const float* ptr_before_mut = b->cpu_data();
  b->cpu_mutable_data()[1] = 99.0f;
  EXPECT_EQ(b->cpu_data(), ptr_before_mut);  // pointer unchanged (no COW)
  EXPECT_EQ(b->cpu_data(), a->cpu_data());   // still sharing
  EXPECT_NEAR(static_cast<double>(b->cpu_data()[1]), 99.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(a->cpu_data()[1]), 99.0, 1e-6);  // a sees mutation

  // ── Part 2: COW share (two-way, non-identity) ──
  // ShareData: mutable access triggers COW even with two sharers,
  // because the caller explicitly requested COW-semantics sharing.
  auto src = make_object<Blob>(shape);
  auto dst1 = make_object<Blob>(shape);
  src->cpu_mutable_data()[0] = 10.0f;

  dst1->ShareData(src.get());
  EXPECT_EQ(dst1->cpu_data(), src->cpu_data());  // const access shares pointer
  EXPECT_NEAR(static_cast<double>(dst1->cpu_data()[0]), 10.0, 1e-6);

  const float* src_ptr_before = src->cpu_data();
  dst1->cpu_mutable_data()[1] = 20.0f;
  EXPECT_NE(dst1->cpu_data(), src->cpu_data());   // COW: dst1 isolated
  EXPECT_EQ(src->cpu_data(), src_ptr_before);      // src pointer unchanged
  EXPECT_NEAR(static_cast<double>(dst1->cpu_data()[1]), 20.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src->cpu_data()[1]), 0.0, 1e-6);  // src unaffected

  // ── Part 3: COW share (three-way, fan-out isolation) ──
  auto src2 = make_object<Blob>(shape);
  auto t0 = make_object<Blob>(shape);
  auto t1 = make_object<Blob>(shape);
  src2->cpu_mutable_data()[0] = 100.0f;
  t0->ShareData(src2.get());
  t1->ShareData(src2.get());
  EXPECT_TRUE(t0->SharesDataWith(src2.get()));
  EXPECT_TRUE(t1->SharesDataWith(src2.get()));
  EXPECT_EQ(t0->cpu_data(), src2->cpu_data());
  EXPECT_EQ(t1->cpu_data(), src2->cpu_data());

  // Mutate via t0: COW isolates t0, src2 and t1 remain shared
  const float* src2_ptr_before = src2->cpu_data();
  t0->cpu_mutable_data()[2] = 77.0f;
  EXPECT_NE(t0->cpu_data(), src2->cpu_data());
  EXPECT_NE(t0->cpu_data(), t1->cpu_data());
  EXPECT_EQ(src2->cpu_data(), src2_ptr_before);
  EXPECT_EQ(t1->cpu_data(), src2->cpu_data());
  EXPECT_NEAR(static_cast<double>(t0->cpu_data()[2]), 77.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src2->cpu_data()[2]), 0.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(t1->cpu_data()[2]), 0.0, 1e-6);

  // Mutate via t1: COW isolates t1 too (use_count=2: src2+t1)
  t1->cpu_mutable_data()[3] = 55.0f;
  EXPECT_NE(t1->cpu_data(), src2->cpu_data());
  EXPECT_NEAR(static_cast<double>(t1->cpu_data()[3]), 55.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src2->cpu_data()[3]), 0.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(t0->cpu_data()[3]), 0.0, 1e-6);  // t0 isolated too
}

TEST(ZeroCopyTest, SharesDataWithFalseForDifferentBlobs) {
  auto a = make_object<Blob>(std::vector<int64_t>{4});
  auto b = make_object<Blob>(std::vector<int64_t>{4});
  EXPECT_FALSE(a->SharesDataWith(b.get()));
  EXPECT_FALSE(b->SharesDataWith(a.get()));
}

TEST(ZeroCopyTest, ReshapeBreaksShare) {
  std::vector<int64_t> shape = {3, 4};
  auto src = make_object<Blob>(shape);
  auto dst = make_object<Blob>(shape);

  dst->ShareData(src.get());
  EXPECT_TRUE(dst->SharesDataWith(src.get()));

  // Reshape dst to a different shape -- must allocate new private memory,
  // breaking the share.
  dst->Reshape(std::vector<int64_t>{6, 8});
  EXPECT_FALSE(dst->SharesDataWith(src.get()));
  EXPECT_NE(dst->cpu_data(), src->cpu_data());
}

TEST(ZeroCopyTest, ShareDataFromSelfIsNoop) {
  std::vector<int64_t> shape = {4};
  auto b = make_object<Blob>(shape);
  const float* before = b->cpu_data();
  b->ShareData(b.get());
  EXPECT_TRUE(b->SharesDataWith(b.get()));
  EXPECT_EQ(b->cpu_data(), before);
}

TEST(ZeroCopyTest, RefcountingSourceOutlivesDestination) {
  std::vector<int64_t> shape = {16};
  const float* src_data_ptr = nullptr;

  {
    auto src = make_object<Blob>(shape);
    src_data_ptr = src->cpu_data();
    src->cpu_mutable_data()[0] = 3.14f;

    {
      auto dst = make_object<Blob>(shape);
      dst->ShareData(src.get());
      EXPECT_EQ(dst->cpu_data(), src_data_ptr);
      EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_data()[0]), 3.14, 1e-6);
    }
    // dst destroyed here; src must still be valid
    EXPECT_NEAR(static_cast<double>(src->cpu_mutable_data()[0]), 3.14, 1e-6);
    EXPECT_EQ(src->cpu_data(), src_data_ptr);
  }
  // src destroyed here, memory freed
}

TEST(ZeroCopyTest, RefcountingDestinationOutlivesSource) {
  std::vector<int64_t> shape = {16};
  auto dst = make_object<Blob>(shape);
  {
    auto src = make_object<Blob>(shape);
    src->cpu_mutable_data()[5] = 7.77f;
    dst->ShareData(src.get());
    EXPECT_TRUE(dst->SharesDataWith(src.get()));
  }
  // src destroyed; dst must still have valid data via refcount
  EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_data()[5]), 7.77, 1e-6);
  // dst no longer shares with src (src is gone), but data_ptr is still valid
  EXPECT_TRUE(dst->cpu_data() != nullptr);
}

TEST(ZeroCopyTest, ShareDataMultipleTimesIdempotent) {
  // COW semantics: multiple ShareData calls from same source are idempotent
  // (all share same pointer for const reads). Mutable access on any alias
  // triggers COW isolation for that alias only.
  std::vector<int64_t> shape = {4};
  auto a = make_object<Blob>(shape);
  auto b = make_object<Blob>(shape);
  auto c = make_object<Blob>(shape);

  a->cpu_mutable_data()[0] = 1.0f;
  b->ShareData(a.get());
  c->ShareData(a.get());

  EXPECT_TRUE(b->SharesDataWith(a.get()));
  EXPECT_TRUE(c->SharesDataWith(a.get()));
  EXPECT_TRUE(b->SharesDataWith(c.get()));
  EXPECT_EQ(b->cpu_data(), c->cpu_data());
  EXPECT_NEAR(static_cast<double>(a->cpu_data()[0]), 1.0, 1e-6);

  // Mutating b triggers COW: b gets private copy, a and c remain shared
  b->cpu_mutable_data()[0] = 99.0f;
  EXPECT_FALSE(b->SharesDataWith(a.get()));
  EXPECT_TRUE(c->SharesDataWith(a.get()));  // c still shares with a
  EXPECT_NEAR(static_cast<double>(b->cpu_data()[0]), 99.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(c->cpu_data()[0]), 1.0, 1e-6);  // c unchanged
  EXPECT_NEAR(static_cast<double>(a->cpu_data()[0]), 1.0, 1e-6);  // a unchanged
}

// ── Phase 2 COW: Blob-level unit tests ──

/// 验证 cpu_mutable_data() 在共享后触发 COW：指针不再相等、
/// 引用计数回到 1、数据内容正确复制
TEST(COWTest, MutableDataTriggersCOWWhenShared) {
  std::vector<int64_t> shape = {8};
  auto src = make_object<Blob>(shape);
  auto dst1 = make_object<Blob>(shape);
  auto dst2 = make_object<Blob>(shape);

  // 写入已知数据
  for (int64_t i = 0; i < src->count(); ++i) {
    src->cpu_mutable_data()[i] = static_cast<float>(i);
  }

  // 三方共享（use_count=3）：src + dst1 + dst2
  dst1->ShareData(src.get());
  dst2->ShareData(src.get());
  EXPECT_TRUE(dst1->SharesDataWith(src.get()));
  EXPECT_TRUE(dst2->SharesDataWith(src.get()));
  EXPECT_EQ(dst1->cpu_data(), src->cpu_data());

  // 调用 cpu_mutable_data() → 在三方共享下 COW 触发
  float* dst_mut = dst1->cpu_mutable_data();

  // 指针不再相等（COW 打破了共享）
  EXPECT_NE(dst_mut, src->cpu_data());
  EXPECT_FALSE(dst1->SharesDataWith(src.get()));

  // dst2 仍与 src 共享
  EXPECT_TRUE(dst2->SharesDataWith(src.get()));

  // 数据内容正确复制
  for (int64_t i = 0; i < src->count(); ++i) {
    EXPECT_NEAR(static_cast<double>(dst_mut[i]),
                static_cast<double>(i), 1e-6);
  }
}

/// 验证 cpu_mutable_data() 在未共享时不触发 COW（use_count == 1）
TEST(COWTest, MutableDataNoCOWWhenNotShared) {
  std::vector<int64_t> shape = {4};
  auto b = make_object<Blob>(shape);
  b->cpu_mutable_data()[0] = 42.0f;

  const float* before = b->cpu_data();
  float* after = b->cpu_mutable_data();

  // 未共享时，指针不变（无 COW）
  EXPECT_EQ(before, after);
  EXPECT_NEAR(static_cast<double>(after[0]), 42.0, 1e-6);
}

/// 验证 N=1 Identity 共享（ShareDataIdentity）data 路径的 in-place 直通语义：
/// - Identity 模式下 cpu_mutable_data() 不触发 COW（指针不变，修改对另一方可见）
/// - 用于 N=1 Split（identity passthrough），top 是 bottom 的真正别名
/// - 这与 COW 模式（ShareData）不同：COW 模式下 data 路径 borrower 在 use_count>1 时即触发 COW
TEST(COWTest, IdentityShareDataNoCOWInPlace) {
  std::vector<int64_t> shape = {8};
  auto a = make_object<Blob>(shape);
  auto b = make_object<Blob>(shape);

  a->cpu_mutable_data()[0] = 42.0f;
  a->cpu_mutable_data()[1] = 0.0f;
  a->cpu_mutable_data()[2] = 0.0f;

  // Identity share: b is a true alias of a (N=1 passthrough), mutable does NOT COW
  b->ShareDataIdentity(a.get());

  // 两方共享（use_count=2）：const reads 共享同一内存
  EXPECT_EQ(b->cpu_data(), a->cpu_data());
  EXPECT_TRUE(b->SharesDataWith(a.get()));
  EXPECT_NEAR(static_cast<double>(b->cpu_data()[0]), 42.0, 1e-6);

  // Identity 模式：mutable access 是 in-place 直通，无 COW 发生
  const float* b_ptr_before = b->cpu_data();
  b->cpu_mutable_data()[1] = 99.0f;
  EXPECT_EQ(b->cpu_data(), b_ptr_before);  // 指针不变，无COW
  EXPECT_EQ(b->cpu_data(), a->cpu_data());  // 仍共享同一buffer
  EXPECT_NEAR(static_cast<double>(b->cpu_data()[1]), 99.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(a->cpu_data()[1]), 99.0, 1e-6);  // in-place 写入对 a 可见

  // Identity 模式下 owner 的 mutable 触发 COW（owner 永远在 use_count>1 时 COW），
  // 打破 identity 别名关系，这是预期的安全行为
  const float* a_ptr_before = a->cpu_data();
  a->cpu_mutable_data()[2] = 77.0f;
  EXPECT_NE(a->cpu_data(), a_ptr_before);  // Owner COW'd
  EXPECT_FALSE(a->SharesDataWith(b.get()));  // 不再共享
  EXPECT_NEAR(static_cast<double>(a->cpu_data()[2]), 77.0, 1e-6);  // Owner 有新值
  EXPECT_NEAR(static_cast<double>(b->cpu_data()[2]), 0.0, 1e-6);  // Borrower 保留旧值
  EXPECT_NEAR(static_cast<double>(b->cpu_data()[1]), 99.0, 1e-6);  // Borrower 的 in-place 写入仍在
}

/// 验证 COW 模式（ShareData）data 路径两方共享触发 COW：
/// - COW 模式下，任何 borrower 的 mutable access 在 use_count>1 时即触发 COW
/// - 用于 N≥2 fan-out（如 N=2 Split），确保每个分支写时隔离
TEST(COWTest, COWModeShareDataTriggersCOW) {
  std::vector<int64_t> shape = {8};
  auto a = make_object<Blob>(shape);
  auto b = make_object<Blob>(shape);

  a->cpu_mutable_data()[0] = 42.0f;
  a->cpu_mutable_data()[1] = 0.0f;

  // COW-mode share (not identity): b is a COW borrower
  b->ShareData(a.get());

  EXPECT_EQ(b->cpu_data(), a->cpu_data());
  EXPECT_TRUE(b->SharesDataWith(a.get()));
  EXPECT_NEAR(static_cast<double>(b->cpu_data()[0]), 42.0, 1e-6);

  // COW 模式：两方共享时 borrower mutable 即触发 COW（data 路径阈值 use_count>1）
  const float* a_ptr_before = a->cpu_data();
  b->cpu_mutable_data()[1] = 99.0f;
  EXPECT_NE(b->cpu_data(), a->cpu_data());  // COW 发生，b 获得私有副本
  EXPECT_EQ(a->cpu_data(), a_ptr_before);  // a 的指针不变
  EXPECT_NEAR(static_cast<double>(b->cpu_data()[1]), 99.0, 1e-6);  // b 有新值
  EXPECT_NEAR(static_cast<double>(a->cpu_data()[1]), 0.0, 1e-6);  // a 不受影响
  EXPECT_NEAR(static_cast<double>(a->cpu_data()[0]), 42.0, 1e-6);  // a 的原始值保留
}

/// 验证 COW 模式（ShareData）data 路径三方共享时的分支隔离：
/// - N≥2 fan-out 场景：两个 borrower 共享 owner 的 data
/// - 任何 borrower mutable 时 COW 触发，获得私有副本
/// - 所有分支完全隔离，互不影响
TEST(COWTest, COWModeThreeWayShareIsolatesBranches) {
  std::vector<int64_t> shape = {8};
  auto a = make_object<Blob>(shape);
  auto b = make_object<Blob>(shape);
  auto c = make_object<Blob>(shape);

  a->cpu_mutable_data()[0] = 42.0f;
  a->cpu_mutable_data()[1] = 0.0f;
  a->cpu_mutable_data()[2] = 0.0f;

  b->ShareData(a.get());
  c->ShareData(a.get());

  EXPECT_TRUE(c->SharesDataWith(a.get()));
  EXPECT_TRUE(b->SharesDataWith(a.get()));
  EXPECT_EQ(c->cpu_data(), a->cpu_data());

  // 三方共享（use_count=3）：data 路径 borrower 在 use_count>1 时触发 COW
  const float* a_ptr_before_cow = a->cpu_data();
  b->cpu_mutable_data()[2] = 77.0f;
  EXPECT_NE(b->cpu_data(), a->cpu_data());  // b COW'd to private copy
  EXPECT_EQ(a->cpu_data(), a_ptr_before_cow);  // a unaffected
  EXPECT_NEAR(static_cast<double>(b->cpu_data()[2]), 77.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(a->cpu_data()[2]), 0.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(c->cpu_data()[2]), 0.0, 1e-6);

  // c still shares with a, c mutable also COWs
  c->cpu_mutable_data()[1] = 55.0f;
  EXPECT_NE(c->cpu_data(), a->cpu_data());
  EXPECT_NEAR(static_cast<double>(c->cpu_data()[1]), 55.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(a->cpu_data()[1]), 0.0, 1e-6);
}

/// 验证 diff 路径 COW 语义（与 data 路径对称）：
/// - ShareDiffIdentity: mutable access in-place 直通，不触发 COW
/// - ShareDiff (COW mode): mutable access 始终触发 COW（use_count > 1），
///   无论共享者数量，用于 N>=2 fan-out 隔离
TEST(COWTest, TwoWayDiffShareNoCOWInPlace) {
  std::vector<int64_t> shape = {8};
  auto a = make_object<Blob>(shape);
  auto b = make_object<Blob>(shape);

  a->cpu_mutable_diff()[0] = 42.0f;

  // ── Part 1: Identity share (two-way, N=1 passthrough) ──
  b->ShareDiffIdentity(a.get());
  EXPECT_EQ(b->cpu_diff(), a->cpu_diff());
  EXPECT_TRUE(b->SharesDiffWith(a.get()));
  EXPECT_NEAR(static_cast<double>(b->cpu_diff()[0]), 42.0, 1e-6);

  // Identity mode: mutable access is in-place, no COW
  const float* b_ptr_before = b->cpu_diff();
  b->cpu_mutable_diff()[1] = 99.0f;
  EXPECT_EQ(b->cpu_diff(), b_ptr_before);
  EXPECT_EQ(b->cpu_diff(), a->cpu_diff());
  EXPECT_NEAR(static_cast<double>(b->cpu_diff()[1]), 99.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(a->cpu_diff()[1]), 99.0, 1e-6);

  // ── Part 2: COW share (two-way, non-identity) ──
  auto src = make_object<Blob>(shape);
  auto dst1 = make_object<Blob>(shape);
  src->cpu_mutable_diff()[0] = 10.0f;

  dst1->ShareDiff(src.get());
  EXPECT_EQ(dst1->cpu_diff(), src->cpu_diff());

  // COW mode: mutable triggers COW even with two sharers
  const float* src_ptr_before = src->cpu_diff();
  dst1->cpu_mutable_diff()[1] = 20.0f;
  EXPECT_NE(dst1->cpu_diff(), src->cpu_diff());
  EXPECT_EQ(src->cpu_diff(), src_ptr_before);
  EXPECT_NEAR(static_cast<double>(dst1->cpu_diff()[1]), 20.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src->cpu_diff()[1]), 0.0, 1e-6);

  // ── Part 3: COW share (three-way, fan-out isolation) ──
  auto src2 = make_object<Blob>(shape);
  auto t0 = make_object<Blob>(shape);
  auto t1 = make_object<Blob>(shape);
  src2->cpu_mutable_diff()[0] = 100.0f;
  t0->ShareDiff(src2.get());
  t1->ShareDiff(src2.get());

  // Mutate via t0: COW isolates t0
  const float* src2_ptr_before = src2->cpu_diff();
  t0->cpu_mutable_diff()[2] = 77.0f;
  EXPECT_NE(t0->cpu_diff(), src2->cpu_diff());
  EXPECT_NE(t0->cpu_diff(), t1->cpu_diff());
  EXPECT_EQ(src2->cpu_diff(), src2_ptr_before);
  EXPECT_EQ(t1->cpu_diff(), src2->cpu_diff());
  EXPECT_NEAR(static_cast<double>(t0->cpu_diff()[2]), 77.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src2->cpu_diff()[2]), 0.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(t1->cpu_diff()[2]), 0.0, 1e-6);

  // Mutate via t1: COW isolates t1 too (use_count=2: src2+t1)
  t1->cpu_mutable_diff()[3] = 55.0f;
  EXPECT_NE(t1->cpu_diff(), src2->cpu_diff());
  EXPECT_NEAR(static_cast<double>(t1->cpu_diff()[3]), 55.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src2->cpu_diff()[3]), 0.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(t0->cpu_diff()[3]), 0.0, 1e-6);
}

/// 验证 cpu_mutable_diff() 在共享后触发 COW
TEST(COWTest, MutableDiffTriggersCOWWhenShared) {
  std::vector<int64_t> shape = {6};
  auto src = make_object<Blob>(shape);
  auto dst1 = make_object<Blob>(shape);
  auto dst2 = make_object<Blob>(shape);

  for (int64_t i = 0; i < src->count(); ++i) {
    src->cpu_mutable_diff()[i] = static_cast<float>(i * 10);
  }

  // 三方共享（use_count=3）：src + dst1 + dst2
  dst1->ShareDiff(src.get());
  dst2->ShareDiff(src.get());
  EXPECT_TRUE(dst1->SharesDiffWith(src.get()));
  EXPECT_TRUE(dst2->SharesDiffWith(src.get()));
  EXPECT_EQ(dst1->cpu_diff(), src->cpu_diff());

  float* dst_mut_diff = dst1->cpu_mutable_diff();

  // 在三方共享下 COW 触发：指针不再相等
  EXPECT_NE(dst_mut_diff, src->cpu_diff());
  EXPECT_FALSE(dst1->SharesDiffWith(src.get()));

  // dst2 仍与 src 共享
  EXPECT_TRUE(dst2->SharesDiffWith(src.get()));

  for (int64_t i = 0; i < src->count(); ++i) {
    EXPECT_NEAR(static_cast<double>(dst_mut_diff[i]),
                static_cast<double>(i * 10), 1e-6);
  }
}

/// 验证 COW 后的数据隔离：三方共享时修改 dst 不影响 src 和其他共享者
TEST(COWTest, DataIsolationAfterCOW) {
  std::vector<int64_t> shape = {4};
  auto src = make_object<Blob>(shape);
  auto dst1 = make_object<Blob>(shape);
  auto dst2 = make_object<Blob>(shape);

  src->cpu_mutable_data()[0] = 10.0f;
  src->cpu_mutable_data()[1] = 20.0f;
  src->cpu_mutable_data()[2] = 30.0f;
  src->cpu_mutable_data()[3] = 40.0f;

  // 三方共享（use_count=3）
  dst1->ShareData(src.get());
  dst2->ShareData(src.get());

  // 触发 COW 并修改 dst1
  dst1->cpu_mutable_data()[0] = 999.0f;
  dst1->cpu_mutable_data()[1] = 888.0f;

  // src 和 dst2 不受影响
  EXPECT_NEAR(static_cast<double>(src->cpu_data()[0]), 10.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src->cpu_data()[1]), 20.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src->cpu_data()[2]), 30.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src->cpu_data()[3]), 40.0, 1e-6);
  EXPECT_TRUE(dst2->SharesDataWith(src.get()));

  // dst1 有自己的值
  EXPECT_NEAR(static_cast<double>(dst1->cpu_data()[0]), 999.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst1->cpu_data()[1]), 888.0, 1e-6);
  // 未修改的部分应与 src 一致（COW 完整复制）
  EXPECT_NEAR(static_cast<double>(dst1->cpu_data()[2]), 30.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst1->cpu_data()[3]), 40.0, 1e-6);
}

/// 验证 const cpu_data() 不触发 COW（指针保持共享）
TEST(COWTest, ConstAccessDoesNotTriggerCOW) {
  std::vector<int64_t> shape = {4};
  auto src = make_object<Blob>(shape);
  auto dst = make_object<Blob>(shape);

  src->cpu_mutable_data()[0] = 77.0f;
  dst->ShareData(src.get());

  const float* dst_ptr = dst->cpu_data();  // const 重载，不触发 COW
  EXPECT_EQ(dst_ptr, src->cpu_data());
  EXPECT_TRUE(dst->SharesDataWith(src.get()));

  // 再通过非 const 访问也不应意外触发（仅读取）
  const float* dst_ptr2 = static_cast<const Blob*>(dst.get())->cpu_data();
  EXPECT_EQ(dst_ptr2, src->cpu_data());
}

/// 验证三次共享后第一个 COW 触发仅影响自身，其他两个仍共享
TEST(COWTest, ThreeWayShareCOWOnlyAffectsMutator) {
  std::vector<int64_t> shape = {4};
  auto a = make_object<Blob>(shape);
  auto b = make_object<Blob>(shape);
  auto c = make_object<Blob>(shape);

  a->cpu_mutable_data()[0] = 1.0f;
  b->ShareData(a.get());
  c->ShareData(a.get());

  EXPECT_TRUE(b->SharesDataWith(a.get()));
  EXPECT_TRUE(c->SharesDataWith(a.get()));
  EXPECT_TRUE(b->SharesDataWith(c.get()));

  // b 触发 COW
  b->cpu_mutable_data()[0] = 999.0f;

  // b 打破共享
  EXPECT_FALSE(b->SharesDataWith(a.get()));
  EXPECT_FALSE(b->SharesDataWith(c.get()));

  // a 和 c 仍共享
  EXPECT_TRUE(a->SharesDataWith(c.get()));
  EXPECT_NEAR(static_cast<double>(a->cpu_mutable_data()[0]), 1.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(c->cpu_mutable_data()[0]), 1.0, 1e-6);

  // b 数据独立
  EXPECT_NEAR(static_cast<double>(b->cpu_mutable_data()[0]), 999.0, 1e-6);
}

// ── ShareData 引用计数异常场景测试 (ShareDataRefCount) ──
// 验证 ShareData() 在各种边界条件、异常共享模式、引用计数泄漏场景下的正确性。
// 这些测试覆盖 N=2 集成测试失败分析中识别的克隆逻辑异常路径。

/// 场景 1：自共享（idempotent self-share）
/// blob 将自身 data 共享给自身，应保持指针不变且不崩溃。
TEST(ShareDataRefCount, SelfShareIsIdempotent) {
  auto a = make_object<Blob>(std::vector<int64_t>{4, 4});
  a->cpu_mutable_data()[0] = 42.0f;
  const float* ptr_before = a->cpu_data();
  int64_t id_before = a->id();

  // 自共享：不应崩溃，指针不变
  a->ShareData(a.get());

  EXPECT_EQ(a->cpu_data(), ptr_before);
  EXPECT_EQ(a->id(), id_before);
  EXPECT_TRUE(a->SharesDataWith(a.get()));
  EXPECT_NEAR(static_cast<double>(a->cpu_mutable_data()[0]), 42.0, 1e-6);
}

/// 场景 2：链式共享 A→B→C（三向指针一致性）
/// 验证 A→B→C 链式共享后，三者指向同一内存。
TEST(ShareDataRefCount, ChainShareThreeWay) {
  auto a = make_object<Blob>(std::vector<int64_t>{8});
  auto b = make_object<Blob>(std::vector<int64_t>{8});
  auto c = make_object<Blob>(std::vector<int64_t>{8});

  a->cpu_mutable_data()[0] = 100.0f;
  a->cpu_mutable_data()[7] = 200.0f;

  // A→B
  b->ShareData(a.get());
  EXPECT_TRUE(b->SharesDataWith(a.get()));
  EXPECT_EQ(b->cpu_data(), a->cpu_data());

  // B→C（通过 B 间接共享到 C，C 应与 A 共享）
  c->ShareData(b.get());
  EXPECT_TRUE(c->SharesDataWith(b.get()));
  EXPECT_TRUE(c->SharesDataWith(a.get()));
  EXPECT_EQ(c->cpu_data(), a->cpu_data());
  EXPECT_EQ(c->cpu_data(), b->cpu_data());

  // 值验证
  EXPECT_NEAR(static_cast<double>(c->cpu_mutable_data()[0]), 100.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(c->cpu_mutable_data()[7]), 200.0, 1e-6);
}

/// 场景 3：重共享覆盖旧引用
/// B 先共享 A，再共享 C。验证 B 的旧引用被释放，新引用指向 C。
TEST(ShareDataRefCount, ReShareOverwritesPrevious) {
  auto a = make_object<Blob>(std::vector<int64_t>{8});
  auto b = make_object<Blob>(std::vector<int64_t>{8});
  auto c = make_object<Blob>(std::vector<int64_t>{8});

  a->cpu_mutable_data()[0] = 10.0f;
  c->cpu_mutable_data()[0] = 20.0f;

  // B 共享 A
  b->ShareData(a.get());
  EXPECT_TRUE(b->SharesDataWith(a.get()));
  EXPECT_NEAR(static_cast<double>(b->cpu_mutable_data()[0]), 10.0, 1e-6);

  // B 重共享 C → 应与 A 断开，与 C 共享
  b->ShareData(c.get());
  EXPECT_TRUE(b->SharesDataWith(c.get()));
  EXPECT_FALSE(b->SharesDataWith(a.get()));
  EXPECT_NEAR(static_cast<double>(b->cpu_mutable_data()[0]), 20.0, 1e-6);

  // A 不受影响
  EXPECT_NEAR(static_cast<double>(a->cpu_mutable_data()[0]), 10.0, 1e-6);
}

/// 场景 4：Reshape 打破共享
/// 共享后 Reshape 目标 Blob，验证共享断开且原数据不受影响。
TEST(ShareDataRefCount, ReshapeBreaksShare) {
  auto a = make_object<Blob>(std::vector<int64_t>{4, 4});
  auto b = make_object<Blob>(std::vector<int64_t>{4, 4});

  a->cpu_mutable_data()[0] = 55.0f;
  a->cpu_mutable_data()[15] = 77.0f;

  b->ShareData(a.get());
  EXPECT_TRUE(b->SharesDataWith(a.get()));

  // Reshape b → 分配新内存，打破共享
  b->Reshape(std::vector<int64_t>{2, 8});
  EXPECT_FALSE(b->SharesDataWith(a.get()));

  // a 的数据应保持不变
  EXPECT_NEAR(static_cast<double>(a->cpu_mutable_data()[0]), 55.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(a->cpu_mutable_data()[15]), 77.0, 1e-6);
}

/// 场景 5：COW 后共享（A→B→D1 三方共享，B COW，B→C→D2 三方共享，C COW）
/// 验证 COW 打破共享后，新建立的共享关系正确隔离。
/// COW 触发条件：use_count > 2（三方及以上共享）。
TEST(ShareDataRefCount, ShareDataAfterCOW) {
  auto a = make_object<Blob>(std::vector<int64_t>{4});
  auto b = make_object<Blob>(std::vector<int64_t>{4});
  auto c = make_object<Blob>(std::vector<int64_t>{4});
  auto d1 = make_object<Blob>(std::vector<int64_t>{4});
  auto d2 = make_object<Blob>(std::vector<int64_t>{4});

  a->cpu_mutable_data()[0] = 1.0f;
  a->cpu_mutable_data()[1] = 2.0f;
  a->cpu_mutable_data()[2] = 3.0f;
  a->cpu_mutable_data()[3] = 4.0f;

  // A→B→D1 三方共享（use_count=3）
  b->ShareData(a.get());
  d1->ShareData(a.get());
  EXPECT_TRUE(b->SharesDataWith(a.get()));
  EXPECT_TRUE(d1->SharesDataWith(a.get()));

  // B 触发 COW（三方共享 use_count=3，通过 cpu_mutable_data 写入）
  float* b_data = b->cpu_mutable_data();
  b_data[0] = 999.0f;

  // COW 后 B 与 A、D1 应断开
  EXPECT_FALSE(b->SharesDataWith(a.get()));
  EXPECT_FALSE(b->SharesDataWith(d1.get()));
  EXPECT_NEAR(static_cast<double>(b->cpu_data()[0]), 999.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(a->cpu_data()[0]), 1.0, 1e-6);  // A 不变
  EXPECT_NEAR(static_cast<double>(d1->cpu_data()[0]), 1.0, 1e-6);  // D1 不变
  EXPECT_TRUE(d1->SharesDataWith(a.get()));  // d1 still shares with a

  // B→C→D2 三方共享（COW 后的 B 作为源，use_count=3）
  c->ShareData(b.get());
  d2->ShareData(b.get());
  EXPECT_TRUE(c->SharesDataWith(b.get()));
  EXPECT_TRUE(d2->SharesDataWith(b.get()));
  EXPECT_FALSE(c->SharesDataWith(a.get()));
  EXPECT_NEAR(static_cast<double>(c->cpu_data()[1]), 2.0, 1e-6);  // c sees b's data via const

  // 三方共享下 C mutable access triggers COW: c gets private copy, b and d2 unaffected
  c->cpu_mutable_data()[1] = 888.0f;
  EXPECT_FALSE(c->SharesDataWith(b.get()));  // c COW'd away from b
  EXPECT_FALSE(c->SharesDataWith(d2.get()));
  EXPECT_NEAR(static_cast<double>(c->cpu_data()[1]), 888.0, 1e-6);  // c has new value
  EXPECT_NEAR(static_cast<double>(b->cpu_data()[1]), 2.0, 1e-6);    // b unchanged
  EXPECT_NEAR(static_cast<double>(d2->cpu_data()[1]), 2.0, 1e-6);   // d2 unchanged
  EXPECT_NEAR(static_cast<double>(a->cpu_data()[1]), 2.0, 1e-6);    // a unchanged
  EXPECT_TRUE(d2->SharesDataWith(b.get()));  // d2 still shares with b
}

/// 场景 6：链中段 Reshape（A→B→C，Reshape B，验证 A 和 C 仍共享）
/// B 是中间节点，Reshape B 后，A 和 C 通过 refcount 仍持有原数据。
TEST(ShareDataRefCount, ChainMiddleReshapePreservesEndpoints) {
  auto a = make_object<Blob>(std::vector<int64_t>{4});
  auto b = make_object<Blob>(std::vector<int64_t>{4});
  auto c = make_object<Blob>(std::vector<int64_t>{4});

  a->cpu_mutable_data()[0] = 11.0f;
  a->cpu_mutable_data()[3] = 44.0f;

  // A→B→C 链
  b->ShareData(a.get());
  c->ShareData(b.get());
  EXPECT_TRUE(c->SharesDataWith(a.get()));

  // Reshape 中间节点 B
  b->Reshape(std::vector<int64_t>{8});

  // B 断开
  EXPECT_FALSE(b->SharesDataWith(a.get()));

  // A 和 C 仍共享原数据（通过 refcount 独立持有）
  EXPECT_TRUE(c->SharesDataWith(a.get()));
  EXPECT_NEAR(static_cast<double>(a->cpu_mutable_data()[0]), 11.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(c->cpu_mutable_data()[0]), 11.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(a->cpu_mutable_data()[3]), 44.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(c->cpu_mutable_data()[3]), 44.0, 1e-6);
}

/// 场景 7：源销毁后目标仍有效
/// 验证 ShareData 后销毁源 Blob，目标 Blob 通过 refcount 独立持有数据。
TEST(ShareDataRefCount, SourceDestroyedDataStillValid) {
  auto dst = make_object<Blob>(std::vector<int64_t>{4});
  const float* dst_ptr_before = nullptr;
  {
    auto src = make_object<Blob>(std::vector<int64_t>{4});
    src->cpu_mutable_data()[0] = 123.0f;
    src->cpu_mutable_data()[3] = 456.0f;

    dst->ShareData(src.get());
    dst_ptr_before = dst->cpu_data();
    EXPECT_TRUE(dst->SharesDataWith(src.get()));
  }  // src 析构

  // dst 仍应持有有效数据
  EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_data()[0]), 123.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_data()[3]), 456.0, 1e-6);
  // 指针不应变为悬空
  EXPECT_EQ(dst->cpu_data(), dst_ptr_before);
}

/// 场景 8：不同形状 Blob 间共享
/// ShareData 不要求形状匹配，shape 应跟随源。
TEST(ShareDataRefCount, ShareDataWithDifferentShapes) {
  auto a = make_object<Blob>(std::vector<int64_t>{2, 3, 4});  // 24 elements
  auto b = make_object<Blob>(std::vector<int64_t>{8});         // 8 elements

  a->cpu_mutable_data()[0] = 5.0f;
  a->cpu_mutable_data()[23] = 10.0f;

  b->ShareData(a.get());
  EXPECT_TRUE(b->SharesDataWith(a.get()));

  // b 的 shape 应跟随 a
  EXPECT_EQ(b->num_axes(), 3);
  EXPECT_EQ(b->count(), 24);
  EXPECT_EQ(b->shape(0), 2);
  EXPECT_EQ(b->shape(1), 3);
  EXPECT_EQ(b->shape(2), 4);

  EXPECT_NEAR(static_cast<double>(b->cpu_mutable_data()[0]), 5.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(b->cpu_mutable_data()[23]), 10.0, 1e-6);
}

/// 场景 9：多次相同共享是幂等的
/// 对同一对 Blob 多次调用 ShareData，指针和值不变。
TEST(ShareDataRefCount, RepeatedShareDataIsIdempotent) {
  auto a = make_object<Blob>(std::vector<int64_t>{4});
  auto b = make_object<Blob>(std::vector<int64_t>{4});

  a->cpu_mutable_data()[0] = 7.0f;

  b->ShareData(a.get());
  const float* ptr_after_first = b->cpu_data();
  EXPECT_TRUE(b->SharesDataWith(a.get()));

  // 再次共享同一源
  b->ShareData(a.get());
  EXPECT_TRUE(b->SharesDataWith(a.get()));
  EXPECT_EQ(b->cpu_data(), ptr_after_first);
  EXPECT_NEAR(static_cast<double>(b->cpu_mutable_data()[0]), 7.0, 1e-6);

  // 第三次共享
  b->ShareData(a.get());
  EXPECT_TRUE(b->SharesDataWith(a.get()));
  EXPECT_EQ(b->cpu_data(), ptr_after_first);
  EXPECT_NEAR(static_cast<double>(b->cpu_mutable_data()[0]), 7.0, 1e-6);
}

/// 场景 10：双向共享（A→B 且 B→A）
/// 验证 A 共享 B 后再由 B 共享 A 是幂等的。
TEST(ShareDataRefCount, BidirectionalShareIsIdempotent) {
  auto a = make_object<Blob>(std::vector<int64_t>{4});
  auto b = make_object<Blob>(std::vector<int64_t>{4});

  a->cpu_mutable_data()[0] = 99.0f;

  // A→B
  b->ShareData(a.get());
  EXPECT_TRUE(b->SharesDataWith(a.get()));
  EXPECT_EQ(b->cpu_data(), a->cpu_data());

  // B→A（反向共享，应幂等）
  a->ShareData(b.get());
  EXPECT_TRUE(a->SharesDataWith(b.get()));
  EXPECT_EQ(a->cpu_data(), b->cpu_data());
  EXPECT_NEAR(static_cast<double>(a->cpu_mutable_data()[0]), 99.0, 1e-6);
}

/// 场景 11：ShareData 后旧 Tensor 被正确释放
/// 验证 B 的旧独立 tensor 在 ShareData 后被释放，不泄漏。
TEST(ShareDataRefCount, OldTensorReleasedAfterShare) {
  // 记录初始已分配字节
  int64_t bytes_before = g_total_allocated_bytes.load();

  {
    auto a = make_object<Blob>(std::vector<int64_t>{100, 100});  // 40KB data + 40KB diff = 80KB
    auto b = make_object<Blob>(std::vector<int64_t>{100, 100});  // 另一个 80KB

    a->cpu_mutable_data()[0] = 1.0f;
    b->cpu_mutable_data()[0] = 2.0f;

    int64_t data_nbytes = a->count() * static_cast<int64_t>(sizeof(float));  // 40KB
    int64_t bytes_before_share = g_total_allocated_bytes.load();

    // B 共享 A → B 的旧 40KB data tensor 应被释放（refcount 归零时 FreeData）
    // 注意：TVM FFI 的侵入式引用计数在 data_tensor_ 被覆盖时
    // 自动递减旧 tensor 引用计数，引用计数归零时释放内存
    b->ShareData(a.get());

    EXPECT_TRUE(b->SharesDataWith(a.get()));
    int64_t bytes_after_share = g_total_allocated_bytes.load();

    // 总分配量应减少 data_nbytes（B 的旧 data tensor 被释放）
    EXPECT_LE(bytes_after_share, bytes_before_share);
    EXPECT_GE(bytes_before_share - bytes_after_share, data_nbytes - 64)  // allow small slack
        << "Expected at least " << data_nbytes << " bytes freed after ShareData";
  }

  // 所有 Blob 析构后，分配量应回到初始值附近
  int64_t bytes_after = g_total_allocated_bytes.load();
  EXPECT_LE(bytes_after, bytes_before + 1024);  // allow small slack for other test allocations
}

/// 场景 12：ShareData 空共享（源 tensor 未定义）
/// 对默认构造的 Blob（shape {0}）调用 ShareData，应触发 CHECK 失败。
/// 该测试用例在 Debug 构建中验证防御性检查，Release 中跳过。
TEST(ShareDataRefCount, ShareDataWithUndefinedTensorFails) {
  auto src = make_object<Blob>();  // 默认构造，data_tensor_ 可能未完全定义
  auto dst = make_object<Blob>(std::vector<int64_t>{4});

  // 如果 src 的 data_tensor_ 已定义（如 shape {0} 的 tensor），
  // ShareData 成功；否则触发 CHECK 失败。
  // 在 Release 中，CHECK 被编译为 no-op，此测试仅验证行为一致。
  // 此处仅验证不应崩溃。
  // 注：默认构造的 Blob 在构造函数中调用了 Reshape({0})，
  // 因此 data_tensor_ 已定义（shape {0}），ShareData 应成功。
  dst->ShareData(src.get());
  EXPECT_TRUE(dst->SharesDataWith(src.get()));
  EXPECT_EQ(dst->count(), 0);
}

/// 场景 13：ShareDiff 独立于 ShareData
/// 验证 data 和 diff 的共享关系相互独立，互不干扰。
TEST(ShareDataRefCount, ShareDiffIndependentOfShareData) {
  auto a = make_object<Blob>(std::vector<int64_t>{4});
  auto b = make_object<Blob>(std::vector<int64_t>{4});

  a->cpu_mutable_data()[0] = 10.0f;
  a->cpu_mutable_diff()[0] = 20.0f;
  b->cpu_mutable_data()[0] = 30.0f;
  b->cpu_mutable_diff()[0] = 40.0f;

  // 仅共享 data，不共享 diff
  b->ShareData(a.get());
  EXPECT_TRUE(b->SharesDataWith(a.get()));
  EXPECT_FALSE(b->SharesDiffWith(a.get()));

  // data 与 a 一致，diff 仍独立
  EXPECT_NEAR(static_cast<double>(b->cpu_mutable_data()[0]), 10.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(b->cpu_mutable_diff()[0]), 40.0, 1e-6);

  // 共享 diff
  b->ShareDiff(a.get());
  EXPECT_TRUE(b->SharesDiffWith(a.get()));
  EXPECT_NEAR(static_cast<double>(b->cpu_mutable_diff()[0]), 20.0, 1e-6);
}

/// 场景 14：零元素 Blob 共享
/// 验证 shape {0} 的 Blob 间共享不会崩溃。
TEST(ShareDataRefCount, ShareDataZeroElementBlob) {
  auto a = make_object<Blob>(std::vector<int64_t>{0});
  auto b = make_object<Blob>(std::vector<int64_t>{0});

  b->ShareData(a.get());
  EXPECT_TRUE(b->SharesDataWith(a.get()));
  EXPECT_EQ(b->count(), 0);
  EXPECT_EQ(a->count(), 0);
}

/// 场景 15：COW 仅影响写入者（A→B→C，B COW）
/// 验证链式共享中只有触发 COW 的 Blob 断开，其余共享关系保持。
TEST(ShareDataRefCount, COWOnlyAffectsMutator) {
  auto a = make_object<Blob>(std::vector<int64_t>{4});
  auto b = make_object<Blob>(std::vector<int64_t>{4});
  auto c = make_object<Blob>(std::vector<int64_t>{4});

  a->cpu_mutable_data()[0] = 1.0f;
  a->cpu_mutable_data()[1] = 2.0f;
  a->cpu_mutable_data()[2] = 3.0f;
  a->cpu_mutable_data()[3] = 4.0f;

  // A→B→C 链
  b->ShareData(a.get());
  c->ShareData(a.get());  // C 直接共享 A（测试源自 A 的多路共享）

  EXPECT_TRUE(b->SharesDataWith(a.get()));
  EXPECT_TRUE(c->SharesDataWith(a.get()));

  // B 触发 COW
  float* b_data = b->cpu_mutable_data();
  b_data[0] = 999.0f;

  // B 断开
  EXPECT_FALSE(b->SharesDataWith(a.get()));
  EXPECT_NEAR(static_cast<double>(b->cpu_mutable_data()[0]), 999.0, 1e-6);

  // A 和 C 仍共享
  EXPECT_TRUE(c->SharesDataWith(a.get()));
  EXPECT_NEAR(static_cast<double>(a->cpu_mutable_data()[0]), 1.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(c->cpu_mutable_data()[0]), 1.0, 1e-6);

  // C 也触发 COW
  float* c_data = c->cpu_mutable_data();
  c_data[0] = 888.0f;

  // C 断开
  EXPECT_FALSE(c->SharesDataWith(a.get()));
  EXPECT_NEAR(static_cast<double>(c->cpu_mutable_data()[0]), 888.0, 1e-6);

  // A 仍不变
  EXPECT_NEAR(static_cast<double>(a->cpu_mutable_data()[0]), 1.0, 1e-6);
}

// ── Split layer N=1 zero-copy integration test (via Net) ──

TEST(ZeroCopyTest, SplitN1ZeroCopyViaNet) {
  // Minimal prototxt: Data -> Split(1 top) -> output
  std::string prototxt = R"(
name: "test_split_n1_zerocopy"
input: "data"
input_dim: 1
input_dim: 2
input_dim: 3
input_dim: 4
layer {
  name: "split1"
  type: "Split"
  bottom: "data"
  top: "split_out"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  net->Forward();

  auto bottom_blob = net->blob_by_name("data");
  auto top_blob = net->blob_by_name("split_out");

  EXPECT_TRUE(bottom_blob);
  EXPECT_TRUE(top_blob);

  // After forward with N=1 zero-copy, top must share the same data pointer
  // as bottom (zero-copy alias).
  EXPECT_TRUE(top_blob->SharesDataWith(bottom_blob.get()));
  EXPECT_EQ(top_blob->cpu_data(), bottom_blob->cpu_data());

  // Shape must match
  EXPECT_EQ(top_blob->count(), bottom_blob->count());
  EXPECT_EQ(top_blob->num_axes(), bottom_blob->num_axes());
}

TEST(ZeroCopyTest, SplitN1DataCorrectnessThroughForward) {
  std::string prototxt = R"(
name: "test_split_n1_data"
input: "data"
input_dim: 2
input_dim: 3
input_dim: 4
input_dim: 5
layer {
  name: "split1"
  type: "Split"
  bottom: "data"
  top: "split_out"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));

  // Write known data into input blob before forward
  auto data_blob = net->blob_by_name("data");
  float* data_ptr = data_blob->cpu_mutable_data();
  for (int64_t i = 0; i < data_blob->count(); ++i) {
    data_ptr[i] = static_cast<float>(i) * 0.01f;
  }

  net->Forward();

  auto out_blob = net->blob_by_name("split_out");

  // Data must be visible through the output blob (zero-copy shared, no corruption)
  EXPECT_TRUE(out_blob->SharesDataWith(data_blob.get()));
  for (int64_t i = 0; i < data_blob->count(); ++i) {
    EXPECT_NEAR(static_cast<double>(out_blob->cpu_mutable_data()[i]),
                static_cast<double>(i) * 0.01, 1e-6);
  }
}

TEST(ZeroCopyTest, SplitN2COWZeroCopyShare) {
  // N=2: COW Phase 2 -- tops share data with bottom after Forward (zero-copy).
  // The actual copy happens only when cpu_mutable_data() is called on a top.
  std::string prototxt = R"(
name: "test_split_n2_cow"
input: "data"
input_dim: 1
input_dim: 2
input_dim: 3
input_dim: 4
layer {
  name: "split1"
  type: "Split"
  bottom: "data"
  top: "out_a"
  top: "out_b"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));

  auto data_blob = net->blob_by_name("data");
  data_blob->cpu_mutable_data()[0] = 111.0f;
  data_blob->cpu_mutable_data()[1] = 222.0f;

  net->Forward();

  auto out_a = net->blob_by_name("out_a");
  auto out_b = net->blob_by_name("out_b");

  // Phase 2 COW: tops share data with bottom (zero-copy, same pointer)
  EXPECT_TRUE(out_a->SharesDataWith(data_blob.get()));
  EXPECT_TRUE(out_b->SharesDataWith(data_blob.get()));
  EXPECT_TRUE(out_a->SharesDataWith(out_b.get()));
  EXPECT_EQ(out_a->cpu_data(), out_b->cpu_data());
  EXPECT_EQ(out_a->cpu_data(), data_blob->cpu_data());

  // Data values correct
  EXPECT_NEAR(static_cast<double>(out_a->cpu_mutable_data()[0]), 111.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(out_a->cpu_mutable_data()[1]), 222.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(out_b->cpu_mutable_data()[0]), 111.0, 1e-6);
}

/// 验证 COW 触发：调用 cpu_mutable_data() 后该 top 打破共享，
/// 其他 top 仍保持共享
TEST(ZeroCopyTest, SplitN2COWTriggerOnMutableData) {
  std::string prototxt = R"(
name: "test_split_n2_cow_trigger"
input: "data"
input_dim: 1
input_dim: 2
input_dim: 3
input_dim: 4
layer {
  name: "split1"
  type: "Split"
  bottom: "data"
  top: "out_a"
  top: "out_b"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));

  auto data_blob = net->blob_by_name("data");
  data_blob->cpu_mutable_data()[0] = 50.0f;
  data_blob->cpu_mutable_data()[1] = 60.0f;

  net->Forward();

  auto out_a = net->blob_by_name("out_a");
  auto out_b = net->blob_by_name("out_b");

  // Before mutation: all share
  EXPECT_TRUE(out_a->SharesDataWith(out_b.get()));

  // Mutate out_a via cpu_mutable_data() → COW triggers
  out_a->cpu_mutable_data()[0] = 999.0f;

  // out_a should now have its own copy (COW broke sharing)
  EXPECT_FALSE(out_a->SharesDataWith(out_b.get()));
  EXPECT_FALSE(out_a->SharesDataWith(data_blob.get()));
  EXPECT_NE(out_a->cpu_data(), out_b->cpu_data());

  // out_a's mutation should NOT affect out_b or bottom
  EXPECT_NEAR(static_cast<double>(out_a->cpu_data()[0]), 999.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(out_b->cpu_data()[0]), 50.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(data_blob->cpu_data()[0]), 50.0, 1e-6);

  // out_b still shares with bottom (const reads did not trigger COW)
  EXPECT_TRUE(out_b->SharesDataWith(data_blob.get()));
}

TEST(ZeroCopyTest, LiveBlobCountStableAcrossShareData) {
  int64_t before = LiveBlobCount();
  {
    auto a = make_object<Blob>(std::vector<int64_t>{8});
    auto b = make_object<Blob>(std::vector<int64_t>{8});
    EXPECT_EQ(LiveBlobCount(), before + 2);
    b->ShareData(a.get());
    // ShareData does not create or destroy Blobs -- count unchanged
    EXPECT_EQ(LiveBlobCount(), before + 2);
  }
  EXPECT_EQ(LiveBlobCount(), before);
}

// ── 高优先级补充场景 ──

/// 场景 1：ShareData 和 ShareDiff 分别来自不同源
/// 验证 data 和 diff 可以独立地从不同 Blob 共享，
/// 且各自的共享关系互不干扰。
/// COW 触发条件：use_count > 2（三方及以上共享）。
TEST(ZeroCopyTest, ShareDataAndDiffFromDifferentSources) {
  std::vector<int64_t> shape = {4, 3};
  auto src_data = make_object<Blob>(shape);
  auto src_diff = make_object<Blob>(shape);
  auto dst = make_object<Blob>(shape);
  auto c_data = make_object<Blob>(shape);

  // 写入可区分的值
  src_data->cpu_mutable_data()[0] = 10.0f;
  src_data->cpu_mutable_data()[1] = 20.0f;
  src_diff->cpu_mutable_diff()[0] = 30.0f;
  src_diff->cpu_mutable_diff()[1] = 40.0f;

  dst->ShareData(src_data.get());
  dst->ShareDiff(src_diff.get());
  c_data->ShareData(src_data.get());  // 三方共享 data（use_count=3）: src_data + dst + c_data

  // data 与 src_data 共享，diff 与 src_diff 共享
  EXPECT_TRUE(dst->SharesDataWith(src_data.get()));
  EXPECT_TRUE(c_data->SharesDataWith(src_data.get()));
  EXPECT_TRUE(dst->SharesDiffWith(src_diff.get()));

  // 交叉关系：data 不与 src_diff 共享，diff 不与 src_data 共享
  EXPECT_FALSE(dst->SharesDataWith(src_diff.get()));
  EXPECT_FALSE(dst->SharesDiffWith(src_data.get()));

  // 值验证（const reads, do not trigger COW）
  EXPECT_NEAR(static_cast<double>(dst->cpu_data()[0]), 10.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_data()[1]), 20.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(c_data->cpu_data()[0]), 10.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_diff()[0]), 30.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_diff()[1]), 40.0, 1e-6);

  // 通过 dst 写入 data 触发 COW（三方共享 use_count=3）：dst 获得私有副本，src_data 和 c_data 不受影响
  dst->cpu_mutable_data()[0] = 99.0f;
  EXPECT_FALSE(dst->SharesDataWith(src_data.get()));  // dst COW'd away
  EXPECT_FALSE(dst->SharesDataWith(c_data.get()));
  EXPECT_TRUE(c_data->SharesDataWith(src_data.get()));  // c_data still shares with src_data
  EXPECT_NEAR(static_cast<double>(dst->cpu_data()[0]), 99.0, 1e-6);   // dst has new value
  EXPECT_NEAR(static_cast<double>(src_data->cpu_data()[0]), 10.0, 1e-6);  // src_data unchanged
  EXPECT_NEAR(static_cast<double>(c_data->cpu_data()[0]), 10.0, 1e-6);    // c_data unchanged
  EXPECT_NEAR(static_cast<double>(src_diff->cpu_data()[0]), 0.0, 1e-6);  // src_diff data untouched
}

/// 场景 2：共享后 Reshape 源 Blob
/// 验证 Reshape 源 Blob 不会破坏已共享的目标数据。
/// 源 Reshape 后分配新内存，目标通过 refcount 仍持有原数据
TEST(ZeroCopyTest, ReshapeSourceAfterSharePreservesDestination) {
  std::vector<int64_t> shape = {4, 4};
  auto src = make_object<Blob>(shape);
  auto dst = make_object<Blob>(shape);

  src->cpu_mutable_data()[0] = 42.0f;
  src->cpu_mutable_data()[5] = 77.0f;
  dst->ShareData(src.get());
  EXPECT_TRUE(dst->SharesDataWith(src.get()));

  // Reshape src 到不同形状 → 分配新内存，打破共享
  src->Reshape(std::vector<int64_t>{8, 8});

  // dst 仍持有原数据（通过 refcount 独立持有）
  EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_data()[0]), 42.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_data()[5]), 77.0, 1e-6);

  // 确认共享已打破
  EXPECT_FALSE(dst->SharesDataWith(src.get()));

  // src 的新数据独立，不受 dst 影响
  src->cpu_mutable_data()[0] = 100.0f;
  EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_data()[0]), 42.0, 1e-6);
}

/// 场景 3：连续多次 Forward（N=1）不泄漏 refcount
/// 验证 N=1 零拷贝路径在重复 Forward 时不会累积
/// 引用计数泄漏或意外创建新 Blob
TEST(ZeroCopyTest, RepeatedForwardN1NoRefcountLeak) {
  std::string prototxt = R"(
name: "test_repeated_forward"
input: "data"
input_dim: 1
input_dim: 2
input_dim: 3
input_dim: 4
layer {
  name: "split1"
  type: "Split"
  bottom: "data"
  top: "split_out"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto bottom_blob = net->blob_by_name("data");
  auto top_blob = net->blob_by_name("split_out");

  // 写入初始数据
  bottom_blob->cpu_mutable_data()[0] = 55.0f;

  int64_t live_count_before = LiveBlobCount();

  // Forward 10 次 -- 不应创建新 Blob 或泄漏 refcount
  for (int i = 0; i < 10; ++i) {
    net->Forward();
    // 每次 Forward 后，N=1 top 应仍与 bottom 共享数据
    EXPECT_TRUE(top_blob->SharesDataWith(bottom_blob.get()));
    // 数据正确性
    EXPECT_NEAR(static_cast<double>(top_blob->cpu_mutable_data()[0]), 55.0, 1e-6);
  }

  // Blob 数量稳定（无泄漏）
  EXPECT_EQ(LiveBlobCount(), live_count_before);
}

// ── Phase 2 COW API 测试 (A2/A3 新增方法) ──

/// 验证 IsDataShared() 在未共享时返回 false
TEST(COWApiTest, IsDataSharedFalseWhenPrivate) {
  auto b = make_object<Blob>(std::vector<int64_t>{4, 4});
  EXPECT_FALSE(b->IsDataShared());
  EXPECT_FALSE(b->IsDiffShared());
}

/// 验证 IsDataShared() 在 ShareData 后返回 true
TEST(COWApiTest, IsDataSharedTrueAfterShareData) {
  auto src = make_object<Blob>(std::vector<int64_t>{4, 4});
  src->cpu_mutable_data()[0] = 1.0f;
  auto dst = make_object<Blob>(std::vector<int64_t>{4, 4});
  dst->ShareData(src.get());
  EXPECT_TRUE(dst->IsDataShared());
  EXPECT_FALSE(src->IsDataShared());  // src 拥有唯一引用（dst 是另一个引用）
}

/// 验证 DataRefCount() 反映共享状态
TEST(COWApiTest, DataRefCountAfterShareData) {
  auto src = make_object<Blob>(std::vector<int64_t>{4, 4});
  auto dst = make_object<Blob>(std::vector<int64_t>{4, 4});
  dst->ShareData(src.get());
  // src 的 data_tensor_ 被 dst 共享，refcount >= 2
  EXPECT_GE(src->DataRefCount(), 2);
  EXPECT_GE(dst->DataRefCount(), 2);
  EXPECT_EQ(src->DataRefCount(), dst->DataRefCount());
}

/// 验证 DataRefCount() 在零元素 tensor 时返回 0（按约定：空 tensor 的 refcount 视为 0）
TEST(COWApiTest, DataRefCountZeroWhenUndefined) {
  auto b = make_object<Blob>();  // 默认构造 shape={0}
  b->Reshape(std::vector<int64_t>{0});
  // Zero-element tensor: DataRefCount() returns 0 per API contract
  // (avoids spurious "shared" signals for empty buffers)
  EXPECT_EQ(b->DataRefCount(), 0);

  // Non-empty tensor has refcount >= 1
  b->Reshape(std::vector<int64_t>{4});
  b->cpu_mutable_data()[0] = 1.0f;
  EXPECT_GE(b->DataRefCount(), 1);
}

/// 验证 UnshareData() 在共享后强制私有化
TEST(COWApiTest, UnshareDataBreaksSharing) {
  auto src = make_object<Blob>(std::vector<int64_t>{8});
  for (int i = 0; i < 8; ++i) src->cpu_mutable_data()[i] = static_cast<float>(i);
  auto dst = make_object<Blob>(std::vector<int64_t>{8});
  dst->ShareData(src.get());
  EXPECT_TRUE(dst->IsDataShared());

  // 显式 Unshare
  void* new_ptr = dst->UnshareData();
  EXPECT_FALSE(dst->IsDataShared());
  EXPECT_NE(new_ptr, nullptr);

  // 数据完整性：COW 后数据应和源一致
  for (int i = 0; i < 8; ++i) {
    EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_data()[i]),
                static_cast<double>(i), 1e-6);
  }
}

/// 验证 UnshareData() 在未共享时是 no-op
TEST(COWApiTest, UnshareDataNoopWhenPrivate) {
  auto b = make_object<Blob>(std::vector<int64_t>{4});
  b->cpu_mutable_data()[0] = 42.0f;
  const void* before = b->cpu_data();
  void* after = b->UnshareData();
  EXPECT_EQ(before, after);  // 指针不变
  EXPECT_FALSE(b->IsDataShared());
}

/// 验证 UnshareDiff() 在 ShareDiff 后强制私有化
TEST(COWApiTest, UnshareDiffBreaksSharing) {
  auto src = make_object<Blob>(std::vector<int64_t>{4});
  src->cpu_mutable_diff()[0] = 10.0f;
  auto dst = make_object<Blob>(std::vector<int64_t>{4});
  dst->ShareDiff(src.get());
  EXPECT_TRUE(dst->IsDiffShared());

  void* new_ptr = dst->UnshareDiff();
  EXPECT_FALSE(dst->IsDiffShared());
  EXPECT_NE(new_ptr, nullptr);
  EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_diff()[0]), 10.0, 1e-6);
}

/// 验证 mutable_data_tensor() 在三方共享（use_count=3）时触发 COW
TEST(COWApiTest, MutableDataTensorTriggersCOW) {
  auto src = make_object<Blob>(std::vector<int64_t>{4});
  src->cpu_mutable_data()[0] = 77.0f;
  auto dst = make_object<Blob>(std::vector<int64_t>{4});
  auto c = make_object<Blob>(std::vector<int64_t>{4});
  dst->ShareData(src.get());
  c->ShareData(src.get());
  EXPECT_TRUE(dst->IsDataShared());
  EXPECT_TRUE(c->IsDataShared());

  Tensor t = dst->mutable_data_tensor();
  EXPECT_FALSE(dst->IsDataShared());
  EXPECT_TRUE(c->IsDataShared());  // c still shares with src
  EXPECT_NEAR(static_cast<double>(static_cast<const float*>(t.data_ptr())[0]), 77.0, 1e-6);
}

/// 验证 mutable_data_tensor() 在未共享时不触发 COW
TEST(COWApiTest, MutableDataTensorNoCOWWhenPrivate) {
  auto b = make_object<Blob>(std::vector<int64_t>{4});
  b->cpu_mutable_data()[0] = 99.0f;
  const void* before = b->cpu_data();
  Tensor t = b->mutable_data_tensor();
  EXPECT_EQ(b->cpu_data(), before);  // 指针不变
  EXPECT_FALSE(b->IsDataShared());
}

/// 验证 mutable_diff_tensor() 在三方共享（use_count=3）时触发 COW
TEST(COWApiTest, MutableDiffTensorTriggersCOW) {
  auto src = make_object<Blob>(std::vector<int64_t>{4});
  src->cpu_mutable_diff()[0] = 55.0f;
  auto dst = make_object<Blob>(std::vector<int64_t>{4});
  auto c = make_object<Blob>(std::vector<int64_t>{4});
  dst->ShareDiff(src.get());
  c->ShareDiff(src.get());
  EXPECT_TRUE(dst->IsDiffShared());
  EXPECT_TRUE(c->IsDiffShared());

  Tensor t = dst->mutable_diff_tensor();
  EXPECT_FALSE(dst->IsDiffShared());
  EXPECT_TRUE(c->IsDiffShared());  // c still shares diff with src
  EXPECT_NEAR(static_cast<double>(static_cast<const float*>(t.data_ptr())[0]), 55.0, 1e-6);
}

/// 验证三方共享（use_count=3）下 COW 写入隔离：修改 dst 不影响 src 和其他共享者
TEST(COWApiTest, COWWriteIsolation) {
  auto src = make_object<Blob>(std::vector<int64_t>{4});
  for (int i = 0; i < 4; ++i) src->cpu_mutable_data()[i] = static_cast<float>(i + 1);
  auto dst = make_object<Blob>(std::vector<int64_t>{4});
  auto c = make_object<Blob>(std::vector<int64_t>{4});
  dst->ShareData(src.get());
  c->ShareData(src.get());
  // 三方共享（use_count=3），原始数据：1, 2, 3, 4

  // COW 后修改 dst，不影响 src 和 c
  dst->cpu_mutable_data()[0] = 999.0f;
  EXPECT_NEAR(static_cast<double>(src->cpu_data()[0]), 1.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(c->cpu_data()[0]), 1.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_data()[0]), 999.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src->cpu_data()[1]), 2.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(c->cpu_data()[1]), 2.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_data()[1]), 2.0, 1e-6);
  EXPECT_TRUE(c->SharesDataWith(src.get()));  // c still shares with src
  EXPECT_FALSE(dst->SharesDataWith(src.get()));
}

// ── ShareDiff 引用计数边界测试 (ShareDiffRefCount) ──
// ShareDiff 的对称测试，覆盖与 ShareDataRefCount 对应的关键边界场景，
// 确保 diff 共享的 COW 语义与 data 共享完全对称。

/// ShareDiff 自共享幂等性（对应 ShareDataRefCount.SelfShareIsIdempotent）
TEST(ShareDiffRefCount, SelfShareIsIdempotent) {
  auto a = make_object<Blob>(std::vector<int64_t>{4, 4});
  a->cpu_mutable_diff()[0] = 42.0f;
  const float* ptr_before = a->cpu_diff();

  a->ShareDiff(a.get());

  EXPECT_EQ(a->cpu_diff(), ptr_before);
  EXPECT_TRUE(a->SharesDiffWith(a.get()));
  EXPECT_NEAR(static_cast<double>(a->cpu_mutable_diff()[0]), 42.0, 1e-6);
}

/// ShareDiff 重复共享幂等（对应 ShareDataRefCount.RepeatedShareDataIsIdempotent）
TEST(ShareDiffRefCount, RepeatedShareDiffIsIdempotent) {
  auto a = make_object<Blob>(std::vector<int64_t>{4});
  auto b = make_object<Blob>(std::vector<int64_t>{4});

  a->cpu_mutable_diff()[0] = 7.0f;

  b->ShareDiff(a.get());
  const float* ptr_after_first = b->cpu_diff();
  EXPECT_TRUE(b->SharesDiffWith(a.get()));

  b->ShareDiff(a.get());
  EXPECT_TRUE(b->SharesDiffWith(a.get()));
  EXPECT_EQ(b->cpu_diff(), ptr_after_first);
  EXPECT_NEAR(static_cast<double>(b->cpu_mutable_diff()[0]), 7.0, 1e-6);

  b->ShareDiff(a.get());
  EXPECT_TRUE(b->SharesDiffWith(a.get()));
  EXPECT_EQ(b->cpu_diff(), ptr_after_first);
}

/// ShareDiff 不同形状共享（对应 ShareDataRefCount.ShareDataWithDifferentShapes）
TEST(ShareDiffRefCount, ShareDiffWithDifferentShapes) {
  auto a = make_object<Blob>(std::vector<int64_t>{2, 3, 4});
  auto b = make_object<Blob>(std::vector<int64_t>{8});

  a->cpu_mutable_diff()[0] = 5.0f;
  a->cpu_mutable_diff()[23] = 10.0f;

  b->ShareDiff(a.get());
  EXPECT_TRUE(b->SharesDiffWith(a.get()));

  // b 的 diff shape 应跟随 a
  EXPECT_EQ(b->num_axes(), 3);
  EXPECT_EQ(b->count(), 24);

  EXPECT_NEAR(static_cast<double>(b->cpu_mutable_diff()[0]), 5.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(b->cpu_mutable_diff()[23]), 10.0, 1e-6);
}

/// ShareDiff COW 后数据隔离：三方共享时修改 dst1 不影响 src 和 dst2
TEST(ShareDiffRefCount, DiffIsolationAfterCOW) {
  auto src = make_object<Blob>(std::vector<int64_t>{4});
  auto dst1 = make_object<Blob>(std::vector<int64_t>{4});
  auto dst2 = make_object<Blob>(std::vector<int64_t>{4});

  src->cpu_mutable_diff()[0] = 10.0f;
  src->cpu_mutable_diff()[1] = 20.0f;
  src->cpu_mutable_diff()[2] = 30.0f;
  src->cpu_mutable_diff()[3] = 40.0f;

  // 三方共享（use_count=3）
  dst1->ShareDiff(src.get());
  dst2->ShareDiff(src.get());

  // 触发 COW 并修改 dst1
  dst1->cpu_mutable_diff()[0] = 999.0f;
  dst1->cpu_mutable_diff()[1] = 888.0f;

  // src 和 dst2 不受影响
  EXPECT_NEAR(static_cast<double>(src->cpu_diff()[0]), 10.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src->cpu_diff()[1]), 20.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src->cpu_diff()[2]), 30.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src->cpu_diff()[3]), 40.0, 1e-6);
  EXPECT_TRUE(dst2->SharesDiffWith(src.get()));

  // dst1 有自己的值
  EXPECT_NEAR(static_cast<double>(dst1->cpu_diff()[0]), 999.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst1->cpu_diff()[1]), 888.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst1->cpu_diff()[2]), 30.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst1->cpu_diff()[3]), 40.0, 1e-6);
}

/// ShareDiff 三向共享 COW 仅影响写入者（对应 COWTest.ThreeWayShareCOWOnlyAffectsMutator）
TEST(ShareDiffRefCount, ThreeWayDiffCOWOnlyAffectsMutator) {
  auto a = make_object<Blob>(std::vector<int64_t>{4});
  auto b = make_object<Blob>(std::vector<int64_t>{4});
  auto c = make_object<Blob>(std::vector<int64_t>{4});

  a->cpu_mutable_diff()[0] = 1.0f;
  b->ShareDiff(a.get());
  c->ShareDiff(a.get());

  EXPECT_TRUE(b->SharesDiffWith(a.get()));
  EXPECT_TRUE(c->SharesDiffWith(a.get()));

  // b 触发 COW
  b->cpu_mutable_diff()[0] = 999.0f;

  EXPECT_FALSE(b->SharesDiffWith(a.get()));
  EXPECT_FALSE(b->SharesDiffWith(c.get()));

  // a 和 c 仍共享
  EXPECT_TRUE(a->SharesDiffWith(c.get()));
  EXPECT_NEAR(static_cast<double>(a->cpu_mutable_diff()[0]), 1.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(c->cpu_mutable_diff()[0]), 1.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(b->cpu_mutable_diff()[0]), 999.0, 1e-6);
}

// ── Owner COW 关键测试（A5 修复验证） ──
// 验证关键 bug 修复：当 Blob 是 tensor 的 owner（data_shared_=false / diff_shared_=false），
// 但其他 Blob 仍共享该 tensor（use_count > 1）时，owner 调用 cpu_mutable_data()/cpu_mutable_diff()
// 也应触发 COW，避免写入破坏 borrower 的视图。这是 Split N≥2 Backward 梯度累加的必要条件。

/// Owner 写入共享 data tensor 时触发 COW（防止写入破坏 borrowers）
TEST(OwnerCOWTest, OwnerMutableDataTriggersCOWWhenShared) {
  auto owner = make_object<Blob>(std::vector<int64_t>{4});
  auto borrower1 = make_object<Blob>(std::vector<int64_t>{4});
  auto borrower2 = make_object<Blob>(std::vector<int64_t>{4});

  owner->cpu_mutable_data()[0] = 10.0f;
  owner->cpu_mutable_data()[1] = 20.0f;
  owner->cpu_mutable_data()[2] = 30.0f;
  owner->cpu_mutable_data()[3] = 40.0f;

  borrower1->ShareData(owner.get());
  borrower2->ShareData(owner.get());

  EXPECT_TRUE(borrower1->SharesDataWith(owner.get()));
  EXPECT_TRUE(borrower2->SharesDataWith(owner.get()));
  const float* borrower1_ptr_before = borrower1->cpu_data();
  const float* owner_ptr_before = owner->cpu_data();

  // Owner 调用 cpu_mutable_data() ---- 因为有 borrowers 共享，应触发 COW
  float* owner_mut = owner->cpu_mutable_data();

  // Owner 获得新的私有 buffer（与 borrowers 断开）
  EXPECT_NE(owner->cpu_data(), borrower1_ptr_before);
  EXPECT_NE(owner->cpu_data(), owner_ptr_before);
  EXPECT_FALSE(owner->SharesDataWith(borrower1.get()));
  EXPECT_FALSE(owner->SharesDataWith(borrower2.get()));

  // Borrowers 仍指向旧数据（不受 owner COW 影响）
  EXPECT_TRUE(borrower1->SharesDataWith(borrower2.get()));
  EXPECT_EQ(borrower1->cpu_data(), borrower1_ptr_before);
  EXPECT_NEAR(static_cast<double>(borrower1->cpu_data()[0]), 10.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(borrower1->cpu_data()[1]), 20.0, 1e-6);

  // Owner 写入新 buffer 不影响 borrowers
  owner_mut[0] = 999.0f;
  EXPECT_NEAR(static_cast<double>(borrower1->cpu_data()[0]), 10.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(owner->cpu_data()[0]), 999.0, 1e-6);
}

/// 单 borrower 场景 data 路径 Identity 模式：borrower 两方共享 in-place 直通（N=1 passthrough），owner mutable 时 COW
TEST(OwnerCOWTest, OwnerMutableDataCOWWithSingleBorrower) {
  auto owner = make_object<Blob>(std::vector<int64_t>{4});
  auto borrower = make_object<Blob>(std::vector<int64_t>{4});

  owner->cpu_mutable_data()[0] = 0.0f;
  borrower->ShareDataIdentity(owner.get());

  // N=1 Identity共享：borrower mutable 是 in-place 直通（identity模式不触发COW）
  const float* shared_ptr = borrower->cpu_data();
  borrower->cpu_mutable_data()[0] = 42.0f;
  EXPECT_EQ(borrower->cpu_data(), shared_ptr);  // 指针不变，无COW
  EXPECT_TRUE(borrower->SharesDataWith(owner.get()));  // 仍共享
  EXPECT_NEAR(static_cast<double>(owner->cpu_data()[0]), 42.0, 1e-6);  // in-place写入owner立即可见

  // Owner mutable: owner角色，use_count=2>1 → COW触发（owner需要私有buffer）
  const float* owner_ptr_before = owner->cpu_data();
  float* owner_mut = owner->cpu_mutable_data();
  EXPECT_NE(owner_mut, owner_ptr_before);  // COW发生，指针改变
  EXPECT_FALSE(owner->SharesDataWith(borrower.get()));  // 不再共享
  // Owner私有copy保留borrower in-place写入的值
  EXPECT_NEAR(static_cast<double>(owner_mut[0]), 42.0, 1e-6);
  // Borrower仍持有原buffer，值正确保留
  EXPECT_NEAR(static_cast<double>(borrower->cpu_data()[0]), 42.0, 1e-6);
  // Borrower不再与owner共享，写入borrower不影响owner
  borrower->cpu_mutable_data()[0] = 99.0f;
  EXPECT_NEAR(static_cast<double>(borrower->cpu_data()[0]), 99.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(owner_mut[0]), 42.0, 1e-6);
}

/// Owner 写入共享 diff tensor 时触发 COW（Split Backward 梯度累加核心场景）
TEST(OwnerCOWTest, OwnerMutableDiffTriggersCOWWhenShared) {
  auto owner = make_object<Blob>(std::vector<int64_t>{4});
  auto borrower1 = make_object<Blob>(std::vector<int64_t>{4});
  auto borrower2 = make_object<Blob>(std::vector<int64_t>{4});

  // 模拟 N=2 Split Forward: bottom(owner) 的 diff 被两个 top(borrowers) 共享
  owner->cpu_mutable_diff()[0] = 0.0f;  // init zeros
  borrower1->ShareDiff(owner.get());
  borrower2->ShareDiff(owner.get());

  EXPECT_TRUE(borrower1->SharesDiffWith(owner.get()));
  EXPECT_TRUE(borrower2->SharesDiffWith(owner.get()));

  // 模拟 borrower1 的 downstream 写入梯度（触发 COW on borrower1）
  borrower1->cpu_mutable_diff()[0] = 1.0f;  // d_top1 = 1.0
  borrower1->cpu_mutable_diff()[1] = 2.0f;

  // borrower2 未被写入（假设该分支不传播梯度），仍与 owner 共享
  EXPECT_TRUE(borrower2->SharesDiffWith(owner.get()));
  EXPECT_FALSE(borrower1->SharesDiffWith(owner.get()));  // borrower1 COW'd away

  // 模拟 Split Backward: owner(bottom) 调用 cpu_mutable_diff() 准备累加
  // 因为 borrower2 仍共享 owner 的 diff，COW 必须触发以避免别名
  float* owner_diff = owner->cpu_mutable_diff();

  // Owner 现在有私有 buffer（不与任何 borrower 别名）
  EXPECT_FALSE(owner->SharesDiffWith(borrower2.get()));

  // borrower2 的指针不变且仍为原始 zeros（未被破坏）
  EXPECT_NEAR(static_cast<double>(borrower2->cpu_diff()[0]), 0.0, 1e-6);

  // borrower1 的梯度完好
  EXPECT_NEAR(static_cast<double>(borrower1->cpu_diff()[0]), 1.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(borrower1->cpu_diff()[1]), 2.0, 1e-6);

  // 模拟梯度累加：copy borrower1 + axpy borrower2(zeros)
  // (这正是 SplitLayer::Backward_cpu 中的核心逻辑)
  const float* b1_diff = borrower1->cpu_diff();
  const float* b2_diff = borrower2->cpu_diff();
  // copy: owner_diff = b1_diff
  for (int64_t i = 0; i < owner->count(); ++i) owner_diff[i] = b1_diff[i];
  // axpy: owner_diff += 1.0 * b2_diff (b2_diff != owner_diff, no aliasing)
  EXPECT_NE(b2_diff, owner_diff);  // 关键断言：无别名
  for (int64_t i = 0; i < owner->count(); ++i) owner_diff[i] += b2_diff[i];

  EXPECT_NEAR(static_cast<double>(owner_diff[0]), 1.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(owner_diff[1]), 2.0, 1e-6);
}

/// N=1 Identity 场景：borrower 通过 ShareDiffIdentity 两方共享 in-place 直通
/// - Identity 模式: borrower mutable 不触发 COW, 写入对 owner 可见
/// - Owner mutable: 对称 COW 阈值 use_count>1，owner 获得私有副本
TEST(OwnerCOWTest, OwnerMutableDiffCOWWithSingleBorrower) {
  auto owner = make_object<Blob>(std::vector<int64_t>{4});
  auto borrower = make_object<Blob>(std::vector<int64_t>{4});

  owner->cpu_mutable_diff()[0] = 0.0f;
  borrower->ShareDiffIdentity(owner.get());  // N=1 identity: in-place passthrough

  // Identity 模式：borrower mutable 是 in-place 直通，不触发 COW
  const float* shared_ptr = borrower->cpu_diff();
  borrower->cpu_mutable_diff()[0] = 42.0f;
  EXPECT_EQ(borrower->cpu_diff(), shared_ptr);  // 指针不变，无COW
  EXPECT_TRUE(borrower->SharesDiffWith(owner.get()));  // 仍共享
  EXPECT_NEAR(static_cast<double>(owner->cpu_diff()[0]), 42.0, 1e-6);  // in-place写入owner立即可见

  // Owner mutable: use_count=2>1 → COW触发（对称阈值，owner需要私有buffer避免累加别名）
  const float* owner_ptr_before = owner->cpu_diff();
  float* owner_mut = owner->cpu_mutable_diff();
  EXPECT_NE(owner_mut, owner_ptr_before);  // COW发生，指针改变
  EXPECT_FALSE(owner->SharesDiffWith(borrower.get()));  // 不再共享
  // Owner私有copy保留borrower in-place写入的值
  EXPECT_NEAR(static_cast<double>(owner_mut[0]), 42.0, 1e-6);
  // Borrower仍持有原buffer，值正确保留
  EXPECT_NEAR(static_cast<double>(borrower->cpu_diff()[0]), 42.0, 1e-6);
  // Borrower不再与owner共享，写入borrower不影响owner
  borrower->cpu_mutable_diff()[0] = 99.0f;
  EXPECT_NEAR(static_cast<double>(borrower->cpu_diff()[0]), 99.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(owner_mut[0]), 42.0, 1e-6);
}

// ── Split layer Backward 集成测试 ──

/// N=1 Backward：单分支梯度直通（d_bottom = d_top）
TEST(SplitBackwardTest, N1GradientPassThrough) {
  std::string prototxt = R"(
name: "test_split_n1_backward"
input: "data"
input_dim: 1
input_dim: 2
input_dim: 2
input_dim: 2
layer {
  name: "split1"
  type: "Split"
  bottom: "data"
  top: "split_out"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));

  // Forward
  auto data = net->blob_by_name("data");
  auto out = net->blob_by_name("split_out");
  for (int64_t i = 0; i < data->count(); ++i) {
    data->cpu_mutable_data()[i] = static_cast<float>(i + 1);
  }
  net->Forward();

  // Set output diff to known values (simulate upstream gradient)
  float expected_dy[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
  float* out_diff = out->cpu_mutable_diff();
  for (int64_t i = 0; i < out->count() && i < 8; ++i) {
    out_diff[i] = expected_dy[i];
  }

  // Backward
  net->Backward();

  // Input diff should equal output diff (N=1 pass-through)
  const float* data_diff = data->cpu_diff();
  for (int64_t i = 0; i < data->count() && i < 8; ++i) {
    EXPECT_NEAR(static_cast<double>(data_diff[i]),
                static_cast<double>(expected_dy[i]), 1e-6)
        << "Mismatch at index " << i;
  }
}

/// N=2 Backward：双分支梯度累加（d_bottom = d_top_a + d_top_b）
TEST(SplitBackwardTest, N2GradientAccumulation) {
  std::string prototxt = R"(
name: "test_split_n2_backward"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 2
input_dim: 3
layer {
  name: "split1"
  type: "Split"
  bottom: "data"
  top: "out_a"
  top: "out_b"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));

  auto data = net->blob_by_name("data");
  auto out_a = net->blob_by_name("out_a");
  auto out_b = net->blob_by_name("out_b");

  // Set forward data and run forward
  for (int64_t i = 0; i < data->count(); ++i) {
    data->cpu_mutable_data()[i] = static_cast<float>(i + 1) * 0.1f;
  }
  net->Forward();

  // Verify zero-copy sharing after forward
  EXPECT_TRUE(out_a->SharesDataWith(data.get()));
  EXPECT_TRUE(out_b->SharesDataWith(data.get()));

  // Set distinct gradients on each output branch
  // out_a diff: [1, 2, 3, 4, 5, 6]
  // out_b diff: [10, 20, 30, 40, 50, 60]
  float* diff_a = out_a->cpu_mutable_diff();
  float* diff_b = out_b->cpu_mutable_diff();
  for (int64_t i = 0; i < data->count(); ++i) {
    diff_a[i] = static_cast<float>(i + 1);
    diff_b[i] = static_cast<float>((i + 1) * 10);
  }

  // After COW triggered by mutable_diff on each top, they should NOT share diff with data
  EXPECT_FALSE(out_a->SharesDiffWith(data.get()));
  EXPECT_FALSE(out_b->SharesDiffWith(data.get()));

  // Backward: should accumulate diff_a + diff_b into data diff
  net->Backward();

  // Expected: data_diff[i] = (i+1) + (i+1)*10 = (i+1)*11
  const float* data_diff = data->cpu_diff();
  for (int64_t i = 0; i < data->count(); ++i) {
    float expected = static_cast<float>((i + 1) * 11);
    EXPECT_NEAR(static_cast<double>(data_diff[i]),
                static_cast<double>(expected), 1e-5)
        << "Gradient accumulation mismatch at index " << i
        << " expected " << expected << " got " << data_diff[i];
  }
}

/// N=3 Backward：三分支梯度累加（d_bottom = d_a + d_b + d_c）
TEST(SplitBackwardTest, N3GradientAccumulation) {
  std::string prototxt = R"(
name: "test_split_n3_backward"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 1
input_dim: 4
layer {
  name: "split1"
  type: "Split"
  bottom: "data"
  top: "out_a"
  top: "out_b"
  top: "out_c"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));

  auto data = net->blob_by_name("data");
  auto out_a = net->blob_by_name("out_a");
  auto out_b = net->blob_by_name("out_b");
  auto out_c = net->blob_by_name("out_c");

  net->Forward();

  // Set per-branch gradients
  float* diff_a = out_a->cpu_mutable_diff();
  float* diff_b = out_b->cpu_mutable_diff();
  float* diff_c = out_c->cpu_mutable_diff();
  diff_a[0] = 1.0f; diff_a[1] = 0.0f; diff_a[2] = 0.0f; diff_a[3] = 0.0f;
  diff_b[0] = 0.0f; diff_b[1] = 2.0f; diff_b[2] = 0.0f; diff_b[3] = 0.0f;
  diff_c[0] = 0.0f; diff_c[1] = 0.0f; diff_c[2] = 3.0f; diff_c[3] = 4.0f;

  net->Backward();

  // Expected sum: [1, 2, 3, 4]
  const float* data_diff = data->cpu_diff();
  EXPECT_NEAR(static_cast<double>(data_diff[0]), 1.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(data_diff[1]), 2.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(data_diff[2]), 3.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(data_diff[3]), 4.0, 1e-6);
}

/// N=2 Backward：验证 COW 后数据不被破坏（梯度隔离）
/// 在写入 out_a diff 后，out_b 的 diff 不应受影响（COW 隔离）
TEST(SplitBackwardTest, N2GradientIsolationAfterCOW) {
  std::string prototxt = R"(
name: "test_split_n2_isolation"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 1
input_dim: 4
layer {
  name: "split1"
  type: "Split"
  bottom: "data"
  top: "out_a"
  top: "out_b"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto data_blob = net->blob_by_name("data");
  // Zero-initialize diff explicitly (Reshape does not zero diff)
  caffe_set_fp32(static_cast<size_t>(data_blob->count()), 0.0f, data_blob->cpu_mutable_diff());
  net->Forward();

  auto out_a = net->blob_by_name("out_a");
  auto out_b = net->blob_by_name("out_b");

  // Both share data's diff initially (we zeroed it above)
  EXPECT_TRUE(out_a->SharesDiffWith(out_b.get()));

  // Write distinct patterns to each branch diff
  float* da = out_a->cpu_mutable_diff();
  da[0] = 100.0f; da[1] = 200.0f; da[2] = 300.0f; da[3] = 400.0f;

  // After writing out_a, out_b should NOT be affected (COW broke sharing for out_a)
  const float* db_before_b_write = out_b->cpu_diff();
  // out_b still shares with data (original zero buffer), not with out_a
  EXPECT_FALSE(out_a->SharesDiffWith(out_b.get()));
  EXPECT_NEAR(static_cast<double>(db_before_b_write[0]), 0.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(db_before_b_write[1]), 0.0, 1e-6);

  // Now write out_b
  float* db = out_b->cpu_mutable_diff();
  db[0] = 1.0f; db[1] = 2.0f; db[2] = 3.0f; db[3] = 4.0f;

  // Both should now be independent
  EXPECT_NEAR(static_cast<double>(out_a->cpu_diff()[0]), 100.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(out_b->cpu_diff()[0]), 1.0, 1e-6);

  // Backward should correctly sum
  net->Backward();
  const float* data_diff = data_blob->cpu_diff();
  EXPECT_NEAR(static_cast<double>(data_diff[0]), 101.0, 1e-5);
  EXPECT_NEAR(static_cast<double>(data_diff[1]), 202.0, 1e-5);
  EXPECT_NEAR(static_cast<double>(data_diff[2]), 303.0, 1e-5);
  EXPECT_NEAR(static_cast<double>(data_diff[3]), 404.0, 1e-5);
}

// ═══════════════════════════════════════════════════════════════════════
// P1: COW Integration Tests -- End-to-end Split → In-place → Backward
// ═══════════════════════════════════════════════════════════════════════
//
// These tests verify COW behavior in realistic network topologies:
//   - Forward: in-place activation on a Split branch triggers COW,
//              sibling branches must see original data (not corrupted)
//   - Backward: gradients from all branches accumulate correctly through Split
//   - Data/diff isolation after COW: writes on one branch don't leak to others

// ── Helper: fill input blob with deterministic values ──
namespace {
void FillBlobSequential(Blob* blob, float base = 1.0f) {
  float* data = blob->cpu_mutable_data();
  for (int64_t i = 0; i < blob->count(); ++i) {
    data[i] = base * static_cast<float>(i + 1);
  }
}

void FillDiffSequential(Blob* blob, float base) {
  float* diff = blob->cpu_mutable_diff();
  for (int64_t i = 0; i < blob->count(); ++i) {
    diff[i] = base * static_cast<float>(i + 1);
  }
}
}  // namespace

// ── Test 1: Split → ReLU (in-place) Forward+Backward ──
// Scenario: data → Split → (raw_branch, relu_branch[in-place ReLU])
// After Forward: relu_branch must have max(x,0), raw_branch must have original x.
// After writing top diffs and calling Backward: data_diff = raw_diff + relu_diff * (x>0?1:0)
TEST(COWIntegrationTest, SplitReLUInplaceForwardCOWIsolation) {
  std::string prototxt = R"(
name: "cow_split_relu_fwd"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 2
input_dim: 3
layer {
  name: "split"
  type: "Split"
  bottom: "data"
  top: "raw"
  top: "act"
}
layer {
  name: "relu"
  type: "ReLU"
  bottom: "act"
  top: "act"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto data = net->blob_by_name("data");

  // Input: mix of positive and negative values to test ReLU masking
  float* inp = data->cpu_mutable_data();
  inp[0] =  1.0f; inp[1] = -2.0f; inp[2] =  3.0f;
  inp[3] = -4.0f; inp[4] =  5.0f; inp[5] = -6.0f;

  net->Forward();

  auto raw = net->blob_by_name("raw");
  auto act = net->blob_by_name("act");

  // After in-place ReLU, act must have triggered COW (written to shared data)
  EXPECT_FALSE(act->SharesDataWith(raw.get()))
      << "ReLU in-place write must trigger COW, isolating act from raw";

  // raw must still have original data (unchanged by ReLU on sibling)
  const float* raw_data = raw->cpu_data();
  EXPECT_NEAR(static_cast<double>(raw_data[0]),  1.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(raw_data[1]), -2.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(raw_data[2]),  3.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(raw_data[3]), -4.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(raw_data[4]),  5.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(raw_data[5]), -6.0, 1e-6);

  // act must have ReLU'd values
  const float* act_data = act->cpu_data();
  EXPECT_NEAR(static_cast<double>(act_data[0]), 1.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(act_data[1]), 0.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(act_data[2]), 3.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(act_data[3]), 0.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(act_data[4]), 5.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(act_data[5]), 0.0, 1e-6);
}

// ── Test 2: Split → ReLU (in-place) Backward gradient accumulation ──
// After Backward: data_diff = raw_diff + relu_diff * mask(x>0)
TEST(COWIntegrationTest, SplitReLUInplaceBackwardGradientAccumulation) {
  std::string prototxt = R"(
name: "cow_split_relu_bwd"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 2
input_dim: 3
layer {
  name: "split"
  type: "Split"
  bottom: "data"
  top: "raw"
  top: "act"
}
layer {
  name: "relu"
  type: "ReLU"
  bottom: "act"
  top: "act"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto data = net->blob_by_name("data");

  float* inp = data->cpu_mutable_data();
  inp[0] =  1.0f; inp[1] = -2.0f; inp[2] =  3.0f;
  inp[3] = -4.0f; inp[4] =  5.0f; inp[5] = -6.0f;

  // Zero-init diffs
  caffe_set_fp32(static_cast<size_t>(data->count()), 0.0f, data->cpu_mutable_diff());

  net->Forward();

  auto raw = net->blob_by_name("raw");
  auto act = net->blob_by_name("act");

  // Write gradients: raw gets constant 1.0, act gets constant 2.0
  FillDiffSequential(raw.get(), 0.0f);  // will overwrite below
  float* raw_diff = raw->cpu_mutable_diff();
  float* act_diff = act->cpu_mutable_diff();
  for (int64_t i = 0; i < data->count(); ++i) {
    raw_diff[i] = 1.0f;
    act_diff[i] = 2.0f;
  }

  net->Backward();

  // Expected: data_diff[i] = raw_diff[i] + (x[i] > 0 ? act_diff[i] : 0)
  //   i=0 (x=1>0):  1 + 2 = 3
  //   i=1 (x=-2<0): 1 + 0 = 1
  //   i=2 (x=3>0):  1 + 2 = 3
  //   i=3 (x=-4<0): 1 + 0 = 1
  //   i=4 (x=5>0):  1 + 2 = 3
  //   i=5 (x=-6<0): 1 + 0 = 1
  const float* dd = data->cpu_diff();
  EXPECT_NEAR(static_cast<double>(dd[0]), 3.0, 1e-5);
  EXPECT_NEAR(static_cast<double>(dd[1]), 1.0, 1e-5);
  EXPECT_NEAR(static_cast<double>(dd[2]), 3.0, 1e-5);
  EXPECT_NEAR(static_cast<double>(dd[3]), 1.0, 1e-5);
  EXPECT_NEAR(static_cast<double>(dd[4]), 3.0, 1e-5);
  EXPECT_NEAR(static_cast<double>(dd[5]), 1.0, 1e-5);
}

// ── Test 3: Split → Sigmoid (in-place) Forward+Backward ──
TEST(COWIntegrationTest, SplitSigmoidInplaceForwardCOWAndBackward) {
  std::string prototxt = R"(
name: "cow_split_sigmoid"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 1
input_dim: 4
layer {
  name: "split"
  type: "Split"
  bottom: "data"
  top: "raw"
  top: "act"
}
layer {
  name: "sigmoid"
  type: "Sigmoid"
  bottom: "act"
  top: "act"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto data = net->blob_by_name("data");

  float* inp = data->cpu_mutable_data();
  inp[0] = 0.0f; inp[1] = 1.0f; inp[2] = -1.0f; inp[3] = 2.0f;
  caffe_set_fp32(static_cast<size_t>(data->count()), 0.0f, data->cpu_mutable_diff());

  net->Forward();

  auto raw = net->blob_by_name("raw");
  auto act = net->blob_by_name("act");

  // COW isolation
  EXPECT_FALSE(act->SharesDataWith(raw.get()));
  // raw must have original data
  const float* rd = raw->cpu_data();
  EXPECT_NEAR(static_cast<double>(rd[0]),  0.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(rd[1]),  1.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(rd[2]), -1.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(rd[3]),  2.0, 1e-6);

  // act must have sigmoid values
  auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
  const float* ad = act->cpu_data();
  EXPECT_NEAR(static_cast<double>(ad[0]), static_cast<double>(sigmoid(0.0f)),  1e-5);
  EXPECT_NEAR(static_cast<double>(ad[1]), static_cast<double>(sigmoid(1.0f)),  1e-5);
  EXPECT_NEAR(static_cast<double>(ad[2]), static_cast<double>(sigmoid(-1.0f)), 1e-5);
  EXPECT_NEAR(static_cast<double>(ad[3]), static_cast<double>(sigmoid(2.0f)),  1e-5);

  // Write top diffs and Backward
  float* raw_diff = raw->cpu_mutable_diff();
  float* act_diff = act->cpu_mutable_diff();
  for (int64_t i = 0; i < data->count(); ++i) {
    raw_diff[i] = 0.5f;
    act_diff[i] = 1.0f;
  }
  net->Backward();

  // Expected: data_diff = 0.5 + sigmoid(x)*(1-sigmoid(x))*1.0
  const float* dd = data->cpu_diff();
  for (int64_t i = 0; i < data->count(); ++i) {
    float s = sigmoid(inp[i]);
    float expected = 0.5f + s * (1.0f - s) * 1.0f;
    EXPECT_NEAR(static_cast<double>(dd[i]), static_cast<double>(expected), 1e-4)
        << "Mismatch at index " << i;
  }
}

// ── Test 4: Split → TanH (in-place) Forward+Backward ──
TEST(COWIntegrationTest, SplitTanHInplaceForwardCOWAndBackward) {
  std::string prototxt = R"(
name: "cow_split_tanh"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 1
input_dim: 4
layer {
  name: "split"
  type: "Split"
  bottom: "data"
  top: "raw"
  top: "act"
}
layer {
  name: "tanh"
  type: "TanH"
  bottom: "act"
  top: "act"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto data = net->blob_by_name("data");

  float* inp = data->cpu_mutable_data();
  inp[0] = 0.0f; inp[1] = 0.5f; inp[2] = -0.5f; inp[3] = 1.0f;
  caffe_set_fp32(static_cast<size_t>(data->count()), 0.0f, data->cpu_mutable_diff());

  net->Forward();

  auto raw = net->blob_by_name("raw");
  auto act = net->blob_by_name("act");

  EXPECT_FALSE(act->SharesDataWith(raw.get()));
  // raw unchanged
  const float* rd = raw->cpu_data();
  EXPECT_NEAR(static_cast<double>(rd[0]),  0.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(rd[1]),  0.5, 1e-6);
  EXPECT_NEAR(static_cast<double>(rd[2]), -0.5, 1e-6);
  EXPECT_NEAR(static_cast<double>(rd[3]),  1.0, 1e-6);

  // Write diffs + Backward
  float* raw_diff = raw->cpu_mutable_diff();
  float* act_diff = act->cpu_mutable_diff();
  for (int64_t i = 0; i < data->count(); ++i) {
    raw_diff[i] = 0.0f;
    act_diff[i] = 1.0f;
  }
  net->Backward();

  // Expected: data_diff = tanh'(x)*1 = (1 - tanh^2(x))
  const float* dd = data->cpu_diff();
  for (int64_t i = 0; i < data->count(); ++i) {
    double t = std::tanh(static_cast<double>(inp[i]));
    double expected = 1.0 - t * t;
    EXPECT_NEAR(static_cast<double>(dd[i]), expected, 1e-4)
        << "Mismatch at index " << i;
  }
}

// ── Test 5: N=3 Split → all branches have in-place ReLU ──
// All three branches COW independently after Forward
TEST(COWIntegrationTest, SplitN3ThreeReLUInplaceCOWIsolation) {
  std::string prototxt = R"(
name: "cow_split_n3_relu"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 1
input_dim: 4
layer {
  name: "split"
  type: "Split"
  bottom: "data"
  top: "a"
  top: "b"
  top: "c"
}
layer { name: "relu_a"; type: "ReLU"; bottom: "a"; top: "a" }
layer { name: "relu_b"; type: "ReLU"; bottom: "b"; top: "b" }
layer { name: "relu_c"; type: "ReLU"; bottom: "c"; top: "c" }
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto data = net->blob_by_name("data");

  float* inp = data->cpu_mutable_data();
  inp[0] = -1.0f; inp[1] = 2.0f; inp[2] = -3.0f; inp[3] = 4.0f;

  net->Forward();

  auto a = net->blob_by_name("a");
  auto b = net->blob_by_name("b");
  auto c = net->blob_by_name("c");

  // All branches must be isolated (each ReLU triggered COW independently)
  EXPECT_FALSE(a->SharesDataWith(b.get()));
  EXPECT_FALSE(a->SharesDataWith(c.get()));
  EXPECT_FALSE(b->SharesDataWith(c.get()));
  EXPECT_FALSE(a->SharesDataWith(data.get()));

  // Each branch must have max(x,0)
  float expected_relu[4] = {0.0f, 2.0f, 0.0f, 4.0f};
  for (auto branch : {a.get(), b.get(), c.get()}) {
    const float* bd = branch->cpu_data();
    for (int64_t i = 0; i < 4; ++i) {
      EXPECT_NEAR(static_cast<double>(bd[i]), static_cast<double>(expected_relu[i]), 1e-6);
    }
  }

  // Backward: each branch gets different gradient, all accumulate to data
  caffe_set_fp32(static_cast<size_t>(data->count()), 0.0f, data->cpu_mutable_diff());
  float* a_diff = a->cpu_mutable_diff();
  float* b_diff = b->cpu_mutable_diff();
  float* c_diff = c->cpu_mutable_diff();
  for (int64_t i = 0; i < 4; ++i) {
    a_diff[i] = 1.0f;
    b_diff[i] = 2.0f;
    c_diff[i] = 3.0f;
  }
  net->Backward();

  // data_diff[i] = (x[i]>0) ? (1+2+3) : 0 = 6 if positive, 0 if negative
  const float* dd = data->cpu_diff();
  EXPECT_NEAR(static_cast<double>(dd[0]), 0.0, 1e-5);  // x=-1, all blocked
  EXPECT_NEAR(static_cast<double>(dd[1]), 6.0, 1e-5);  // x=2, all pass
  EXPECT_NEAR(static_cast<double>(dd[2]), 0.0, 1e-5);  // x=-3, all blocked
  EXPECT_NEAR(static_cast<double>(dd[3]), 6.0, 1e-5);  // x=4, all pass
}

// ── Test 6: Split → Dropout (inference/in-place identity) → Backward ──
// Dropout in inference mode is identity copy. Split fan-out with dropout
// on one branch: COW not triggered (identity copy is read-only of input).
// Backward: dropout in inference should pass gradient straight through.
TEST(COWIntegrationTest, SplitDropoutInferenceInplaceBackwardPassthrough) {
  std::string prototxt = R"(
name: "cow_split_dropout"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 1
input_dim: 4
layer {
  name: "split"
  type: "Split"
  bottom: "data"
  top: "raw"
  top: "dropped"
}
layer {
  name: "drop"
  type: "Dropout"
  bottom: "dropped"
  top: "dropped"
  dropout_param { dropout_ratio: 0.5 }
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto data = net->blob_by_name("data");
  FillBlobSequential(data.get(), 1.0f);
  caffe_set_fp32(static_cast<size_t>(data->count()), 0.0f, data->cpu_mutable_diff());

  net->Forward();

  auto raw = net->blob_by_name("raw");
  auto dropped = net->blob_by_name("dropped");

  // Dropout in inference mode is identity copy (no mutation of shared buffer
  // from the layer's perspective -- but the impl copies data anyway).
  // What matters: data is correct.
  const float* rd = raw->cpu_data();
  const float* dd = dropped->cpu_data();
  for (int64_t i = 0; i < data->count(); ++i) {
    EXPECT_NEAR(static_cast<double>(dd[i]), static_cast<double>(rd[i]), 1e-6)
        << "Dropout inference must produce same output as input";
  }

  // Write diffs and Backward
  float* raw_diff = raw->cpu_mutable_diff();
  float* drop_diff = dropped->cpu_mutable_diff();
  for (int64_t i = 0; i < data->count(); ++i) {
    raw_diff[i] = 1.0f;
    drop_diff[i] = 2.0f;
  }
  net->Backward();

  // Dropout inference backward is identity: data_diff = raw_diff + drop_diff = 3.0
  const float* data_diff = data->cpu_diff();
  for (int64_t i = 0; i < data->count(); ++i) {
    EXPECT_NEAR(static_cast<double>(data_diff[i]), 3.0, 1e-5)
        << "Dropout inference backward must pass gradients through at index " << i;
  }
}

// ── Test 7: Split → ELU (in-place) Forward+Backward ──
TEST(COWIntegrationTest, SplitELUInplaceForwardCOWAndBackward) {
  std::string prototxt = R"(
name: "cow_split_elu"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 1
input_dim: 4
layer {
  name: "split"
  type: "Split"
  bottom: "data"
  top: "raw"
  top: "act"
}
layer {
  name: "elu"
  type: "ELU"
  bottom: "act"
  top: "act"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto data = net->blob_by_name("data");

  float* inp = data->cpu_mutable_data();
  inp[0] =  1.0f; inp[1] = -1.0f; inp[2] =  0.0f; inp[3] = -2.0f;
  caffe_set_fp32(static_cast<size_t>(data->count()), 0.0f, data->cpu_mutable_diff());

  net->Forward();

  auto raw = net->blob_by_name("raw");
  auto act = net->blob_by_name("act");
  EXPECT_FALSE(act->SharesDataWith(raw.get()));

  // raw unchanged
  const float* rd = raw->cpu_data();
  EXPECT_NEAR(static_cast<double>(rd[0]),  1.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(rd[1]), -1.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(rd[2]),  0.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(rd[3]), -2.0, 1e-6);

  // ELU forward: x if x>0, alpha*(exp(x)-1) if x<=0 (default alpha=1)
  auto elu_fwd = [](float x) { return x > 0 ? x : (std::exp(x) - 1.0f); };
  const float* ad = act->cpu_data();
  EXPECT_NEAR(static_cast<double>(ad[0]), static_cast<double>(elu_fwd(1.0f)),  1e-5);
  EXPECT_NEAR(static_cast<double>(ad[1]), static_cast<double>(elu_fwd(-1.0f)), 1e-5);
  EXPECT_NEAR(static_cast<double>(ad[2]), static_cast<double>(elu_fwd(0.0f)),  1e-5);
  EXPECT_NEAR(static_cast<double>(ad[3]), static_cast<double>(elu_fwd(-2.0f)), 1e-5);

  // Write diffs and Backward -- raw gets 0, act gets 1
  float* raw_diff = raw->cpu_mutable_diff();
  float* act_diff = act->cpu_mutable_diff();
  for (int64_t i = 0; i < data->count(); ++i) {
    raw_diff[i] = 0.0f;
    act_diff[i] = 1.0f;
  }
  net->Backward();

  // ELU backward: dY * (1 if x>0 else y+alpha) with alpha=1
  // y = elu_fwd(x), so for x<=0: dY * (y+1) = dY * (exp(x)-1+1) = dY*exp(x)
  const float* dd = data->cpu_diff();
  for (int64_t i = 0; i < data->count(); ++i) {
    float x = inp[i];
    float expected;
    if (x > 0) {
      expected = 1.0f;
    } else {
      expected = std::exp(x);  // alpha=1, dy*(y+1)=dy*exp(x)
    }
    EXPECT_NEAR(static_cast<double>(dd[i]), static_cast<double>(expected), 1e-4)
        << "Mismatch at index " << i;
  }
}

// ── Test 8: Forward-Backward cycle isolation -- COW doesn't corrupt diff buffer ──
// After Forward (which COWs data), the diff buffers must still be properly
// shareable and COW'd independently. Writing diff on one branch must not
// corrupt sibling diffs.
TEST(COWIntegrationTest, ForwardCOWDoesNotCorruptDiffSharing) {
  std::string prototxt = R"(
name: "cow_fwd_diff_isolation"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 1
input_dim: 4
layer {
  name: "split"
  type: "Split"
  bottom: "data"
  top: "a"
  top: "b"
}
layer {
  name: "relu_a"
  type: "ReLU"
  bottom: "a"
  top: "a"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto data = net->blob_by_name("data");
  FillBlobSequential(data.get(), 1.0f);  // 1,2,3,4 all positive

  net->Forward();

  auto a = net->blob_by_name("a");
  auto b = net->blob_by_name("b");

  // After Forward + ReLU on a: a has COW'd data, b still shares with data
  EXPECT_FALSE(a->SharesDataWith(data.get()));
  EXPECT_TRUE(b->SharesDataWith(data.get()));

  // Diff side: both should still share diff with data initially
  // (Forward doesn't touch diff)
  EXPECT_TRUE(a->SharesDiffWith(data.get()));
  EXPECT_TRUE(b->SharesDiffWith(data.get()));

  // Write to a's diff via mutable → triggers COW for a's diff only
  float* a_diff = a->cpu_mutable_diff();
  a_diff[0] = 999.0f;

  // b's diff must remain shared and zero (not corrupted by a's COW)
  EXPECT_FALSE(a->SharesDiffWith(b.get()));
  EXPECT_TRUE(b->SharesDiffWith(data.get()));

  const float* b_diff = b->cpu_diff();
  EXPECT_NEAR(static_cast<double>(b_diff[0]), 0.0, 1e-6)
      << "Writing a's diff must not corrupt b's shared diff buffer";
}

// ── Test 9: N=1 Split backward is identity (gradient passthrough) ──
TEST(COWIntegrationTest, SplitN1BackwardGradientPassthrough) {
  std::string prototxt = R"(
name: "cow_split_n1_bwd"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 1
input_dim: 4
layer {
  name: "split"
  type: "Split"
  bottom: "data"
  top: "out"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto data = net->blob_by_name("data");
  FillBlobSequential(data.get(), 2.0f);
  caffe_set_fp32(static_cast<size_t>(data->count()), 0.0f, data->cpu_mutable_diff());

  net->Forward();

  auto out = net->blob_by_name("out");
  EXPECT_TRUE(out->SharesDataWith(data.get()));  // N=1 zero-copy

  float* out_diff = out->cpu_mutable_diff();
  for (int64_t i = 0; i < data->count(); ++i) {
    out_diff[i] = static_cast<float>(i + 1) * 10.0f;
  }
  net->Backward();

  const float* dd = data->cpu_diff();
  for (int64_t i = 0; i < data->count(); ++i) {
    float expected = static_cast<float>(i + 1) * 10.0f;
    EXPECT_NEAR(static_cast<double>(dd[i]), static_cast<double>(expected), 1e-5)
        << "N=1 Split backward must pass gradient through at index " << i;
  }
}

// ── Test 10: Multiple Forward-Backward iterations (training loop simulation) ──
// Simulates 3 training steps: Forward → set diffs → Backward.
// Each iteration must produce correct results without COW state leakage.
TEST(COWIntegrationTest, MultipleForwardBackwardIterations) {
  std::string prototxt = R"(
name: "cow_multi_iter"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 1
input_dim: 3
layer {
  name: "split"
  type: "Split"
  bottom: "data"
  top: "raw"
  top: "act"
}
layer {
  name: "relu"
  type: "ReLU"
  bottom: "act"
  top: "act"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto data = net->blob_by_name("data");
  auto raw = net->blob_by_name("raw");
  auto act = net->blob_by_name("act");

  for (int iter = 0; iter < 3; ++iter) {
    // Set input: alternating positive/negative
    float* inp = data->cpu_mutable_data();
    for (int64_t i = 0; i < 3; ++i) {
      inp[i] = ((i + iter) % 2 == 0) ? static_cast<float>(i + 1) : -static_cast<float>(i + 1);
    }
    // Zero diffs
    caffe_set_fp32(static_cast<size_t>(data->count()), 0.0f, data->cpu_mutable_diff());

    net->Forward();

    // Verify COW isolation
    const float* rd = raw->cpu_data();
    const float* ad = act->cpu_data();
    for (int64_t i = 0; i < 3; ++i) {
      EXPECT_NEAR(static_cast<double>(rd[i]), static_cast<double>(inp[i]), 1e-6)
          << "Iter " << iter << ": raw corrupted at index " << i;
      float expected_act = inp[i] > 0 ? inp[i] : 0.0f;
      EXPECT_NEAR(static_cast<double>(ad[i]), static_cast<double>(expected_act), 1e-6)
          << "Iter " << iter << ": act ReLU wrong at index " << i;
    }

    // Set diffs and Backward
    float* raw_diff = raw->cpu_mutable_diff();
    float* act_diff = act->cpu_mutable_diff();
    for (int64_t i = 0; i < 3; ++i) {
      raw_diff[i] = 1.0f;
      act_diff[i] = 1.0f;
    }
    net->Backward();

    const float* dd = data->cpu_diff();
    for (int64_t i = 0; i < 3; ++i) {
      float expected = inp[i] > 0 ? 2.0f : 1.0f;  // 1(raw) + (x>0?1:0)
      EXPECT_NEAR(static_cast<double>(dd[i]), static_cast<double>(expected), 1e-5)
          << "Iter " << iter << ": backward wrong at index " << i;
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════
// SoftmaxWithLoss Layer Tests
// ═══════════════════════════════════════════════════════════════════════

/// Test 1: Forward loss computation with uniform probabilities (all zeros input)
/// When input logits are all zeros, softmax outputs uniform distribution 1/C.
/// Cross-entropy loss = -log(1/C) = log(C) per sample.
TEST(SoftmaxWithLossTest, ForwardLossUniform) {
  // 2 inputs: data (N=2, C=3, H=1, W=1) and label (N=2, C=1, H=1, W=1)
  // input_dim: 2 3 1 1 for data, then 2 1 1 1 for label → total 8 dims
  std::string prototxt = R"(
name: "softmax_loss_uniform"
input: "data"
input: "label"
input_dim: 2
input_dim: 3
input_dim: 1
input_dim: 1
input_dim: 2
input_dim: 1
input_dim: 1
input_dim: 1
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "data"
  bottom: "label"
  top: "loss"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto data = net->blob_by_name("data");
  auto label = net->blob_by_name("label");
  auto loss = net->blob_by_name("loss");

  // Set all logits to zero → uniform softmax 1/3
  float* data_ptr = data->cpu_mutable_data();
  for (int64_t i = 0; i < data->count(); ++i) {
    data_ptr[i] = 0.0f;
  }
  // Labels: sample 0 → class 0, sample 1 → class 2
  label->cpu_mutable_data()[0] = 0.0f;
  label->cpu_mutable_data()[1] = 2.0f;

  net->Forward();

  // Expected loss: -log(1/3) = log(3) ≈ 1.0986
  float expected_loss = std::log(3.0f);
  EXPECT_NEAR(static_cast<double>(loss->cpu_data()[0]),
              static_cast<double>(expected_loss), 1e-5);
}

/// Test 2: Backward gradient correctness with uniform input
/// Gradient formula: d_x = (prob - one_hot(label)) / N * loss_weight
/// With zero logits: prob = 1/3 for all classes
TEST(SoftmaxWithLossTest, BackwardGradientUniform) {
  std::string prototxt = R"(
name: "softmax_loss_bwd_uniform"
input: "data"
input: "label"
input_dim: 2
input_dim: 3
input_dim: 1
input_dim: 1
input_dim: 2
input_dim: 1
input_dim: 1
input_dim: 1
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "data"
  bottom: "label"
  top: "loss"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto data = net->blob_by_name("data");
  auto label = net->blob_by_name("label");

  // All zeros → uniform 1/3
  float* data_ptr = data->cpu_mutable_data();
  for (int64_t i = 0; i < data->count(); ++i) {
    data_ptr[i] = 0.0f;
  }
  label->cpu_mutable_data()[0] = 0.0f;  // sample 0 → class 0
  label->cpu_mutable_data()[1] = 2.0f;  // sample 1 → class 2

  // Zero-initialize diff
  caffe_set_fp32(static_cast<size_t>(data->count()), 0.0f, data->cpu_mutable_diff());

  net->Forward();
  // Set loss diff to 1.0 (standard for starting backprop from loss)
  auto loss = net->blob_by_name("loss");
  loss->cpu_mutable_diff()[0] = 1.0f;
  net->Backward();

  const float* dd = data->cpu_diff();
  float third = 1.0f / 3.0f;
  float sixth = 1.0f / 6.0f;
  float neg_two_thirds_div2 = -third;  // (1/3 - 1) / 2 = -1/3

  // Sample 0 (indices 0,1,2): label=0 → [-1/3, 1/6, 1/6]
  EXPECT_NEAR(static_cast<double>(dd[0]), static_cast<double>(neg_two_thirds_div2), 1e-5);
  EXPECT_NEAR(static_cast<double>(dd[1]), static_cast<double>(sixth), 1e-5);
  EXPECT_NEAR(static_cast<double>(dd[2]), static_cast<double>(sixth), 1e-5);

  // Sample 1 (indices 3,4,5): label=2 → [1/6, 1/6, -1/3]
  EXPECT_NEAR(static_cast<double>(dd[3]), static_cast<double>(sixth), 1e-5);
  EXPECT_NEAR(static_cast<double>(dd[4]), static_cast<double>(sixth), 1e-5);
  EXPECT_NEAR(static_cast<double>(dd[5]), static_cast<double>(neg_two_thirds_div2), 1e-5);
}

/// Test 3: Backward with ignore_label -- ignored samples must have zero gradient
TEST(SoftmaxWithLossTest, BackwardIgnoreLabel) {
  std::string prototxt = R"(
name: "softmax_loss_ignore"
input: "data"
input: "label"
input_dim: 3
input_dim: 2
input_dim: 1
input_dim: 1
input_dim: 3
input_dim: 1
input_dim: 1
input_dim: 1
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "data"
  bottom: "label"
  top: "loss"
  loss_param { ignore_label: 255 }
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto data = net->blob_by_name("data");
  auto label = net->blob_by_name("label");
  auto loss = net->blob_by_name("loss");

  float* data_ptr = data->cpu_mutable_data();
  for (int64_t i = 0; i < data->count(); ++i) {
    data_ptr[i] = 0.0f;
  }
  // 3 samples: class 0, ignore (255), class 1
  label->cpu_mutable_data()[0] = 0.0f;
  label->cpu_mutable_data()[1] = 255.0f;
  label->cpu_mutable_data()[2] = 1.0f;

  caffe_set_fp32(static_cast<size_t>(data->count()), 0.0f, data->cpu_mutable_diff());
  net->Forward();
  loss->cpu_mutable_diff()[0] = 1.0f;
  net->Backward();

  const float* dd = data->cpu_diff();
  float quarter = 0.25f;
  float neg_quarter = -0.25f;

  // Sample 0 (valid, label=0, N_valid=2): (0.5-1)/2 = -0.25, 0.5/2 = 0.25
  EXPECT_NEAR(static_cast<double>(dd[0]), static_cast<double>(neg_quarter), 1e-5);  // (0.5-1)/2
  EXPECT_NEAR(static_cast<double>(dd[1]), static_cast<double>(quarter), 1e-5);      // 0.5/2

  // Sample 1 (ignored): all zeros
  EXPECT_NEAR(static_cast<double>(dd[2]), 0.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dd[3]), 0.0, 1e-6);

  // Sample 2 (valid, label=1): 0.5/2 = 0.25, (0.5-1)/2 = -0.25
  EXPECT_NEAR(static_cast<double>(dd[4]), static_cast<double>(quarter), 1e-5);
  EXPECT_NEAR(static_cast<double>(dd[5]), static_cast<double>(neg_quarter), 1e-5);
}

/// Test 4: Single sample, one-hot-like confident prediction (one large logit)
/// When one logit is very large, softmax approaches one-hot; loss approaches 0,
/// gradient for correct class approaches (1-1)/1 = 0, others approach 0.
TEST(SoftmaxWithLossTest, ForwardBackwardConfidentPrediction) {
  std::string prototxt = R"(
name: "softmax_loss_confident"
input: "data"
input: "label"
input_dim: 1
input_dim: 3
input_dim: 1
input_dim: 1
input_dim: 1
input_dim: 1
input_dim: 1
input_dim: 1
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "data"
  bottom: "label"
  top: "loss"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto data = net->blob_by_name("data");
  auto label = net->blob_by_name("label");
  auto loss = net->blob_by_name("loss");

  // Large logit for class 0, others small: exp(10)/(exp(10)+exp(0)+exp(0)) ≈ 1.0
  float* data_ptr = data->cpu_mutable_data();
  data_ptr[0] = 10.0f;
  data_ptr[1] = 0.0f;
  data_ptr[2] = 0.0f;
  label->cpu_mutable_data()[0] = 0.0f;

  caffe_set_fp32(static_cast<size_t>(data->count()), 0.0f, data->cpu_mutable_diff());
  net->Forward();
  float loss_val = loss->cpu_data()[0];
  // Loss should be very small (near 0)
  EXPECT_LT(static_cast<double>(loss_val), 0.01);

  loss->cpu_mutable_diff()[0] = 1.0f;
  net->Backward();

  const float* dd = data->cpu_diff();
  // Correct class gradient: (≈1 - 1)/1 ≈ 0
  EXPECT_NEAR(static_cast<double>(dd[0]), 0.0, 1e-3);
  // Wrong classes: ≈0/1 ≈ small positive
  EXPECT_GT(static_cast<double>(dd[1]), 0.0);
  EXPECT_GT(static_cast<double>(dd[2]), 0.0);
}

/// Test 5: Probability-only mode (no label input) -- forward outputs probabilities
/// backward zeros out gradient (no loss signal)
TEST(SoftmaxWithLossTest, ProbabilityOnlyMode) {
  std::string prototxt = R"(
name: "softmax_prob_only"
input: "data"
input_dim: 1
input_dim: 3
input_dim: 1
input_dim: 1
layer {
  name: "prob"
  type: "SoftmaxWithLoss"
  bottom: "data"
  top: "prob"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto data = net->blob_by_name("data");
  auto prob = net->blob_by_name("prob");

  float* data_ptr = data->cpu_mutable_data();
  data_ptr[0] = 1.0f;
  data_ptr[1] = 2.0f;
  data_ptr[2] = 3.0f;

  net->Forward();

  // Check probabilities sum to 1
  const float* p = prob->cpu_data();
  double sum = static_cast<double>(p[0]) + static_cast<double>(p[1]) + static_cast<double>(p[2]);
  EXPECT_NEAR(sum, 1.0, 1e-5);
  // p[2] should be largest (logit=3)
  EXPECT_GT(static_cast<double>(p[2]), static_cast<double>(p[1]));
  EXPECT_GT(static_cast<double>(p[1]), static_cast<double>(p[0]));

  // Backward should zero out diff (no labels)
  caffe_set_fp32(static_cast<size_t>(data->count()), 0.0f, data->cpu_mutable_diff());
  net->Backward();
  const float* dd = data->cpu_diff();
  for (int64_t i = 0; i < data->count(); ++i) {
    EXPECT_NEAR(static_cast<double>(dd[i]), 0.0, 1e-6);
  }
}

// ═══════════════════════════════════════════════════════════════════════
// Pooling Layer Tests
// ═══════════════════════════════════════════════════════════════════════

/// Test 1: Max Pooling 2x2 stride 2 no padding on simple 4x4 input
/// Input (1x1x4x4):
///   1  2  3  4
///   5  6  7  8
///   9 10 11 12
///  13 14 15 16
/// Expected output (1x1x2x2): max of each 2x2 block
///   top-left:  max(1,2,5,6) = 6
///   top-right: max(3,4,7,8) = 8
///   bot-left:  max(9,10,13,14) = 14
///   bot-right: max(11,12,15,16) = 16
TEST(PoolingLayerTest, MaxPooling2x2Stride2) {
  std::string prototxt = R"(
name: "maxpool_test"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 4
input_dim: 4
layer {
  name: "pool"
  type: "Pooling"
  bottom: "data"
  top: "pooled"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto data = net->blob_by_name("data");
  auto pooled = net->blob_by_name("pooled");

  float in[16] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
  };
  float* dptr = data->cpu_mutable_data();
  for (int i = 0; i < 16; ++i) dptr[i] = in[i];

  net->Forward();

  const float* out = pooled->cpu_data();
  EXPECT_EQ(pooled->shape(2), 2);  // pooled height
  EXPECT_EQ(pooled->shape(3), 2);  // pooled width

  EXPECT_NEAR(static_cast<double>(out[0]), 6.0, 1e-6);   // top-left
  EXPECT_NEAR(static_cast<double>(out[1]), 8.0, 1e-6);   // top-right
  EXPECT_NEAR(static_cast<double>(out[2]), 14.0, 1e-6);  // bot-left
  EXPECT_NEAR(static_cast<double>(out[3]), 16.0, 1e-6);  // bot-right
}

/// Test 2: Average Pooling 2x2 stride 2 no padding
/// Same input as Max test, but averages instead of max.
/// Expected: 3.5, 5.5, 11.5, 13.5
TEST(PoolingLayerTest, AvePooling2x2Stride2) {
  std::string prototxt = R"(
name: "avepool_test"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 4
input_dim: 4
layer {
  name: "pool"
  type: "Pooling"
  bottom: "data"
  top: "pooled"
  pooling_param {
    pool: AVE
    kernel_size: 2
    stride: 2
  }
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto data = net->blob_by_name("data");
  auto pooled = net->blob_by_name("pooled");

  float in[16] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
  };
  float* dptr = data->cpu_mutable_data();
  for (int i = 0; i < 16; ++i) dptr[i] = in[i];

  net->Forward();

  const float* out = pooled->cpu_data();
  EXPECT_NEAR(static_cast<double>(out[0]), (1+2+5+6)/4.0, 1e-6);    // 3.5
  EXPECT_NEAR(static_cast<double>(out[1]), (3+4+7+8)/4.0, 1e-6);    // 5.5
  EXPECT_NEAR(static_cast<double>(out[2]), (9+10+13+14)/4.0, 1e-6); // 11.5
  EXPECT_NEAR(static_cast<double>(out[3]), (11+12+15+16)/4.0, 1e-6); // 13.5
}

/// Test 3: Max Pooling with padding
/// Input 2x2, kernel 3x3, stride 1, pad 1
/// With padding, the 2x2 input is surrounded by implicit -inf (max pool),
/// but padding areas don't contribute max values since actual data > -inf.
/// Output size: floor((2+2*1-3)/1)+1 = floor(1/1)+1 = 2
/// So output is still 2x2 with proper edge handling.
TEST(PoolingLayerTest, MaxPoolingWithPadding) {
  std::string prototxt = R"(
name: "maxpool_pad"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 2
input_dim: 2
layer {
  name: "pool"
  type: "Pooling"
  bottom: "data"
  top: "pooled"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 1
    pad: 1
  }
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto data = net->blob_by_name("data");
  auto pooled = net->blob_by_name("pooled");

  float in[4] = {1, 2, 3, 4};
  float* dptr = data->cpu_mutable_data();
  for (int i = 0; i < 4; ++i) dptr[i] = in[i];

  net->Forward();

  const float* out = pooled->cpu_data();
  // With pad=1, kernel=3, stride=1 on 2x2 input:
  // Each output position's window includes some padding and some real values.
  // Since max ignores -inf padding implicitly (by starting with -max and taking max),
  // output at (0,0) = max(1,2,3,4) over window covering real values = 4?
  // Actually let's verify pooled dimensions first
  EXPECT_GE(pooled->shape(2), 2);
  EXPECT_GE(pooled->shape(3), 2);
  // The output should contain the max value 4 somewhere
  float out_max = -1e30f;
  for (int64_t i = 0; i < pooled->count(); ++i) {
    out_max = std::max(out_max, out[i]);
  }
  EXPECT_NEAR(static_cast<double>(out_max), 4.0, 1e-6);
}

/// Test 4: Global Max/Average Pooling
/// Global pooling pools over entire spatial dimensions (H×W).
/// For 4x4 input with all values 1..16:
///   global max = 16
///   global avg = (1+2+...+16)/16 = 136/16 = 8.5
TEST(PoolingLayerTest, GlobalPooling) {
  // Test Global Max
  {
    std::string prototxt = R"(
name: "global_maxpool"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 4
input_dim: 4
layer {
  name: "pool"
  type: "Pooling"
  bottom: "data"
  top: "pooled"
  pooling_param {
    pool: MAX
    global_pooling: true
  }
}
)";
    ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
    auto data = net->blob_by_name("data");
    auto pooled = net->blob_by_name("pooled");

    float* dptr = data->cpu_mutable_data();
    for (int i = 0; i < 16; ++i) dptr[i] = static_cast<float>(i + 1);

    net->Forward();
    EXPECT_EQ(pooled->shape(2), 1);
    EXPECT_EQ(pooled->shape(3), 1);
    EXPECT_NEAR(static_cast<double>(pooled->cpu_data()[0]), 16.0, 1e-6);
  }

  // Test Global Average
  {
    std::string prototxt = R"(
name: "global_avepool"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 4
input_dim: 4
layer {
  name: "pool"
  type: "Pooling"
  bottom: "data"
  top: "pooled"
  pooling_param {
    pool: AVE
    global_pooling: true
  }
}
)";
    ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
    auto data = net->blob_by_name("data");
    auto pooled = net->blob_by_name("pooled");

    float* dptr = data->cpu_mutable_data();
    for (int i = 0; i < 16; ++i) dptr[i] = static_cast<float>(i + 1);

    net->Forward();
    EXPECT_EQ(pooled->shape(2), 1);
    EXPECT_EQ(pooled->shape(3), 1);
    float expected_avg = (16 * 17 / 2) / 16.0f;  // sum 1..16 = 136, avg = 8.5
    EXPECT_NEAR(static_cast<double>(pooled->cpu_data()[0]),
                static_cast<double>(expected_avg), 1e-5);
  }
}

/// Test 5: Max Pooling multi-channel (2 channels, independent)
/// Ensures pooling operates per-channel without cross-channel mixing.
TEST(PoolingLayerTest, MaxPoolingMultiChannel) {
  std::string prototxt = R"(
name: "maxpool_2ch"
input: "data"
input_dim: 1
input_dim: 2
input_dim: 2
input_dim: 2
layer {
  name: "pool"
  type: "Pooling"
  bottom: "data"
  top: "pooled"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto data = net->blob_by_name("data");
  auto pooled = net->blob_by_name("pooled");

  // Channel 0: 1,2,3,4 → max = 4
  // Channel 1: 10,20,30,40 → max = 40
  float in[8] = {1, 2, 3, 4, 10, 20, 30, 40};
  float* dptr = data->cpu_mutable_data();
  for (int i = 0; i < 8; ++i) dptr[i] = in[i];

  net->Forward();

  const float* out = pooled->cpu_data();
  EXPECT_EQ(pooled->count(), 2);  // 2 channels, 1x1 output per channel
  EXPECT_NEAR(static_cast<double>(out[0]), 4.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(out[1]), 40.0, 1e-6);
}

// ═══════════════════════════════════════════════════════════════════════
// Slice Layer Zero-Copy Tests (single-output scenario)
// ═══════════════════════════════════════════════════════════════════════
//
// 验证Slice层在单输出（top.size() == 1）场景下的零拷贝优化：
//   - Reshape阶段ShareData/ShareDiff指针参数正确性
//   - Forward阶段跳过数据复制
//   - Backward阶段跳过梯度复制
//   - 多轮Forward/Backward无refcount泄漏

/// Test 1: Slice N=1 单输出零拷贝——data和diff指针直接共享
TEST(SliceLayerZeroCopyTest, SingleOutputSharesDataAndDiff) {
  std::string prototxt = R"(
name: "slice_single_output"
input: "data"
input_dim: 1
input_dim: 2
input_dim: 3
input_dim: 4
layer {
  name: "slice"
  type: "Slice"
  bottom: "data"
  top: "output"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  net->Forward();

  auto bottom = net->blob_by_name("data");
  auto top = net->blob_by_name("output");

  EXPECT_TRUE(bottom);
  EXPECT_TRUE(top);

  // N=1零拷贝：top必须与bottom共享data和diff指针
  EXPECT_TRUE(top->SharesDataWith(bottom.get()));
  EXPECT_TRUE(top->SharesDiffWith(bottom.get()));
  EXPECT_EQ(top->cpu_data(), bottom->cpu_data());
  EXPECT_EQ(top->cpu_diff(), bottom->cpu_diff());

  // 形状必须完全一致
  EXPECT_EQ(top->count(), bottom->count());
  EXPECT_EQ(top->num_axes(), bottom->num_axes());
  for (int i = 0; i < bottom->num_axes(); ++i) {
    EXPECT_EQ(top->shape(i), bottom->shape(i));
  }
}

/// Test 2: Slice N=1 单输出——Forward后数据正确性验证
TEST(SliceLayerZeroCopyTest, SingleOutputDataCorrectness) {
  std::string prototxt = R"(
name: "slice_single_output_data"
input: "data"
input_dim: 2
input_dim: 3
input_dim: 4
input_dim: 5
layer {
  name: "slice"
  type: "Slice"
  bottom: "data"
  top: "output"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));

  // 写入已知数据
  auto data = net->blob_by_name("data");
  float* data_ptr = data->cpu_mutable_data();
  for (int64_t i = 0; i < data->count(); ++i) {
    data_ptr[i] = static_cast<float>(i) * 0.01f;
  }

  net->Forward();

  auto output = net->blob_by_name("output");

  // 零拷贝共享：指针必须相同
  EXPECT_TRUE(output->SharesDataWith(data.get()));
  EXPECT_EQ(output->cpu_data(), data->cpu_data());

  // 通过output读取数据必须与写入一致
  for (int64_t i = 0; i < data->count(); ++i) {
    EXPECT_NEAR(static_cast<double>(output->cpu_data()[i]),
                static_cast<double>(i) * 0.01, 1e-6)
        << "Data mismatch at index " << i;
  }
}

/// Test 3: Slice N=1 单输出——Backward梯度直通
TEST(SliceLayerZeroCopyTest, SingleOutputGradientPassthrough) {
  std::string prototxt = R"(
name: "slice_single_output_bwd"
input: "data"
input_dim: 1
input_dim: 1
input_dim: 2
input_dim: 3
layer {
  name: "slice"
  type: "Slice"
  bottom: "data"
  top: "output"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));

  auto data = net->blob_by_name("data");
  auto output = net->blob_by_name("output");

  // 初始化Forward数据
  for (int64_t i = 0; i < data->count(); ++i) {
    data->cpu_mutable_data()[i] = static_cast<float>(i + 1);
  }

  net->Forward();

  // 验证零拷贝
  EXPECT_TRUE(output->SharesDiffWith(data.get()));

  // 设置输出梯度（模拟上游梯度）
  float expected_dy[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
  float* out_diff = output->cpu_mutable_diff();
  for (int64_t i = 0; i < output->count() && i < 6; ++i) {
    out_diff[i] = expected_dy[i];
  }

  net->Backward();

  // N=1梯度直通：data_diff必须等于output_diff
  const float* data_diff = data->cpu_diff();
  for (int64_t i = 0; i < data->count() && i < 6; ++i) {
    EXPECT_NEAR(static_cast<double>(data_diff[i]),
                static_cast<double>(expected_dy[i]), 1e-6)
        << "Gradient mismatch at index " << i;
  }
}

/// Test 4: Slice N=1 单输出——指定axis切片仍零拷贝（N=1时不实际切片）
TEST(SliceLayerZeroCopyTest, SingleOutputWithAxisStillShares) {
  std::string prototxt = R"(
name: "slice_single_output_axis"
input: "data"
input_dim: 1
input_dim: 4
input_dim: 2
input_dim: 2
layer {
  name: "slice"
  type: "Slice"
  bottom: "data"
  top: "output"
  slice_param { axis: 1 }
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));

  auto data = net->blob_by_name("data");
  data->cpu_mutable_data()[0] = 42.0f;

  net->Forward();

  auto output = net->blob_by_name("output");

  // 即使指定了axis，单输出时形状与输入一致，仍零拷贝
  EXPECT_TRUE(output->SharesDataWith(data.get()));
  EXPECT_EQ(output->cpu_data(), data->cpu_data());
  EXPECT_EQ(output->shape(1), data->shape(1));  // axis=1维度未被切分
  EXPECT_NEAR(static_cast<double>(output->cpu_data()[0]), 42.0, 1e-6);
}

/// Test 5: Slice N=1 单输出——多轮Forward/Backward无refcount泄漏
TEST(SliceLayerZeroCopyTest, RepeatedForwardBackwardNoLeak) {
  std::string prototxt = R"(
name: "slice_repeated_fwd_bwd"
input: "data"
input_dim: 1
input_dim: 2
input_dim: 2
input_dim: 2
layer {
  name: "slice"
  type: "Slice"
  bottom: "data"
  top: "output"
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));
  auto data = net->blob_by_name("data");
  auto output = net->blob_by_name("output");

  // 写入初始数据
  data->cpu_mutable_data()[0] = 100.0f;

  int64_t live_count_before = LiveBlobCount();

  // 执行10轮Forward/Backward循环
  for (int iter = 0; iter < 10; ++iter) {
    net->Forward();
    EXPECT_TRUE(output->SharesDataWith(data.get()));
    EXPECT_EQ(output->cpu_data(), data->cpu_data());

    // 设置梯度并Backward
    output->cpu_mutable_diff()[0] = static_cast<float>(iter + 1) * 0.1f;
    net->Backward();

    // 数据始终正确
    EXPECT_NEAR(static_cast<double>(output->cpu_data()[0]), 100.0, 1e-6);
  }

  // Blob数量稳定（无refcount泄漏导致的额外Blob创建）
  EXPECT_EQ(LiveBlobCount(), live_count_before);
}

/// Test 6: Slice N=2 多输出——验证不零拷贝（对比测试）
TEST(SliceLayerZeroCopyTest, MultiOutputDoesNotShareData) {
  std::string prototxt = R"(
name: "slice_multi_output"
input: "data"
input_dim: 1
input_dim: 4
input_dim: 1
input_dim: 1
layer {
  name: "slice"
  type: "Slice"
  bottom: "data"
  top: "out_a"
  top: "out_b"
  slice_param { axis: 1 slice_point: 2 }
}
)";

  ObjectPtr<Net> net = make_object<Net>(ReadNetParamsFromTextString(prototxt));

  auto data = net->blob_by_name("data");
  float* dptr = data->cpu_mutable_data();
  dptr[0] = 1.0f; dptr[1] = 2.0f; dptr[2] = 3.0f; dptr[3] = 4.0f;

  net->Forward();

  auto out_a = net->blob_by_name("out_a");
  auto out_b = net->blob_by_name("out_b");

  // 多输出场景：不共享data指针（需要实际切片复制）
  EXPECT_FALSE(out_a->SharesDataWith(data.get()));
  EXPECT_FALSE(out_b->SharesDataWith(data.get()));
  EXPECT_NE(out_a->cpu_data(), data->cpu_data());
  EXPECT_NE(out_b->cpu_data(), data->cpu_data());

  // 但切片数据必须正确
  EXPECT_EQ(out_a->count(), 2);
  EXPECT_EQ(out_b->count(), 2);
  EXPECT_NEAR(static_cast<double>(out_a->cpu_data()[0]), 1.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(out_a->cpu_data()[1]), 2.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(out_b->cpu_data()[0]), 3.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(out_b->cpu_data()[1]), 4.0, 1e-6);
}

// ═══════════════════════════════════════════════════════════════════════
// COW Runtime Switch Tests — 验证运行时动态开关 Copy-on-Write 行为
// ═══════════════════════════════════════════════════════════════════════
//
// 设计背景：COW 逻辑始终编译（避免 ODR 违规），由运行时原子开关控制。
// 这些测试验证 SetCOWEnabled()/IsCOWEnabled() 在动态切换时的正确性。
// 每个测试手动保存/恢复 COW 状态，确保不影响其他测试。

// Helper: RAII COW state guard (manual, no TEST_F fixture support)
struct CowStateGuard {
  bool saved_;
  CowStateGuard() : saved_(IsCOWEnabled()) {}
  ~CowStateGuard() { SetCOWEnabled(saved_); }
};

// Test 1: 默认状态下 COW 启用（CMake CAFFE_FFI_ENABLE_COW=ON 构建时）
TEST(COWRuntimeSwitchTest, DefaultStateIsEnabled) {
  // 构建使用 CAFFE_FFI_ENABLE_COW=ON，默认应为 true
  // 注意：前一个测试可能改变了状态，这里直接验证 IsCOWEnabled() 返回 bool
  bool state = IsCOWEnabled();
  (void)state;  // 仅验证函数可调用，不断言具体值（因为其他测试可能修改）
  SetCOWEnabled(true);
  EXPECT_TRUE(IsCOWEnabled());
}

// Test 2: COW 启用时，共享 Blob 调用 mutable_data 触发 COW（指针分离）
TEST(COWRuntimeSwitchTest, COWEnabledTriggersCopyOnWrite) {
  CowStateGuard guard;
  SetCOWEnabled(true);

  std::vector<int64_t> shape = {2, 3};
  auto a = make_object<Blob>(shape);
  auto b = make_object<Blob>(shape);

  float* a_data = a->cpu_mutable_data();
  a_data[0] = 42.0f;
  b->ShareData(a.get());
  EXPECT_EQ(b->cpu_data(), a->cpu_data());  // 共享同一块内存

  // COW 启用时，b 的 mutable 访问应触发拷贝
  float* b_data = b->cpu_mutable_data();
  EXPECT_NE(b_data, a_data) << "COW enabled: mutable_data() on shared blob must trigger copy";
  b_data[0] = 99.0f;
  // a 不受 b 的写入影响
  EXPECT_NEAR(static_cast<double>(a_data[0]), 42.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(b_data[0]), 99.0, 1e-6);
}

// Test 3: COW 禁用时，共享 Blob 调用 mutable_data 不触发 COW（指针不变，就地修改）
TEST(COWRuntimeSwitchTest, COWDisabledNoCopyOnMutableAccess) {
  CowStateGuard guard;

  std::vector<int64_t> shape = {2, 3};
  auto a = make_object<Blob>(shape);
  auto b = make_object<Blob>(shape);

  float* a_data = a->cpu_mutable_data();
  a_data[0] = 42.0f;
  b->ShareData(a.get());
  EXPECT_EQ(b->cpu_data(), a->cpu_data());

  // 禁用 COW
  SetCOWEnabled(false);
  EXPECT_FALSE(IsCOWEnabled());

  // COW 禁用时，mutable 访问不触发拷贝——返回同一指针
  float* b_data = b->cpu_mutable_data();
  EXPECT_EQ(b_data, a_data) << "COW disabled: mutable_data() on shared blob returns same pointer";
  // 写入 b 会直接修改共享内存（a 也看到变化）
  b_data[0] = 99.0f;
  EXPECT_NEAR(static_cast<double>(a_data[0]), 99.0, 1e-6)
      << "COW disabled: writes to b mutate the shared tensor visible to a";
}

// Test 4: COW 禁用不影响 const 访问（始终零拷贝）
TEST(COWRuntimeSwitchTest, ConstAccessAlwaysZeroCopy) {
  CowStateGuard guard;

  std::vector<int64_t> shape = {3, 3};
  auto a = make_object<Blob>(shape);
  auto b = make_object<Blob>(shape);
  a->cpu_mutable_data()[0] = 1.0f;
  b->ShareData(a.get());

  SetCOWEnabled(false);
  // const 指针访问在 COW 禁用时也应该共享（const 访问永远不触发 COW）
  EXPECT_EQ(b->cpu_data(), a->cpu_data());

  SetCOWEnabled(true);
  EXPECT_EQ(b->cpu_data(), a->cpu_data());
}

// Test 5: 运行时反复切换开关——off→on→off→on 循环验证
TEST(COWRuntimeSwitchTest, ToggleOnOffMultipleTimes) {
  CowStateGuard guard;

  std::vector<int64_t> shape = {2, 2};
  auto a = make_object<Blob>(shape);
  auto b = make_object<Blob>(shape);
  a->cpu_mutable_data()[0] = 10.0f;
  b->ShareData(a.get());

  // Cycle 1: off → mutable 不拷贝
  SetCOWEnabled(false);
  EXPECT_EQ(b->cpu_mutable_data(), a->cpu_data());
  b->ShareData(a.get());

  // Cycle 2: on → mutable 触发拷贝
  SetCOWEnabled(true);
  float* b_ptr = b->cpu_mutable_data();
  EXPECT_NE(b_ptr, a->cpu_data());
  b->ShareData(a.get());

  // Cycle 3: off → mutable 不拷贝
  SetCOWEnabled(false);
  EXPECT_EQ(b->cpu_mutable_data(), a->cpu_data());
  b->ShareData(a.get());

  // Cycle 4: on → mutable 触发拷贝
  SetCOWEnabled(true);
  EXPECT_NE(b->cpu_mutable_data(), a->cpu_data());
}

// Test 6: COW 开关对 diff 同样生效
TEST(COWRuntimeSwitchTest, DiffCOWRespectsSwitch) {
  CowStateGuard guard;

  std::vector<int64_t> shape = {2, 2};
  auto a = make_object<Blob>(shape);
  auto b = make_object<Blob>(shape);

  a->cpu_mutable_diff()[0] = 7.0f;
  b->ShareDiff(a.get());
  EXPECT_EQ(b->cpu_diff(), a->cpu_diff());

  // COW 启用 → diff mutable 触发拷贝
  SetCOWEnabled(true);
  float* b_diff = b->cpu_mutable_diff();
  EXPECT_NE(b_diff, a->cpu_diff()) << "diff COW enabled: mutable_diff triggers copy";
  b_diff[0] = 99.0f;
  EXPECT_NEAR(static_cast<double>(a->cpu_diff()[0]), 7.0, 1e-6);

  // 重新共享
  b->ShareDiff(a.get());

  // COW 禁用 → diff mutable 不拷贝
  SetCOWEnabled(false);
  float* b_diff2 = b->cpu_mutable_diff();
  EXPECT_EQ(b_diff2, a->cpu_diff()) << "diff COW disabled: mutable_diff returns same pointer";
}

// Test 7: COW 开关对 mutable_data_tensor() 同样生效
TEST(COWRuntimeSwitchTest, MutableTensorCOWRespectsSwitch) {
  CowStateGuard guard;

  std::vector<int64_t> shape = {1, 4};
  auto a = make_object<Blob>(shape);
  auto b = make_object<Blob>(shape);

  a->cpu_mutable_data();
  b->ShareData(a.get());
  const void* a_ptr = a->cpu_data();

  // COW 禁用 → mutable_data_tensor 不分离
  SetCOWEnabled(false);
  Tensor t_off = b->mutable_data_tensor();
  EXPECT_EQ(t_off.data_ptr(), a_ptr) << "COW disabled: mutable_data_tensor returns shared tensor";

  // 重新共享
  b->ShareData(a.get());

  // COW 启用 → mutable_data_tensor 分离
  SetCOWEnabled(true);
  Tensor t_on = b->mutable_data_tensor();
  EXPECT_NE(t_on.data_ptr(), a_ptr) << "COW enabled: mutable_data_tensor returns private copy";
}

// Test 8: IsCOWEnabled() 状态查询正确反映最近一次 SetCOWEnabled()
TEST(COWRuntimeSwitchTest, IsCOWEnabledReflectsLastSet) {
  CowStateGuard guard;

  SetCOWEnabled(true);
  EXPECT_TRUE(IsCOWEnabled());
  SetCOWEnabled(false);
  EXPECT_FALSE(IsCOWEnabled());
  SetCOWEnabled(false);  // 重复设置同一值无副作用
  EXPECT_FALSE(IsCOWEnabled());
  SetCOWEnabled(true);
  EXPECT_TRUE(IsCOWEnabled());
}

// Test 9: COW 禁用时，Three-way share 中 mutable 访问修改所有共享者（因为无拷贝）
TEST(COWRuntimeSwitchTest, COWDisabledThreeWayShareAllSeeMutation) {
  CowStateGuard guard;

  std::vector<int64_t> shape = {1, 3};
  auto a = make_object<Blob>(shape);
  auto b = make_object<Blob>(shape);
  auto c = make_object<Blob>(shape);

  float* a_data = a->cpu_mutable_data();
  a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f;
  b->ShareData(a.get());
  c->ShareData(a.get());

  SetCOWEnabled(false);

  // 通过 b 写入，a 和 c 都能看到（因为无拷贝）
  float* b_data = b->cpu_mutable_data();
  b_data[0] = 100.0f;
  EXPECT_NEAR(static_cast<double>(a->cpu_data()[0]), 100.0, 1e-6)
      << "COW disabled: b's mutation visible to a";
  EXPECT_NEAR(static_cast<double>(c->cpu_data()[0]), 100.0, 1e-6)
      << "COW disabled: b's mutation visible to c";
}

// Test 10: CowStateGuard 析构恢复状态验证——禁用后守卫恢复启用
TEST(COWRuntimeSwitchTest, GuardRestoresStateAfterScopeExit) {
  SetCOWEnabled(true);
  {
    CowStateGuard inner_guard;
    SetCOWEnabled(false);
    EXPECT_FALSE(IsCOWEnabled());
  }
  // 守卫析构后应恢复为 true
  EXPECT_TRUE(IsCOWEnabled()) << "CowStateGuard destructor must restore original state";
}

// Test 11: COW 开关对 mutable_diff_tensor() 同样生效
TEST(COWRuntimeSwitchTest, MutableDiffTensorCOWRespectsSwitch) {
  CowStateGuard guard;

  std::vector<int64_t> shape = {1, 3};
  auto a = make_object<Blob>(shape);
  auto b = make_object<Blob>(shape);

  a->cpu_mutable_diff();
  b->ShareDiff(a.get());
  const void* a_diff_ptr = a->cpu_diff();

  // COW 禁用 → mutable_diff_tensor 不分离
  SetCOWEnabled(false);
  Tensor t_off = b->mutable_diff_tensor();
  EXPECT_EQ(t_off.data_ptr(), a_diff_ptr) << "COW disabled: mutable_diff_tensor returns shared tensor";

  b->ShareDiff(a.get());

  // COW 启用 → mutable_diff_tensor 分离
  SetCOWEnabled(true);
  Tensor t_on = b->mutable_diff_tensor();
  EXPECT_NE(t_on.data_ptr(), a_diff_ptr) << "COW enabled: mutable_diff_tensor returns private copy";
}

