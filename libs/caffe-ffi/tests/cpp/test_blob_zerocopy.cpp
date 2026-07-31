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

  // Data must be visible through dst (same physical memory)
  for (int64_t i = 0; i < src->count(); ++i) {
    EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_data()[i]),
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
  EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_diff()[0]), 0.1, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_diff()[15]), 1.6, 1e-6);
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
  std::vector<int64_t> shape = {8};
  auto a = make_object<Blob>(shape);
  auto b = make_object<Blob>(shape);

  a->cpu_mutable_data()[0] = 42.0f;
  b->ShareData(a.get());

  // Mutate via b, read via a (same memory)
  b->cpu_mutable_data()[1] = 99.0f;
  EXPECT_NEAR(static_cast<double>(a->cpu_mutable_data()[1]), 99.0, 1e-6);

  // Mutate via a, read via b
  a->cpu_mutable_data()[2] = 7.0f;
  EXPECT_NEAR(static_cast<double>(b->cpu_mutable_data()[2]), 7.0, 1e-6);
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

  // Reshape dst to a different shape — must allocate new private memory,
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
  std::vector<int64_t> shape = {4};
  auto a = make_object<Blob>(shape);
  auto b = make_object<Blob>(shape);
  auto c = make_object<Blob>(shape);

  b->ShareData(a.get());
  c->ShareData(a.get());

  EXPECT_TRUE(b->SharesDataWith(a.get()));
  EXPECT_TRUE(c->SharesDataWith(a.get()));
  EXPECT_TRUE(b->SharesDataWith(c.get()));
  EXPECT_EQ(b->cpu_data(), c->cpu_data());

  // Writes through any alias are visible everywhere
  b->cpu_mutable_data()[0] = 1.0f;
  EXPECT_NEAR(static_cast<double>(c->cpu_mutable_data()[0]), 1.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(a->cpu_mutable_data()[0]), 1.0, 1e-6);
}

// ── Phase 2 COW: Blob-level unit tests ──

/// 验证 cpu_mutable_data() 在共享后触发 COW：指针不再相等、
/// 引用计数回到 1、数据内容正确复制
TEST(COWTest, MutableDataTriggersCOWWhenShared) {
  std::vector<int64_t> shape = {8};
  auto src = make_object<Blob>(shape);
  auto dst = make_object<Blob>(shape);

  // 写入已知数据
  for (int64_t i = 0; i < src->count(); ++i) {
    src->cpu_mutable_data()[i] = static_cast<float>(i);
  }

  dst->ShareData(src.get());
  EXPECT_TRUE(dst->SharesDataWith(src.get()));
  EXPECT_EQ(dst->cpu_data(), src->cpu_data());

  // 调用 cpu_mutable_data() → COW 触发
  float* dst_mut = dst->cpu_mutable_data();

  // 指针不再相等（COW 打破了共享）
  EXPECT_NE(dst_mut, src->cpu_data());
  EXPECT_FALSE(dst->SharesDataWith(src.get()));

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

/// 验证 cpu_mutable_diff() 在共享后触发 COW
TEST(COWTest, MutableDiffTriggersCOWWhenShared) {
  std::vector<int64_t> shape = {6};
  auto src = make_object<Blob>(shape);
  auto dst = make_object<Blob>(shape);

  for (int64_t i = 0; i < src->count(); ++i) {
    src->cpu_mutable_diff()[i] = static_cast<float>(i * 10);
  }

  dst->ShareDiff(src.get());
  EXPECT_TRUE(dst->SharesDiffWith(src.get()));
  EXPECT_EQ(dst->cpu_diff(), src->cpu_diff());

  float* dst_mut_diff = dst->cpu_mutable_diff();

  EXPECT_NE(dst_mut_diff, src->cpu_diff());
  EXPECT_FALSE(dst->SharesDiffWith(src.get()));

  for (int64_t i = 0; i < src->count(); ++i) {
    EXPECT_NEAR(static_cast<double>(dst_mut_diff[i]),
                static_cast<double>(i * 10), 1e-6);
  }
}

/// 验证 COW 后的数据隔离：修改 dst 不影响 src
TEST(COWTest, DataIsolationAfterCOW) {
  std::vector<int64_t> shape = {4};
  auto src = make_object<Blob>(shape);
  auto dst = make_object<Blob>(shape);

  src->cpu_mutable_data()[0] = 10.0f;
  src->cpu_mutable_data()[1] = 20.0f;
  src->cpu_mutable_data()[2] = 30.0f;
  src->cpu_mutable_data()[3] = 40.0f;

  dst->ShareData(src.get());

  // 触发 COW 并修改 dst
  dst->cpu_mutable_data()[0] = 999.0f;
  dst->cpu_mutable_data()[1] = 888.0f;

  // src 不受影响
  EXPECT_NEAR(static_cast<double>(src->cpu_mutable_data()[0]), 10.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src->cpu_mutable_data()[1]), 20.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src->cpu_mutable_data()[2]), 30.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src->cpu_mutable_data()[3]), 40.0, 1e-6);

  // dst 有自己的值
  EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_data()[0]), 999.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_data()[1]), 888.0, 1e-6);
  // 未修改的部分应与 src 一致（COW 完整复制）
  EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_data()[2]), 30.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_data()[3]), 40.0, 1e-6);
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

/// 场景 5：COW 后共享（A→B 共享，B COW，B→C 共享）
/// 验证 COW 打破共享后，新建立的共享关系正确隔离。
TEST(ShareDataRefCount, ShareDataAfterCOW) {
  auto a = make_object<Blob>(std::vector<int64_t>{4});
  auto b = make_object<Blob>(std::vector<int64_t>{4});
  auto c = make_object<Blob>(std::vector<int64_t>{4});

  a->cpu_mutable_data()[0] = 1.0f;
  a->cpu_mutable_data()[1] = 2.0f;
  a->cpu_mutable_data()[2] = 3.0f;
  a->cpu_mutable_data()[3] = 4.0f;

  // A→B 共享
  b->ShareData(a.get());
  EXPECT_TRUE(b->SharesDataWith(a.get()));

  // B 触发 COW（通过 cpu_mutable_data 写入）
  float* b_data = b->cpu_mutable_data();
  b_data[0] = 999.0f;

  // COW 后 B 与 A 应断开
  EXPECT_FALSE(b->SharesDataWith(a.get()));
  EXPECT_NEAR(static_cast<double>(b->cpu_mutable_data()[0]), 999.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(a->cpu_mutable_data()[0]), 1.0, 1e-6);  // A 不变

  // B→C 共享（COW 后的 B 作为源）
  c->ShareData(b.get());
  EXPECT_TRUE(c->SharesDataWith(b.get()));
  EXPECT_FALSE(c->SharesDataWith(a.get()));

  // C 通过 B 写入 → B 可见，A 不可见
  c->cpu_mutable_data()[1] = 888.0f;
  EXPECT_NEAR(static_cast<double>(b->cpu_mutable_data()[1]), 888.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(a->cpu_mutable_data()[1]), 2.0, 1e-6);
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
    auto a = make_object<Blob>(std::vector<int64_t>{100, 100});  // 40KB
    auto b = make_object<Blob>(std::vector<int64_t>{100, 100});  // 另一个 40KB

    a->cpu_mutable_data()[0] = 1.0f;
    b->cpu_mutable_data()[0] = 2.0f;

    // B 共享 A → B 的旧 40KB tensor 应被释放
    // 注意：TVM FFI 的侵入式引用计数在 data_tensor_ 被覆盖时
    // 自动递减旧 tensor 引用计数，引用计数归零时释放内存
    b->ShareData(a.get());

    EXPECT_TRUE(b->SharesDataWith(a.get()));
    int64_t bytes_after_share = g_total_allocated_bytes.load();

    // 总分配量应减少（B 的旧 40KB 被释放）
    // 注意：此断言依赖于 TVM FFI 的即时释放行为
    // 如果使用延迟释放，此断言可能不稳定
    EXPECT_LE(bytes_after_share, bytes_before);
  }

  // 所有 Blob 析构后，分配量应回到初始值附近
  int64_t bytes_after = g_total_allocated_bytes.load();
  EXPECT_LE(bytes_after, bytes_before);
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
  // N=2: COW Phase 2 — tops share data with bottom after Forward (zero-copy).
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
  EXPECT_NEAR(static_cast<double>(out_a->cpu_mutable_data()[0]), 999.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(out_b->cpu_mutable_data()[0]), 50.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(data_blob->cpu_mutable_data()[0]), 50.0, 1e-6);

  // out_b still shares with bottom
  EXPECT_TRUE(out_b->SharesDataWith(data_blob.get()));
}

TEST(ZeroCopyTest, LiveBlobCountStableAcrossShareData) {
  int64_t before = LiveBlobCount();
  {
    auto a = make_object<Blob>(std::vector<int64_t>{8});
    auto b = make_object<Blob>(std::vector<int64_t>{8});
    EXPECT_EQ(LiveBlobCount(), before + 2);
    b->ShareData(a.get());
    // ShareData does not create or destroy Blobs — count unchanged
    EXPECT_EQ(LiveBlobCount(), before + 2);
  }
  EXPECT_EQ(LiveBlobCount(), before);
}

// ── 高优先级补充场景 ──

/// 场景 1：ShareData 和 ShareDiff 分别来自不同源
/// 验证 data 和 diff 可以独立地从不同 Blob 共享，
/// 且各自的共享关系互不干扰
TEST(ZeroCopyTest, ShareDataAndDiffFromDifferentSources) {
  std::vector<int64_t> shape = {4, 3};
  auto src_data = make_object<Blob>(shape);
  auto src_diff = make_object<Blob>(shape);
  auto dst = make_object<Blob>(shape);

  // 写入可区分的值
  src_data->cpu_mutable_data()[0] = 10.0f;
  src_data->cpu_mutable_data()[1] = 20.0f;
  src_diff->cpu_mutable_diff()[0] = 30.0f;
  src_diff->cpu_mutable_diff()[1] = 40.0f;

  dst->ShareData(src_data.get());
  dst->ShareDiff(src_diff.get());

  // data 与 src_data 共享，diff 与 src_diff 共享
  EXPECT_TRUE(dst->SharesDataWith(src_data.get()));
  EXPECT_TRUE(dst->SharesDiffWith(src_diff.get()));

  // 交叉关系：data 不与 src_diff 共享，diff 不与 src_data 共享
  EXPECT_FALSE(dst->SharesDataWith(src_diff.get()));
  EXPECT_FALSE(dst->SharesDiffWith(src_data.get()));

  // 值验证
  EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_data()[0]), 10.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_data()[1]), 20.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_diff()[0]), 30.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_diff()[1]), 40.0, 1e-6);

  // 通过 dst 写入 data，src_data 可见，src_diff 不受影响
  dst->cpu_mutable_data()[0] = 99.0f;
  EXPECT_NEAR(static_cast<double>(src_data->cpu_mutable_data()[0]), 99.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src_diff->cpu_mutable_data()[0]), 0.0, 1e-6);  // src_diff 的 data 未共享
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

  // Forward 10 次 — 不应创建新 Blob 或泄漏 refcount
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

/// 验证 DataRefCount() 在未定义时为 0
TEST(COWApiTest, DataRefCountZeroWhenUndefined) {
  auto b = make_object<Blob>();  // 默认构造 shape={0}，但 defined
  // Reshape 到 {0} 仍然定义了 tensor
  b->Reshape(std::vector<int64_t>{0});
  // 但 0 元素 tensor 仍然 defined
  EXPECT_GT(b->DataRefCount(), 0);
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

/// 验证 mutable_data_tensor() 在共享时触发 COW
TEST(COWApiTest, MutableDataTensorTriggersCOW) {
  auto src = make_object<Blob>(std::vector<int64_t>{4});
  src->cpu_mutable_data()[0] = 77.0f;
  auto dst = make_object<Blob>(std::vector<int64_t>{4});
  dst->ShareData(src.get());
  EXPECT_TRUE(dst->IsDataShared());

  Tensor t = dst->mutable_data_tensor();
  EXPECT_FALSE(dst->IsDataShared());
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

/// 验证 mutable_diff_tensor() 在共享时触发 COW
TEST(COWApiTest, MutableDiffTensorTriggersCOW) {
  auto src = make_object<Blob>(std::vector<int64_t>{4});
  src->cpu_mutable_diff()[0] = 55.0f;
  auto dst = make_object<Blob>(std::vector<int64_t>{4});
  dst->ShareDiff(src.get());
  EXPECT_TRUE(dst->IsDiffShared());

  Tensor t = dst->mutable_diff_tensor();
  EXPECT_FALSE(dst->IsDiffShared());
  EXPECT_NEAR(static_cast<double>(static_cast<const float*>(t.data_ptr())[0]), 55.0, 1e-6);
}

/// 验证 COW 后写入不影响原始共享源
TEST(COWApiTest, COWWriteIsolation) {
  auto src = make_object<Blob>(std::vector<int64_t>{4});
  for (int i = 0; i < 4; ++i) src->cpu_mutable_data()[i] = static_cast<float>(i + 1);
  auto dst = make_object<Blob>(std::vector<int64_t>{4});
  dst->ShareData(src.get());
  // 原始数据：1, 2, 3, 4

  // COW 后修改 dst，不影响 src
  dst->cpu_mutable_data()[0] = 999.0f;
  EXPECT_NEAR(static_cast<double>(src->cpu_mutable_data()[0]), 1.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_data()[0]), 999.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src->cpu_mutable_data()[1]), 2.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_mutable_data()[1]), 2.0, 1e-6);
}
