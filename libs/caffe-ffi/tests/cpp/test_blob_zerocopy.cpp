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
  float* src_data = src->cpu_data();
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
    EXPECT_NEAR(static_cast<double>(dst->cpu_data()[i]),
                static_cast<double>(i) * 0.5, 1e-6);
  }
}

TEST(ZeroCopyTest, ShareDiffMakesDiffPointersEqual) {
  std::vector<int64_t> shape = {4, 4};
  auto src = make_object<Blob>(shape);
  auto dst = make_object<Blob>(shape);

  float* src_diff = src->cpu_diff();
  for (int64_t i = 0; i < src->count(); ++i) {
    src_diff[i] = static_cast<float>(i + 1) * 0.1f;
  }

  EXPECT_NE(src->cpu_diff(), dst->cpu_diff());
  EXPECT_FALSE(dst->SharesDiffWith(src.get()));

  dst->ShareDiff(src.get());

  EXPECT_TRUE(dst->SharesDiffWith(src.get()));
  EXPECT_EQ(dst->cpu_diff(), src->cpu_diff());
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
  std::vector<int64_t> shape = {8};
  auto a = make_object<Blob>(shape);
  auto b = make_object<Blob>(shape);

  a->cpu_data()[0] = 42.0f;
  b->ShareData(a.get());

  // Mutate via b, read via a (same memory)
  b->cpu_data()[1] = 99.0f;
  EXPECT_NEAR(static_cast<double>(a->cpu_data()[1]), 99.0, 1e-6);

  // Mutate via a, read via b
  a->cpu_data()[2] = 7.0f;
  EXPECT_NEAR(static_cast<double>(b->cpu_data()[2]), 7.0, 1e-6);
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
  float* src_data_ptr = nullptr;

  {
    auto src = make_object<Blob>(shape);
    src_data_ptr = src->cpu_data();
    src->cpu_data()[0] = 3.14f;

    {
      auto dst = make_object<Blob>(shape);
      dst->ShareData(src.get());
      EXPECT_EQ(dst->cpu_data(), src_data_ptr);
      EXPECT_NEAR(static_cast<double>(dst->cpu_data()[0]), 3.14, 1e-6);
    }
    // dst destroyed here; src must still be valid
    EXPECT_NEAR(static_cast<double>(src->cpu_data()[0]), 3.14, 1e-6);
    EXPECT_EQ(src->cpu_data(), src_data_ptr);
  }
  // src destroyed here, memory freed
}

TEST(ZeroCopyTest, RefcountingDestinationOutlivesSource) {
  std::vector<int64_t> shape = {16};
  auto dst = make_object<Blob>(shape);
  {
    auto src = make_object<Blob>(shape);
    src->cpu_data()[5] = 7.77f;
    dst->ShareData(src.get());
    EXPECT_TRUE(dst->SharesDataWith(src.get()));
  }
  // src destroyed; dst must still have valid data via refcount
  EXPECT_NEAR(static_cast<double>(dst->cpu_data()[5]), 7.77, 1e-6);
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
  b->cpu_data()[0] = 1.0f;
  EXPECT_NEAR(static_cast<double>(c->cpu_data()[0]), 1.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(a->cpu_data()[0]), 1.0, 1e-6);
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
    src->cpu_data()[i] = static_cast<float>(i);
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
  b->cpu_data()[0] = 42.0f;

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
    src->cpu_diff()[i] = static_cast<float>(i * 10);
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

  src->cpu_data()[0] = 10.0f;
  src->cpu_data()[1] = 20.0f;
  src->cpu_data()[2] = 30.0f;
  src->cpu_data()[3] = 40.0f;

  dst->ShareData(src.get());

  // 触发 COW 并修改 dst
  dst->cpu_mutable_data()[0] = 999.0f;
  dst->cpu_mutable_data()[1] = 888.0f;

  // src 不受影响
  EXPECT_NEAR(static_cast<double>(src->cpu_data()[0]), 10.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src->cpu_data()[1]), 20.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src->cpu_data()[2]), 30.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src->cpu_data()[3]), 40.0, 1e-6);

  // dst 有自己的值
  EXPECT_NEAR(static_cast<double>(dst->cpu_data()[0]), 999.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_data()[1]), 888.0, 1e-6);
  // 未修改的部分应与 src 一致（COW 完整复制）
  EXPECT_NEAR(static_cast<double>(dst->cpu_data()[2]), 30.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_data()[3]), 40.0, 1e-6);
}

/// 验证 const cpu_data() 不触发 COW（指针保持共享）
TEST(COWTest, ConstAccessDoesNotTriggerCOW) {
  std::vector<int64_t> shape = {4};
  auto src = make_object<Blob>(shape);
  auto dst = make_object<Blob>(shape);

  src->cpu_data()[0] = 77.0f;
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

  a->cpu_data()[0] = 1.0f;
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
  EXPECT_NEAR(static_cast<double>(a->cpu_data()[0]), 1.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(c->cpu_data()[0]), 1.0, 1e-6);

  // b 数据独立
  EXPECT_NEAR(static_cast<double>(b->cpu_data()[0]), 999.0, 1e-6);
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
  float* data_ptr = data_blob->cpu_data();
  for (int64_t i = 0; i < data_blob->count(); ++i) {
    data_ptr[i] = static_cast<float>(i) * 0.01f;
  }

  net->Forward();

  auto out_blob = net->blob_by_name("split_out");

  // Data must be visible through the output blob (zero-copy shared, no corruption)
  EXPECT_TRUE(out_blob->SharesDataWith(data_blob.get()));
  for (int64_t i = 0; i < data_blob->count(); ++i) {
    EXPECT_NEAR(static_cast<double>(out_blob->cpu_data()[i]),
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
  data_blob->cpu_data()[0] = 111.0f;
  data_blob->cpu_data()[1] = 222.0f;

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
  EXPECT_NEAR(static_cast<double>(out_a->cpu_data()[0]), 111.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(out_a->cpu_data()[1]), 222.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(out_b->cpu_data()[0]), 111.0, 1e-6);
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
  data_blob->cpu_data()[0] = 50.0f;
  data_blob->cpu_data()[1] = 60.0f;

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
  src_data->cpu_data()[0] = 10.0f;
  src_data->cpu_data()[1] = 20.0f;
  src_diff->cpu_diff()[0] = 30.0f;
  src_diff->cpu_diff()[1] = 40.0f;

  dst->ShareData(src_data.get());
  dst->ShareDiff(src_diff.get());

  // data 与 src_data 共享，diff 与 src_diff 共享
  EXPECT_TRUE(dst->SharesDataWith(src_data.get()));
  EXPECT_TRUE(dst->SharesDiffWith(src_diff.get()));

  // 交叉关系：data 不与 src_diff 共享，diff 不与 src_data 共享
  EXPECT_FALSE(dst->SharesDataWith(src_diff.get()));
  EXPECT_FALSE(dst->SharesDiffWith(src_data.get()));

  // 值验证
  EXPECT_NEAR(static_cast<double>(dst->cpu_data()[0]), 10.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_data()[1]), 20.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_diff()[0]), 30.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_diff()[1]), 40.0, 1e-6);

  // 通过 dst 写入 data，src_data 可见，src_diff 不受影响
  dst->cpu_data()[0] = 99.0f;
  EXPECT_NEAR(static_cast<double>(src_data->cpu_data()[0]), 99.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(src_diff->cpu_data()[0]), 0.0, 1e-6);  // src_diff 的 data 未共享
}

/// 场景 2：共享后 Reshape 源 Blob
/// 验证 Reshape 源 Blob 不会破坏已共享的目标数据。
/// 源 Reshape 后分配新内存，目标通过 refcount 仍持有原数据
TEST(ZeroCopyTest, ReshapeSourceAfterSharePreservesDestination) {
  std::vector<int64_t> shape = {4, 4};
  auto src = make_object<Blob>(shape);
  auto dst = make_object<Blob>(shape);

  src->cpu_data()[0] = 42.0f;
  src->cpu_data()[5] = 77.0f;
  dst->ShareData(src.get());
  EXPECT_TRUE(dst->SharesDataWith(src.get()));

  // Reshape src 到不同形状 → 分配新内存，打破共享
  src->Reshape(std::vector<int64_t>{8, 8});

  // dst 仍持有原数据（通过 refcount 独立持有）
  EXPECT_NEAR(static_cast<double>(dst->cpu_data()[0]), 42.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(dst->cpu_data()[5]), 77.0, 1e-6);

  // 确认共享已打破
  EXPECT_FALSE(dst->SharesDataWith(src.get()));

  // src 的新数据独立，不受 dst 影响
  src->cpu_data()[0] = 100.0f;
  EXPECT_NEAR(static_cast<double>(dst->cpu_data()[0]), 42.0, 1e-6);
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
  bottom_blob->cpu_data()[0] = 55.0f;

  int64_t live_count_before = LiveBlobCount();

  // Forward 10 次 — 不应创建新 Blob 或泄漏 refcount
  for (int i = 0; i < 10; ++i) {
    net->Forward();
    // 每次 Forward 后，N=1 top 应仍与 bottom 共享数据
    EXPECT_TRUE(top_blob->SharesDataWith(bottom_blob.get()));
    // 数据正确性
    EXPECT_NEAR(static_cast<double>(top_blob->cpu_data()[0]), 55.0, 1e-6);
  }

  // Blob 数量稳定（无泄漏）
  EXPECT_EQ(LiveBlobCount(), live_count_before);
}
