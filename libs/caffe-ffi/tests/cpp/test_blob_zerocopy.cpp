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

TEST(ZeroCopyTest, SplitN2StillCopiesData) {
  // N=2: NOT zero-copy in Phase 1 — must still memcpy to independent buffers
  std::string prototxt = R"(
name: "test_split_n2_copy"
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

  net->Forward();

  auto out_a = net->blob_by_name("out_a");
  auto out_b = net->blob_by_name("out_b");

  // In Phase 1 N>=2, tops must NOT share data with bottom or each other
  // (full memcpy to independent buffers, as COW is deferred to Phase 2).
  EXPECT_FALSE(out_a->SharesDataWith(data_blob.get()));
  EXPECT_FALSE(out_b->SharesDataWith(data_blob.get()));
  EXPECT_FALSE(out_a->SharesDataWith(out_b.get()));
  EXPECT_NE(out_a->cpu_data(), out_b->cpu_data());

  // But data values must be correctly copied
  EXPECT_NEAR(static_cast<double>(out_a->cpu_data()[0]), 111.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(out_b->cpu_data()[0]), 111.0, 1e-6);
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
