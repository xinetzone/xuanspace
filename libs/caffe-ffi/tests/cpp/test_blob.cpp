#include "test_harness.hpp"

#include "caffe_ffi/blob.hpp"
#include "caffe_ffi/common.hpp"

#include <cstring>
#include <vector>

using namespace caffe_ffi;

// ---- Constructor & shape tests ----

TEST(BlobTest, DefaultConstructor) {
  Blob b;
  EXPECT_EQ(b.num_axes(), 1);  // Default Blob has shape {0}
  EXPECT_EQ(b.count(), 0);
  EXPECT_TRUE(b.name().empty());
}

TEST(BlobTest, ShapeConstructor) {
  std::vector<int64_t> shape = {2, 3, 4, 5};
  Blob b(shape);
  EXPECT_EQ(b.num_axes(), 4);
  EXPECT_EQ(b.count(), 2 * 3 * 4 * 5);
  EXPECT_EQ(b.shape(0), 2);
  EXPECT_EQ(b.shape(1), 3);
  EXPECT_EQ(b.shape(2), 4);
  EXPECT_EQ(b.shape(3), 5);
}

TEST(BlobTest, LegacyShapeAccessors) {
  std::vector<int64_t> shape = {8, 16, 32, 64};
  Blob b(shape);
  EXPECT_EQ(b.num(), 8);
  EXPECT_EQ(b.channels(), 16);
  EXPECT_EQ(b.height(), 32);
  EXPECT_EQ(b.width(), 64);
}

// ---- Reshape tests ----

TEST(BlobTest, Reshape) {
  Blob b;
  EXPECT_EQ(b.num_axes(), 1);  // Default: shape {0}
  std::vector<int64_t> shape = {3, 4};
  b.Reshape(shape);
  EXPECT_EQ(b.num_axes(), 2);
  EXPECT_EQ(b.count(), 12);
  EXPECT_EQ(b.shape(0), 3);
  EXPECT_EQ(b.shape(1), 4);
}

TEST(BlobTest, ReshapeLike) {
  std::vector<int64_t> shape = {2, 3};
  Blob a(shape);
  Blob b;
  b.ReshapeLike(a);
  EXPECT_EQ(b.num_axes(), 2);
  EXPECT_EQ(b.count(), 6);
  EXPECT_EQ(b.shape(0), 2);
  EXPECT_EQ(b.shape(1), 3);
}

TEST(BlobTest, CountWithAxis) {
  std::vector<int64_t> shape = {2, 3, 4, 5};
  Blob b(shape);
  EXPECT_EQ(b.count(0), 2 * 3 * 4 * 5);
  EXPECT_EQ(b.count(1), 3 * 4 * 5);
  EXPECT_EQ(b.count(2), 4 * 5);
  EXPECT_EQ(b.count(3), 5);
  EXPECT_EQ(b.count(1, 3), 3 * 4);
  EXPECT_EQ(b.count(0, 2), 2 * 3);
}

TEST(BlobTest, CanonicalAxisIndex) {
  std::vector<int64_t> shape = {2, 3, 4};
  Blob b(shape);
  EXPECT_EQ(b.CanonicalAxisIndex(0), 0);
  EXPECT_EQ(b.CanonicalAxisIndex(2), 2);
  EXPECT_EQ(b.CanonicalAxisIndex(-1), 2);
  EXPECT_EQ(b.CanonicalAxisIndex(-3), 0);
}

// ---- Name tests ----

TEST(BlobTest, NameGetterSetter) {
  Blob b;
  EXPECT_TRUE(b.name().empty());
  b.set_name("test_blob");
  EXPECT_EQ(b.name(), std::string("test_blob"));
}

// ---- Memory/data tests ----

TEST(BlobTest, CpuDataNotNullAfterAllocation) {
  std::vector<int64_t> shape = {2, 3};
  Blob b(shape);
  EXPECT_TRUE(b.cpu_data() != nullptr);
  EXPECT_TRUE(b.cpu_diff() != nullptr);
}

TEST(BlobTest, DataTensorReturnsValidTensor) {
  std::vector<int64_t> shape = {2, 3};
  Blob b(shape);
  Tensor dt = b.data_tensor();
  EXPECT_EQ(dt.ndim(), 2);
  EXPECT_EQ(dt.numel(), 6);
  EXPECT_TRUE(dt.data_ptr() != nullptr);
}

TEST(BlobTest, DiffTensorReturnsValidTensor) {
  std::vector<int64_t> shape = {2, 3};
  Blob b(shape);
  Tensor dt = b.diff_tensor();
  EXPECT_EQ(dt.ndim(), 2);
  EXPECT_EQ(dt.numel(), 6);
}

TEST(BlobTest, SetDataWithTensorRoundTrip) {
  std::vector<int64_t> shape = {2, 3};
  Blob b(shape);

  // Write test data via cpu_data
  float* data = b.cpu_data();
  for (int i = 0; i < 6; ++i) {
    data[i] = static_cast<float>(i) * 1.5f;
  }

  // Read back via get_data
  Array<float> arr = b.get_data();
  EXPECT_EQ(static_cast<int64_t>(arr.size()), 6);
  EXPECT_NEAR(static_cast<double>(arr[0]), 0.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(arr[5]), 7.5, 1e-6);
}

TEST(BlobTest, CpuDataWriteAndRead) {
  std::vector<int64_t> shape = {4};
  Blob b(shape);
  float* data = b.cpu_data();
  data[0] = 1.0f;
  data[1] = 2.0f;
  data[2] = 3.0f;
  data[3] = 4.0f;

  const float* read = b.cpu_data();
  EXPECT_NEAR(static_cast<double>(read[0]), 1.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(read[3]), 4.0, 1e-6);
}

TEST(BlobTest, CpuDiffWriteAndRead) {
  std::vector<int64_t> shape = {3};
  Blob b(shape);
  float* diff = b.cpu_diff();
  diff[0] = 0.1f;
  diff[1] = 0.2f;
  diff[2] = 0.3f;

  const float* read = b.cpu_diff();
  EXPECT_NEAR(static_cast<double>(read[0]), 0.1, 1e-6);
  EXPECT_NEAR(static_cast<double>(read[2]), 0.3, 1e-6);
}

// ---- ObjectPtr reference counting tests ----

TEST(BlobTest, ObjectPtrRefCounting) {
  int64_t before = LiveBlobCount();
  {
    ObjectPtr<Blob> b = make_object<Blob>(std::vector<int64_t>{2, 3});
    EXPECT_EQ(LiveBlobCount(), before + 1);
  }
  EXPECT_EQ(LiveBlobCount(), before);
}

TEST(BlobTest, UpdateSubtractsDiffFromData) {
  std::vector<int64_t> shape = {2};
  Blob b(shape);
  b.cpu_data()[0] = 10.0f;
  b.cpu_data()[1] = 20.0f;
  b.cpu_diff()[0] = 1.0f;
  b.cpu_diff()[1] = 2.0f;
  b.Update();
  EXPECT_NEAR(static_cast<double>(b.cpu_data()[0]), 9.0, 1e-6);
  EXPECT_NEAR(static_cast<double>(b.cpu_data()[1]), 18.0, 1e-6);
}

// ---- ID and backtrace tests ----

TEST(BlobTest, IdIsUniqueAndPositive) {
  ObjectPtr<Blob> b1 = make_object<Blob>();
  ObjectPtr<Blob> b2 = make_object<Blob>();
  EXPECT_TRUE(b1->id() > 0);
  EXPECT_TRUE(b2->id() > 0);
  EXPECT_TRUE(b2->id() > b1->id());
}

TEST(BlobTest, ConstructionBacktraceNotEmpty) {
  Blob b;
  std::string bt = b.construction_backtrace();
  EXPECT_FALSE(bt.empty());
}

// ---- Negative axis tests ----

TEST(BlobTest, NegativeAxisIndex) {
  std::vector<int64_t> shape = {2, 3, 4, 5};
  Blob b(shape);
  EXPECT_EQ(b.shape(-1), 5);
  EXPECT_EQ(b.shape(-2), 4);
  EXPECT_EQ(b.shape(-3), 3);
  EXPECT_EQ(b.shape(-4), 2);
  EXPECT_EQ(b.count(-1), 5);
  EXPECT_EQ(b.count(-2), 4 * 5);
}

// ---- Diff round-trip test ----

TEST(BlobTest, GetDiffSetDiffRoundTrip) {
  std::vector<int64_t> shape = {2, 2};
  Blob b(shape);
  float* diff = b.cpu_diff();
  diff[0] = 0.5f;
  diff[1] = 1.5f;
  diff[2] = 2.5f;
  diff[3] = 3.5f;

  Array<float> arr = b.get_diff();
  EXPECT_EQ(static_cast<int64_t>(arr.size()), 4);
  EXPECT_NEAR(static_cast<double>(arr[0]), 0.5, 1e-6);
  EXPECT_NEAR(static_cast<double>(arr[3]), 3.5, 1e-6);
}

// ---- Shape as TVM FFI Shape ----

TEST(BlobTest, ShapeReturnsTVMShape) {
  std::vector<int64_t> shape = {2, 3};
  Blob b(shape);
  Shape s = b.shape();
  EXPECT_EQ(s.size(), static_cast<size_t>(2));
  EXPECT_EQ(s[0], 2);
  EXPECT_EQ(s[1], 3);
}

// ---- Error handling tests ----

TEST(BlobTest, NegativeDimensionThrows) {
  Blob b;
  std::vector<int64_t> bad_shape = {2, -3, 4};
  bool threw = false;
  try {
    b.Reshape(bad_shape);
  } catch (const std::exception&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

// ---- Legacy shape accessor with missing dimensions ----

TEST(BlobTest, LegacyShapeAccessorsWithMissingDims) {
  std::vector<int64_t> shape = {8};
  Blob b(shape);
  EXPECT_EQ(b.num(), 8);
  EXPECT_EQ(b.channels(), 1);
  EXPECT_EQ(b.height(), 1);
  EXPECT_EQ(b.width(), 1);
}
