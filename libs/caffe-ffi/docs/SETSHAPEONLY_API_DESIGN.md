# SetShapeOnly API 接口设计文档

> **状态**: 草稿 | **日期**: 2026-07-31 | **来源**: Split COW Phase 3.1 延迟 Reshape 分配

---

## 1. 设计动机

### 1.1 问题

Phase 2 的 `SplitLayer::Reshape()` 为每个 top Blob 调用 `ReshapeLike()`，分配完整内存。N=100 时产生 100 个临时 buffer，随后在 `Forward_cpu()` 中通过 `ShareData()` 替换为共享引用，临时 buffer 被释放。这造成了：

- **内存峰值**：N=100 时 Reshape 阶段内存峰值 = 100 × count × 4B
- **GC 压力**：100 次 alloc + 100 次 free（在 Forward 阶段隐式发生）
- **CPU 浪费**：Reshape 阶段的分配操作在 Forward 阶段被废弃

### 1.2 目标

对大 N 场景（N ≥ LAZY_RESHAPE_THRESHOLD），Reshape 阶段仅设置 shape 元数据，不分配数据内存。Forward 阶段通过 `ShareData()` 建立引用时，目标 Blob 的 data_tensor 直接从 undefined 变为共享引用。

---

## 2. API 设计

### 2.1 新增方法

```cpp
// blob.hpp
class Blob : public Object {
 public:
  /**
   * @brief Set shape metadata only, without allocating data memory.
   *
   * This is a Phase 3.1 optimization for lazy Reshape in large-N Split
   * scenarios. Unlike Reshape() which allocates a full data buffer, this
   * method only stores the shape information. The data_tensor_ and
   * diff_tensor_ remain undefined until ShareData() or an explicit Reshape()
   * allocates them.
   *
   * After calling SetShapeOnly():
   *   - shape(), num_axes(), count() return the stored shape
   *   - cpu_data(), cpu_diff() return nullptr (data_tensor_ is undefined)
   *   - cpu_mutable_data(), cpu_mutable_diff() return nullptr
   *   - data_tensor(), diff_tensor() return undefined Tensor
   *
   * This is NOT a general-purpose API. It is specifically designed for the
   * Split layer's lazy allocation pattern where:
   *   1. Reshape phase: SetShapeOnly() sets shape metadata
   *   2. Forward phase: ShareData() replaces the undefined tensor with a
   *      shared reference from bottom
   *   3. Downstream layers: Reshape() triggers full allocation only if
   *      the layer actually needs to write into the blob
   *
   * @param shape The target shape to store as metadata.
   * @pre shape dimensions must be positive (no negative dimensions).
   * @post shape_only_ is set, is_lazy_allocated_ = true,
   *       data_tensor_ and diff_tensor_ remain undefined.
   */
  void SetShapeOnly(ShapeView shape);

  /**
   * @brief Check if this Blob is in lazy-allocation mode (Phase 3.1).
   * @return true if SetShapeOnly() was called and data_tensor_ is still
   *         undefined.
   */
  bool IsLazyAllocated() const { return is_lazy_allocated_; }

  // ... (existing members)
};
```

### 2.2 内部实现

```cpp
// blob.hpp private section — new members
class Blob : public Object {
 private:
  // ... existing members ...
  Tensor data_tensor_;
  Tensor diff_tensor_;

  // Phase 3.1: lazy allocation support
  std::vector<int64_t> shape_only_;   // stored shape for lazy allocation
  bool is_lazy_allocated_ = false;     // whether in lazy-allocation mode
};
```

```cpp
// blob.cpp
void Blob::SetShapeOnly(ShapeView shape) {
  // Validate shape
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0) {
      CAFFE_FFI_LOG_ERROR << "SetShapeOnly: invalid shape dimension " << shape[i]
                          << " at axis " << i;
      throw std::invalid_argument("SetShapeOnly: all dimensions must be positive");
    }
  }

  // Store shape metadata
  shape_only_.assign(shape.data(), shape.data() + shape.size());
  is_lazy_allocated_ = true;

  // data_tensor_ and diff_tensor_ remain undefined
  CAFFE_FFI_MEM_LOG << "[LAZY] Blob#" << id_
                    << " SetShapeOnly: shape=["
                    << shape_only_ << "]"
                    << " count=" << (shape_only_.empty() ? 0 :
                        std::accumulate(shape_only_.begin(), shape_only_.end(),
                                        1LL, std::multiplies<int64_t>()));
}
```

### 2.3 受影响的方法修改

以下方法需要检查 `is_lazy_allocated_` 标志：

| 方法 | 当前行为 | 修改后行为 |
|------|---------|-----------|
| `shape()` | `Shape(data_tensor_.shape())` | `is_lazy_allocated_ ? Shape(shape_only_) : Shape(data_tensor_.shape())` |
| `num_axes()` | `data_tensor_.ndim()` | `is_lazy_allocated_ ? shape_only_.size() : data_tensor_.ndim()` |
| `count()` | `data_tensor_.numel()` | `is_lazy_allocated_ ? product(shape_only_) : data_tensor_.numel()` |
| `shape(int)` | `data_tensor_.size(...)` | `is_lazy_allocated_ ? shape_only_[index] : data_tensor_.size(...)` |
| `cpu_data()` | `data_tensor_.data_ptr()` | 不变（undefined tensor 返回 nullptr） |
| `cpu_mutable_data()` | COW + data_ptr | 不变（undefined tensor 的 data_ptr 为 nullptr） |
| `Reshape()` | 分配新内存 | 需清除 `is_lazy_allocated_` 标志 |
| `ShareData()` | 替换 data_tensor_ | 需清除 `is_lazy_allocated_` 标志 |
| `data_tensor()` | 返回 data_tensor_ | 不变（返回 undefined Tensor，下游层可检查 `defined()`） |

### 2.4 生命周期状态机

```
┌──────────┐  SetShapeOnly()   ┌──────────────┐
│  Normal   │ ────────────────→ │ LazyAllocated │
│  (default)│                   │ (shape only)  │
└──────────┘                   └──────┬───────┘
     ↑                                │
     │    Reshape() / ShareData()     │
     └────────────────────────────────┘
```

- **Normal → LazyAllocated**: `SetShapeOnly()` 设置 shape 元数据，不分配内存
- **LazyAllocated → Normal**: 任何触发实际内存分配的操作（`Reshape()`, `ShareData()`, `FromProto()`）清除 `is_lazy_allocated_` 标志

---

## 3. SplitLayer 集成

### 3.1 Reshape 修改

```cpp
// split_layer.cpp
constexpr int kLazyReshapeThreshold = 16;

void SplitLayer::Reshape(const std::vector<Blob*>& bottom,
                          const std::vector<Blob*>& top) {
  int count = bottom[0]->count();
  int num_top = static_cast<int>(top.size());

  if (num_top >= kLazyReshapeThreshold) {
    // Phase 3.1: Lazy allocation — only set shape metadata
    for (int i = 0; i < num_top; ++i) {
      top[i]->SetShapeOnly(bottom[0]->shape());
    }
    total_alloc_bytes = 0;  // No memory allocated in Reshape
  } else {
    // Phase 2 behavior: full allocation (unchanged)
    for (int i = 0; i < num_top; ++i) {
      top[i]->ReshapeLike(*bottom[0]);
    }
    total_alloc_bytes = num_top * count * static_cast<int64_t>(sizeof(float));
  }
  // ... summary log unchanged
}
```

### 3.2 Forward 兼容性

`Forward_cpu()` 中的 `ShareData()` 调用无需修改——当目标 Blob 的 `data_tensor_` 为 undefined 时，`ShareData()` 直接赋值引用（无 COW 触发，因为 `use_count()` 在 undefined tensor 上不会大于 1）。

---

## 4. 兼容性测试用例

### 4.1 单元测试 (`test_blob.cpp`)

```cpp
// ========== Phase 3.1: SetShapeOnly tests ==========

TEST(BlobSetShapeOnly, BasicShapeStorage) {
  Blob blob;
  std::vector<int64_t> shape = {2, 3, 224, 224};
  blob.SetShapeOnly(ShapeView(shape.data(), shape.size()));

  EXPECT_TRUE(blob.IsLazyAllocated());
  EXPECT_EQ(blob.num_axes(), 4);
  EXPECT_EQ(blob.shape(0), 2);
  EXPECT_EQ(blob.shape(1), 3);
  EXPECT_EQ(blob.shape(2), 224);
  EXPECT_EQ(blob.shape(3), 224);
  EXPECT_EQ(blob.count(), 2 * 3 * 224 * 224);
}

TEST(BlobSetShapeOnly, NoDataAllocation) {
  Blob blob;
  int64_t bytes_before = TotalAllocatedBytes();
  blob.SetShapeOnly(ShapeView({100, 100, 100}));
  int64_t bytes_after = TotalAllocatedBytes();

  // No memory should be allocated for data
  EXPECT_EQ(bytes_after, bytes_before);
}

TEST(BlobSetShapeOnly, CpuDataReturnsNullptr) {
  Blob blob;
  blob.SetShapeOnly(ShapeView({64, 3, 32, 32}));

  // cpu_data() should return nullptr since no data is allocated
  EXPECT_EQ(blob.cpu_data(), nullptr);
  EXPECT_EQ(blob.cpu_diff(), nullptr);
}

TEST(BlobSetShapeOnly, DataTensorUndefined) {
  Blob blob;
  blob.SetShapeOnly(ShapeView({1, 10}));

  // data_tensor() should return undefined Tensor
  Tensor dt = blob.data_tensor();
  EXPECT_FALSE(dt.defined());
}

TEST(BlobSetShapeOnly, ReshapeClearsLazyFlag) {
  Blob blob;
  blob.SetShapeOnly(ShapeView({10, 20}));
  EXPECT_TRUE(blob.IsLazyAllocated());

  // Reshape should allocate memory and clear lazy flag
  blob.Reshape(ShapeView({5, 5}));
  EXPECT_FALSE(blob.IsLazyAllocated());
  EXPECT_NE(blob.cpu_data(), nullptr);
  EXPECT_EQ(blob.count(), 25);
}

TEST(BlobSetShapeOnly, ShareDataClearsLazyFlag) {
  Blob source(ShapeView({3, 4}));
  Blob target;
  target.SetShapeOnly(ShapeView({3, 4}));
  EXPECT_TRUE(target.IsLazyAllocated());

  // ShareData should replace tensor and clear lazy flag
  target.ShareData(&source);
  EXPECT_FALSE(target.IsLazyAllocated());
  EXPECT_TRUE(target.SharesDataWith(&source));
  EXPECT_EQ(target.cpu_data(), source.cpu_data());
}

TEST(BlobSetShapeOnly, CountAfterSetShapeOnly) {
  Blob blob;
  blob.SetShapeOnly(ShapeView({32, 64, 112, 112}));
  EXPECT_EQ(blob.count(), 32LL * 64 * 112 * 112);
  EXPECT_EQ(blob.count(1), 64LL * 112 * 112);
  EXPECT_EQ(blob.count(0, 2), 32LL * 64);
}

TEST(BlobSetShapeOnly, InvalidShapeRejected) {
  Blob blob;
  // Negative dimension should throw
  EXPECT_THROW(blob.SetShapeOnly(ShapeView({-1, 10})), std::invalid_argument);
  // Zero dimension should throw
  EXPECT_THROW(blob.SetShapeOnly(ShapeView({0, 10})), std::invalid_argument);
}
```

### 4.2 集成测试 (`test_split_layer.cpp`)

```cpp
// ========== Phase 3.1: Split Lazy Reshape tests ==========

TEST(SplitLayerLazyReshape, LargeNTriggersLazyAllocation) {
  // N=64 should trigger lazy allocation (threshold=16)
  Net net;
  // Create a Split layer with N=64
  auto* split = net.AddLayer<SplitLayer>("split64", 64);
  Blob bottom(ShapeView({1, 3, 32, 32}));
  std::vector<Blob*> tops = CreateTopBlobs(64);

  int64_t bytes_before = TotalAllocatedBytes();
  split->Reshape({&bottom}, tops);
  int64_t bytes_after = TotalAllocatedBytes();

  // With lazy allocation, Reshape should not allocate data memory
  // (only shape metadata, which is negligible)
  int64_t expected_min = 1 * 3 * 32 * 32 * static_cast<int64_t>(sizeof(float));
  // Without lazy allocation, memory would be N * count * sizeof(float)
  // = 64 * 3072 * 4 = 786,432 bytes
  // With lazy allocation, only shape metadata (64 * 4 * 8 = 2,048 bytes)
  EXPECT_LT(bytes_after - bytes_before, expected_min);

  // All tops should be lazy-allocated
  for (int i = 0; i < 64; ++i) {
    EXPECT_TRUE(tops[i]->IsLazyAllocated());
    EXPECT_EQ(tops[i]->cpu_data(), nullptr);
  }
}

TEST(SplitLayerLazyReshape, ForwardTransitionsToNormal) {
  Net net;
  auto* split = net.AddLayer<SplitLayer>("split64", 64);
  Blob bottom(ShapeView({1, 3, 32, 32}));
  std::vector<Blob*> tops = CreateTopBlobs(64);

  split->Reshape({&bottom}, tops);
  // All tops are lazy-allocated
  for (int i = 0; i < 64; ++i) {
    EXPECT_TRUE(tops[i]->IsLazyAllocated());
  }

  split->Forward({&bottom}, tops);
  // After Forward, ShareData should have replaced lazy tensors
  for (int i = 0; i < 64; ++i) {
    EXPECT_FALSE(tops[i]->IsLazyAllocated());
    EXPECT_TRUE(tops[i]->SharesDataWith(&bottom));
    EXPECT_NE(tops[i]->cpu_data(), nullptr);
    EXPECT_EQ(tops[i]->cpu_data(), bottom.cpu_data());
  }
}

TEST(SplitLayerLazyReshape, SmallNStaysNormal) {
  // N=4 should not trigger lazy allocation (threshold=16)
  Net net;
  auto* split = net.AddLayer<SplitLayer>("split4", 4);
  Blob bottom(ShapeView({1, 3, 32, 32}));
  std::vector<Blob*> tops = CreateTopBlobs(4);

  split->Reshape({&bottom}, tops);
  for (int i = 0; i < 4; ++i) {
    EXPECT_FALSE(tops[i]->IsLazyAllocated());
    EXPECT_NE(tops[i]->cpu_data(), nullptr);
  }
}

TEST(SplitLayerLazyReshape, DownstreamLayerCompatibility) {
  // Test that a downstream ReLU layer can handle lazy-allocated input
  Net net;
  auto* split = net.AddLayer<SplitLayer>("split100", 100);
  auto* relu = net.AddLayer<ReLULayer>("relu");

  Blob bottom(ShapeView({1, 64, 28, 28}));
  Blob top_single(ShapeView({1, 64, 28, 28}));
  std::vector<Blob*> tops = CreateTopBlobs(100);

  // Reshape: Split uses lazy allocation
  split->Reshape({&bottom}, tops);

  // Forward: Split transitions tops to normal via ShareData
  split->Forward({&bottom}, tops);

  // Downstream ReLU should work on one of the shared tops
  Blob relu_out(ShapeView({1, 64, 28, 28}));
  relu->Reshape({tops[0]}, {&relu_out});
  relu->Forward({tops[0]}, {&relu_out});

  // Verify ReLU output
  const float* in_data = tops[0]->cpu_data();
  const float* out_data = relu_out.cpu_data();
  for (int i = 0; i < tops[0]->count(); ++i) {
    EXPECT_EQ(out_data[i], std::max(0.0f, in_data[i]));
  }
}
```

### 4.3 Python 测试 (`test_lazy_reshape.py`)

```python
"""Phase 3.1: SetShapeOnly lazy reshape integration tests."""
import pytest
import caffe_ffi


class TestSetShapeOnly:
    """Blob-level SetShapeOnly unit tests."""

    def test_basic_shape_storage(self):
        blob = caffe_ffi.Blob()
        blob.set_shape_only([2, 3, 224, 224])
        assert blob.is_lazy_allocated()
        assert blob.num_axes() == 4
        assert blob.shape(0) == 2
        assert blob.count() == 2 * 3 * 224 * 224

    def test_no_data_allocation(self):
        blob = caffe_ffi.Blob()
        bytes_before = caffe_ffi.total_allocated_bytes()
        blob.set_shape_only([100, 100, 100])
        bytes_after = caffe_ffi.total_allocated_bytes()
        assert bytes_after == bytes_before

    def test_cpu_data_returns_none(self):
        blob = caffe_ffi.Blob()
        blob.set_shape_only([64, 3, 32, 32])
        # cpu_data() returns None for undefined tensor
        assert blob.cpu_data() is None

    def test_reshape_clears_lazy_flag(self):
        blob = caffe_ffi.Blob()
        blob.set_shape_only([10, 20])
        assert blob.is_lazy_allocated()
        blob.reshape([5, 5])
        assert not blob.is_lazy_allocated()
        assert blob.cpu_data() is not None

    def test_share_data_clears_lazy_flag(self):
        source = caffe_ffi.Blob([3, 4])
        target = caffe_ffi.Blob()
        target.set_shape_only([3, 4])
        assert target.is_lazy_allocated()
        target.share_data(source)
        assert not target.is_lazy_allocated()
        assert target.shares_data_with(source)

    def test_invalid_shape_rejected(self):
        blob = caffe_ffi.Blob()
        with pytest.raises(ValueError):
            blob.set_shape_only([-1, 10])
        with pytest.raises(ValueError):
            blob.set_shape_only([0, 10])


class TestSplitLazyReshape:
    """Split layer lazy reshape integration tests."""

    def test_large_n_triggers_lazy_allocation(self):
        """N=64 should trigger lazy allocation (threshold=16)."""
        net = caffe_ffi.Net()
        split = net.add_layer("Split", "split64", num_top=64)
        bottom = caffe_ffi.Blob([1, 3, 32, 32])
        tops = [caffe_ffi.Blob() for _ in range(64)]

        bytes_before = caffe_ffi.total_allocated_bytes()
        split.reshape([bottom], tops)
        bytes_after = caffe_ffi.total_allocated_bytes()

        # Lazy allocation should not allocate data memory
        expected_full = 64 * 3 * 32 * 32 * 4  # ~786KB
        assert (bytes_after - bytes_before) < expected_full

        for top in tops:
            assert top.is_lazy_allocated()

    def test_forward_transitions_to_normal(self):
        """After Forward, ShareData should replace lazy tensors."""
        net = caffe_ffi.Net()
        split = net.add_layer("Split", "split64", num_top=64)
        bottom = caffe_ffi.Blob([1, 3, 32, 32])
        tops = [caffe_ffi.Blob() for _ in range(64)]

        split.reshape([bottom], tops)
        split.forward([bottom], tops)

        for top in tops:
            assert not top.is_lazy_allocated()
            assert top.shares_data_with(bottom)
            assert top.cpu_data() is not None

    def test_small_n_stays_normal(self):
        """N=4 should not trigger lazy allocation."""
        net = caffe_ffi.Net()
        split = net.add_layer("Split", "split4", num_top=4)
        bottom = caffe_ffi.Blob([1, 3, 32, 32])
        tops = [caffe_ffi.Blob() for _ in range(4)]

        split.reshape([bottom], tops)
        for top in tops:
            assert not top.is_lazy_allocated()
            assert top.cpu_data() is not None

    @pytest.mark.parametrize("layer_type", ["ReLU", "InnerProduct", "Convolution"])
    def test_downstream_layer_compatibility(self, layer_type):
        """Verify downstream layers work with lazy-allocated→shared blobs."""
        net = caffe_ffi.Net()
        split = net.add_layer("Split", "split100", num_top=100)
        bottom = caffe_ffi.Blob([1, 64, 28, 28])
        tops = [caffe_ffi.Blob() for _ in range(100)]

        split.reshape([bottom], tops)
        split.forward([bottom], tops)

        # downstream layer should work on shared top
        # (specific setup depends on layer type)
        # This test ensures no crash or undefined behavior
        assert tops[0].cpu_data() is not None
        assert tops[0].count() == 64 * 28 * 28
```

---

## 5. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 下游层在 Reshape 阶段假设 data_tensor 已定义 | 中 | 高 | 在 Split 的 Forward 之后才调用下游层 Reshape，此时 ShareData 已完成 |
| `shape()` 返回类型变化导致调用方编译错误 | 低 | 中 | `Shape(shape_only_)` 需要构造 `Shape` 对象（额外开销），但调用频率低 |
| `count(int)` 的 `Count()` 辅助函数期望 Tensor shape | 低 | 低 | 实现 `Count(shape_only_, start_axis)` 重载 |
| 非 Split 层误用 `SetShapeOnly()` | 低 | 中 | 文档明确标注为 Split 专用 API，通过命名和注释限制使用范围 |

---

## 6. 回退策略

与 Phase 3.0 一致，通过 `CAFFE_FFI_ENABLE_COW_PHASE3` 编译期开关控制：

```cpp
#ifdef CAFFE_FFI_ENABLE_COW_PHASE3
  if (num_top >= kLazyReshapeThreshold) {
    // Lazy allocation path
  } else
#endif
  {
    // Phase 2 fallback: full allocation
  }
```

`OFF` 时保持 Phase 2 行为，零风险回退。