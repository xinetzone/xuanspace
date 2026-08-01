# Split 层 Phase 2 (N≥2) Copy-on-Write 优化方案草稿

> **版本**: v0.1-draft  
> **日期**: 2026-07-31  
> **前置**: Phase 1 N=1 零拷贝捷径已实现并验证通过  
> **目标**: 将零拷贝从 N=1 扩展到 N≥2 场景，通过 COW 语义在首次写入时延迟拷贝

---

## 1. 背景与动机

### 1.1 Phase 1 成果回顾

Phase 1 为 N=1 Split 层实现了零拷贝捷径：
- `SplitLayer::Forward_cpu()` 在 `num_top == 1` 时直接调用 `ShareData()/ShareDiff()`
- 通过 TVM FFI Tensor 的侵入式引用计数共享底层缓冲区，完全消除 memcpy
- **验证结果**: `Forward(n1_split) Δmem=-64B`，C++ 单元测试确认指针相等，`[SPLIT-PERF] ZEROCOPY` 日志正确输出

### 1.2 Phase 2 问题陈述

当前 N≥2 Split 层的 `Forward_cpu()` 对每个 top Blob 执行 `std::memcpy`：
- 对于 N=4 扇出，一次 Forward 产生 4 次 memcpy
- 在大张量场景下（如 batch=32, feat_dim=2048，单 Blob 256KB），N=4 扇出一次 Forward 复制 1MB 数据
- 但很多分支**只读取不修改**数据（如残差连接的 identity 分支、Concat 输入）

### 1.3 COW 核心思想

> "Everyone gets a shared view until someone writes — then that someone pays the copy cost."

- Split 后所有 top Blob **共享** bottom 的 data/diff tensor（refcount++）
- 非 const 访问（写入意图）触发 **Copy-on-Write**：为当前 Blob 分配私有副本并复制数据
- const 访问始终零拷贝，不触发复制
- 这是经典的"惰性拷贝"优化，在多读取、少写入场景下收益显著

---

## 2. 设计方案

### 2.1 核心机制：COW 触发点

COW 的核心是在**获取可写指针时**检查引用计数：

```
┌─────────────────────────────────────────────────────────────┐
│  Blob::cpu_mutable_data()  ← 新增方法，所有写入路径走这里    │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  if (data_tensor_.use_count() > 1) {                  │  │
│  │      // 有人共享，需要私有化                            │  │
│  │      Tensor private_copy = CloneTensor(data_tensor_); │  │
│  │      data_tensor_ = private_copy;  // refcount--      │  │
│  │      CAFFE_FFI_MEM_LOG << "[COW] Unshared data";      │  │
│  │  }                                                    │  │
│  │  return static_cast<float*>(data_tensor_.data_ptr()); │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Blob 类 API 变更

| 方法 | Phase 1 | Phase 2 | 说明 |
|------|---------|---------|------|
| `const float* cpu_data() const` | 直接返回 data_ptr | **不变** | 只读访问，永远不触发 COW |
| `float* cpu_data()` | 直接返回 data_ptr | **触发 COW 检查** | 写入意图，use_count>1 时克隆 |
| `const float* cpu_diff() const` | 直接返回 diff_ptr | **不变** | 只读访问 |
| `float* cpu_diff()` | 直接返回 diff_ptr | **触发 COW 检查** | 写入意图 |
| `void ShareData(const Blob*)` | 直接赋值 tensor | **不变** | 建立共享关系 |
| `void ShareDiff(const Blob*)` | 直接赋值 tensor | **不变** | 建立共享关系 |
| `bool SharesDataWith(const Blob*)` | 比较 data_ptr | **不变** | 查询共享状态 |
| **新增** `bool IsDataShared() const` | - | 返回 `data_tensor_.use_count() > 1` | 查询是否在共享中 |
| **新增** `int DataRefCount() const` | - | 返回 `data_tensor_.use_count()` | 调试用 |
| **新增** `void* UnshareData()` | - | 强制克隆私有化 | 显式 COW |
| **新增** `void* UnshareDiff()` | - | 强制克隆私有化 | 显式 COW |
| `void Reshape(ShapeView)` | 分配新 tensor | **必须中断共享** | Reshape 总是分配私有内存 |

### 2.3 Tensor 克隆实现

```cpp
// 新增到 common.hpp 或 blob.cpp 的辅助函数
inline Tensor CloneTensor(const Tensor& src) {
  CAFFE_FFI_CHECK_TYPE(src.defined()) << "CloneTensor: source tensor is undefined";
  
  // 1. 分配新的私有 CPU tensor（同 shape、同 dtype）
  Tensor dst = NewCPUTensor(src.shape());
  
  // 2. 执行 memcpy（这是 COW 的唯一拷贝点）
  int64_t nbytes = src.numel() * (src.dtype().bits / 8);
  std::memcpy(dst.data_ptr(), src.data_ptr(), nbytes);
  
  CAFFE_FFI_MEM_LOG << "[COW] Cloned tensor: " << nbytes << "B ("
                    << FormatBytes(nbytes) << ")"
                    << " shape=" << ShapeToString(src.shape())
                    << " src_ptr=" << PtrToString(src.data_ptr())
                    << " dst_ptr=" << PtrToString(dst.data_ptr());
  
  return dst;
}
```

### 2.4 Split 层修改

**Phase 1 N=1 路径保持不变**（直接 ShareData/ShareDiff，无 COW 必要）。

**Phase 2 N≥2 路径修改**：

```cpp
// split_layer.cpp Forward_cpu() N≥2 分支
if (num_top == 1) {
  // Phase 1: N=1 零拷贝（保持不变）
  // ... ShareData + ShareDiff ...
  return;
}

// Phase 2: N≥2 COW 路径（替换原 memcpy 循环）
auto t0 = std::chrono::high_resolution_clock::now();

// 所有 top 共享 bottom 的 data/diff tensor（refcount 原子递增，无 memcpy）
for (int i = 0; i < num_top; ++i) {
  bool was_shared = top[i]->SharesDataWith(bottom[0]);
  top[i]->ShareData(bottom[0]);
  top[i]->ShareDiff(bottom[0]);
}

auto t1 = std::chrono::high_resolution_clock::now();
double share_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

// 计算预期内存节省：(N-1) 个 tensor 不再需要独立副本
int64_t bytes_saved_immediate = (num_top - 1) * copy_bytes_per_top;

CAFFE_FFI_LOG_WARN() << "[SPLIT-PERF] " << this->name()
                     << " Forward(N=" << num_top << " COW-SHARED): count=" << count
                     << " shared_bytes_per_top=" << copy_bytes_per_top << "B"
                     << " immediate_mem_saved=" << bytes_saved_immediate << "B"
                     << " share_time=" << share_us << "us"
                     << " data_refcount=" << bottom[0]->DataRefCount()
                     << " (cow: memcpy deferred until first write)";

// 注意：不再执行 memcpy！实际拷贝将在下游层首次写入某个 top 时通过 COW 触发
```

### 2.5 COW 触发场景审计（下游层分类）

Split 后的 top Blob 被各层消费时，按是否写入分为两类：

#### 只读层（不触发 COW，零拷贝贯穿）

| 层类型 | 访问模式 | COW 触发？ |
|--------|----------|-----------|
| `SplitLayer` 输入读取 | `const cpu_data()` | ❌ 不触发 |
| `ConcatLayer` 输入读取 | `const cpu_data()` | ❌ 不触发 |
| `EltwiseLayer`（Sum/Prod） | `const cpu_data()` | ❌ 不触发（输出到新 Blob） |
| `SoftmaxLayer` 前向 | `const cpu_data()` | ❌ 不触发（输出到 top） |
| `AccuracyLayer` | `const cpu_data()` | ❌ 不触发 |
| `SoftmaxWithLossLayer` | `const cpu_data()` | ❌ 不触发 |
| `BatchNormLayer` 前向（inference） | `const cpu_data()` | ❌ 不触发 |
| `ScaleLayer` 前向 | `const cpu_data()` | ❌ 不触发 |

#### 写入层（触发 COW，按需复制）

| 层类型 | 访问模式 | COW 触发？ | 备注 |
|--------|----------|-----------|------|
| `ReLULayer`（in-place） | `cpu_mutable_data()` → in-place 修改 | ✅ 触发 | 就地 ReLU 写自己的 top，触发 COW |
| `DropoutLayer`（in-place） | `cpu_mutable_data()` | ✅ 触发 | 就地 Dropout 写 mask |
| `ConvLayer`/`InnerProductLayer` | 输入 `const cpu_data()`，输出到新 Blob | ❌ 不触发 | 输入只读，输出是独立 Blob |
| `PoolingLayer` | 输入 `const cpu_data()`，输出到新 Blob | ❌ 不触发 | |
| `ReshapeLayer` | `Reshape()` 中断共享 | ✅ 触发（Reshape 时） | Reshape 强制分配私有内存 |
| `BatchNormLayer`（训练） | 写 diff/中间结果 | ⚠️ 需审计 | 可能通过 temp Blob 隔离 |
| 反向传播 `Backward_cpu()` | `cpu_mutable_diff()` | ✅ 触发 | 梯度写入触发 diff COW |

### 2.6 反向传播的 COW 处理

反向传播中 `diff_tensor_` 的 COW 逻辑与 data 对称：

- **Bottom Blob 的 diff**：多个 top 的梯度需要累加（`caffe_cpu_axpby`）到 bottom diff
- **COW 场景**：当多个 top 共享同一个 bottom diff 引用时，第一个写入者会 COW 出私有副本
- **问题**：梯度累加场景下，所有 top 都需要写入**同一个** bottom diff（不是各自写私有副本）
- **解决方案**：
  1. Split 层 `Backward_cpu()` 中，不共享 diff，而是显式分配独立的 bottom diff（或用第一个 top 的 diff）
  2. 或者：Net 在 backward 时为累加梯度的 Blob 调用 `UnshareDiff()` 预先私有化
  3. **推荐方案**：Split backward 中，将 bottom diff 直接设为第一个 top 的 diff（独占所有权），其余 top 的 diff 通过 axpby 累加到它——这是传统 Caffe 的做法，天然不需要 COW

### 2.7 Reshape() 对共享的中断

当前 Reshape 已经分配新的 data_tensor_ 和 diff_tensor_，这会自然中断共享：
- 如果 Blob 处于共享状态，Reshape 释放旧 tensor（refcount--），分配新 tensor（refcount=1）
- Reshape 后 Blob 必然处于私有状态
- 这符合语义：改变形状后不可能还共享旧形状的缓冲区

### 2.8 性能埋点增强

在现有 `[SPLIT-PERF]` 和 `[ZEROCOPY]` 日志基础上新增：

| 日志标签 | 触发点 | 字段 |
|----------|--------|------|
| `[COW] Unshared data` | cpu_mutable_data() 触发克隆 | blob_id, nbytes, src_refcount, copy_time_us |
| `[COW] Unshared diff` | cpu_mutable_diff() 触发克隆 | blob_id, nbytes, src_refcount, copy_time_us |
| `[SPLIT-PERF] COW-SHARED` | N≥2 Forward 共享完成 | num_top, shared_bytes, immediate_mem_saved, share_time_us, refcount_after |
| `[SPLIT-PERF] COW-COPY` | 下游层触发 COW（在 Blob 层记录，可关联 split 名） | split_name, cow_branch_index, layer_name, copy_bytes, copy_time_us |

CSV 性能日志新增字段：
- `operation` 列新增 `COW-Data`/`COW-Diff` 事件
- `extra_fields` 中包含 `refcount_before`, `copy_bytes`, `copy_us`

---

## 3. 关键安全约束

### 3.1 Rust 风格的"别名 XOR 可变性"原则

COW 本质上在 C++ 中模拟了 Rust 的借用规则：
- **多个只读引用**（`const float*`）：安全，可以任意共享
- **单个可写引用**（`float*`）：独占，必须私有化

```
Phase 1 (N=1): 只有一个 consumer，永远不写（Identity passthrough）→ 安全
Phase 2 (N≥2): 所有 consumer 一开始都是 Reader → 安全
               第一个 Writer 触发 COW → 变成私有 Writer + N-1 个 Reader → 安全
               后续 Writer 各自 COW → 多个私有 Writer + 剩余 Reader → 安全
```

### 3.2 禁止绕过 COW 的 API 路径

审计所有获取可写指针的入口，确保都经过 COW 检查：

1. ✅ `float* cpu_data()` → 调用 `UnshareData()` 后返回
2. ✅ `float* cpu_diff()` → 调用 `UnshareDiff()` 后返回
3. ✅ `void set_data(Tensor)` → 已经 memcpy 到本 Blob 的 data_tensor_，但需要确保先 Unshare
4. ✅ `void set_diff(Tensor)` → 同上
5. ⚠️ `void FromProto(...)` → 调用 `cpu_data()`/`cpu_diff()`，自动触发 COW
6. ⚠️ `void Update()` → 调用 `cpu_data()` 和 `cpu_diff()`，自动触发 COW
7. ⚠️ `Tensor data_tensor()` → **危险**：返回 Tensor 副本，外部可通过 `data_ptr()` 获取可写指针绕过 COW
   - **方案**：新增 `Tensor mutable_data_tensor()` 方法触发 COW 后返回；`data_tensor() const` 保持只读语义
   - 或者在文档中标注：通过 `data_tensor()` 获取的 Tensor 写入是 UB（undefined behavior）

### 3.3 FFI 边界安全

Python FFI 层通过 `_caffe_ffi.cc` 注册的方法：
- `get_data()` → 返回 `Array<float>`（拷贝），安全
- `set_data(Tensor)` → 走 memcpy 到本 Blob，需确保 Unshare
- Python 用户通过 `blob.cpu_data()` （如果暴露）获取指针时，同样触发 COW

---

## 4. 实现步骤与里程碑

### Phase 2a: COW 基础机制（预计 1-2 天）

- [ ] **Step 1**: 在 `blob.hpp` 声明 `cpu_mutable_data()`/`cpu_mutable_diff()`/`UnshareData()`/`UnshareDiff()`/`IsDataShared()`/`DataRefCount()`
- [ ] **Step 2**: 在 `blob.cpp` 实现 CloneTensor() 辅助函数和 Unshare*() 方法
- [ ] **Step 3**: 修改 `float* cpu_data()` 和 `float* cpu_diff()` 内联调用 Unshare
- [ ] **Step 4**: 添加 COW 日志埋点（`[COW]` 标签）
- [ ] **Step 5**: C++ 单元测试：`test_blob_cow.cpp`
  - 测试 Shared → Write → COW 触发后指针分离
  - 测试 use_count() 在 COW 前后的变化
  - 测试 const 访问不触发 COW
  - 测试 Reshape 中断共享
  - 测试多轮 COW（shared → cow1 → cow2 各自独立）

### Phase 2b: Split 层 N≥2 COW 路径（预计 1 天）

- [ ] **Step 6**: 修改 `SplitLayer::Forward_cpu()` N≥2 分支为 ShareData/ShareDiff 循环
- [ ] **Step 7**: 更新 Split 层 `[SPLIT-PERF]` 日志为 COW-SHARED 格式
- [ ] **Step 8**: 修改 `SplitLayer::Reshape()` 的 bytes_copied_per_fwd 计算逻辑（N≥2 时不再是 N*count*sizeof(float)）
- [ ] **Step 9**: 反向传播审计：确保 Backward 中 diff 累加的正确性

### Phase 2c: 下游层审计与修复（预计 1-2 天）

- [ ] **Step 10**: 审计所有 Layer 的 Forward_cpu/Backward_cpu，确认：
  - 只读访问使用 `const float*`（`cpu_data() const`）
  - 写入访问使用 `float*`（触发 COW 的 `cpu_data()`）
- [ ] **Step 11**: 特别检查 in-place 层（ReLU、Dropout）：
  - 确认它们正确调用 mutable 版本触发 COW
  - 确保 COW 后 top 和 bottom 的指针不再相等（防止意外的 in-place 语义问题）
- [ ] **Step 12**: 检查 Net::Forward() 中是否有直接指针操作绕过 Blob API

### Phase 2d: 测试与性能验证（预计 1 天）

- [ ] **Step 13**: 扩展 P2-B 回归测试：
  - 新增 `test_split_cow_n2_readonly`：N=2 两个分支都只读，验证无 COW 发生，refcount=N+1
  - 新增 `test_split_cow_n2_one_writer`：一个分支 ReLU（in-place），验证一次 COW，其他分支仍共享
  - 新增 `test_split_cow_n2_all_writers`：N=2 两个分支都写，验证两次 COW，最终完全分离
  - 新增 `test_split_cow_reshape_breaks`：Reshape 后验证共享中断
- [ ] **Step 14**: 性能基准测试：
  - 对比 Phase 1 (memcpy N≥2) vs Phase 2 (COW N≥2) 在不同 read/write ratio 下的性能
  - 记录 CSV 性能日志中的 COW 事件数量和总 COW 字节数
  - 计算"有效内存节省率" = (immediate_mem_saved - total_cow_bytes) / (N * per_top_bytes)
- [ ] **Step 15**: C++ 单元测试全部通过
- [ ] **Step 16**: Python P2-B 回归测试全部 29 项通过（原有）+ 新增 COW 测试通过

---

## 5. 预期收益分析

### 5.1 内存节省

| 场景 | Phase 1 (N≥2 memcpy) | Phase 2 (COW) | 节省 |
|------|---------------------|---------------|------|
| N=2, 1写1读（残差连接） | 2份副本 | 1.5份（1共享+1COW） | 25% |
| N=4, 1写3读（多任务学习） | 4份副本 | 1.25份（1共享+1COW） | 69% |
| N=4, 全只读（Concat前置） | 4份副本 | 1份（全共享） | 75% |
| N=8, 全只读（Deep Supervision） | 8份副本 | 1份（全共享） | 87.5% |

### 5.2 性能提升（前向传播）

- **Split 层本身**：N×memcpy → N×refcount 原子递增（纳秒级）
- **对于只读分支**：整个数据通路零拷贝
- **对于写入分支**：多一次 COW 检查 + 一次克隆 memcpy（等同于原来的 memcpy，无额外开销）
- **最坏情况**（所有分支都写）：等价于原 memcpy（一次 refcount 原子操作的额外开销，可忽略）
- **最好情况**（全只读）：N×memcpy → 0 memcpy

### 5.3 典型模型收益预估

- **ResNet 残差块**：每个残差 Split N=2，一个分支（conv）写，一个分支（identity）读 → 约 25% 内存节省，Split 前向延迟接近 0
- **Inception 模块**：N=4-6 分支，部分只读（1x1, 3x3, 5x5, pool 都各自写输出，但输入读取共享）→ 输入侧零拷贝
- **FPN/SSD 多尺度检测头**：N=4-6 个检测头共享 backbone 特征 → 显著内存节省

---

## 6. 风险与回退

### 6.1 潜在风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 某个 Layer 绕过 Blob API 直接写共享 tensor | 静默数据损坏 | 审计所有 Layer，添加 debug 模式下的 refcount 断言 |
| COW 触发时机不正确（该触发时没触发） | 数据污染 | 在 `float* cpu_data()` 唯一入口触发，代码量小易审计 |
| COW 触发过于激进（不该触发时触发） | 性能回退 | const 重载保证只读路径零开销；添加 COW 计数监控 |
| 反向传播梯度累加错误 | 训练不收敛 | Split backward 保守处理：diff 不共享，显式累加 |
| Python FFI 暴露 mutable_tensor 逃逸 | Python 端绕过 COW | FFI 层只暴露 `get_data()`（拷贝）和 `set_data()`（memcpy 到私有） |

### 6.2 回退策略

- **编译期开关**：在 `cmake/Options.cmake` 添加 `CAFFE_FFI_ENABLE_COW=ON/OFF`，默认 OFF
- **运行期开关**：环境变量 `CAFFE_FFI_DISABLE_COW=1` 强制走 memcpy 路径（用于紧急回退）
- **A/B 测试**：通过开关对比 COW 开启/关闭的正确性和性能

---

## 7. 与现有机制的兼容性

### 7.1 Phase 1 N=1 路径

**完全不受影响**。N=1 时仍然走现有的 ShareData/ShareDiff 捷径，不经过 N≥2 的 COW 逻辑。

### 7.2 TVM FFI DLPack 互操作

- `data_tensor() const` 返回共享 Tensor 的只读视图：安全，DLPack 消费者只读
- 若需通过 DLPack 写入，应先调用 `mutable_data_tensor()` 触发 COW

### 7.3 内存统计（g_total_allocated_bytes）

- COW 克隆时需要正确更新全局计数器：
  - CloneTensor 分配新内存：`g_total_allocated_bytes += nbytes`
  - 旧 tensor 释放（refcount--）：由 Tensor 析构时自动处理（需确认 CPUMemAlloc 的 deleter 中是否递减）
  - **当前审计点**：确认 `CPUMemAlloc` 的 deleter 中是否有 `g_total_allocated_bytes -= size`

### 7.4 Reshape() 零拷贝中断

现有 Reshape 已经自然处理：分配新 tensor → 旧 tensor refcount-- → 新 tensor refcount=1（私有）。无需额外修改。

---

## 8. 附录

### 8.1 参考实现先例

- **Rust `std::sync::Arc`** + `make_mut()`：标准库 COW 模式
- **PyTorch `Tensor.storage()`** + 隐式 COW（历史版本）
- **Caffe2 原始 Split 层**：总是 memcpy，但 Caffe2 后来引入了 "unsafe share" 供优化器使用
- **TVM Relay `TupleGetItem`**：函数式 IR 天然共享，写入通过 op 产生新 tensor

### 8.2 TVM FFI ObjectPtr 引用计数 API

```cpp
// 来自 tvm::ffi::ObjectRef（Tensor 的基类）
int use_count() const;     // 返回当前强引用计数
bool unique() const;       // 等价于 use_count() == 1
bool defined() const;      // 是否已关联对象
```

原子操作保证线程安全（但 caffe-ffi 当前假设单线程前向/反向，暂不考虑多线程）。

### 8.3 CloneTensor 注意事项

- 必须保持 strides/byte_offset 一致（当前 caffe-ffi 只用 contiguous tensor，所以直接 memcpy 即可）
- 只支持 CPU float32（当前 Blob 只支持 CPU float32，未来扩展 GPU/其他 dtype 时再泛化）
- 克隆后的 tensor 使用独立的 CPUMemAlloc 分配，生命周期独立

---

## 9. 开放问题（需决策）

1. **Q**: `data_tensor()` 方法返回 Tensor 是否需要在 mutable 上下文中触发 COW？
   - **选项 A**: 保持现状（返回 Tensor 副本不触发 COW），文档标注写入 UB
   - **选项 B**: 新增 `mutable_data_tensor()` 触发 COW，`data_tensor() const` 保留只读
   - **建议**: 选项 B（更安全）

2. **Q**: 反向传播中 diff 是否也走 COW？
   - **选项 A**: diff 不共享（保守，Split backward 显式累加）
   - **选项 B**: diff 也走 COW（对称但增加复杂度）
   - **建议**: 选项 A（Phase 2a/b 先只优化 data，diff 走保守路径；Phase 3 再考虑 diff COW）

3. **Q**: 是否需要在 debug 模式下检测"共享 tensor 被意外写入"？
   - **选项 A**: 添加 mprotect/VirtualProtect 写保护（复杂，跨平台难）
   - **选项 B**: 仅依赖代码审计和 use_count 断言
   - **建议**: 选项 B（保持简单，核心路径只有 cpu_mutable_data() 一个写入入口）
