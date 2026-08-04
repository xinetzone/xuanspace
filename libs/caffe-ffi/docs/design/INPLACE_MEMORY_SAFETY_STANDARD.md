# In-Place 操作内存安全规范

> **日期**: 2026-08-04
> **来源**: Task 17b ASan 内存安全验证（发现并修复 in-place InnerProduct 堆越界读）
> **适用范围**: caffe-ffi 所有 Layer 的 in-place 操作实现与测试
> **级别**: 内部技术规范

---

## 一、背景与动机

ASan 内存安全验证在 `test_inplace_chain_forward` 中捕获了 **1 处真实堆越界读**（heap-buffer-overflow）：in-place InnerProduct 层在 `num_output` 改变尺寸时，`Reshape` 截断共享缓冲区后，`Forward_cpu` 仍按旧尺寸读取数据，越界访问缓冲区末尾之外的内存。

本规范萃取该次验证经验，作为 caffe-ffi 所有 Layer 在 in-place 操作上的强制内存安全约束，防止同类缺陷在新增层（尤其 P4 能力扩展）中复现。

## 二、In-Place 操作语义

**In-place 操作**是指 Layer 的 `top[0]` 与 `bottom[0]` 指向**同一个 Blob**（`bottom[0] == top[0]`），即输出复用输入的内存缓冲区。在 proto 中形式为：

```protobuf
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "x"
  top: "x"   # 与 bottom 同名 → in-place
}
```

### 2.1 In-place 的合法性前提

**核心原则：in-place 仅当输出形状与输入形状完全一致（count 相等）时才安全。**

理由：in-place 意味着输出写入输入缓冲区。若输出形状与输入形状不同（count 不同），`Reshape` 会改变共享缓冲区大小，但 `Forward`/`Backward` 仍按各层缓存的 `M_/K_/N_` 等维度读取数据，导致**越界读或写**。

### 2.2 安全 in-place 的典型场景

| 场景 | 说明 | 是否安全 |
|------|------|---------|
| ReLU/Sigmoid/TanH 等逐元素激活 | 输出形状 = 输入形状 | ✅ 安全 |
| Dropout（推理模式恒等） | 输出形状 = 输入形状 | ✅ 安全 |
| 形状不变的 Eltwise/Scale | 输出形状 = 输入形状 | ✅ 安全 |
| **InnerProduct 且 num_output == 输入特征维** | 输出 count = 输入 count | ✅ 安全（罕见） |
| **InnerProduct 且 num_output ≠ 输入特征维** | 输出 count ≠ 输入 count | ❌ **不安全，必须拒绝** |
| Conv/Deconv/Pooling（形状变化） | 输出形状 ≠ 输入形状 | ❌ **不安全，必须拒绝** |

## 三、安全规范（强制约束）

### 规则 1：形状变化层必须拒绝 in-place

任何会改变输出形状（输出 count ≠ 输入 count）的 Layer，在 `Reshape` 中必须检测 in-place 并抛错拒绝。**禁止**静默容忍或尝试"修复"。

参考实现（`InnerProductLayer::Reshape`，`src/caffe_ffi/layers/inner_product_layer.cpp`）：

```cpp
// In-place 安全守卫：InnerProduct 输出尺寸(N_)与输入尺寸(K_)通常不同，
// 若 top 与 bottom 共享同一 Blob（in-place），top[0]->Reshape(top_shape) 会
// 改变共享缓冲区大小，导致 Forward 时按旧尺寸（M*K）读取数据越界。
if (bottom[0] == top[0]) {
  const int64_t bottom_count = bottom[0]->count();
  const int64_t top_count = static_cast<int64_t>(M_) * static_cast<int64_t>(N_);
  if (top_count != bottom_count) {
    CAFFE_FFI_CHECK_VALUE_EQ(top_count, bottom_count)
        << "InnerProduct in-place operation requires input and output "
        << "to have the same total count (M*N == M*K), but got bottom_count="
        << bottom_count << " (M*K=" << M_ << "*" << K_ << ") vs top_count="
        << top_count << " (M*N=" << M_ << "*" << N_ << "). In-place "
        << "InnerProduct with num_output != input feature dim is unsupported.";
  }
}
```

### 规则 2：in-place 判定必须基于 `bottom[0] == top[0]`

守卫判定使用指针比较 `bottom[0] == top[0]`，而非 blob 名称比较。名称相同不代表同一对象（可能被重命名），指针相等才是共享内存的充分条件。

### 规则 3：count 计算必须统一口径

在守卫中，`bottom_count` 用 `bottom[0]->count()`，`top_count` 用 `M_ * N_`（与 `Reshape` 中 `top[0]->Reshape(top_shape)` 生成的输出 count 一致）。避免用 `shape` 各维手工相乘导致与 `count()` 计算不一致。

### 规则 4：守卫必须位于 `Reshape` 破坏缓冲区之前

守卫必须在 `top[0]->Reshape(top_shape)` **之前**执行。一旦 Reshape 已截断共享缓冲区，再做校验为时已晚（缓冲区已损坏）。

### 规则 5：新增层必须覆盖 in-place 负向测试

每个允许/拒绝 in-place 的新 Layer，必须同时提供：
- **正向测试**：合法 in-place 场景（形状不变）正常工作
- **负向测试**：非法 in-place 场景（形状变化）被拒绝（抛异常）

负向测试参考（`tests/python/test_complex_topologies.py`）：

```python
def test_inplace_inner_product_shape_change_rejected(self, ptrace):
    """In-place InnerProduct with output size != input size is rejected (ASan guard)."""
    proto = """name: "inplace_ip_shape_change"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape { dim: 2 dim: 4 } }
}
layer {
  name: "ip"
  type: "InnerProduct"
  bottom: "data"
  top: "data"
  inner_product_param { num_output: 2 bias_term: true }
}
"""
    with ptrace("in-place InnerProduct shape-change → rejected"):
        with pytest.raises((ValueError, RuntimeError)):
            net_from_param(net_param_from_string(proto))
```

## 四、反模式（Anti-patterns）

| 反模式 | 危害 | 正确做法 |
|--------|------|---------|
| 静默容忍 in-place 形状变化 | 缓冲区被截断后越界读/写，ASan 才能检测，普通运行可能产生脏数据 | 显式抛错拒绝 |
| 用名称而非指针判断 in-place | 同名≠同对象，误判导致漏检或误拒 | 用 `bottom[0] == top[0]` |
| 在 `Reshape` 之后才校验 | 缓冲区已损坏，校验无意义 | 在 `Reshape` 之前守卫 |
| 测试只覆盖合法 in-place | 非法场景的守卫逻辑不被覆盖 | 必须加负向测试 |
| 手动计算 count 而非用 `count()` | 维度算错导致守卫误判 | 统一用 `count()` 与 `M_*N_` |

## 五、验证方法（ASan 守卫闭环）

新增/修改 in-place 相关 Layer 后，按下述流程验证：

1. **单元测试**：正向 + 负向 in-place 用例通过
2. **ASan 构建**：`-DCAFFE_FFI_ENABLE_ASAN=ON -O1` 构建，运行全量测试
   ```bash
   unset CFLAGS CXXFLAGS LDFLAGS
   cmake --preset default -DCMAKE_C_FLAGS="-O1 -fno-omit-frame-pointer -fPIC" \
     -DCMAKE_CXX_FLAGS="-O1 -fno-omit-frame-pointer -fPIC" \
     -DCAFFE_FFI_ENABLE_ASAN=ON -DCAFFE_FFI_ENABLE_COW=ON \
     -DCAFFE_FFI_ENABLE_COW_PHASE3=ON -DCAFFE_FFI_BUILD_TESTS=ON
   cmake --build --preset default -j$(nproc)
   export ASAN_OPTIONS="detect_leaks=0:halt_on_error=1:abort_on_error=1"
   LD_PRELOAD=$(gcc -print-file-name=libasan.so) python -m pytest tests/python -q
   ```
3. **验收标准**：`0 ASan 内存安全错误`，全量测试通过

## 六、相关文档

- [ASan 验证报告](setup/ASAN_VERIFICATION_REPORT_20260804.md)（Task 17b 完整结果）
- [ASan 报告堆栈解读指南](setup/ASAN_REPORT_READING_GUIDE.md)
- `src/caffe_ffi/layers/inner_product_layer.cpp`（守卫实现）
- `tests/python/test_complex_topologies.py`（负向测试）