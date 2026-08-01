# Backward_cpu 日志埋点扩展计划

> **日期**: 2026-07-31
> **状态**: 待实施
> **影响范围**: `include/caffe_ffi/layer.hpp`、`src/caffe_ffi/layer.cpp`、所有激活层、所有带参数层

---

## 1. 背景与目标

当前 `Layer` 基类仅声明了 `Forward_cpu` 纯虚方法，框架为纯推理模式。`OPTIMIZATION_REPORT.md` 第6节已将"反向传播"列为后续优化项。本计划聚焦于：在 `Layer` 基类中引入 `Backward_cpu` 接口，并复用前向传播已验证的"计算 + 统计单次遍历"日志埋点模式，为后续训练支持奠定可观测性基础。

**目标**：
1. 在 `layer.hpp` 中声明 `Backward_cpu` 接口及相关基础设施
2. 在 `layer.cpp` 中实现 `Backward` 编排方法（对称于 `Forward`）
3. 为所有已实现层添加 `Backward_cpu` 存根或实现，包含性能日志埋点
4. 确保日志格式与前向传播一致，便于统一性能分析

**非目标**：
- 实现完整的反向传播计算逻辑（仅搭框架，计算逻辑后续迭代）
- 实现 GPU 反向传播（`Backward_gpu`）
- 实现优化器/参数更新逻辑

---

## 2. 当前代码基线

### 2.1 Layer 基类（layer.hpp）

```cpp
class Layer : public tvm::ffi::Object {
 public:
  // ... 现有公共接口 ...
  void Forward(const std::vector<Blob*>& bottom, const std::vector<Blob*>& top);
  // Backward 方法缺失

 protected:
  caffe::LayerParameter layer_param_;
  std::vector<ObjectPtr<Blob>> blobs_;
  std::vector<bool> param_propagate_down_;
  std::vector<float> loss_;

  virtual void Forward_cpu(const std::vector<Blob*>& bottom,
                           const std::vector<Blob*>& top) = 0;
  // Backward_cpu 缺失

  void CheckBlobCounts(const std::vector<Blob*>& bottom,
                       const std::vector<Blob*>& top);
  void SetLossWeights(const std::vector<Blob*>& top);
};
```

### 2.2 Layer 实现（layer.cpp）

`Forward` 方法的编排模式：
1. 检查 Blob 数量
2. 调用 `Forward_cpu(bottom, top)`
3. 设置 loss 权重

### 2.3 前向日志模式参考（以 Sigmoid 为例）

```cpp
void SigmoidLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                               const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->cpu_mutable_data();
  const int64_t count = bottom[0]->count();
  CAFFE_FFI_LAYER_LOG << "Sigmoid Forward_cpu: count=" << count;

  auto t_start = std::chrono::high_resolution_clock::now();

  float in_min = std::numeric_limits<float>::max();
  float in_max = -std::numeric_limits<float>::max();
  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();

  for (int64_t i = 0; i < count; ++i) {
    float x = bottom_data[i];
    float y = 1.0f / (1.0f + std::exp(-x));
    top_data[i] = y;
    in_min = std::min(in_min, x);
    in_max = std::max(in_max, x);
    out_min = std::min(out_min, y);
    out_max = std::max(out_max, y);
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[ACTIVATION-PERF] " << this->name()
                       << " Sigmoid forward: count=" << count
                       << " in=[" << in_min << ", " << in_max << "]"
                       << " out=[" << out_min << ", " << out_max << "]"
                       << " time=" << elapsed_us << "us";
}
```

---

## 3. 修改计划

### 3.1 第一阶段：layer.hpp 接口扩展

**文件**: `include/caffe_ffi/layer.hpp`

在 `protected` 区域添加：

```cpp
/// @brief 反向传播 CPU 实现（纯虚方法，子类必须实现）
/// @param top    顶层 Blob 指针（包含梯度 top[0]->cpu_diff()）
/// @param propagate_down  对每个 bottom 输入是否需要计算梯度
/// @param bottom 底层 Blob 指针（梯度写入 bottom[0]->mutable_cpu_diff()）
virtual void Backward_cpu(const std::vector<Blob*>& top,
                          const std::vector<bool>& propagate_down,
                          const std::vector<Blob*>& bottom) = 0;
```

在 `public` 区域添加：

```cpp
/// @brief 反向传播编排入口（对称于 Forward）
/// @param top    顶层 Blob（含输出值和梯度）
/// @param propagate_down  是否对每个 bottom 传播梯度
/// @param bottom 底层 Blob（梯度写入）
void Backward(const std::vector<Blob*>& top,
              const std::vector<bool>& propagate_down,
              const std::vector<Blob*>& bottom);
```

**设计决策**：
- `Backward_cpu` 参数顺序遵循 Caffe 惯例：`(top, propagate_down, bottom)`，与 Caffe 原始代码一致
- `propagate_down` 允许跳过不需要梯度的 bottom 输入，避免不必要的计算
- 保持纯虚方法，强制所有现有层必须实现（第一阶段可以存根 + NOT_IMPLEMENTED 日志）

### 3.2 第二阶段：layer.cpp Backward 编排实现

**文件**: `src/caffe_ffi/layer.cpp`

实现 `Backward` 方法，结构对称于 `Forward`：

```cpp
void Layer::Backward(const std::vector<Blob*>& top,
                     const std::vector<bool>& propagate_down,
                     const std::vector<Blob*>& bottom) {
  CheckBlobCounts(bottom, top);
  switch (this->phase_) {
  case TRAIN:
    Backward_cpu(top, propagate_down, bottom);
    break;
  case TEST:
    // 测试阶段不需要反向传播
    break;
  default:
    LOG(FATAL) << "Unknown phase.";
  }
}
```

**注意事项**：
- 需要在 `Layer` 类中添加 `phase_` 成员变量（`caffe::Phase`），在 `SetUp`/`LayerSetUp` 中初始化
- 若框架暂不支持 phase 概念，可简化为直接调用 `Backward_cpu`

### 3.3 第三阶段：各层 Backward_cpu 实现

按层类型分批次实现，优先级如下：

| 优先级 | 层类型 | 代表层 | 反向传播复杂度 |
|---|---|---|---|
| P0 | 激活层 | ReLU, Sigmoid, TanH, PReLU | 低（逐元素操作） |
| P1 | 损失层 | SoftmaxWithLoss, EuclideanLoss | 中（需数值稳定性处理） |
| P2 | 线性层 | InnerProduct | 中（矩阵乘） |
| P3 | 规范化层 | BatchNorm, Scale | 中 |
| P4 | 其他层 | Concat, Dropout, Convolution, Pooling | 高 |

#### 3.3.1 激活层 Backward_cpu 实现模板

以 Sigmoid 为例，反向传播公式为 `dX = dY * Y * (1 - Y)`，其中 Y 已在前向传播中计算并存储在 top[0] 中：

```cpp
void SigmoidLayer::Backward_cpu(const std::vector<Blob*>& top,
                                const std::vector<bool>& propagate_down,
                                const std::vector<Blob*>& bottom) {
  if (!propagate_down[0]) return;  // 不需要梯度则跳过

  const float* top_data = top[0]->cpu_data();      // 前向输出 Y
  const float* top_diff = top[0]->cpu_diff();      // 上游梯度 dY
  float* bottom_diff = bottom[0]->cpu_mutable_diff();  // 写入梯度 dX
  const int64_t count = bottom[0]->count();

  CAFFE_FFI_LAYER_LOG << "Sigmoid Backward_cpu: count=" << count;

  auto t_start = std::chrono::high_resolution_clock::now();

  float diff_in_min = std::numeric_limits<float>::max();
  float diff_in_max = -std::numeric_limits<float>::max();
  float diff_out_min = std::numeric_limits<float>::max();
  float diff_out_max = -std::numeric_limits<float>::max();
  int64_t saturated_count = 0;  // 梯度消失计数

  for (int64_t i = 0; i < count; ++i) {
    float dy = top_diff[i];
    float y = top_data[i];
    float dx = dy * y * (1.0f - y);  // sigmoid'(x) = y * (1-y)
    bottom_diff[i] = dx;

    diff_in_min = std::min(diff_in_min, dy);
    diff_in_max = std::max(diff_in_max, dy);
    diff_out_min = std::min(diff_out_min, dx);
    diff_out_max = std::max(diff_out_max, dx);

    // 梯度消失检测：y 接近 0 或 1 时，梯度被严重压缩
    if (y < 1e-4f || y > 1.0f - 1e-4f) {
      saturated_count++;
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  float saturate_ratio = static_cast<float>(saturated_count) / count;

  CAFFE_FFI_LOG_INFO() << "[ACTIVATION-PERF] " << this->name()
                       << " Sigmoid backward: count=" << count
                       << " diff_in=[" << diff_in_min << ", " << diff_in_max << "]"
                       << " diff_out=[" << diff_out_min << ", " << diff_out_max << "]"
                       << " saturate=" << saturated_count << "/" << count
                       << " (" << saturate_ratio << ")"
                       << " time=" << elapsed_us << "us";
}
```

#### 3.3.2 ReLU Backward_cpu

```cpp
// ReLU 反向：dx = dy * (x > 0 ? 1 : 0)
// dead_count 统计 dy != 0 但 x <= 0 导致梯度被杀死的元素
```

#### 3.3.3 PReLU Backward_cpu

```cpp
// PReLU 反向：
//   dx = dy * (x > 0 ? 1 : slope)  （对输入的梯度）
//   dslope = sum(dy * x) for x < 0 （对 slope 参数的梯度）
// 日志额外记录：slope_grad 统计信息
```

### 3.4 第四阶段：Python 绑定扩展

**文件**: `src/caffe_ffi/_caffe_ffi.cc` 和 `python/caffe_ffi/layer.py`

- 在 FFI 注册中添加 `Backward` 方法导出
- Python 端 `Layer` 类添加 `backward()` 方法，透传 C++ 调用

---

## 4. 反向传播日志格式规范

反向传播日志复用 `[ACTIVATION-PERF]` 标签，格式与前向传播对齐：

```
[ACTIVATION-PERF] {layer_name} {LayerType} backward: count={N} \
  diff_in=[{min}, {max}] diff_out=[{min}, {max}] \
  {layer_specific_metrics} \
  time={elapsed_us}us
```

### 4.1 通用指标

| 指标 | 说明 |
|---|---|
| `count` | 元素总数 |
| `diff_in` | 输入梯度（上游梯度）的值域 `[min, max]` |
| `diff_out` | 输出梯度（本层计算的梯度）的值域 `[min, max]` |
| `time` | 计算耗时（微秒） |

### 4.2 层特有指标

| 层 | 特有指标 | 含义 |
|---|---|---|
| Sigmoid/TanH | `saturate=N/M (ratio)` | 饱和区元素数/总数，用于检测梯度消失 |
| ReLU | `dead=N/M (ratio)` | 死亡 ReLU 计数（前向输出 ≤0 且有梯度流入） |
| PReLU | `slope_grad=[min, max]` | slope 参数梯度的值域 |
| InnerProduct | `w_grad_norm`, `b_grad_norm` | 权重/偏置梯度的 L2 范数 |
| Dropout | `mask_density` | dropout mask 的保留比例 |

---

## 5. 测试策略

### 5.1 单元测试

在 `tests/python/test_layers.py`（或新的 `test_p3c_backward.py`）中添加：

1. **梯度数值检验**：数值梯度（finite difference）vs 解析梯度，误差 < 1e-5
2. **梯度消失检测**：Sigmoid/TanH 在极端输入下的梯度压缩比
3. **ReLU dead neuron 检测**：负输入 + 非零梯度 → 零输出梯度
4. **性能日志验证**：Backward 执行后日志中包含 `[ACTIVATION-PERF]` 且 `backward` 关键字存在
5. **propagate_down 跳过验证**：`propagate_down[0]=false` 时不计算梯度，diff 缓冲区不被修改

### 5.2 梯度检查工具函数

```python
def numerical_gradient(f, x, eps=1e-6):
    """数值梯度：(f(x+eps) - f(x-eps)) / (2*eps)"""
    ...

def check_gradient(layer, bottom, top, eps=1e-6, rtol=1e-5):
    """对比数值梯度和反向传播梯度"""
    ...
```

---

## 6. 实施顺序与依赖关系

```
阶段1 (layer.hpp 接口) ──→ 阶段2 (layer.cpp Backward编排)
                              │
                              ├──→ 阶段3.1 激活层 Backward_cpu (P0)
                              │       ├── ReLU
                              │       ├── Sigmoid
                              │       ├── TanH
                              │       └── PReLU
                              │
                              ├──→ 阶段4 Python绑定扩展
                              │
                              ├──→ 阶段5.1 激活层梯度测试
                              │
                              ├──→ 阶段3.2 损失层 Backward_cpu (P1)
                              ├──→ 阶段3.3 线性层 Backward_cpu (P2)
                              └──→ 阶段3.4 其他层 (P3/P4)
```

**最小可行落地（MVP）**：完成阶段1 + 阶段2 + Sigmoid/ReLU Backward_cpu + 梯度测试。

---

## 7. 风险与注意事项

1. **Blob diff 缓冲区生命周期**：反向传播需要读写 `cpu_diff()`，需确保 Blob 在 Reshape 时分配了 diff 内存（当前 Blob 构造时 data 和 diff 都分配了，无需额外修改）
2. **线程安全**：Forward 中使用的 `std::chrono` 是线程局部的，Backward 中同样使用即可
3. **日志性能开销**：`[ACTIVATION-PERF]` 使用 `CAFFE_FFI_LOG_INFO()`，在 Release 构建中若日志级别低于 INFO 则不会产生 I/O 开销。统计变量的 min/max 累加开销为 O(N)，已融合在计算循环中，无额外遍历
4. **数值稳定性**：Sigmoid/TanH 反向传播涉及乘法，极端值可能下溢为 0。饱和计数指标可帮助识别此类问题
5. **纯虚方法破坏性变更**：添加 `Backward_cpu = 0` 后所有现有层必须实现该方法，否则编译失败。建议第一阶段提供默认实现（打 NOT_IMPLEMENTED 日志后返回），待各层实现后再改为纯虚

**建议**：第一阶段 `Backward_cpu` 不设为纯虚，而是提供带默认日志的虚方法：

```cpp
virtual void Backward_cpu(const std::vector<Blob*>& top,
                          const std::vector<bool>& propagate_down,
                          const std::vector<Blob*>& bottom) {
  CAFFE_FFI_LOG_WARN() << "Backward_cpu not implemented for "
                       << this->type() << " layer: " << this->name();
}
```

这样不破坏现有层的编译，各层可独立迭代实现。

---

## 8. 验收标准

- [ ] `layer.hpp` 声明 `Backward` 和 `Backward_cpu`，`Backward_cpu` 有默认实现
- [ ] `layer.cpp` 实现 `Backward` 编排方法
- [ ] 至少 Sigmoid 和 ReLU 实现了 `Backward_cpu` 并带性能日志埋点
- [ ] 反向传播日志格式符合第4节规范
- [ ] Python 绑定暴露 `backward()` 方法
- [ ] 梯度数值检查测试通过（误差 < 1e-5）
- [ ] 所有现有前向传播测试仍然通过（无回归）
- [ ] 编译零警告
