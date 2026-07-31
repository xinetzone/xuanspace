# 激活层性能监控规范：计算 + 统计单次遍历模式

> **版本**: 1.0
> **日期**: 2026-07-31
> **适用范围**: caffe-ffi 所有计算层（激活层、损失层、线性层等）
> **状态**: 生效

---

## 1. 核心原则

### 1.1 单次遍历原则

**黄金法则**：数据遍历循环中同时完成计算和统计，**禁止**为统计而二次遍历数组。

```cpp
// ✅ 正确：计算 + 统计在同一次遍历中完成
for (int64_t i = 0; i < count; ++i) {
    float x = bottom_data[i];
    float y = compute(x);        // 计算
    top_data[i] = y;
    in_min = std::min(in_min, x);  // 统计
    in_max = std::max(in_max, x);
    out_min = std::min(out_min, y);
    out_max = std::max(out_max, y);
}

// ❌ 错误：先计算再遍历统计（O(2N)，cache 不友好）
for (int64_t i = 0; i < count; ++i) {
    top_data[i] = compute(bottom_data[i]);
}
for (int64_t i = 0; i < count; ++i) {  // 二次遍历！
    in_min = std::min(in_min, bottom_data[i]);
    // ...
}
```

**理论依据**：对于大张量（≥1M floats），二次遍历导致 L1/L2 cache miss 翻倍。单次遍历模式在统计开销 < 5% 的前提下，保证 O(N) 时间复杂度且 cache 友好。

### 1.2 零额外分配原则

统计变量全部使用栈上 `float`/`int64_t` 局部变量，禁止在循环中进行堆分配或调用可能分配内存的函数。

### 1.3 日志不阻塞计算原则

计时使用 `std::chrono::high_resolution_clock`，日志 I/O 在循环结束后执行，**禁止**在循环内部调用日志宏。

---

## 2. 标准日志格式

### 2.1 日志标签

所有性能监控日志使用统一标签：`[ACTIVATION-PERF]`

> 注：未来扩展到非激活层时，可引入 `[LAYER-PERF]`、`[LOSS-PERF]` 等子标签。

### 2.2 前向传播日志格式

```
[ACTIVATION-PERF] {layer_name} {LayerType} forward: count={N} \
  [{param_kv_pairs}] \
  in=[{in_min}, {in_max}] out=[{out_min}, {out_max}] \
  time={elapsed_us}us
```

### 2.3 反向传播日志格式

```
[ACTIVATION-PERF] {layer_name} {LayerType} backward: count={N} \
  [{param_kv_pairs}] \
  diff_in=[{d_in_min}, {d_in_max}] diff_out=[{d_out_min}, {d_out_max}] \
  [{layer_specific_metrics}] \
  time={elapsed_us}us
```

### 2.4 字段说明

| 字段 | 类型 | 必选 | 说明 |
|---|---|:---:|---|
| `layer_name` | string | ✅ | 层实例名称（`this->name()`） |
| `LayerType` | string | ✅ | 层类型名（如 ReLU, Sigmoid, PReLU） |
| `forward/backward` | string | ✅ | 传播方向标识 |
| `count` | int | ✅ | 处理的元素总数 |
| `param_kv_pairs` | k=v 对 | 条件 | 层配置参数（如 `channel_shared=true`） |
| `in/out` | [min, max] | ✅ | 输入/输出值域 |
| `diff_in/diff_out` | [min, max] | 反向 | 输入/输出梯度值域 |
| `layer_specific_metrics` | k=v 对 | 条件 | 层特有诊断指标 |
| `time` | float+us | ✅ | 计算耗时（微秒） |

---

## 3. 各层实现规范

### 3.1 无参数激活层（ReLU, Sigmoid, TanH）

**代表实现**：SigmoidLayer

```cpp
void XxxLayer::Forward_cpu(const std::vector<Blob*>& bottom,
                            const std::vector<Blob*>& top) {
  const float* bottom_data = bottom[0]->cpu_data();
  float* top_data = top[0]->mutable_cpu_data();
  const int64_t count = bottom[0]->count();
  CAFFE_FFI_LAYER_LOG << "Xxx Forward_cpu: count=" << count;

  auto t_start = std::chrono::high_resolution_clock::now();

  float in_min = std::numeric_limits<float>::max();
  float in_max = -std::numeric_limits<float>::max();
  float out_min = std::numeric_limits<float>::max();
  float out_max = -std::numeric_limits<float>::max();

  // 可选：层特有诊断计数器
  int64_t special_count = 0;

  for (int64_t i = 0; i < count; ++i) {
    float x = bottom_data[i];
    float y = /* 计算逻辑 */;
    top_data[i] = y;

    in_min = std::min(in_min, x);
    in_max = std::max(in_max, x);
    out_min = std::min(out_min, y);
    out_max = std::max(out_max, y);

    // 可选：层特有条件计数
    if (/* 异常/特殊值条件 */) {
      special_count++;
    }
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_us = std::chrono::duration<double, std::micro>(t_end - t_start).count();

  CAFFE_FFI_LOG_INFO() << "[ACTIVATION-PERF] " << this->name()
                       << " Xxx forward: count=" << count
                       << " in=[" << in_min << ", " << in_max << "]"
                       << " out=[" << out_min << ", " << out_max << "]"
                       // 可选特有指标
                       << " special=" << special_count << "/" << count
                       << " time=" << elapsed_us << "us";
}
```

**各无参数激活层特有诊断指标**：

| 层 | 特有指标 | 检测条件 | 诊断用途 |
|---|---|---|---|
| ReLU | `dead=N/M (ratio)` | `x <= 0` 的元素数 | 死亡 ReLU 检测 |
| Sigmoid | `saturate=N/M (ratio)` | `y < 1e-4 || y > 1-1e-4` | 梯度消失预警 |
| TanH | `saturate=N/M (ratio)` | `|y| > 1-1e-4` | 梯度消失预警 |

### 3.2 带参数激活层（PReLU）

**代表实现**：PReLULayer

额外要求：
- 循环前/循环中统计参数值域（slope min/max）
- per-channel 模式下，slope 统计在循环外独立遍历（slope 数组大小为 channels，远小于 count，开销可忽略）
- 日志中包含参数配置（`channel_shared=true/false`）和参数值域

```cpp
// channel_shared 模式：slope 是标量，直接在循环外获取
if (channel_shared_) {
  slope_min = slope_max = slope_data[0];
  for (...) { /* 单次遍历 */ }
} else {
  // per-channel 模式：先遍历 slope 数组（大小=channels，通常<<count）
  for (int c = 0; c < channels_; ++c) {
    float s = slope_data[c];
    slope_min = std::min(slope_min, s);
    slope_max = std::max(slope_max, s);
  }
  for (...) { /* 主循环：计算 + 输入输出统计 */ }
}
```

### 3.3 带参数层（InnerProduct, BatchNorm 等）

待 Backward_cpu 扩展计划实施后补充。基本原则：
- 权重梯度值域（`w_grad=[min, max]`）和范数（`w_grad_norm=value`）必须记录
- 偏置梯度同理
- 参数量较大时（>1M），范数计算可采用 Welford 在线算法融合进主循环

---

## 4. 初始化值规范

统计极值（min/max）的初始值必须正确设置：

```cpp
// min 初始化为最大值，max 初始化为最小值
float in_min = std::numeric_limits<float>::max();
float in_max = -std::numeric_limits<float>::max();
float out_min = std::numeric_limits<float>::max();
float out_max = -std::numeric_limits<float>::max();
```

**禁止**使用 `0`、`FLT_MAX`（等价但不推荐）、或未初始化变量作为极值初值。

计数指标初始化为 `0`：
```cpp
int64_t dead_count = 0;       // ReLU dead neuron 计数
int64_t saturate_count = 0;  // Sigmoid/TanH 饱和计数
```

---

## 5. 性能与精度注意事项

### 5.1 编译优化

- 统计操作（`std::min`/`std::max`/计数器递增）在 `-O2`/`-O3` 下会被编译器自动向量化（SIMD），无需手动优化
- 禁止在循环中使用 `if (log_enabled)` 分支判断日志级别——日志宏本身已有编译期门控

### 5.2 浮点精度

- min/max 统计对浮点精度无要求，直接使用 `float` 即可
- 耗时统计使用 `double` 存储微秒值，避免大数截断
- 比率计算（如 saturate ratio）在循环外一次性计算：`float ratio = static_cast<float>(special_count) / count;`

### 5.3 多线程安全

- 统计变量是栈上局部变量，天然线程安全
- `CAFFE_FFI_LOG_INFO()` 内部使用互斥锁保护，多线程下日志不会交错
- 禁止将统计变量声明为 `static` 或类成员变量（会导致跨调用污染）

---

## 6. 日志输出示例

### 6.1 Sigmoid 前向传播

```
[ACTIVATION-PERF] sigmoid1 Sigmoid forward: count=784 in=[-3.2, 2.8] out=[0.039, 0.943] saturate=12/784 (0.015) time=2.3us
```

### 6.2 ReLU 前向传播

```
[ACTIVATION-PERF] relu1 ReLU forward: count=256 in=[-1.5, 3.2] out=[0.0, 3.2] dead=45/256 (0.176) time=1.1us
```

### 6.3 PReLU 前向传播（channel_shared）

```
[ACTIVATION-PERF] prelu1 PReLU forward: count=150528 channel_shared=true slope=[0.25, 0.25] in=[-2.1, 4.5] out=[-0.525, 4.5] time=180us
```

### 6.4 Sigmoid 反向传播

```
[ACTIVATION-PERF] sigmoid1 Sigmoid backward: count=784 diff_in=[-0.001, 0.002] diff_out=[-2.5e-4, 5.0e-4] saturate=12/784 (0.015) time=2.8us
```

---

## 7. Python 端验证

测试中应验证性能日志的存在性和格式正确性：

```python
def test_activation_perf_log(self, ptrace):
    """验证激活层输出包含 [ACTIVATION-PERF] 性能日志"""
    with ptrace.span("activation-perf-check"):
        result = self.layer.forward([self.input_tensor])
        logs = ptrace.get_logs()
        perf_logs = [l for l in logs if "[ACTIVATION-PERF]" in l]
        assert len(perf_logs) >= 1, "缺少 [ACTIVATION-PERF] 性能日志"
        assert "in=" in perf_logs[0]
        assert "out=" in perf_logs[0]
        assert "time=" in perf_logs[0]
```

---

## 8. 新增层 Checklist

为新计算层添加性能日志埋点时，确认以下项：

- [ ] 循环前记录 `t_start = std::chrono::high_resolution_clock::now()`
- [ ] 声明 `in_min/in_max/out_min/out_max` 并正确初始化
- [ ] 主循环中：计算 → 写入输出 → 更新极值统计
- [ ] 层特有诊断计数器在循环中正确递增
- [ ] 循环后记录 `t_end` 并计算 `elapsed_us`
- [ ] 输出日志包含 `[ACTIVATION-PERF]` 标签、层名、类型、count、in、out、time
- [ ] 层特有参数和诊断指标已包含在日志中
- [ ] 日志在循环外部输出，不在循环内部调用日志宏
- [ ] 无二次遍历数组的统计代码
- [ ] Python 测试验证日志格式正确性
