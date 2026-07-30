# Split层性能分析报告 & P2-B极端边界测试规划

> **生成时间**: 2025-03-30
> **阶段**: P2-B Split层开发 & 性能验证
> **状态**: 代码已完成，等待编译环境运行后补充实测数据

---

## 一、Split层memcpy瓶颈分析报告

### 1.1 分析结论（基于代码审查的预判断）

**结论：Split层的memcpy在以下场景中确实会成为推理瓶颈：**

| 场景 | memcpy占比预估 | 瓶颈程度 |
|------|---------------|---------|
| 小输入 (batch≤8, feat≤128) | <5% | 🟢 可忽略 |
| 中等输入 (batch=32, feat=256~512, N=2) | 10~25% | 🟡 需关注 |
| 大输入 (batch=32, feat=1024~2048, N=2) | 30~50% | 🟠 显著瓶颈 |
| 超大输入 + 多分支 (batch=64, feat=2048+, N≥4) | 50~70%+ | 🔴 严重瓶颈 |

**判断依据：**

1. **数据复制量**：Split层对每个top blob执行一次完整memcpy，总复制量 = `count * sizeof(float) * num_top`
   - batch=32, feat=1024, N=2 → 32×1024×4×2 = **256KB** 每次forward
   - batch=64, feat=2048, N=4 → 64×2048×4×4 = **2MB** 每次forward

2. **内存带宽瓶颈**：现代CPU的memcpy带宽约10~30GB/s（取决于缓存命中率）
   - 256KB复制耗时约8~25μs（L3缓存内）
   - 2MB复制耗时约67~200μs（可能触发主存访问）

3. **对比InnerProduct（GEMM）**：InnerProduct计算量 = `2*batch*in_dim*out_dim` FLOPs
   - 对于batch=32, in=1024, out=1024: 67M FLOPs ≈ 0.02~0.07ms（现代AVX2可达1~3TFLOPS）
   - memcpy 256KB ≈ 0.008~0.025ms → memcpy与GEMM同量级

4. **零拷贝预期收益**：零拷贝优化（后续实现）理论上可将Split耗时降至接近0（仅指针传递，无数据复制）

### 1.2 性能埋点日志解读指南

Split层在C++中添加了 `[SPLIT-PERF]` WARN级别日志（Release构建也可见）：

#### Reshape阶段日志格式
```
[SPLIT-PERF] <layer_name> Reshape: num_top=<N> count=<C> elem_size=4B total_copied_per_fwd=<B>B reshape_time=<T>ms net_alloc=<A>B
```

| 字段 | 含义 |
|------|------|
| `num_top` | 输出分支数 |
| `count` | bottom blob元素总数 |
| `total_copied_per_fwd` | 每次forward需要复制的总字节数 |
| `reshape_time` | Reshape（内存分配）耗时 |
| `net_alloc` | Reshape中净分配的新内存字节数 |

#### Forward阶段日志格式
```
[SPLIT-PERF] <layer_name> Forward(N=<N>): count=<C> total_copied=<B>B total_memcpy_time=<T>ms avg_per_copy=<avg>us min_copy=<min>us max_copy=<max>us throughput=<GB/s>GB/s num_copies=<n>
```

| 字段 | 含义 |
|------|------|
| `total_memcpy_time` | 所有memcpy总耗时 |
| `avg/min/max_copy` | 单次memcpy的时间统计（微秒） |
| `throughput` | 实测内存带宽（GB/s），可评估缓存效率 |
| `num_copies` | 实际执行复制的top数量（in-place跳过的不计入） |

### 1.3 瓶颈判定标准（实测后填写）

运行 `test_split_perf_scaling` 测试后，从CSV中提取数据：

```python
import pandas as pd
df = pd.read_csv('perf_log.csv')
split_rows = df[df['operation'].str.contains('SplitPerf')]
total_rows = df[df['operation'].str.startswith('Forward(')]
# 计算split memcpy时间占forward总时间的比例
```

**判定阈值**：
- 🟢 memcpy耗时 < forward总耗时10% → 非瓶颈
- 🟡 10~30% → 需要关注，可考虑后续优化
- 🟠 30~50% → 显著瓶颈，建议P3实现零拷贝
- 🔴 >50% → 严重瓶颈，零拷贝优先级提升

### 1.4 实测数据（待运行后填写）

| 配置 (batch, feat, N) | total_bytes | total_memcpy | throughput | Forward总耗时 | memcpy占比 |
|----------------------|-------------|-------------|------------|-------------|-----------|
| (1, 128, 2) | 1KB | - | - | - | - |
| (32, 256, 2) | 64KB | - | - | - | - |
| (32, 512, 2) | 128KB | - | - | - | - |
| (32, 1024, 2) | 256KB | - | - | - | - |
| (16, 2048, 2) | 256KB | - | - | - | - |
| (32, 256, 4) | 128KB | - | - | - | - |

---

## 二、P2-B阶段极端边界条件测试规划

### 2.1 测试矩阵

#### A. 超大输入维度测试

| 测试用例 | 配置 | 验证目标 | 预期结果 |
|---------|------|---------|---------|
| `test_large_input_2048` | batch=64, feat=2048, 2隐层(512) | 大batch+大特征下不崩溃 | Forward成功，prob输出合法 |
| `test_split_large_input_1024` | batch=32, feat=1024, Split 2分支 | Split大内存memcpy正确性 + 稳定性 | 输出正确，无崩溃 |

**扩展建议（手动验证）**：
- batch=128, feat=4096（512MB输入）
- N=8分支Split（feat=512, batch=16）→ 8×16×512×4 = 256KB复制

#### B. 异常输入值测试

| 测试用例 | 输入 | 验证目标 |
|---------|------|---------|
| `test_nan_input_no_crash` | 全NaN | 不segfault，NaN正确传播 |
| `test_inf_input_no_crash` | 全Inf | 不segfault，Inf/NaN传播可接受 |
| `test_zero_input_deterministic` | 全0 | 输出确定性（两次forward结果一致） |

#### C. 极端权重测试

| 测试用例 | 权重 | 验证目标 |
|---------|------|---------|
| `test_extreme_weights_large` | W=1e6 | 不溢出崩溃（允许Inf输出） |
| `test_extreme_weights_tiny` | W=1e-6 | 输出有限值，不下溢为0 |

#### D. 深度网络测试

| 测试用例 | 配置 | 验证目标 |
|---------|------|---------|
| `test_deep_network_20_layers` | 20层MLP (18隐层) | 深层网络前向传播成功 |

**扩展建议**：50层、100层（验证梯度消失/爆炸及内存稳定性）

#### E. 内存稳定性测试

| 测试用例 | 配置 | 验证目标 | 通过标准 |
|---------|------|---------|---------|
| `test_lifecycle_stress_50_creates` | 反复创建-Forward-销毁Net 50次 | 无内存泄漏 | 净泄漏 < 1MB |
| `test_repeated_forward_100_times` | 同一Net重复Forward 100次 | Blob复用无增长 | Blob数不变，内存增长<4KB |

#### F. 高并发/串行压力测试（Python线程模拟）

**建议补充测试**（当前测试为串行执行，模拟高并发场景）：

```python
def test_concurrent_net_creation_stress(ptrace):
    """多线程同时创建/销毁Net，验证内存分配线程安全"""
    import threading
    errors = []
    def worker(seed):
        try:
            for _ in range(10):
                proto = _make_mlp_prototxt(batch=4, input_dim=8, hidden_dim=8, n_hidden=1)
                net = net_from_param(net_param_from_string(proto))
                _set_random_weights_all(net, seed=seed)
                inp = np.random.randn(4, 8).astype(np.float32)
                net.Forward({"data": inp})
        except Exception as e:
            errors.append(e)
    
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert len(errors) == 0, f"Concurrent errors: {errors}"
```

#### G. 边界shape测试

| 测试用例 | 配置 | 验证目标 |
|---------|------|---------|
| `test_minimal_1x1` | batch=1, feat=1 | 最小标量网络正常工作 |

**建议补充**：
- batch=0（空batch）错误处理
- feat=1 极端窄网络
- N=16/32分支Split（扇出压力测试）
- Reshape动态变化（不同shape的连续forward）

### 2.2 P2-B测试执行优先级

| 优先级 | 测试类别 | 原因 |
|-------|---------|------|
| P0 必须通过 | 内存稳定性（E类）、超大输入（A类） | 直接验证内存安全 |
| P1 应该通过 | 异常输入（B类）、极端权重（C类） | 鲁棒性验证 |
| P2 建议通过 | 深度网络（D类）、边界shape（G类） | 覆盖度完善 |
| P3 可选 | 高并发模拟（F类） | 单线程场景为主，多线程为后续增强 |

### 2.3 内存泄漏检测方法

运行测试时，使用 `--leak-check` 标记控制内存泄漏基线检查：

```bash
cd /path/to/caffe-ffi
# 运行P2-B全部测试（含Split拓扑+极端边界）
pytest tests/python/test_split_topologies.py tests/python/test_extreme_boundaries.py -v

# 带详细性能日志
pytest tests/python/test_split_topologies.py::TestSplitTopologies::test_split_perf_scaling -v -s

# 内存泄漏专项
pytest tests/python/test_extreme_boundaries.py::TestExtremeBoundaries::test_lifecycle_stress_50_creates -v
pytest tests/python/test_extreme_boundaries.py::TestExtremeBoundaries::test_repeated_forward_100_times -v
```

性能CSV日志将输出到 `tests/python/perf_log.csv`，包含字段：
- `timestamp`: ISO时间戳
- `test_class`, `test_name`: 测试标识
- `operation`: 操作名称（Net/Forward/SplitPerf等）
- `elapsed_ms`: 耗时（毫秒）
- `delta_mem`: 内存变化（字节）
- `delta_blobs`: Blob数量变化
- `extra`: 额外信息（shape/layers等）

---

## 三、Split层性能埋点设计说明

### 3.1 C++层埋点（已实现）

**文件**: `src/caffe_ffi/layers/split_layer.cpp`

埋点覆盖以下指标：

| 阶段 | 指标 | 日志级别 |
|------|------|---------|
| Reshape | 每个top blob的reshape耗时(μs)、前后内存大小 | DEBUG (CAFFE_FFI_LAYER_LOG) |
| Reshape | 总reshape耗时(ms)、净分配字节数 | WARN ([SPLIT-PERF]) |
| Forward | 每个top blob的memcpy耗时(μs)、指针地址、是否in-place | DEBUG (CAFFE_FFI_LAYER_LOG) |
| Forward | 总memcpy时间(ms)、平均/最小/最大单次时间、吞吐量(GB/s) | WARN ([SPLIT-PERF]) |

关键设计决策：
1. **Reshape计时**：覆盖内存分配路径（Blob::Reshape → NewCPUTensor）
2. **Forward计时**：精确到每个top blob的memcpy，检测in-place优化机会
3. **吞吐量计算**：`throughput = total_bytes / total_time`，可用于判断缓存效率
   - >20GB/s: L1/L2缓存命中
   - 10~20GB/s: L3缓存
   - <10GB/s: 主存带宽（大输入场景）
4. **WARN级别**保证Release构建可见，不依赖DEBUG宏

### 3.2 Python层CSV导出（已实现）

**文件**: `tests/python/conftest.py`

CSV列定义：
```
timestamp, test_class, test_name, operation, elapsed_ms, delta_mem, delta_blobs, extra
```

使用方式：
```python
from .conftest import perf_trace

with perf_trace("Forward(my_config)") as t:
    out = net.Forward({"data": inp})
    t['shape'] = str(inp.shape)  # extra fields
    t['total_bytes'] = inp.nbytes
```

---

## 四、编译运行指南

### 4.1 环境要求

- Linux/WSL2 with CMake ≥ 3.20, Ninja, GCC ≥ 11
- tvm-ffi 0.1.x (compatible version)
- Python 3.10+, NumPy, pytest

### 4.2 编译步骤

```bash
# 使用dev.sh脚本（推荐在conda环境中）
cd /path/to/caffe-ffi
bash scripts/dev.sh -b   # 构建C++
bash scripts/dev.sh -i   # 安装pip包
bash scripts/dev.sh -t   # 运行测试

# 或手动编译
cmake --preset default
cmake --build --preset default
pip install --no-build-isolation -e .
```

### 4.3 已知环境问题

当前WSL Ubuntu-24.04中安装的tvm-ffi (0.1.14.dev4)与代码库不兼容，存在`ObjectPtr<Layer>`类型trait问题。原始编译环境（conda caffe-ffi环境，位于/opt/conda/envs/caffe-ffi/）使用的是兼容版本的tvm-ffi。

**解决方案**：使用项目指定的conda环境或Docker容器编译。

---

## 五、总结与下一步

### 已完成
1. ✅ Split层C++实现（基于memcpy，含完整性能埋点）
2. ✅ CMake自动编译配置（file(GLOB)自动包含layers/*.cpp）
3. ✅ _caffe_ffi.cc注册Split层
4. ✅ CSV性能日志导出功能
5. ✅ Split拓扑测试用例（6个正确性测试+1个性能缩放测试）
6. ✅ 极端边界测试用例（10个测试覆盖大输入/异常值/极端权重/深层/内存稳定性/最小输入）
7. ✅ [SPLIT-PERF] WARN级别性能日志，覆盖Reshape分配+Forward memcpy全链路

### 待执行
- [ ] 在正确编译环境中构建并运行测试
- [ ] 收集CSV性能数据，填写1.4节实测数据表格
- [ ] 基于实测数据确认瓶颈程度
- [ ] （可选）补充多线程并发测试
- [ ] （P3阶段）实现零拷贝优化
