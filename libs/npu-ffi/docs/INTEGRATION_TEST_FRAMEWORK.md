# NPU 指令集集成测试框架扩展设计方案

> **日期**: 2026-07-28
> **基于**: GEMM 端到端集成测试成功经验 (test_gemm_e2e, 9/9 通过)
> **目标**: 为卷积、注意力机制等后续 NPU 指令集扩展提供标准化、可复用的集成测试框架

---

## 1. 现有测试架构回顾

### 1.1 五层 FFI 指令流模型

当前 GEMM 端到端测试覆盖完整的五层指令流水线：

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: 内存分配 (Memory Allocation)                       │
│   Buffer inp(kTensorSize), wgt(kTensorSize), acc(kTensorSize)│
├─────────────────────────────────────────────────────────────┤
│ Layer 2: 数据移动 (Data Movement / Load)                    │
│   read_barrier → load_buffer_2d (DRAM → SRAM: INP/WGT)     │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: 计算 (Compute)                                     │
│   uop_push(GEMM mode, reset_out=1) → uop_loop_begin/end     │
├─────────────────────────────────────────────────────────────┤
│ Layer 4: 结果存储 (Store)                                   │
│   store_buffer_2d (SRAM:ACC → DRAM) → write_barrier        │
├─────────────────────────────────────────────────────────────┤
│ Layer 5: 同步与依赖 (Sync & Dependencies)                   │
│   dep_push/dep_pop → synchronize → read_barrier            │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 当前测试分层

| 测试文件 | 层级 | 测试目标 | 用例数 |
|---------|------|---------|-------|
| test_buffer.cc | L0 (单元) | Buffer RAII、移动语义、生命周期 | 13 |
| test_memory_ops.cc | L1 (操作) | load/store/copy/barrier/cpu_ptr | 10 |
| test_compute_ops.cc | L1 (操作) | uop_push GEMM/ALU 模式、opcode 枚举值 | 13 |
| test_gemm_e2e.cc | L2 (集成) | 完整 GEMM 流水线端到端 | 9 |
| test_control_flow.cc | L1 (操作) | 循环、依赖、同步、debug flag | - |

### 1.3 现有测试工具组件

[test_utils.h](../include/npu_ffi/testing/test_utils.h) 提供：
- `TestRunner`: 零依赖测试注册与执行器
- `ScopedBuffer`: RAII Buffer 自动释放
- `ScopedContext`: RAII CommandContext 自动同步
- `fill_pattern()` / `verify_pattern()`: 缓冲区模式填充与验证
- `TEST_ASSERT_*` 宏族: TEST_ASSERT_EQ/NE/TRUE/FALSE/GE/LE/GT/LT/PATTERN
- `TEST_INSTRUCTION_BEGIN/END`: 指令测试块宏（自动创建 ScopedContext）

---

## 2. 框架扩展设计

### 2.1 核心设计原则

1. **分层递进**: 单元测试 → 操作测试 → 端到端集成测试 → 算子融合测试
2. **参数化测试**: 数据类型、tile 尺寸、循环范围等参数化覆盖
3. **黄金参考比对**: Stub 后端验证指令流正确性；真实硬件验证数值正确性
4. **最小知识原则**: 新指令测试只需关注指令参数，复用五层模板
5. **零外部依赖**: 不依赖 gtest/gmock，保持现有 TestRunner 极简风格

### 2.2 测试工具扩展 (test_utils.h 增量)

#### 2.2.1 端到端测试模板基类

```cpp
/*!
 * \brief Base template for end-to-end instruction tests.
 *
 * Encapsulates the five-layer FFI pipeline so new instructions
 * only need to override:
 *   - setup_buffers(): allocate and initialize input/weight/output buffers
 *   - load_data(): DRAM → SRAM data movement layer
 *   - compute(): issue compute uops (the instruction-specific part)
 *   - store_result(): SRAM → DRAM result storage
 *   - verify(): validate output correctness
 */
class E2ETestBase {
 public:
  virtual ~E2ETestBase() = default;

  void run(CommandHandle cmd) {
    setup_buffers();
    read_barriers(cmd);
    load_data(cmd);
    compute(cmd);
    store_result(cmd);
    write_barriers(cmd);
    synchronize_cmd(cmd);
    read_barriers(cmd);
    verify();
  }

 protected:
  virtual void setup_buffers() = 0;
  virtual void read_barriers(CommandHandle cmd) {}
  virtual void load_data(CommandHandle cmd) = 0;
  virtual void compute(CommandHandle cmd) = 0;
  virtual void store_result(CommandHandle cmd) = 0;
  virtual void write_barriers(CommandHandle cmd) {}
  virtual void synchronize_cmd(CommandHandle cmd) { synchronize(cmd, 0); }
  virtual void verify() = 0;
};
```

#### 2.2.2 数据填充与验证辅助函数

```cpp
/*!
 * \brief Fill buffer with sequential values: data[i] = start + i * stride
 */
inline void fill_sequential(Buffer& buf, int32_t start = 0, int32_t stride = 1);

/*!
 * \brief Fill buffer with random values in [min_val, max_val]
 */
inline void fill_random(Buffer& buf, int32_t min_val, int32_t max_val, uint32_t seed = 42);

/*!
 * \brief Element-wise comparison with tolerance for fixed-point arithmetic
 */
inline bool verify_approx(const Buffer& actual, const Buffer& expected,
                          int32_t tolerance = 1);

/*!
 * \brief Compute reference result on CPU and compare
 */
template <typename RefFunc>
inline bool verify_with_reference(const Buffer& actual, RefFunc ref_func,
                                  size_t elem_count, int32_t tolerance = 1);
```

#### 2.2.3 参数化测试注册宏

```cpp
/*!
 * \brief Register a parameterized test case that runs for multiple parameter sets.
 *
 * Usage:
 *   TEST_PARAMETERIZED_BEGIN("conv/kernel_sizes") {
 *     for (auto kh : {1, 3, 5}) {
 *       for (auto kw : {1, 3, 5}) {
 *         TEST_PARAMETERIZED_CASE(kh, kw) {
 *           run_conv_test(kh, kw);
 *         }
 *       }
 *     }
 *   } TEST_PARAMETERIZED_END
 */
#define TEST_E2E_CASE(name, test_class, ...) \
  runner.add_test(name, []() { \
    TEST_INSTRUCTION_BEGIN(ctx) { \
      test_class tc(__VA_ARGS__); \
      tc.run(ctx.cmd()); \
    } TEST_INSTRUCTION_END \
  })
```

#### 2.2.4 指令序列记录器 (Stub 后端验证)

```cpp
/*!
 * \brief Records the sequence of instructions issued during a test.
 *
 * Works with the stub backend to verify that the correct instructions
 * are issued in the correct order with correct parameters.
 * Useful for validating instruction stream correctness without real hardware.
 */
class InstructionRecorder {
 public:
  struct OpRecord {
    std::string op_name;
    std::vector<int64_t> params;
  };

  void record(const char* op, std::initializer_list<int64_t> params);
  void clear();
  size_t count() const;
  const OpRecord& at(size_t index) const;

  /*!
   * \brief Verify that the recorded sequence matches expected pattern.
   */
  bool expect_sequence(const std::vector<std::string>& expected_ops) const;
  bool expect_param(size_t op_index, int param_index, int64_t expected_val) const;
};
```

---

## 3. 卷积 (Conv2D) 测试方案

### 3.1 卷积指令流五层映射

```
┌──────────────────────────────────────────────────────────────┐
│ Conv2D: output[b][oc][oh][ow] = Σ_{ic,kh,kw} input[b][ic][ih][kw] * weight[oc][ic][kh][kw] │
└──────────────────────────────────────────────────────────────┘

Layer 1: 内存分配
  - input:  N*IC*IH*IW elements (int8)
  - weight: OC*IC*KH*KW elements (int8)
  - bias:   OC elements (int32)
  - output: N*OC*OH*OW elements (int32)
  - 注：当前 stub 模式下使用 kTensorSize 固定大小

Layer 2: 数据移动
  - load_buffer_2d(input_buffer  → SRAM INP tile)
  - load_buffer_2d(weight_buffer → SRAM WGT tile)
  - load_buffer_2d(bias_buffer   → SRAM ACC (bias初始化))
  - 循环遍历 ic: 每次加载 IC_TILE 个输入通道和对应权重

Layer 3: 计算
  - GEMM uop 循环 (im2col 或 direct conv 由硬件决定)
  - 外层循环: batch (N)
  - 中层循环: output channels (OC)
  - 内层循环: kernel spatial (KH*KW) + input channels (IC)
  - ALU uop: bias add (可选)

Layer 4: 存储
  - store_buffer_2d(SRAM ACC → output_buffer DRAM)

Layer 5: 同步
  - dep_push/dep_pop (LOAD→COMPUTE, COMPUTE→STORE)
  - synchronize + barriers
```

### 3.2 卷积测试用例矩阵

| 测试名称 | 场景 | 参数 | 验证目标 |
|---------|------|------|---------|
| conv/single_tile | 单层 tile 卷积 | KH=KW=1, stride=1, pad=0 | 基本指令流正确 |
| conv/3x3_basic | 3x3 标准卷积 | KH=KW=3, stride=1, pad=1 | 空间维度循环 |
| conv/stride_2 | 步长为2 | stride=2 | 步长参数传递 |
| conv/with_padding | 带 padding | pad=1/2 | padding 参数传递 |
| conv/multi_channel | 多通道 | IC=4, OC=8 | 通道维度循环 |
| conv/accumulate | 累加模式 | reset_out=0 | 累加/重置 ACC |
| conv/bias_add | 加 bias | ALU ADD imm/bias | bias 加载与 ALU |
| conv/debug_mode | 调试模式 | set_debug_mode | profiler 路径 |
| conv/dependency_chain | 依赖链 | dep_push/pop 多 conv | 队列依赖正确 |

### 3.3 卷积测试模板代码

```cpp
// tests/cpp/test_conv2d_e2e.cc
#include "npu_ffi/testing/test_utils.h"

using namespace npu_ffi::vta;
using namespace npu_ffi::vta::testing;

static constexpr uint32_t kTensorSize = 1024;  // 16*16*4 int8 values

struct ConvParams {
  uint32_t batch, ic, oc, ih, iw, kh, kw, stride, pad;
};

void test_conv2d_basic() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    ScopedBuffer inp(kTensorSize), wgt(kTensorSize), bias(kTensorSize), out(kTensorSize);
    fill_pattern(inp.get(), 0x01);
    fill_pattern(wgt.get(), 0x02);
    fill_pattern(bias.get(), 0x00);
    fill_pattern(out.get(), 0x00);

    // L2: Load weights and bias
    read_barrier(ctx.cmd(), wgt.get(), 8, 0, static_cast<uint32_t>(kTensorSize));
    load_buffer_2d(ctx.cmd(), wgt.get(), 0, 16, 16, 16, 0, 0, 0, 0, 0, MemoryType::WGT);

    // Bias → ACC (reset and initialize)
    read_barrier(ctx.cmd(), bias.get(), 8, 0, static_cast<uint32_t>(kTensorSize));
    load_buffer_2d(ctx.cmd(), bias.get(), 0, 16, 16, 16, 0, 0, 0, 0, 0, MemoryType::ACC);

    // Load input tile
    read_barrier(ctx.cmd(), inp.get(), 8, 0, static_cast<uint32_t>(kTensorSize));
    load_buffer_2d(ctx.cmd(), inp.get(), 0, 16, 16, 16, 0, 0, 0, 0, 0, MemoryType::INP);

    // L3: Compute conv (GEMM-based: IC_TILE=16, OC_TILE=16)
    uop_push(0, 1, 0, 0, 0, ALUOpcode::ADD, false, 0);

    // L4: Store result
    store_buffer_2d(ctx.cmd(), 0, MemoryType::ACC, out.get(), 0, 16, 16, 16);
    write_barrier(ctx.cmd(), out.get(), 8, 0, static_cast<uint32_t>(kTensorSize));
  } TEST_INSTRUCTION_END
}

int main() {
  printf("Running npu-ffi Conv2D end-to-end tests...\n\n");
  TestRunner runner;
  runner.add_test("conv/basic_3x3", test_conv2d_basic);
  // ... more tests ...
  return runner.run_all() ? 0 : 1;
}
```

---

## 4. 注意力机制 (Attention) 测试方案

### 4.1 注意力指令流特点

注意力机制比 GEMM/Conv 更复杂，涉及多步计算：

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

五层指令流:
┌──────────────────────────────────────────────────────────────┐
│ Layer 1: 内存分配                                           │
│   Q:  batch*seq_len*d_k, K: batch*seq_len*d_k,              │
│   V:  batch*seq_len*d_v, scores: batch*seq_len*seq_len,     │
│   output: batch*seq_len*d_v                                 │
├──────────────────────────────────────────────────────────────┤
│ Layer 2: 数据移动                                           │
│   Load Q → INP, Load K → WGT (for Q·K^T)                    │
│   需要多次 load (sequence tiled by chunks)                  │
├──────────────────────────────────────────────────────────────┤
│ Layer 3: 计算 (多阶段)                                      │
│   Stage 3a: GEMM → Q·K^T (scores matrix)                   │
│   Stage 3b: ALU  → scale by 1/√d_k (SHR or MUL imm)        │
│   Stage 3c: ALU  → row-wise softmax (max/sub/exp/sum/div)*  │
│   Stage 3d: Load V → WGT (reuse INP for scores)            │
│   Stage 3e: GEMM → scores·V (attention output)             │
│   *注：softmax 可能需要 CPU 辅助或专用 NPU 指令             │
├──────────────────────────────────────────────────────────────┤
│ Layer 4: 存储                                               │
│   store_buffer_2d(ACC → output)                             │
├──────────────────────────────────────────────────────────────┤
│ Layer 5: 同步 (多阶段依赖)                                  │
│   dep_push: LOAD_QK → COMPUTE_SCORES                        │
│   dep_push: COMPUTE_SCORES → LOAD_V                         │
│   dep_push: LOAD_V → COMPUTE_ATTN                           │
│   dep_push: COMPUTE_ATTN → STORE                            │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 注意力测试用例矩阵

| 测试名称 | 场景 | 验证目标 |
|---------|------|---------|
| attn/qk_matmul | Q·K^T GEMM | 矩阵乘法指令流 |
| attn/scale | 缩放因子 1/√d_k | ALU MUL/SHR 参数 |
| attn/softmax_row | 单行 softmax | max/exp/sum/div 序列 |
| attn/sv_matmul | scores·V GEMM | 第二阶段 GEMM |
| attn/single_head_full | 单头完整注意力 | QK→scale→softmax→SV 全链路 |
| attn/multi_head | 多头注意力 | 头维度循环，权重切换 |
| attn/causal_mask | 因果掩码 | 三角 mask 加载与应用 |
| attn/multi_seq_tile | 长序列分块 | seq_tile 循环与 tile 边界处理 |

---

## 5. 通用 E2E 测试框架实现计划

### 5.1 新增/扩展文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `include/npu_ffi/testing/test_utils.h` | 扩展 | 添加 E2ETestBase、数据填充辅助、指令记录器 |
| `tests/cpp/test_conv2d_e2e.cc` | 新增 | 卷积端到端测试 |
| `tests/cpp/test_attention_e2e.cc` | 新增 | 注意力端到端测试 |
| `tests/cpp/test_template.cc` | 新增 | 新指令测试模板（供开发者复制） |
| `tests/cpp/CMakeLists.txt` | 修改 | 添加新测试目标 |
| `docs/TESTING_GUIDE.md` | 新增 | 测试编写指南（如何添加新指令测试） |

### 5.2 分阶段实施路线图

**Phase 1: 测试工具增强 (当前迭代可做)**
1. 扩展 test_utils.h 添加 `fill_sequential`、`fill_random`、`verify_approx`
2. 添加 `E2ETestBase` 基类，封装五层模板
3. 创建 `tests/cpp/test_template.cc` 作为新指令的起始模板
4. 添加 `TESTING_GUIDE.md` 文档

**Phase 2: 卷积测试 (Conv2D 指令可用时)**
1. 基于 E2ETestBase 创建 Conv2DE2ETest 子类
2. 实现 9 个核心测试用例（见 3.2 矩阵）
3. 添加 CMake 目标并验证通过

**Phase 3: 注意力测试 (Attention 指令可用时)**
1. 扩展 E2ETestBase 支持多阶段计算（softmax 中间步骤）
2. 添加 InstructionRecorder 验证多阶段指令序列
3. 实现 8 个核心测试用例（见 4.2 矩阵）

**Phase 4: 算子融合测试 (后续扩展)**
1. Conv→ReLU→BN 融合测试
2. QKV projection + Attention 融合测试
3. 添加性能统计 (cycle count) 验证

### 5.3 测试命名规范

```
test_<instruction>_<scope>.cc

其中:
- <instruction>: 指令名称 (gemm, conv2d, attention, alu, etc.)
- <scope>: 测试范围
    - e2e: 端到端集成测试
    - unit: 单元测试（单指令参数验证）
    - perf: 性能测试（cycle count）

测试用例命名: "<category>/<op_name>_<variant>"
例如:
  "conv/3x3_stride1_pad1"
  "attn/single_head_causal"
  "gemm/multi_tile_accumulate"
```

### 5.4 CMake 集成模式

```cmake
# tests/cpp/CMakeLists.txt (扩展后)

# === 已有测试 ===
add_executable(test_buffer test_buffer.cc)
target_link_libraries(test_buffer PRIVATE npu_ffi_vta)
add_test(NAME test_buffer COMMAND test_buffer)
# ... 其他已有测试 ...

# === Phase 1: 模板 ===
add_executable(test_template test_template.cc)
target_link_libraries(test_template PRIVATE npu_ffi_vta)
add_test(NAME test_template COMMAND test_template)

# === Phase 2: 卷积 ===
add_executable(test_conv2d_e2e test_conv2d_e2e.cc)
target_link_libraries(test_conv2d_e2e PRIVATE npu_ffi_vta)
add_test(NAME test_conv2d_e2e COMMAND test_conv2d_e2e)

# === Phase 3: 注意力 ===
add_executable(test_attention_e2e test_attention_e2e.cc)
target_link_libraries(test_attention_e2e PRIVATE npu_ffi_vta)
add_test(NAME test_attention_e2e COMMAND test_attention_e2e)
```

---

## 6. 新指令接入 Checklist

添加新 NPU 指令集测试时，按以下 Checklist 执行：

- [ ] **L0 单元测试**: 创建或扩展 `test_<instruction>_unit.cc`，验证枚举值、参数边界
- [ ] **L1 操作测试**: 验证单指令 uop_push 不崩溃，参数正确传递
- [ ] **L2 端到端测试**: 基于 E2ETestBase 创建 `test_<instruction>_e2e.cc`，覆盖五层流水线
- [ ] **场景覆盖**: 至少覆盖 (a) 单 tile 基础 (b) 多 tile 循环 (c) 累加模式 (d) 调试模式
- [ ] **CMake 注册**: 在 tests/cpp/CMakeLists.txt 添加测试目标
- [ ] **验证通过**: 本地运行 `ctest --output-on-failure`，所有测试通过
- [ ] **文档更新**: 在 TESTING_GUIDE.md 记录新指令的特殊注意事项
