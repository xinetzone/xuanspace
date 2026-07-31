# InsertSplits 图变换维护文档

> **维护位置**：`src/caffe_ffi/net.cpp` → `InsertSplits()` 函数（L97-L375）
> **日志宏**：`CAFFE_FFI_SPLIT_LOG`（`include/caffe_ffi/log.hpp`，WARN 级别，默认可见）
> **原生参考**：`vendor/caffe/caffex/src/caffe/util/insert_splits.cpp`

---

## 1. 功能概述

InsertSplits 是 caffe-ffi 网络初始化阶段的**图变换 pass**，在 `Net::Init()` 中于层初始化之前执行。它的核心职责是：当一个 blob 被多个层消费（fan-out > 1）时，自动插入一个显式的 Split 层来实现数据分发，否则会触发 `Unknown bottom blob` 错误（`AppendBottom` 在第一次消费后会从 `available_blobs` 中移除 blob）。

```
输入 prototxt（隐式共享）              输出 prototxt（显式 Split）
┌─────────────┐                      ┌─────────────┐
│  data       │                      │  data       │
└──────┬──────┘                      └──────┬──────┘
       │                                    │
   ┌───┴───┐                          ┌─────┴─────┐
   │       │                          │  Split    │
┌──▼──┐ ┌──▼──┐                       └──┬─────┬──┘
│ fc1 │ │ fc2 │                          │     │
└─────┘ └─────┘                     ┌────▼┐ ┌──▼────┐
                                    │ fc1 │ │  fc2  │
                                    └─────┘ └───────┘
```

---

## 2. 算法详解（两趟扫描）

### Pass 1：计数 + 映射构建（L127-L178）

按拓扑序遍历所有层，构建以下数据结构：

| 数据结构 | 类型 | 用途 |
|---|---|---|
| `blob_name_to_last_top_idx` | `map<string, pair<int,int>>` | blob 名称 → (生产层索引, 输出索引)。**In-place 层会更新此映射**，使 Split 以最后生产者命名 |
| `bottom_idx_to_source_top_idx` | `map<pair<int,int>, pair<int,int>>` | (消费层, bottom索引) → (生产层, top索引)。Pass 2 重写 bottom 引用时查此表 |
| `top_idx_to_bottom_count` | `map<pair<int,int>, int>` | (生产层, top索引) → 消费者数量。**>1 时触发 Split 插入** |
| `top_idx_to_loss_weight` | `map<pair<int,int>, float>` | (生产层, top索引) → loss_weight。非零 loss 权重本身计为一个消费者 |
| `top_idx_to_bottom_split_idx` | `map<pair<int,int>, int>` | (生产层, top索引) → 当前已分配的 split 输出序号。Pass 2 重写 bottom 时递增 |
| `layer_idx_to_layer_name` | `map<int, string>` | 层索引 → 层名称。用于 Split 命名 |

**外部输入注册**（L118-L124）：`param.input()` 声明的外部输入注册为虚拟生产者 `(-1, input_idx)`。

**Loss 权重处理**（L165-L177）：若 top 有非零 loss_weight，loss 本身算一个消费者（`++top_idx_to_bottom_count`），这与原生 Caffe 行为一致。

### Pass 2：重写 + Split 插入（L216-L280）

再次按拓扑序遍历，对每一层：

1. **2a - 重写 bottom 引用**（L225-L254）：若 bottom 来源的 top 有 `split_count > 1`，将 bottom 名重写为 `<blob>_<producer>_<idx>_split_<k>`，并递增 split 分配序号。
2. **2b - 在层后插入 Split**（L257-L279）：若该层的某个 top 需要分裂（`split_count > 1`），立即在当前层之后追加一个 Split 层。有 loss_weight 时，清除原层的 loss_weight（第一个 split top 继承）。

### Pass 2b：外部输入 Split 前置（L284-L364）

外部输入（`param.input()`）的 Split 不能插入在某个生产层之后（因为没有生产层），需要单独处理并移到网络最前面。

**关键算法（批量头部插入四步法）**：
1. **收集**：按 `param.input()` 声明顺序遍历，收集需要 split 的外部输入 Split 层配置到 `input_splits` vector
2. **扩容**：在 `out_param->layer` 末尾追加 `n_ext` 个空槽位
3. **右移**：从后向前将原有层复制到 `i + n_ext` 位置
4. **写入**：将 `input_splits` 按顺序写入位置 `0..n_ext-1`

> ⚠️ **历史修复**：早期版本使用"逐个冒泡到头部"策略，导致多个外部输入 split 顺序反转（LIFO）。修复后采用收集→移位→写入模式，保持 FIFO 顺序，与 `param.input()` 声明一致。

---

## 3. 命名约定

| 对象 | 命名格式 | 示例 |
|---|---|---|
| Split 层名 | `<blob_name>_<producer_name>_<top_idx>_split` | `data_input_0_split`, `fc1_out_relu1_0_split` |
| Split 输出 blob 名 | `<blob_name>_<producer_name>_<top_idx>_split_<k>` | `data_input_0_split_0`, `fc1_out_relu1_0_split_1` |
| 外部输入 producer_name | 固定为 `"input"` | `data_input_0_split` 中的 `input` 即为虚拟生产者名 |

**In-place 层命名规则**：Split 层以**最后一个生产者**（即最近一次覆盖该 blob 的 in-place 层）命名。例如：
```
fc1 → x → relu1(in-place x) → relu2(in-place x) → [fc2, fc3]
```
Split 层名为 `x_relu2_0_split`，不是 `x_fc1_0_split`。

---

## 4. 边界情况处理

| 边界场景 | 处理逻辑 | 测试用例 |
|---|---|---|
| **零消费者（dead-end blob）** | `split_count == 0`，不进入 `>1` 分支，不插入 Split | `test_insert_splits_edge.py::Test1_dead_end` |
| **单消费者 blob** | `split_count == 1`，不进入 `>1` 分支，不重写、不插入 Split | `test_insert_splits_edge.py::Test2_single_consumer_no_split` |
| **线性链路（无 fan-out）** | 每个 blob 恰好 1 个消费者，0 个 Split 插入 | `test_insert_splits_edge.py::Test9_linear_no_splits` |
| **In-place ReLU → 双消费者** | 追踪 `blob_name_to_last_top_idx` 更新到 ReLU，Split 以 ReLU 命名 | `test_insert_splits_edge.py::Test3_inplace_relu` |
| **双重 In-place（ReLU+Dropout）** | 连续 in-place 更新生产者，Split 以最后一个 in-place 层命名 | `test_insert_splits_edge.py::Test10_double_inplace` |
| **Loss weight 导致 split** | 非零 loss_weight 使消费者计数 +1（loss 本身是消费者），若总 count>1 则插入 Split；第一个 split top 继承 loss_weight，其余为 0 | `test_insert_splits_edge.py::Test4_loss_weight_split` |
| **链式分裂（fan-out 后 fan-out）** | 内层 split 输出各 1 消费者，不会重复 split；外层多消费者 blob 各插入自己的 Split | `test_insert_splits_edge.py::Test5_chained_splits` |
| **幂等性（已显式 Split 的网络）** | 显式 Split 的每个输出恰好 1 个消费者，不会触发额外 Split 插入 | `test_insert_splits_edge.py::Test6_idempotence` |
| **多外部输入各需 split** | 收集所有需要 split 的外部输入，批量头插保持声明顺序 | `test_insert_splits_edge.py::Test8_multi_input_splits` |
| **前向正确性** | Split 层 N=1 时做 identity 转发（share_data），N>1 时 memcpy 复制 | `test_insert_splits_edge.py::Test7_forward_correctness` |

---

## 5. 日志调试指南

`CAFFE_FFI_SPLIT_LOG` 设为 WARN 级别，默认构建即输出。以下是关键日志标签：

| 标签 | 含义 |
|---|---|
| `=== InsertSplits BEGIN ===` | 图变换开始，打印输入网络名/层数/输入数 |
| `--- Pass 1a: registering external inputs ---` | 注册外部输入 |
| `--- Pass 1b: counting bottom references ---` | 逐层统计消费者计数，每层打印每个 bottom 的来源和计数 |
| `*** NEEDS SPLIT ***` | Pass 1 总结中标记需要 split 的 blob |
| `=== InsertSplits END (no splits inserted) ===` | 快速路径：无需任何 split，直接复制所有层 |
| `Rewriting bottom[j] 'X' -> 'X_split_k'` | Pass 2a 重写 bottom 引用 |
| `*** Inserting Split after 'L' for top[j]='X'` | Pass 2b 在层 L 后插入 Split |
| `Pass2b BEFORE/AFTER/Step1/2/3` | 外部输入 split 移动过程的详细顺序日志 |
| `--- Transformed layer list ---` | 变换后完整层列表 |
| `=== InsertSplits END ===` | 最终统计：原始层数 + 自动插入 Split 数 |

**使用示例**：若遇到 `Unknown bottom blob` 错误，搜索日志中 `NEEDS SPLIT` 和 `Rewriting bottom` 确认 Split 插入是否正确；搜索 `Pass2b AFTER` 确认外部输入 split 顺序。

---

## 6. 调用关系与依赖

```
Net::Init() [net.cpp:~L417]
  └── InsertSplits(in_param, &param)     [net.cpp:422]  ← 唯一调用点
        ├── ConfigureSplitLayer()       [net.cpp:69]   ← 辅助函数，构造 Split LayerParameter
        │     ├── SplitLayerName()      [net.cpp:50]   ← 命名辅助
        │     └── SplitBlobName()       [net.cpp:58]   ← 命名辅助
        └── Split 层运行时由 SplitLayer 类处理
              └── layers/split_layer.cpp/.hpp
```

**外部依赖检查结论**（2026-07-31 验证）：

| 模块 | 依赖类型 | 影响评估 |
|---|---|---|
| `tests/python/test_p3c_transformer.py` | 隐式依赖（残差连接需要 auto-split） | ✅ 13 测试全通过 |
| `test_insert_splits.py` / `test_insert_splits_edge.py` | 显式测试 InsertSplits | ✅ 10+10 测试全通过 |
| `tests/python/test_split_topologies.py` | 显式 Split 层测试（不依赖自动插入） | ✅ 7 测试全通过 |
| `tests/python/test_cow.py` | Split 层 COW 行为（显式 Split） | ✅ 21 测试全通过 |
| `tests/python/test_extreme_boundaries.py` | 极端边界（显式 Split） | ✅ 11 测试全通过 |
| `tests/python/test_p2b_regression.py` | P2B 回归（含显式 Split） | ✅ 29 测试全通过 |
| `vendor/caffe/caffex/` 和 `vendor/caffe/caffe-slim/` | 独立 git submodule，有自己的 InsertSplits | ❌ 不受影响（禁止本地修改 vendor） |

**总计 91 个相关测试全部通过，无回归。**

---

## 7. 修改 Checklist

修改 InsertSplits 相关代码后，请确认：

- [ ] 所有 10 类边界测试通过（`test_insert_splits_edge.py`）
- [ ] P3-C Transformer 13 个测试通过（`tests/python/test_p3c_transformer.py`）
- [ ] Split topology / COW / extreme boundary / p2b regression 全通过
- [ ] 外部输入 split 顺序与 `param.input()` 声明一致（查 `Pass2b AFTER` 日志）
- [ ] In-place 场景下 Split 以最后生产者命名（非初始生产者）
- [ ] 单消费者 blob 不插入 Split（`split_count == 1` 时跳过）
- [ ] 零消费者 dead-end blob 不插入 Split（`split_count == 0`）
- [ ] 幂等性：对已含显式 Split 的网络运行不产生重复 Split
- [ ] Loss weight 正确传播到 Split 第一个 top
- [ ] 日志中 `=== InsertSplits END ===` 统计数字与预期一致
