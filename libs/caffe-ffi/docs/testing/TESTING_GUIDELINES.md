---
id: "testing-guidelines"
version: "1.2.0"
date: "2026-08-01"
status: "active"
source: "P3-B milestone (Scale/Bias/Eltwise/Concat/Dropout/SoftmaxWithLoss/Accuracy testing) + Sigmoid saturation precision fix + InsertSplits edge case testing + test helper library extraction"
---

# Caffe-FFI 测试用例编写规范

> **适用范围**：caffe-ffi 所有 Python 单元测试（`tests/python/test_*.py`）
>
> **核心原则**：先验证明、边界优先、阈值断言、前向验证、泄漏自动检测

---

## 1. 文件结构规范

### 1.1 文件命名与位置

```
tests/python/
├── conftest.py                          # 全局fixture与性能追踪基础设施
├── caffe_test_helpers.py                # ⭐ 通用测试辅助函数库（断言/构造/验证）
├── test_python_api.py                   # P0 基础API测试
├── test_blob.py                         # P1 Blob生命周期测试
├── test_layers.py                       # P1 单层功能测试
├── test_net.py                          # P1 网络构造测试
├── test_cow.py                          # P2 COW机制测试
├── test_split_topologies.py             # P2 Split拓扑测试
├── test_split_concat_bench.py           # P2 Split/Concat嵌套性能基准
├── test_p3a_conv_pool_bn.py             # P3 组合层测试（Conv/Pool/BN）
├── test_p3b_eltwise_scale.py            # P3 组合层测试（Eltwise/Scale/Concat）
├── test_p3c_activations_ip.py           # P3 激活层+InnerProduct测试
├── test_p3c_transformer.py              # P3 Transformer组件测试
└── test_insert_splits.py                # 图变换边界测试
```

### 1.2 测试类结构

```python
"""模块docstring：列出覆盖场景编号与描述（1-2行/场景）。

Covered scenarios:
1.  场景一描述
2.  场景二描述
...
"""
from __future__ import annotations

import numpy as np
import pytest

from .conftest import require_cpp_extension
from .caffe_test_helpers import (       # ⭐ 优先使用辅助函数库
    make_net, count_splits,
    assert_split_exists, assert_split_after_producer,
    assert_split_at_position, assert_split_order,
    assert_no_split, assert_exact_split_name,
    assert_forward_shapes, assert_finite,
)


# ── Test class ───────────────────────────────────────────────────────

@require_cpp_extension
class TestXxx:
    """测试类docstring：一句话说明测试范围。"""

    def test_yyy(self):
        """测试方法docstring：描述测试意图（断言什么）。"""
        prototxt = """..."""
        net = make_net(prototxt)        # 使用make_net而非手动构造
        names = list(net.layer_names())
        # 结构性断言（使用辅助函数）
        assert count_splits(net) == expected_count
        assert_split_exists(names, "pattern")
        assert_split_after_producer(names, "producer", "split_pattern")
        # 前向传播断言（结构测试之后）
        outputs = net.Forward({...})
        assert_forward_shapes(outputs, {"blob": expected_shape})
        assert_finite(outputs["blob"])
```

### 1.3 强制装饰器

| 装饰器 | 用途 | 何时使用 |
|--------|------|---------|
| `@require_cpp_extension` | 标记需要C++扩展的测试 | **所有**测试类必须添加（除纯Python mock测试外） |
| `@pytest.mark.leak_check(False)` | 跳过该测试的泄漏检测 | 仅当测试故意持有Blob引用时 |

---

## 2. Prototxt构造规范

### 2.1 输入定义的两种方式

**方式一：`param.input()` + `input_shape`（推荐用于简单测试）**

```python
prototxt = """
name: 'test_name'
input: 'data'
input_shape { dim: 2 dim: 4 }    # batch=2, features=4
layer { ... }
"""
```

**方式二：显式Input层（推荐用于多输入/复杂场景）**

```python
prototxt = """
name: 'test_name'
layer { name: 'data' type: 'Input' top: 'data'
  input_param { shape { dim: 2 dim: 4 } } }
layer { name: 'label' type: 'Input' top: 'label'
  input_param { shape { dim: 2 dim: 3 } } }
layer { ... }
"""
```

> ⚠️ **注意**：两种方式可以混合使用，但`param.input()`的split插入位置在网络最开头（position 0），显式Input层的split紧跟在Input层之后。必须分别验证两种位置。

### 2.2 Shape设计原则

| 场景 | batch | 特征维度 | 说明 |
|------|-------|---------|------|
| 最小可行 | 1 | 2-4 | 最快执行，适合纯结构测试 |
| 标准验证 | 2 | 3-8 | 平衡速度与覆盖，Forward测试推荐 |
| 多输出层 | 2 | 3 | num_output=3，避免shape不匹配 |
| Concat测试 | 2 | 各分支匹配 | axis=1时各分支dim[1]可不同 |

### 2.3 强制bias_term: false（无权重测试）

当测试**图变换/拓扑结构**而非层本身计算时，InnerProduct/Convolution层必须设置`bias_term: false`，避免随机初始化权重导致数值不稳定：

```python
layer { name: 'fc1' type: 'InnerProduct' bottom: 'data' top: 'fc1_out'
  inner_product_param { num_output: 3 bias_term: false } }
```

> ✅ 例外：需要验证前向传播数值正确性时，使用固定权重（见§5.2）。

---

## 3. 浮点数断言规范

### 3.1 禁止精确相等（`==`/`!=`）

浮点数受IEEE754精度限制，**禁止**对浮点输出使用精确相等断言。

**反模式**（来自Sigmoid饱和Bug）：
```python
# ❌ 错误：float32中sigmoid(-88)≈6e-39（次正规数），不是精确0
assert result[0,0,0,0] == 0.0
# ❌ 错误：sigmoid(88)在float32中精确等于1.0，但测试期望<1.0
assert result[0,0,0,0] < 1.0  # x=20时已饱和为1.0
```

### 3.2 float32数值边界参考

| 边界类型 | 值 | 说明 |
|---------|-----|------|
| 机器精度（ULP） | ~1.2e-7 | `1.0`附近的最小可表示差 |
| 最小正规数 | ~1.2e-38 | `np.finfo(np.float32).tiny` |
| 最小次正规数 | ~1.4e-45 | `np.finfo(np.float32).smallest_subnormal` |
| Sigmoid负饱和阈值 | x ≤ -88 | 输出 `< 1e-37`（有效为0） |
| Sigmoid正饱和阈值 | x ≥ 17 | 输出 `== 1.0`（精确） |
| Sigmoid过渡区下界 | x ≈ -16 | 输出开始显著偏离0 |
| Sigmoid过渡区上界 | x ≈ 14 | 输出严格 `< 1.0` |

### 3.3 正确的断言方式

```python
# ✅ 负饱和：用阈值而非==0
assert result[0, 0, 0, 0] < 1e-37, (
    f"sigmoid(-88) should be < 1e-37 (effectively zero), got {result[0,0,0,0]}"
)

# ✅ 正饱和（x足够大时精确==1）
assert result[0, 0, 0, 0] == 1.0, (
    f"sigmoid(88) should be exactly 1.0 in float32, got {result[0,0,0,0]}"
)

# ✅ 过渡区：选择x≤14验证严格<1
x = np.array([[-10.0, 0.0, 10.0, 14.0]], dtype=np.float32)
result = _forward_sigmoid(x)
assert (result < 1.0).all(), "sigmoid(x) for x<=14 should be < 1.0"
assert (result > 0.0).all(), "sigmoid(x) for x>=-14 should be > 0.0"

# ✅ 非饱和值：用np.allclose
assert np.allclose(output, expected, rtol=1e-5, atol=1e-6)

# ✅ NaN/Inf防护
assert not np.any(np.isnan(result)), "sigmoid output contains NaN"
assert not np.any(np.isinf(result)), "sigmoid output contains Inf"
```

> ⭐ **快捷方式**：使用 `caffe_test_helpers` 中的预封装断言：
> ```python
> from .caffe_test_helpers import (
>     assert_finite, assert_all_between,
>     assert_sigmoid_negative_saturated,
>     assert_sigmoid_positive_saturated,
>     assert_sigmoid_transition,
> )
> assert_sigmoid_negative_saturated(result_neg)  # < 1e-37 + NaN/Inf guard
> assert_sigmoid_positive_saturated(result_pos)  # == 1.0 + NaN/Inf guard
> assert_sigmoid_transition(result_mid)          # 严格在(0,1)区间 + NaN/Inf guard
> assert_finite(arr, label="blob_name")          # 通用NaN/Inf防护
> ```

### 3.4 饱和函数测试模板

```python
def test_saturation_behavior(self):
    """测试饱和函数在极限值、过渡区、中间值的行为。"""
    # 1. 极限值测试（使用阈值比较）
    # 2. 过渡区边界测试（验证不早熟饱和）
    # 3. 中间值单调性测试
    # 4. NaN/Inf防护
    # 5. 批量大张量测试（确保无数组越界）
```

---

## 4. 核心架构约束：Single-Consumer Blob模型（必读！）

> **⚠️ 这是caffe-ffi与标准Caffe最重要的行为差异，也是最容易踩坑的地方。所有测试编写者必须首先理解本节。**

### 4.1 约束说明

caffe-ffi的Net实现采用**严格的Single-Consumer（单消费）Blob模型**：

> 每个blob在被一个layer的bottom引用后，立即从`available_blobs`集合中删除，不能被第二个layer直接消费。

**代码位置**：[src/caffe_ffi/net.cpp#L615](file:///d:/spaces/SpecWeave/projects/xuanspace/libs/caffe-ffi/src/caffe_ffi/net.cpp#L615)
```cpp
// AppendBottom函数中：
available_blobs->erase(blob_name);  // blob被消费后立即从可用集合中移除
```

**与标准Caffe的差异**：

| 行为 | 标准Caffe | caffe-ffi |
|------|----------|-----------|
| 同一bottom blob被多layer引用 | ✅ 隐式共享（Caffe内部自动in-place或copy） | ❌ 立即报错`Unknown bottom blob` |
| 多消费者blob | 无需任何处理 | **必须显式插入Split层** |
| 内存模型 | 隐式copy | 零拷贝/COW优化，显式Split |

### 4.2 错误症状

当违反Single-Consumer约束时，会出现以下错误：

```
Unknown bottom blob '<blob_name>' (layer '<layer_name>', bottom index <idx>). See [BLOB-NOT-FOUND] above.
```

**P3-B阶段三次触发场景**（均已修复）：

| 场景 | 错误触发点 | 修复方式 |
|------|----------|---------|
| 同一net中三个Eltwise操作（SUM/PROD/MAX）共享input | 第二个Eltwise引用`a`时 | 拆分为三个独立net，每个net一个操作 |
| 分类全链路中score同时喂给Loss和Accuracy | Accuracy引用`score`时 | 添加`split_score`层：score→score_loss/score_acc |
| 分类全链路中label同时喂给Loss和Accuracy | Accuracy引用`label`时 | 添加`split_label`层：label→label_loss/label_acc |

### 4.3 多消费者场景的正确处理方式

**方式一：独立操作分离Net（推荐用于参数对比测试）**

当需要对同一个输入测试多种操作变体时，为每个操作创建独立的net：

```python
# ✅ 正确：每个Eltwise操作独立Net
ops = [
    ("SUM",  "EltwiseOp_SUM"),
    ("PROD", "EltwiseOp_PROD"),
    ("MAX",  "EltwiseOp_MAX"),
]
for op_name, op_type in ops:
    prototxt = f"""
name: 'test_eltwise_{op_name}'
input: 'a' input_shape {{ dim: 2 dim: 3 }}
input: 'b' input_shape {{ dim: 2 dim: 3 }}
layer {{ name: 'elt' type: 'Eltwise' bottom: 'a' bottom: 'b' top: 'out'
  eltwise_param {{ operation: {op_type} }} }}
"""
    net = make_net(prototxt)
    out = net.Forward({"a": a, "b": b})["out"]
    # 验证out
```

**方式二：显式Split层（组合网络/流水线测试）**

当需要在同一个net中让一个blob被多个layer消费时，**必须显式插入Split层**：

```python
# ✅ 正确：分类全链路 — score和label都需要Split
prototxt = """
name: 'test_classification_pipeline'
input: 'data'  input_shape { dim: 2 dim: 4 }
input: 'label' input_shape { dim: 2 }
layer { name: 'fc' type: 'InnerProduct' bottom: 'data' top: 'score'
  inner_product_param { num_output: 3 bias_term: false } }

# ⭐ Split score: score → score_loss / score_acc
layer { name: 'split_score' type: 'Split' bottom: 'score'
  top: 'score_loss' top: 'score_acc' }

# ⭐ Split label: label → label_loss / label_acc
layer { name: 'split_label' type: 'Split' bottom: 'label'
  top: 'label_loss' top: 'label_acc' }

layer { name: 'loss' type: 'SoftmaxWithLoss'
  bottom: 'score_loss' bottom: 'label_loss' top: 'loss' }
layer { name: 'acc' type: 'Accuracy'
  bottom: 'score_acc' bottom: 'label_acc' top: 'accuracy' }
"""
net = make_net(prototxt)
outputs = net.Forward({"data": data, "label": label})
# outputs包含"loss"和"accuracy"
```

### 4.4 Split命名约定

Split层命名遵循Caffe原生约定：
```
<blob_name>_<producer_name>_<top_idx>_split          # Split层名
<blob_name>_<producer_name>_<top_idx>_split_<k>      # Split输出名（k从0开始）
```

- 外部输入（`param.input()`）：producer名为`"input"`
- 显式Input层：producer名为层自身名称
- in-place链：producer名为**最后一个**in-place层名

### 4.5 编写网络前的自检清单

在编写prototxt网络定义时，**第一步检查**：

- [ ] 列出所有bottom blob，统计每个blob被多少个layer引用
- [ ] 检查是否有blob被引用次数 > 1
- [ ] 如果有多消费者blob：要么拆分为多个独立net（方式一），要么显式添加Split层（方式二）
- [ ] loss_weight ≠ 0的top也计为一个消费者（loss通道）

> **参考模式**：`explicit-split-multi-consumer` 模式文档
> [patterns/code-patterns/explicit-split-multi-consumer.md](file:///d:/spaces/SpecWeave/.agents/docs/retrospective/patterns/code-patterns/explicit-split-multi-consumer.md)

---

## 5. 图变换测试规范（InsertSplits等Pass）

### 5.1 Split命名约定（必须与Caffe原生一致）

```
<blob_name>_<producer_name>_<top_idx>_split          # Split层名
<blob_name>_<producer_name>_<top_idx>_split_<k>      # Split输出名（k从0开始）
```

**特殊producer名称**：
- 外部输入（`param.input()`）：producer名为`"input"`
- 显式Input层：producer名为层自身名称（如`"data"`）
- in-place链：producer名为**最后一个**in-place层名

### 5.2 必测的InsertSplits边界场景

| # | 场景 | 关键断言 |
|---|------|---------|
| 1 | 零消费者blob（死端） | 不插入split |
| 2 | 单消费者 | 不插入split |
| 3 | 外部输入多消费 | split名为`<blob>_input_0_split`，位于position 0 |
| 4 | 普通层top多消费 | split紧跟在producer层之后（idx = producer_idx + 1） |
| 5 | in-place后多消费 | split以最后in-place层命名 |
| 6 | loss_weight作为消费 | loss_weight≠0时count++，split输出数包含loss通道 |
| 7 | 链式split（fan-out后fan-out） | 多层split各自正确命名 |
| 8 | 幂等性（显式split后不重复插入） | 已有split的输出单消费时不二次split |
| 9 | 多外部输入顺序 | split顺序与param.input()声明顺序一致 |
| 10 | 线性链无fan-out | 0个split |
| 11 | 双in-place链 | 最后in-place层后插入split |
| 12 | 混合Input层+param.input() | 两种split位置分别正确 |
| 13 | Split→Concat→Split嵌套 | Inception式拓扑两层split位置均正确 |
| 14 | 多个独立split位置 | 每个split在各自producer之后 |
| 15 | 空网络（0层） | 不崩溃，0 split |
| 16 | Input层3+消费者 | split输出数与消费者数一致 |
| 17 | loss_weight+多downstream | split输出数=downstream数+1（含loss） |
| 18 | 未知bottom blob引用 | 抛出RuntimeError |

### 5.3 结构性断言模板

使用 `caffe_test_helpers` 中的辅助函数，**禁止**手写 `next(i for i,n in enumerate(names) if ...)` 等重复模式：

```python
from .caffe_test_helpers import (
    make_net, count_splits,
    assert_split_exists, assert_split_after_producer,
    assert_split_at_position, assert_split_order,
    assert_no_split, assert_exact_split_name,
    assert_forward_shapes, assert_finite,
)

def test_something(self):
    prototxt = """..."""
    net = make_net(prototxt)
    names = list(net.layer_names())

    # 1. split数量断言
    assert count_splits(net) == expected_count

    # 2. split存在性断言（子串匹配）
    assert_split_exists(names, "expected_name_pattern")

    # 3. 精确名称断言（知道完整名称时）
    assert_exact_split_name(names, "data_data_0_split")

    # 4. split位置断言：紧跟在producer之后
    assert_split_after_producer(names, "producer_name", "split_pattern")

    # 5. 外部输入split位置断言（position 0）
    assert_split_at_position(names, "data_input_0_split", 0)

    # 6. 顺序断言
    assert_split_order(names, "pattern_a", "pattern_b", msg="a should precede b")

    # 7. 不存在断言
    assert_no_split(names, "should_not_exist_pattern")

    # 8. 前向输出shape断言（批量）
    outputs = net.Forward({"data": inp})
    assert_forward_shapes(outputs, {
        "blob_a": (batch, dim_a),
        "blob_b": (batch, dim_b),
    })
    # NaN/Inf防护
    for v in outputs.values():
        assert_finite(v)
```

### 5.4 前向验证（结构断言之后必做）

图变换测试**必须**验证前向传播正确性，不能只做结构检查：

```python
# 在结构断言之后，添加：
inp = np.random.randn(batch, dim).astype(np.float32)
outputs = net.Forward({"data": inp})
assert "output_blob" in outputs
assert outputs["output_blob"].shape == expected_shape
# 必要时验证数值正确性
```

> 💡 **为什么？** 结构正确（split命名/位置对）不代表计算正确（如bottom重写错名导致forward失败）。两层验证缺一不可。

---

## 6. 数值测试规范

### 6.1 确定性输入

测试前向数值正确性时，使用固定种子或手工权重，避免随机初始化导致偶发失败：

```python
np.random.seed(42)  # 固定随机种子
# 或手工指定权重
W = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
b = np.array([0.01, 0.02], dtype=np.float32)
```

### 6.2 权重注入

```python
def test_with_known_weights(self):
    prototxt = """..."""
    net = _make_net(prototxt)
    # 获取层权重blob并注入已知值
    layers = net.layers_array()
    fc_layer = [l for l in layers if l.name == "fc1"][0]
    fc_layer.blobs[0].from_numpy(W)  # weight
    fc_layer.blobs[1].from_numpy(b)  # bias
    # 然后forward验证
```

### 6.3 误差容限选择

| 场景 | rtol | atol |
|------|------|------|
| 激活层（ReLU/Sigmoid等逐元素） | 1e-6 | 1e-7 |
| InnerProduct/Convolution（乘加） | 1e-5 | 1e-6 |
| BatchNorm（除法+epsilon） | 1e-4 | 1e-5 |
| 大张量批量测试 | 1e-4 | 1e-5 |

---

## 7. 错误路径测试

### 7.1 必须测试的异常

```python
def test_unknown_bottom_raises(self):
    """引用未定义的bottom blob必须抛出错误。"""
    prototxt = """
name: 'test_bad'
input: 'data'
input_shape { dim: 1 dim: 4 }
layer { name: 'fc1' type: 'InnerProduct' bottom: 'nonexistent' top: 'out'
  inner_product_param { num_output: 3 } }
"""
    with pytest.raises((RuntimeError, ValueError),
                       match="Unknown bottom blob|nonexistent"):
        _make_net(prototxt)
```

### 7.2 错误消息验证

`pytest.raises`的`match`参数应同时匹配：
1. 错误类型关键词（如"Unknown bottom blob"）
2. 问题对象名称（如"nonexistent"）

---

## 8. 性能与内存规范

### 8.1 自动泄漏检测

conftest.py中的autouse fixture会**自动**在测试之间检查Blob泄漏。测试无需手动写泄漏断言，但需注意：

- 测试中不要在模块级/类级变量中持有Blob/Net引用
- 如果需要故意持有引用（如缓存测试），使用`@pytest.mark.leak_check(False)`跳过检测

### 8.2 perf_trace上下文管理器

对于性能相关测试，使用`perf_trace`记录Δtime/Δmem/Δblobs：

```python
def test_something(self, ptrace):
    with ptrace("Net construction + forward") as t:
        net = _make_net(prototxt)
        out = net.Forward({...})
        t["shape"] = str(out["out"].shape)
        t["layers"] = len(list(net.layer_names()))
    # 自动输出：[PERF] Net construction + forward  Δtime=XXms  Δmem=+XX  Δblobs=+X ...
```

---

## 9. 常见陷阱与反模式

### 9.1 ❌ 反模式清单

| 反模式 | 后果 | 正确做法 |
|--------|------|---------|
| `assert x == 0.0` 对饱和值 | float32次正规数导致断言失败 | 使用阈值 `< 1e-37` |
| `assert x < 1.0` 对正饱和 | x≥17时float32精确等于1.0 | 区分过渡区(x≤14)和饱和区(x≥17) |
| 只验证结构不做forward | split命名对但bottom重写错导致计算图断裂 | 结构断言后必须`net.Forward()`验证 |
| InnerProduct层不设bias_term=false | 随机bias导致数值不稳定 | 结构测试设`bias_term: false` |
| 多bottom shape不匹配 | Eltwise/Concat/EuclideanLoss要求shape一致 | 仔细设计dim，Concat时注意axis |
| **同一blob被多layer消费（无Split）** | `Unknown bottom blob`运行时错误 | 参考§4：多消费者必须显式Split或拆分为独立Net |
| `git add .` 提交临时文件 | 调试脚本污染仓库 | 显式`git add <files>` |
| 根目录放test_xxx.py临时测试 | 文件丢失/不被CI发现 | 直接写在`tests/python/`下 |
| 使用EuclideanLoss验证结构 | 需要label blob且shape必须匹配 | 用Concat/Eltwise替代，或提供正确shape的label |

### 9.2 ✅ Sigmoid饱和修复案例参考

问题：`sigmoid(-88) == 0.0`断言失败，实际值约6e-39。

根因：IEEE754 float32中，极大负值的sigmoid输出是**次正规数**（subnormal），不是精确0.0。

修复：
```python
# Before（错误）
assert result[0, 0, 0, 0] == 0.0
assert result[0, 0, 0, 1] < 1.0  # x=20时已饱和为1.0

# After（正确）
assert result[0, 0, 0, 0] < 1e-37  # 阈值代替==0
# x=14验证过渡区不早熟饱和，x=88验证精确==1
```

### 9.3 ✅ Split位置修复案例参考

问题：多个`param.input()`需要split时，split层顺序倒置。

根因：初始实现在Pass 2遍历层时遇到外部输入消费者即插入split，导致按consumer顺序而非input声明顺序排列。

修复：Pass 2b先收集所有外部输入split，然后整体移位插入到网络开头，保持声明顺序。

测试验证：
```python
data_split_idx = next(i for i, n in enumerate(names)
                      if n.startswith("data_input_") and "_split" in n)
weight_split_idx = next(i for i, n in enumerate(names)
                        if n.startswith("weight_input_") and "_split" in n)
assert data_split_idx < weight_split_idx  # 声明顺序：data在weight前
```

---

## 10. 测试提交前检查清单

- [ ] 文件位于`tests/python/test_*.py`
- [ ] 测试类使用`@require_cpp_extension`装饰器
- [ ] 模块docstring列出所有覆盖场景
- [ ] 每个测试方法有docstring描述意图
- [ ] 浮点数断言使用阈值/`np.allclose`，无`==0.0`/`==1.0`
- [ ] 网络定义前已做Single-Consumer检查（§4.5），多消费者blob已显式Split或分离Net
- [ ] 图变换测试包含结构断言+前向验证
- [ ] prottxt中InnerProduct/Conv层结构测试设`bias_term: false`
- [ ] 多bottom层（Eltwise/Concat/Loss）的shape设计匹配
- [ ] 测试了异常路径（pytest.raises）
- [ ] 运行`pytest tests/python/test_insert_splits.py -v`全部通过
- [ ] 运行相关测试套件无回归

---

## 11. 参考测试文件

| 文件 | 用途 | 参考章节 |
|------|------|---------|
| `tests/python/conftest.py` | fixture定义、perf_trace、泄漏检测 | §1.3, §8 |
| `tests/python/caffe_test_helpers.py` | ⭐ 通用测试辅助函数（断言/构造/验证） | §5.3, §3 |
| `tests/python/test_insert_splits.py` | 图变换边界测试范本（使用辅助函数） | §5 |
| `tests/python/test_split_concat_bench.py` | Split/Concat嵌套性能基准范本 | §8 |
| `tests/python/test_p3c_activations_ip.py` | 浮点数饱和测试范本 | §3 |
| `tests/python/test_p3b_eltwise_scale.py` | ⭐ P3-B阶段测试范本（numpy参考+三层验证+组合网络） | §4, §6 |
| `tests/python/test_layer_template_three_layer_validation.py` | ⭐ 三层验证法测试模板（可直接复制） | §6 |
| `tests/python/test_cow.py` | COW机制+refcount测试 | - |
| `docs/plans/INSERT_SPLITS_GRAPH_TRANSFORM.md` | InsertSplits算法详解+辅助函数索引 | §5 |
