# Attested Computation 契约

*Attested Computation* 是 OKF 中一种特殊的概念类型，用于描述「一次经见证（attested）的计算」。其 `type` 固定为 `Attested Computation`，契约字段由 `src/okf/attested.py` 解析。

## 识别

`is_attested_computation(concept)` 判断标准很直接：`concept.type == "Attested Computation"`。

## frontmatter 字段

一个 Attested Computation 概念的 frontmatter 形如：

```markdown
---
type: Attested Computation
runtime: python-3.14
parameters:
  - name: threshold
    type: float
    required: true
executor:
  resource: executor-01
  receipt: [receipt-a, receipt-b]
attester:
  resource: attester-01
computation: computations/threshold.py
---
```

各字段含义：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `runtime` | `str` | 是 | 计算运行时，非空字符串 |
| `parameters` | `list` | 否 | 计算参数，每项含 `name` / `type` / `required` |
| `computation` | `str` | 否 | 计算逻辑：文件路径，或留空表示内联 |
| `executor` | `dict` | 否 | 执行者：`resource` + `receipt`（回执列表） |
| `attester` | `dict` | 否 | 见证者：`resource` |

## 计算逻辑的两种来源

`parse_attested_computation` 按 `computation` 字段是否为空，选择计算逻辑的来源：

| `computation` | 来源 | 说明 |
|---------------|------|------|
| 空 / 缺省 | 正文内联 | 从正文提取 `# Computation` 标题下的第一个代码围栏 |
| 非空字符串 | 外部文件 | 相对路径基于 `bundle_root` 拼接后读取 |

### 内联计算

正文采用约定结构：

````markdown
# Computation

```python
def compute(threshold):
    return threshold * 2
```
````

`extract_computation_from_body` 用正则匹配 `# Computation` 标题后的**第一个** ``` 代码围栏，返回围栏内内容（去除围栏标记）。

### 文件式计算

当 `computation` 指定了文件路径时，`load_computation_file` 解析路径：相对路径基于 `bundle_root` 拼接（绝对路径直接使用），随后读取文件文本。此时若未提供 `bundle_root` 会抛出 `ValueError`。

## 解析结果

解析成功后返回 `AttestedComputation` 模型：

```text
AttestedComputation(
    runtime = "python-3.14",
    executor = Executor(resource="executor-01", receipt=[...]),
    attester = Attester(resource="attester-01"),
    parameters = [ComputationParameter(name="threshold", type="float", required=True), ...],
    computation = "<计算逻辑文本>",
)
```

其中 `computation` 字段承载的是计算逻辑的**文本内容**（而非路径）。