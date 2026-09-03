# 一致性校验

`okf validate` 对 Bundle 执行一致性校验（规范 §11）。校验分两个层级：**严格项**（产生错误，不通过则 Bundle 不合规）与**宽松项**（产生警告，不拒绝 Bundle）。两者汇总为一份 `ConformanceReport`。

## 严格项（错误）

严格项检查由 `validate_strict` 执行，任一不满足即判定 Bundle 不合规：

| 检查项 | 说明 |
|--------|------|
| frontmatter 存在 | 每个非保留 `.md` 概念文件必须有可解析的 YAML frontmatter（非空） |
| body 非空 | 每个概念的正文不能为空 |
| `type` 非空 | 每个 frontmatter 必须含非空 `type` 字段 |
| `index.md` 存在 | `Bundle.indices` 不能为空 |
| `log.md` 存在 | `Bundle.logs` 不能为空 |

## 宽松项（警告）

宽松项检查由 `validate_lenient` 执行，输出警告但不拒绝 Bundle：

| 检查项 | 说明 |
|--------|------|
| 缺少 `title` | 概念未声明标题 |
| 缺少 `description` | 概念未声明描述 |
| 未登记的 `type` | OKF 不集中注册类型，此条始终提示（不拒绝） |
| 未知扩展键 | frontmatter 含已知字段列表之外的键 |
| 断链 | 调用链接检测，发现的断链以警告输出 |

> **说明**：`type` 字段的「未知类型」提示是**始终产生**的——OKF 采用开放类型体系，不维护集中式的类型注册表，因此每个概念都会得到一条「OKF does not centrally register types」的提示性警告。

## 报告格式

`format_report` 生成的报告形如：

```text
=== OKF Conformance Report ===
Bundle: <bundle_root>

Errors (0):
  (none)

Warnings (2):
  - WARNING: Concept '...' is missing title
  - WARNING: Concept '...' has unknown extension keys: ...

Result: PASS
```

`Result` 取值由 `is_conformant(report)` 决定：`len(errors) == 0` 时为 `PASS`，否则为 `FAIL`。

## 断链检测

宽松项中的断链检测复用链接解析模块（见[索引与日志合成](synthesis) 的链接部分）。指向不存在的文件、指向 Bundle 根之外的文件，或指向「文件存在但并非已注册概念」的链接，都会被记为断链警告。

## 退出码

`okf validate` 在报告包含错误时以退出码 1 结束，否则为 0。这使得它可以方便地接入 CI 流水线作为门禁。