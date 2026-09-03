# Bundle 与概念

Bundle 是 OKF 的最小自包含知识单元：一个目录树，内含携带 YAML frontmatter 的 Markdown 文件。本章说明 Bundle 的目录结构与概念（Concept）的 frontmatter 字段。

## 目录结构

一个标准 Bundle 的布局如下：

```text
my-bundle/
├── index.md       # 保留文件：概念索引，对应规范 §8
├── log.md         # 保留文件：变更日志，对应规范 §9
└── (任意 .md)     # 概念文件，可嵌套在任意子目录中
```

`okf init <path>` 生成的骨架额外包含三个子目录：

| 目录 | 用途 |
|------|------|
| `concepts/` | 概念文件 |
| `playbooks/` | 剧本（playbook） |
| `references/` | 引用材料 |

### 保留文件

加载 Bundle 时，文件名会决定文件如何被归类：

| 文件名 | 归类位置 |
|--------|----------|
| `index.md` | `Bundle.indices` |
| `log.md` | `Bundle.logs` |
| 其他 `.md` | 解析为 `Concept`，归入 `Bundle.concepts` |

### 概念 ID

非保留文件的**概念 ID** 由其**相对 Bundle 根目录的路径去除 `.md` 后缀、路径分隔符统一为 `/`** 得到。例如：

| 文件路径 | 概念 ID |
|----------|---------|
| `concepts/metrics/active_users.md` | `concepts/metrics/active_users` |
| `concepts/tables/customers.md` | `concepts/tables/customers` |

`okf inspect` / `okf trust` 都以概念 ID 作为定位参数。

## 概念数据模型

每个概念在内存中被表示为 `Concept`（一个 `frozen` dataclass），其字段与 frontmatter 的映射如下：

| 字段 | 类型 | 对应 frontmatter 键 |
|------|------|---------------------|
| `path` | `Path` | 文件绝对路径 |
| `type` | `str` | `type`（必填） |
| `title` | `str` | `title` |
| `description` | `str` | `description` |
| `resource` | `str` | `resource` |
| `tags` | `list[str]` | `tags` |
| `frontmatter` | `dict` | 全部原始 frontmatter |
| `body` | `str` | 正文 |
| `extra` | `dict` | 非已知字段的扩展键 |

其中 `extra` 收集所有**不在已知字段列表内**的键，用于承载扩展元数据。

## frontmatter 字段速查

`okf` 内置的 YAML 解析器识别以下**已知字段**：

| 字段 | 说明 |
|------|------|
| `type` | 概念类型，必填 |
| `title` | 标题 |
| `description` | 描述 |
| `resource` | 来源资源标识 |
| `tags` | 标签列表 |
| `sources` | 来源数组（见[信任与保鲜](trust)） |
| `usage_window` | 使用窗口 |
| `generated` | 生成信息 |
| `verified` | 验证记录（见[信任与保鲜](trust)） |
| `status` | 状态（`draft` / `stable` / `deprecated`） |
| `stale_after` | 失效日期（见[信任与保鲜](trust)） |
| `runtime` | 计算运行时（见 [Attested Computation](attested)） |
| `parameters` | 计算参数（见 [Attested Computation](attested)） |
| `computation` | 计算逻辑（见 [Attested Computation](attested)） |
| `executor` | 执行者（见 [Attested Computation](attested)） |
| `attester` | 见证者（见 [Attested Computation](attested)） |

## 解析规则细节

`okf` 不依赖 PyYAML，而是内置了一个**最小 YAML 子集解析器**（`src/okf/frontmatter.py`）。它支持：

- 标量：字符串、整数、布尔值（`true` / `false`）、`null`
- 流程列表 `[a, b]`、流程映射 `{k: v}`
- 块级序列（`- item`）、块级映射、嵌套结构
- 多行字符串续行

几点值得注意的细节：

1. **`type` 为必填**：缺失或为空会抛出 `FrontmatterError`，导致该文件无法被加载为概念。
2. **`tags` 归一化**：既支持 YAML 列表 `[a, b]`，也支持逗号分隔字符串 `"a, b"`，最终统一为字符串列表。
3. **`verified` 归一化**（§5.2）：裸映射 `verified: {by: ..., at: ...}` 等价于单元素列表 `verified: [{by: ..., at: ...}]`。

示例：

```markdown
---
type: Table
title: Customers
description: Customer master data
tags: [crm, master]
---

# Customers

Customer master data table.
```

## 相关页面

- [CLI 命令参考](commands)
- [一致性校验](conformance)
- [信任与保鲜](trust)