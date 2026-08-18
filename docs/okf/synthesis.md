# 索引、日志与链接

本章说明三个相互配合的机制：`index.md` 的合成（§8）、`log.md` 的解析与合成（§9）、Markdown 链接的分类与断链检测（§6）。

## index.md 合成（§8）

`generate_index` 按以下规则生成 `index.md`：

- **不包含 YAML frontmatter**
- 一级标题为 `# <目录名>`
- 按概念的 `type` 字段分组，每个类型一个小节 `## <TypeName>`
- 每个概念条目格式：`- [<title>](<相对路径>.md) — <description>`
- 无 `description` 时省略 `— description` 部分
- 无 `type` 的概念归入 `_` 分组
- 分组名称按字典序排列

示例：

```markdown
# Sample Bundle

## Metric

- [Revenue](concepts/metrics/revenue.md) — Annual revenue by region
- [Active Users](concepts/metrics/active_users.md)

## Table

- [Customers](concepts/tables/customers.md) — Customer master data
```

`okf index` 命令调用该生成器并写回 `<path>/index.md`。

## log.md 解析与合成（§9）

`log.md` 采用「日期标题 + 无序列表」结构：

```markdown
# Change Log

## 2024-02-01

- **Update** Updated revenue metrics for Q1 2024
- **Creation** Added customer table

## 2024-01-15

- **Creation** Initial bundle setup
```

- **解析**（`parse_log`）：日期标题采用 ISO 8601 格式 `## YYYY-MM-DD`，其下条目去除 `- ` / `* ` 列表标记后收集，返回 `{date: [entries]}`。
- **合成**（`generate_log`）：标题为 `# Change Log`，按日期**倒序**排列，每个日期组一个 `## YYYY-MM-DD` 小节。
- 粗体动词（`**Update**`、`**Creation**` 等）原样保留，属于约定俗成的动词标记而非被解析的结构。

## 链接分类（§6）

链接解析器从概念正文提取 Markdown 链接 `[text](target)`，并按 `target` 前缀分类：

| target 形式 | 分类 | 处理 |
|-------------|------|------|
| `/` 前缀 | Bundle 内绝对路径 | `bundle_root / target` 解析为 `Path` |
| `http://` / `https://` | 外部 URL | 保留为字符串 |
| `#` 前缀 | 锚点 | 保留为字符串 |
| 相对路径 | 相对链接 | 相对当前文件父目录解析 |

`parse_links` 与 `parse_links_with_context` 的区别在于：后者读取文件并携带「当前文件路径」上下文，从而能把相对路径解析为基于当前文件父目录的 `Path`。

## 断链检测

`check_broken_links` 对每个解析为 `Path` 的目标执行两层检查：

1. 文件是否存在（不存在 → 断链）
2. 文件虽存在，但其对应的概念 ID（相对 Bundle 根、去 `.md` 后缀）是否已注册到 `bundle.concepts`（未注册 → 断链）

指向 Bundle 根之外文件的链接同样记为断链。断链在 OKF 中是**警告而非错误**（§6.1），因此不会导致 `validate` 失败，但会出现在宽松项警告中。