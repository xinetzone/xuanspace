# 文档元数据二分法规范

玄境项目采用 **YAML/TOML 内容-元数据二分法** 管理文档：
- **Markdown 文件**：只存放正文内容和极简 YAML frontmatter
- **TOML 文件**：存放所有结构化元数据，位于 `.meta/toml/` 镜像路径

**注意：当前阶段（Task 20 之前）不要添加 YAML frontmatter，本规范为后续任务准备。**

## 1. YAML frontmatter 规范

Markdown 文件顶部的 YAML frontmatter 仅允许以下 4 个字段：

```yaml
---
id: "unique-document-id"
x-toml-ref: "path/to/metadata.toml"
source: "original-source-if-applicable"
version: "1.0.0"
---
```

### 允许字段说明

| 字段 | 必填 | 说明 |
|---|---|---|
| `id` | 是 | 文档唯一标识符，建议使用 kebab-case |
| `x-toml-ref` | 是 | 对应 TOML 元数据文件的相对路径（相对于 .meta/toml/） |
| `source` | 否 | 原始来源（如引用自外部文档） |
| `version` | 否 | 文档版本号，遵循 semver |

### 禁止字段列表

YAML frontmatter 中**禁止**出现以下字段（这些应放在 TOML 元数据中）：
- ❌ `title`（标题应作为 Markdown 正文的 H1）
- ❌ `description`、`summary`
- ❌ `tags`、`categories`、`keywords`
- ❌ `author`、`created_at`、`updated_at`
- ❌ `status`、`draft`
- ❌ 任何其他自定义元数据字段

## 2. TOML 元数据规范

所有扩展元数据存放于 `.meta/toml/` 目录下，路径与原 Markdown 文件镜像对应。

### 路径计算方法

| Markdown 文件路径 | TOML 元数据路径 |
|---|---|
| `README.md` | `.meta/toml/README.toml` |
| `doc/guide.md` | `.meta/toml/doc/guide.toml` |
| `libs/xuan-core/README.md` | `.meta/toml/libs/xuan-core/README.toml` |
| `.agents/ONBOARDING.md` | `.meta/toml/.agents/ONBOARDING.toml` |

**计算规则**：
1. 保持原有的目录结构
2. 文件扩展名从 `.md` 改为 `.toml`
3. 根目录下的文件直接放在 `.meta/toml/` 下

### TOML 文件结构

```toml
# 文档标题（对应Markdown的H1）
title = "文档标题"

# 文档描述
description = "文档的简短描述"

# 标签
tags = ["tag1", "tag2"]

# 分类
categories = ["category"]

# 状态：draft（草稿）、review（审核中）、stable（稳定）、deprecated（废弃）
status = "stable"

# 作者信息
[author]
name = "作者名"
email = "e******@*********"

# 时间戳
[timestamps]
created = 2025-01-01T00:00:00Z
updated = 2025-01-15T12:00:00Z

# 自定义扩展字段（放在此处）
[extra]
custom_field = "value"
```

## 3. 使用原则

1. **内容与元数据分离**：Markdown 专注于写作，TOML 专注于结构化数据
2. **极简 frontmatter**：YAML 部分只保留连接信息，不重复内容
3. **单一数据源**：每个元数据字段只在 TOML 中出现一次，不重复
4. **路径镜像**：TOML 路径严格对应 Markdown 路径，便于查找

## 4. 当前阶段说明

在 Task 20 统一处理 frontmatter 之前：
- ✅ 可以正常编写 Markdown 内容（带 H1 标题）
- ❌ 不要手动添加 YAML frontmatter
- ❌ 不要创建 TOML 元数据文件
- ✅ 后续任务会统一处理元数据添加
