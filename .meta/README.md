# .meta/ - 元数据目录

## 概述

`.meta/` 目录存放外部 TOML 元数据文件，实现**内容-元数据二分法**架构。文档内容（Markdown）与元数据（TOML）分离存储，保持文档正文的纯粹性。

## 目录结构

```
.meta/
└── toml/              # TOML 元数据文件目录
    └── <镜像路径>/    # 按文档镜像路径存放
        └── <doc>.toml
```

`.meta/toml/` 下的目录结构与项目根目录下的 Markdown 文档路径一一对应（镜像路径）。

## 内容-元数据二分法原则

- **内容**：Markdown 文档正文，关注"写什么"，不包含配置信息
- **元数据**：TOML 文件，关注"如何组织/展示"，与正文分离

### YAML Frontmatter 规范

Markdown 文档的 YAML frontmatter **仅保留以下四个字段**：

```yaml
---
id: unique-document-id
x-toml-ref: .meta/toml/path/to/doc.toml
source: original-source-url (optional)
version: 1.0.0
---
```

| 字段 | 必填 | 说明 |
|------|------|------|
| `id` | 是 | 文档唯一标识符 |
| `x-toml-ref` | 是 | 对应 TOML 元数据文件的相对路径 |
| `source` | 否 | 原始来源 URL（适用于转载/引用文档） |
| `version` | 否 | 文档版本号 |

**其他所有元数据（标题、作者、日期、标签、分类等）一律存放在对应的 `.meta/toml/` TOML 文件中，不放在 YAML frontmatter 里。**

## TOML 元数据文件示例

对应文档 `docs/quickstart.md` 的元数据文件 `.meta/toml/docs/quickstart.toml`：

```toml
[meta]
title = "快速开始"
description = "5 分钟上手 Xuanspace"
date = 2026-01-01
authors = ["xinetzone"]
tags = ["getting-started", "quickstart"]
category = "getting-started"
order = 1
draft = false

[nav]
parent = "user-guide"
label = "快速开始"

[seo]
keywords = ["玄境", "快速开始", "安装"]
```

## 为什么这样设计？

1. **关注点分离**：写作者专注内容，元数据由工具/编辑器管理
2. **易于批量处理**：TOML 格式更适合程序读写，便于批量更新元数据
3. **避免污染正文**：Markdown 文件保持干净，只包含实际内容
4. **版本控制友好**：元数据变更与内容变更分离，便于 Code Review
