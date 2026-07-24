# docs/ - 项目文档目录

## 概述

`docs/` 目录存放 Xuanspace（玄境）项目的所有文档源码。项目使用 **Sphinx + MyST Markdown** 构建文档系统，支持 Markdown 编写、自动 API 文档生成、交叉引用等功能。

## 文档技术栈

- **构建工具**：Sphinx
- **Markdown 支持**：MyST Parser（支持标准 Markdown + Sphinx 扩展语法）
- **API 文档**：sphinx-autodoc2（从 Python 源码自动生成）
- **主题**：（待配置，推荐 furo 或 sphinx-rtd-theme）
- **输出格式**：HTML（默认）、PDF、Markdown（单文件导出）

## 文档结构

```
docs/
├── getting-started/      # 快速开始
│   ├── installation.md   # 安装指南
│   ├── quickstart.md     # 快速上手
│   └── first-project.md  # 第一个项目
├── user-guide/           # 用户指南
│   ├── cli.md            # CLI 使用
│   ├── configuration.md  # 配置说明
│   └── tutorials/        # 教程集合
├── api-reference/        # API 参考
│   ├── xuan-core.md      # xuan-core 库 API
│   ├── xuan-config.md    # xuan-config 库 API
│   └── ...               # 其他库 API
├── developer-guide/      # 开发者指南
│   ├── setup.md          # 开发环境搭建
│   ├── workflow.md       # 开发工作流
│   ├── testing.md        # 测试指南
│   └── coding-style.md   # 编码规范
├── architecture/         # 架构设计
│   ├── overview.md       # 架构总览
│   ├── directory.md      # 目录结构说明
│   ├── monorepo.md       # Monorepo 管理
│   └── decisions/        # 架构决策记录 (ADR)
├── conf.py               # Sphinx 配置
├── index.md              # 文档首页
└── _static/              # 静态资源（图片、CSS、JS）
```

### 各目录内容说明

| 目录 | 目标读者 | 内容 |
|------|----------|------|
| `getting-started/` | 新用户 | 5 分钟快速上手、安装、Hello World |
| `user-guide/` | 用户 | 功能使用说明、教程、最佳实践 |
| `api-reference/` | 开发者 | 从代码自动生成的 API 文档 |
| `developer-guide/` | 贡献者 | 开发环境、工作流、测试、规范 |
| `architecture/` | 核心开发者 | 系统设计、架构决策、技术选型 |

## 构建命令

项目通过 `xs` CLI 提供文档操作命令（推荐方式）：

```bash
# 构建 HTML 文档
xs docs build

# 构建并启动本地预览服务器（支持自动刷新）
xs docs serve

# 清理构建产物
xs docs clean

# 检查文档链接有效性
xs docs linkcheck
```

### 直接使用 Sphinx 命令

如果未安装 CLI，也可以直接使用 Sphinx 命令：

```bash
# 安装文档依赖
pdm install --dev -G docs

# 构建 HTML
pdm run sphinx-build -b html docs/ docs/_build/html

# 启动预览服务器
pdm run sphinx-autobuild docs/ docs/_build/html
```

构建产物输出到 `docs/_build/` 目录（该目录已在 .gitignore 中）。

## 内容-元数据二分法规范

所有文档必须遵循 **YAML/TOML 内容-元数据二分法** 规范：

### 原则

- **内容（正文）** 使用 Markdown 编写，关注"是什么"
- **元数据（配置）** 使用 YAML frontmatter 或 TOML 配置块编写，关注"如何组织"
- 二者严格分离，不混合使用

### 文档 YAML Frontmatter

每个文档文件开头应包含 YAML frontmatter：

```yaml
---
title: 安装指南
description: 如何在各种环境下安装 Xuanspace
date: 2024-01-01
authors:
  - Author Name
tags:
  - getting-started
  - installation
category: getting-started
order: 1
---
```

### 代码示例的元数据分离

Markdown 正文（内容）：
```markdown
# 配置文件示例

这是一个基本的项目配置文件：
```

独立 TOML 代码块（元数据），使用 `toml` 标签并与正文分离：

```toml
# 这是配置示例的元数据/代码内容
[project]
name = "my-xuan-project"
version = "0.1.0"

[xuan]
debug = true
```

### 规范要点

1. **不使用 Markdown 内嵌 HTML** 来表达配置或元数据
2. **配置示例统一使用 TOML 代码块**，配置文件本身也是 TOML 格式
3. **描述性内容用 Markdown**，结构性配置用 YAML/TOML
4. **API 文档中的类型注解**通过 Python 源码的类型提示自动生成，不要在 Markdown 中重复

## MyST Markdown 常用语法

MyST 支持标准 CommonMark Markdown，并添加了 Sphinx 扩展：

### 交叉引用

```markdown
<!-- 引用其他文档 -->
参见 [快速开始指南](getting-started/quickstart.md)

<!-- 引用 API -->
参见 {py:func}`xuan_core.some_function`

<!-- 引用任意标题（自动生成锚点） -->
参见 [安装章节](#installation)
```

### 提示框（Admonitions）

```markdown
:::{note}
这是一个提示。
:::

:::{warning}
这是一个警告。
:::

:::{tip}
这是一个建议。
:::
```

### 代码块增强

````markdown
```{code-block} python
:linenos:
:emphasize-lines: 2,5

def hello():
    print("Hello, Xuanspace!")  # 这行会被高亮
    return True
```
````

## 文档编写指南

1. **面向读者写作**：明确该文档的目标读者（新用户/进阶用户/开发者）
2. **提供可运行示例**：所有代码示例应该可以直接复制运行
3. **保持更新**：代码变更时同步更新对应文档
4. **从简到繁**：教程/指南类文档遵循从简单到复杂的递进顺序
5. **中英文混排**：中文与英文/数字之间加空格，提升可读性
6. **API 文档自动生成**：不要手动编写 API 文档，通过 sphinx-autodoc2 从代码注释生成

## 文档 PR 要求

提交文档变更时需满足：

- [ ] `xs docs build` 构建无错误和警告（允许 nitpick 警告）
- [ ] `xs docs linkcheck` 链接检查通过
- [ ] 新增文档已正确添加到对应目录的 index.md 目录树
- [ ] 所有代码示例可正常运行（如有必要，包含对应的测试）
- [ ] 图片等静态资源放入 `_static/` 目录
