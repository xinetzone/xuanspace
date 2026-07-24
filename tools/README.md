# tools/ - 工具集目录

## 概述

`tools/` 目录存放 Xuanspace（玄境）项目的开发和构建工具。这些工具面向开发者，用于项目创建、构建、文档生成等开发流程，不属于项目运行时依赖。

## 目录结构

```
tools/
├── xs/              # xs CLI 工具主项目（核心实现）
└── templates/       # 项目模板目录
    ├── python/      # Python 库/应用模板
    ├── native/      # C/C++ 原生扩展/应用模板
    └── static/      # 纯 HTML/JS 静态项目模板
```

## 主要组件

### tools/xs/ - CLI 工具主项目

`tools/xs/` 是 `xs` 命令行工具的核心实现，提供以下命令：

- `xs new` - 创建新项目（使用 templates/ 中的模板）
- `xs build` - 构建项目
- `xs doctor` - 检查开发环境
- `xs list` - 列出所有项目
- `xs docs` - 文档相关操作
- 其他开发辅助命令

`xs` CLI 是项目的主要开发入口，apps/xs-cli/ 只是入口点包装，实际实现在 tools/xs/ 中。

### tools/templates/ - 项目模板

`templates/` 目录包含用于 `xs new` 命令的项目脚手架模板：

| 模板目录 | 用途 | 对应命令 |
|----------|------|----------|
| `templates/python/` | Python 库和应用模板 | `xs new --type python` |
| `templates/native/` | C/C++ 原生扩展和应用模板 | `xs new --type native` |
| `templates/static/` | 纯 HTML/JS 静态项目模板 | `xs new --type static` |

每个模板目录包含该类型项目的标准结构、pyproject.toml 配置、示例代码和 README。

## 准入标准

一个子项目应放入 `tools/` 当且仅当满足以下条件：

1. **面向开发者**：是开发/构建工具，而非最终用户使用的应用
2. **正式工具链**：是核心开发工具，不是一次性脚本（一次性脚本放 scripts/）
3. **有独立结构**：包含完整的项目结构，有自己的 pyproject.toml（如 xs/）或是组织化的模板集（如 templates/）

## tools/ 与 scripts/ 的区别

| 特性 | tools/ | scripts/ |
|------|--------|----------|
| 定位 | 正式的开发工具链 | 一次性/维护脚本 |
| 复用性 | 高，被 xs CLI 或其他工具调用 | 低，特定任务专用 |
| 用户 | 所有开发者通过 xs 命令使用 | 项目维护者手动执行 |
| 结构 | 多文件包，有 pyproject.toml | 通常单文件脚本 |
| 示例 | xs CLI、项目模板 | 发布脚本、迁移脚本 |

## 依赖规则

- **可以依赖**：`libs/` 中的共享库、PyPI 开发类依赖
- **可以被依赖**：`apps/` 中的入口应用（如 apps/xs-cli/ 调用 tools/xs/）
- **不应依赖**：`apps/` 中的业务应用
- **不进入生产环境**：tools 中的依赖不应出现在运行时依赖列表中
