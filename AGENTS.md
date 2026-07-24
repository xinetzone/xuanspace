# 玄境（Xuanspace）智能体协作入口

> **🚨 启动协议（PRIORITY ZERO — 所有智能体必须在收到任务后立即执行）**
>
> **步骤 1**：读取本文件全文
>
> **步骤 2**：按「上下文路由表」确定本次任务需要读取的规范文件
> - **步骤 2.0**（子项目预检·必做）：检查当前工作目录是否在子项目目录下（`apps/*/`、`libs/*/`），如果是则：
>   - 读取该子项目自身的 `README.md`
>   - 读取该子项目的 `pyproject.toml`（Python项目）或 `CMakeLists.txt`（原生项目）
>   - 遵循该子项目目录下的特殊规范（若有 `AGENTS.md`）
>
> **步骤 3**：读取对应的规范文件（按需读取，不要一次加载全部）
>
> **步骤 3.5**（自检·必做）：在执行任何操作之前，逐项确认：
> - □ 是否已完成子项目预检？是否需要读取子项目自身规范？
> - □ 是否已读取上下文路由表中与当前任务直接相关的入口？
> - □ 是否理解当前任务所属的内容敏感度级别？
>
> **步骤 4**：在规范指导下执行任务
>
> ⚠️ **禁止在完成步骤 1-3.5 之前生成任何产出物。跳过此协议将导致输出格式错误、文件路径错误、项目结构不符合规范。**

## 项目概述

**Xuanspace（玄境）** 是一个 Python 3.13+ monorepo 项目，用于管理多个子项目。

- **核心理念**：技术为器、思想为道，器以载道
- **技术栈**：
  - Python 3.13+（PEP 517/621）
  - typer CLI（xs 命令行工具）
  - CMake + Ninja + scikit-build-core（C++ 原生扩展）
  - Sphinx + MyST（文档系统）

本文件是玄境项目 AI 智能体的最高优先级入口与上下文路由。所有智能体在启动时必须首先读取本文件，依据上下文路由表定位到具体的 `.agents/` 规范后执行任务。

## 核心规范入口表

| 规范 | 入口 | 说明 |
|---|---|---|
| 🚀 入门指南 | [.agents/ONBOARDING.md](.agents/ONBOARDING.md) | 快速开始、能力速查表、常用 xs 命令 |
| 📜 全局核心规则 | [.agents/global-core-rules.md](.agents/global-core-rules.md) | 启动协议、内容分流、版本要求、工作流规范 |
| 🧭 上下文路由表 | [.agents/context-routing.md](.agents/context-routing.md) | 任务类型→必读规范映射表 |
| 📄 文档元数据规范 | [.agents/rules/frontmatter.md](.agents/rules/frontmatter.md) | YAML/TOML 内容-元数据二分法 |
| 🔍 工作区发现协议 | [.agents/protocols/workspace-discovery.md](.agents/protocols/workspace-discovery.md) | 五步发现流程，从任意位置定位工作区 |
| 🚀 提示词自举协议 | [.agents/protocols/prompt-bootstrap.md](.agents/protocols/prompt-bootstrap.md) | 一句话装载，零配置接入 |
| 🤝 协作协议 | [.agents/protocols/README.md](.agents/protocols/README.md) | 会话启动、任务交接、工作区发现 |
| 💬 提示词库 | [.agents/prompts/README.md](.agents/prompts/README.md) | 角色提示词集合 |
| 📦 项目模板 | [.agents/templates/README.md](.agents/templates/README.md) | 模板索引（Python/C++/静态项目） |
| 📋 模板源码 | [tools/templates/](tools/templates/) | 模板文件存放位置 |

## 目录结构说明

```
xuanspace/
├── apps/              # 可执行应用和CLI工具
├── libs/              # 可复用Python库和C++原生扩展
├── vendor/            # 第三方依赖和外部项目（只读）
├── tools/             # 项目工具链（xs CLI、模板）
├── docs/              # 文档目录（Sphinx+MyST）
├── scripts/           # 构建和维护脚本
├── attic/             # 归档/废弃内容
├── .agents/           # AI智能体规范目录（本规范所在目录）
├── .meta/toml/        # TOML元数据镜像目录
└── AGENTS.md          # 本文件 - 智能体入口
```

### 子项目目录分类

| 目录 | 适用场景 |
|---|---|
| `apps/` | 可执行应用、CLI工具、面向最终用户的程序 |
| `libs/` | 可复用的Python库、C++原生扩展 |
| `vendor/` | 第三方依赖、fork的外部项目（只读） |
| `tools/` | 项目内部工具链、模板、构建辅助 |

## 开发规范要点

- **Python版本**：所有子项目 `requires-python>=3.13`，严格使用Python 3.13+特性
- **包管理器**：不强制PDM，支持uv/pip等标准Python工具
- **代码风格**：遵循ruff+black+isort配置（行宽120，py313目标版本），配置见根目录pyproject.toml
- **提交规范**：遵循Conventional Commits（`type(scope): subject`），主体使用中文
- **文档规范**：遵循YAML/TOML内容-元数据二分法，详见 [.agents/rules/frontmatter.md](.agents/rules/frontmatter.md)
- **路径引用**：Markdown文档交叉引用使用相对路径，禁止`file:///`绝对路径

## 开发操作规范

- **新增子项目**：使用 `xs new --type python|native|static <name>` 从模板创建
- **C++原生扩展**：必须在`libs/`下创建，使用CMake+Ninja+scikit-build-core构建系统
- **文档构建**：使用Sphinx+MyST构建，配置见`docs/conf.py`
- **环境检查**：使用 `xs doctor` 检查开发环境完整性
- **列出子项目**：使用 `xs list` 查看所有子项目状态
