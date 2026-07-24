# 项目介绍

## 命名溯源

**玄境（Xuanspace）** 取自《老子》开篇：

> 道可道，非常道；名可名，非常名。无名天地之始，有名万物之母。故常无欲，以观其妙；常有欲，以观其徼。此两者同出而异名，**同谓之玄，玄之又玄，众妙之门**。

"玄"意为深远、幽微、不可穷尽；"境"意为境界、空间、领域。玄境即是一个探索技术深境的空间，承载着我们对代码与思想的双重追求。

## 设计哲学

### 技术为器，思想为道，器以载道

- **器**：Python 库、C++ 扩展、CLI 工具——这些是载体
- **道**：设计理念、架构哲学、文化内涵——这些是灵魂
- **器以载道**：技术服务于思想，工具承载理念

我们相信，好的代码不仅是能运行的，更是有思想的。每一行代码背后，都应当有清晰的设计意图和哲学思考。

## 项目定位

玄境是一个 **Python 3.13+ Monorepo** 项目，旨在统一管理多个子项目，包括：

| 项目类型 | 目录 | 说明 |
|---|---|---|
| 可执行应用 | `apps/` | CLI 工具、Web 应用、文化项目 |
| 可复用库 | `libs/` | Python 库、C++ 原生扩展 |
| 第三方依赖 | `vendor/` | 需要 patch 的外部项目 |
| 内部工具 | `tools/` | 项目构建和开发辅助工具 |

## 核心能力

- **Monorepo 管理**：统一的项目结构，workspace 自动链接，依赖关系可视化
- **多语言支持**：Python 库 + C++ 原生扩展，通过 CMake + Ninja + scikit-build-core 无缝集成
- **多包管理器**：PDM（推荐）、uv（快速）、pip（标准）均可使用，不强制绑定
- **版本管理**：`xs version` 命令统一管理子项目版本，自动生成 CHANGELOG
- **文档系统**：Sphinx + MyST Markdown，支持 Mermaid 图表，美观的文档主题
- **AI 协作**：内置 `AGENTS.md` 和 `.agents/` 规范，AI 智能体可直接参与开发

## 项目状态

Xuanspace 目前处于活跃开发阶段，核心 CLI 工具 `xs` 已实现以下命令：

| 命令 | 功能 | 状态 |
|---|---|---|
| `xs list` | 列出所有子项目 | ✅ |
| `xs new` | 从模板创建新项目 | ✅ |
| `xs build` | 构建项目 | ✅ |
| `xs doctor` | 环境诊断 | ✅ |
| `xs init` | 初始化工作区 | ✅ |
| `xs deps` | 依赖管理（check/tree/outdated/update） | ✅ |
| `xs version` | 版本管理（show/bump） | ✅ |
| `xs docs` | 文档管理（build/serve/clean） | ✅ |
| `xs meta` | 元数据管理（init/validate/scan/sync） | ✅ |
| `xs toolchain` | 工具链管理（check/list/install） | ✅ |
| `xs py-compat` | Python 兼容性检查 | ✅ |
| `xs update` | 子模块与依赖更新 | ✅ |

## 相关资源

- [GitHub 仓库](https://github.com/xinetzone/xuanspace)
- [快速开始](quickstart)
- [架构设计](architecture)
- [构建系统](build-system)
- [贡献指南](contributing)