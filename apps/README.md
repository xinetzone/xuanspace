# apps/ - 应用程序目录

## 概述

`apps/` 目录存放 Xuanspace（玄境）项目中所有可直接运行/部署的应用程序。这些子项目是最终用户直接交互的入口点。

## 项目类型

本目录支持三种类型的应用：

### 1. 纯 HTML/JS 静态项目

纯前端静态 Web 项目，无需后端服务：

- 包含 `index.html` 作为入口点
- 使用原生 HTML/CSS/JavaScript 或前端框架
- 可直接部署到静态托管服务
- 示例：竹简悟道（zhujian-wudao）

### 2. Python 可执行应用

Python 编写的命令行工具、Web 服务、GUI 应用等：

- 包含 `main.py` 或在 `pyproject.toml` 中定义入口点
- 可以组合多个 `libs/` 中的共享库完成业务功能
- 支持 CLI 工具、Web 服务、数据管道、GUI 应用等形态

### 3. C/C++ 原生可执行程序

使用 C/C++ 编写的高性能原生应用：

- 包含 CMakeLists.txt 构建配置
- 通过 scikit-build-core + CMake + Ninja 构建
- 可独立编译为可执行文件

## 准入标准

一个子项目应放入 `apps/` 当且仅当其具备**独立运行入口点**：

- 静态项目：有 `index.html` 文件
- Python 应用：有 `main.py` 或在 `pyproject.toml` 中定义 `[project.scripts]`/`[project.gui-scripts]`
- 原生应用：有 CMakeLists.txt 且定义了可执行目标

**判断原则**：如果一个包的主要价值在于被其他代码 `import` 使用，它应该放在 `libs/`；如果它的主要价值在于被直接执行/部署，它应该放在 `apps/`。

## 创建方式

使用 `xs` CLI 创建新应用：

```bash
# 创建静态 HTML/JS 项目
xs new --type static --app <project-name>

# 创建 Python 应用
xs new --type python --app <project-name>

# 创建 C/C++ 原生应用
xs new --type native --app <project-name>
```

## 命名规范

- 使用 **kebab-case** 命名风格
- 名称应清晰反映应用的功能定位
- 命令行工具建议以 `xs-` 为前缀（如 `xs-cli`）
- 避免使用过于通用或模糊的名称（如 `app`、`tool`、`service`）

**正确示例**：
- `xs-cli` - 项目主命令行工具
- `zhujian-wudao` - 竹简悟道静态应用
- `demo-app` - 示例演示应用

**错误示例**：
- `myapp`（不清晰）
- `test`（过于通用）
- `cli_tool`（应使用 kebab-case）

## 项目结构要求

每个 `apps/` 下的子项目**必须**满足以下要求：

1. 拥有独立的入口点文件（index.html、main.py 或 CLI 入口）
2. 通过 PDM workspace 被根目录识别（根 `pyproject.toml` 中已配置 `packages`）
3. 有自己的 README.md 说明文档
4. 静态项目不依赖后端服务

## 现有子项目示例

| 目录名 | 类型 | 说明 | 入口点 |
|--------|------|------|--------|
| `xs-cli/` | Python | 项目主 CLI 工具，提供项目管理、构建、文档等命令 | `xs` |
| `zhujian-wudao/` | static | 竹简悟道纯前端静态应用 | index.html |

## 与其他目录的关系

- **依赖 libs/**：Python 和原生应用可以依赖 `libs/` 下的共享库
- **使用 tools/**：可以调用 `tools/` 中的开发工具实现功能
- **不反向依赖**：`libs/` 和 `tools/` 中的代码**不应该**依赖 `apps/` 中的任何项目
