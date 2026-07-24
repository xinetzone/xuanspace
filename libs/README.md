# libs/ - 共享库目录

## 概述

`libs/` 目录存放 Xuanspace（玄境）项目中的共享代码库。这些库是项目的核心构建块，可被 `apps/` 中的应用程序以及其他 `libs/` 中的库导入使用，提供可复用的功能模块。

## 库类型

### 1. Python 库

纯 Python 共享库，无原生扩展：

- 使用 `setuptools` 或 `hatchling` 作为构建后端
- 无需额外编译工具链，跨平台兼容性好
- 对外提供 importable API，无独立运行入口

### 2. C/C++ 扩展库

包含原生代码的高性能库：

- 使用 `scikit-build-core + CMake + Ninja` 构建
- 需要正确配置 `CMakeLists.txt`
- 提供预编译 wheel 或确保 CI 环境有完整编译工具链

## 准入标准

一个子项目应放入 `libs/` 当且仅当满足以下条件：

1. **对外提供 importable API**：主要用途是被其他 Python 代码通过 `import` 语句使用
2. **无独立运行入口**：`pyproject.toml` 中不定义 `[project.scripts]` 或 `[project.gui-scripts]` 入口点
3. **封装可复用逻辑**：封装了可复用的业务逻辑、工具函数或基础框架能力

**判断原则**：如果一个包包含 `if __name__ == "__main__"` 块，这并不意味着它属于 `apps/`——关键看它是否对外提供可导入的 API，且没有正式的可执行入口。

## 创建方式

使用 `xs` CLI 创建新库（注意：不加 `--app` 参数）：

```bash
# 创建纯 Python 库
xs new --type python <library-name>

# 创建 C/C++ 扩展库
xs new --type native <library-name>
```

## 命名规范

- 使用 **kebab-case** 命名风格
- 核心库通常以 `xuan-` 为前缀（如 `xuan-core`、`xuan-config`）
- 工具类库可以以 `xs-` 为前缀（如 `xs-utils`）
- 名称应清晰反映库提供的功能领域

**正确示例**：
- `xuan-core` - 核心工具库，提供基础数据结构和通用函数
- `xuan-config` - 配置管理库，统一处理 YAML/TOML 配置加载
- `xs-io` - IO 工具库，封装文件读写操作

**错误示例**：
- `core`（过于通用，容易冲突）
- `utils`（含义模糊）
- `ConfigLib`（应使用 kebab-case）

## 项目结构要求

每个 `libs/` 下的子项目**必须**满足以下要求：

1. 拥有独立的 `pyproject.toml` 文件，正确声明依赖和构建配置
2. 通过 PDM workspace 被根目录识别
3. 公开 API 通过 `__init__.py` 清晰导出，避免内部实现细节泄露
4. 包含完整的类型注解（`py.typed` 标记文件推荐）
5. 有对应的单元测试

### pyproject.toml 配置示例（纯 Python）

```toml
[project]
name = "xuan-core"
version = "0.1.0"
description = "Xuanspace 核心工具库"
dependencies = [
    "pydantic>=2.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### pyproject.toml 配置示例（含 C/C++ 扩展）

```toml
[project]
name = "xuan-accelerate"
version = "0.1.0"
description = "高性能计算加速库"
dependencies = [
    "numpy>=1.24",
]

[build-system]
requires = ["scikit-build-core>=0.5"]
build-backend = "scikit_build_core.build"
```

## 现有子项目示例

| 目录名 | 类型 | 说明 |
|--------|------|------|
| `xuan-core/` | 纯 Python | 核心工具库，提供基础类型、通用函数、错误处理 |
| `xuan-ext-demo/` | 原生扩展 | C/C++ 扩展示例库 |

## 依赖规则

- **可以依赖**：其他 `libs/` 中的库、PyPI 第三方包
- **不应依赖**：`apps/` 中的任何应用项目（防止循环依赖）
- **谨慎依赖**：`tools/` 中的工具库（tools 主要面向开发环境）
- **禁止依赖**：`scripts/`、`attic/`、`vendor/` 中的代码
