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
requires = ["scikit-build-core>=0.11"]
build-backend = "scikit_build_core.build"
```

## 现有子项目

> 以下清单由 `scripts/check_declaration_consistency.py` 从 `libs/` 磁盘目录派生（按字母序），**块内内容请勿手改**；新增子项目后须在脚本的 `LIB_PROJECT_META` 中登记类型与说明，再运行 `python scripts/check_declaration_consistency.py --generate` 刷新本块。

<!-- BEGIN GENERATED: libs-projects -->

| 目录名 | 类型 | 说明 |
|--------|------|------|
| `caffe-ffi/` | 原生扩展 | Caffe 深度学习框架 FFI 绑定 |
| `demo-ffi/` | 原生扩展 | FFI 扩展开发模板（Stub/Real 双运行时） |
| `mystx/` | Python（子模块） | MyST 文档工具链/主题 |
| `npu-ffi/` | 原生扩展 | VTA NPU 加速器 FFI 绑定 |
| `tvm-book/` | Python（子模块） | TVM 中文手册 + flexloopy |
| `xuan-cmd/` | 纯 Python | 命令装配与工程路径解析工具库（invoke 任务通用件萃取） |
| `xuan-compose/` | 纯 Python | Compose 规范声明式容器编排引擎（podman-compose 重构） |
| `xuan-core/` | 纯 Python | 核心工具库，提供基础类型、通用函数、错误处理 |
| `xuan-ext-demo/` | 原生扩展 | C/C++ 扩展示例库 |
| `xuan-plugin/` | 纯 Python | 插件内核：可逆装配、能力注册与生命周期管理 |

<!-- END GENERATED: libs-projects -->

## 依赖规则

- **可以依赖**：其他 `libs/` 中的库、PyPI 第三方包
- **不应依赖**：`apps/` 中的任何应用项目（防止循环依赖）
- **谨慎依赖**：`tools/` 中的工具库（tools 主要面向开发环境）
- **禁止依赖**：`scripts/`、`attic/`、`vendor/` 中的代码

> 上述规则并非仅作声明：由 [`scripts/check_dependency_direction.py`](../scripts/check_dependency_direction.py)
> 在 CI `quality` job 中机械断言（解析 `libs/` 源码的 import，命中禁止分区即判失败）。
> 该脚本的 `FORBIDDEN_IMPORTS` 是本规则的单一事实来源，调整规则须同步此表。

## 工作区与跨仓复用边界

`[tool.pdm.workspace]` 只解决**本仓内**解析：`pdm install` 后根环境可 `import` 本目录下的库，
但这依赖工作区路径，不具备可传递性。跨仓复用（如另一个仓库的脚本引用本目录的库）只有两种合法介质：

1. **可安装制品**：本库独立发布 wheel，对方以版本化依赖引用；对方解释器基线低于本库
   `requires-python` 时，须先评估是否下调基线，不能用路径注入绕过。
2. **源码复制**：对方随包携带该源码，并自行承担同步与校验。

不采用的做法：把本仓路径写入对方 `PYTHONPATH` / `sys.path`；在对方配置中以路径源引用工作区成员；
把根发行版聚合为携带本目录业务代码的大包（与根 `pyproject.toml` 中“各独立发行版”的设计相悖，
且放大对方依赖面）。

### 跨仓发布流程（以 `xuan-cmd` 为例）

1. **核对对方基线**：确认消费方的 `requires-python`。本库声明 `>=3.14.6`，若对方达不到，
   先下调本库基线并同步 classifiers，在本库跑完全套测试再发版；不得要求对方用路径注入绕过。
2. **确定版本**：修改 `libs/<库名>/pyproject.toml` 的 `version`，并在 `CHANGELOG.md` 记录变更。
3. **本地构建与校验**（在库目录执行；Windows 用 `py`，Linux 用 `python`）：
   ```bash
   python -m build                 # 产出 sdist 与 py3-none-any wheel
   python -m twine check dist/*
   ```
4. **发布**，二选一：
   - **公开索引**：推送 `xuan-cmd-v<版本>` tag，参照 [`.github/workflows/release.yml`](../.github/workflows/release.yml)
     新增同构 job（改 `working-directory` 与 tag 前缀，复用 `PYPI_TOKEN`）；先用 `workflow_dispatch`
     的 dry-run 上传 TestPyPI 验证。现有流水线只覆盖 `xs-cli-v*`，新增库须用独立 tag 前缀避免误触发。
   - **内网或离线**：`python scripts/build_lib_dist.py --lib <库名>` 构建 wheel 并生成
     `CHECKSUMS`（默认输出到 `dist/libs/<库名>/`，不入库）；该目录可直接作为内网索引的
     文件源，或随交付物附带。消费方安装：
     `pip install --no-index --find-links <目录> <库名>==<版本>`；
     分发前用 `sha256sum -c CHECKSUMS` 校验完整性。
5. **消费方声明固定版本**：`<库名>==<版本>`；交付类场景不用版本范围，避免客户侧漂移。
6. **发布后核验**：在干净虚拟环境中安装成功并能 `import`，消费方跑一次其契约用例。
