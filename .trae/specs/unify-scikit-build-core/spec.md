# 统一玄境 scikit-build-core 构建风格 Spec

## 文档元数据

```yaml
id: "unify-scikit-build-core"
version: "0.1.0"
x-toml-ref: ".meta/toml/specs/unify-scikit-build-core.toml"
```

## Why

玄境（Xuanspace）monorepo 中 `pyproject.toml` 的构建后端**不统一**：

- 根目录 `pyproject.toml` 与 `libs/xuan-core/` 使用 `setuptools`
- `tools/`、`libs/` 下的 FFI 原生扩展子项目已使用 `scikit-build-core`
- `tools/templates/python/` 模板仍使用 `setuptools`
- 已使用 `scikit-build-core` 的子项目间在**键排序、缩进、minimum-version、sdist 清单、wheel.packages** 等细节上存在差异

这种混乱导致：维护者需同时掌握两套构建后端心智模型；新增子项目时风格无统一可循；根目录能力与子项目构建链割裂，无法通过 CIT 工具链做一致性校验。

**目标**：将根目录改造为 `scikit-build-core + ninja + cmake` 形式的**纯 Python 枢纽包**（参照 `tools/xs` 与 `tools/okf` 的 `LANGUAGES NONE` + `install(DIRECTORY)` 模式），并让**整个仓库所有 `pyproject.toml` 统一为同一套 canonical 风格**。

## 总体范围

用户已确认两项决策：

1. **统一范围 = 全部**：根目录、xuan-core、python 模板等所有 `setuptools` 后端统一改造成 `scikit-build-core+ninja+cmake`，并校准已用 `scikit-build-core` 的子项目格式。
2. **根目录构建目标 = 纯 Python 枢纽包**：根 CMakeLists.txt 改为 `LANGUAGES NONE`，维护 dev 依赖与 CLI 入口，无需 C++ 编译器，不聚合原生扩展编译。

## What Changes

建立玄境**官方 canonical `pyproject.toml` 风格**，并应用到全部相关文件：

- 🎯 **根目录 `pyproject.toml`**（核心改造）：`setuptools` → `scikit-build-core`；移除 `[tool.setuptools.packages.find]`；新增 `[tool.scikit-build]` 纯枢纽包配置；保留 `[tool.pdm.workspace]`（monorepo 工作区）与 `[dependency-groups]` / `[project.optional-dependencies]`；删除 `[build-system]` 中 `wheel` 依赖。
- 📄 **根 `CMakeLists.txt`**：`LANGUAGES C CXX` → `LANGUAGES NONE`；移除/注释 `add_subdirectory(libs/xuan-ext-demo)` 聚合；新增 `if(SKBUILD)` + `install(DIRECTORY ...)`（枢纽包无包体则留空并有说明注释）。
- 🧱 **`libs/xuan-core/pyproject.toml`**：`setuptools` → `scikit-build-core`；新增 `CMakeLists.txt`（`LANGUAGES NONE` + `install(DIRECTORY src/xuan_core DESTINATION xuan_core)`）。
- 📦 **`tools/templates/python/pyproject.toml`**：`setuptools` → `scikit-build-core`；模板新增 `CMakeLists.txt`（`LANGUAGES NONE` + `install(DIRECTORY src/{{package_name}} DESTINATION {{package_name}})`），与 native-ffi 模板对齐。
- ⚙️ **`tools/xs` / `tools/okf`**（已用 scikit-build-core）：校准 `wheel.packages` 语义（xs 当前 `["xs"]` 与 okf `["src/okf"]` 不一致）、补齐 `minimum-version`、统一缩进与键排序。
- 🔧 **`libs/caffe-ffi` / `demo-ffi` / `npu-ffi` / `libs/xuan-ext-demo` / `templates/native`（已用 scikit-build-core）**：统一格式（`minimum-version`、键排序、缩进、注释风格），不改变其构建语义。
- 📚 **`docs/build-system.md` 等文档**：更新「pyproject.toml 标准」章节，将 `scikit-build-core` 定为仓库唯一构建后端，删除 setuptools 示例或标注弃用。
- 🧪 **校验工具**：新增/复用校验入口，验证全仓库 `pyproject.toml` 均符合 canonical 风格（构建后端统一、非空要求、无残留 setuptools）。

**BREAKING**：根目录与 xuan-core 从 `setuptools` 切换到 `scikit-build-core`，`pip install -e .` / `python -m build` 行为变化（要求已安装 CMake+Ninja，由 `[build-system].requires` 自动拉取）。

## 影响面

- 受影响规范：`AGENTS.md` 中「包管理器：不强制 PDM」、"C++ 原生扩展必须使用 CMake+Ninja+scikit-build-core"；`.agents/global-core-rules.md` 中构建后端约定；`docs/build-system.md`。
- 受影响代码/文件：
  - `pyproject.toml`（根）、`CMakeLists.txt`（根）
  - `libs/xuan-core/pyproject.toml`、`libs/xuan-core/CMakeLists.txt`（新增）
  - `tools/templates/python/pyproject.toml`、`tools/templates/python/CMakeLists.txt`（新增）
  - `tools/xs/pyproject.toml`、`tools/okf/pyproject.toml`、`tools/xs/CMakeLists.txt`、`tools/okf/CMakeLists.txt`
  - `libs/caffe-ffi/pyproject.toml`、`libs/demo-ffi/pyproject.toml`、`libs/npu-ffi/pyproject.toml`、`libs/xuan-ext-demo/pyproject.toml`
  - `tools/templates/native/CMakeLists.txt`、`tools/templates/native/pyproject.toml`（校准）
  - `docs/build-system.md`
  - 校验工具（`tools/*` 或 `.agents/scripts/` 下的一致性校验脚本）

## Canonical 风格规范（统一标准）

所有 `pyproject.toml` 遵循以下约定（下称 **Canonical Style**）：

1. **构建后端**：`build-backend = "scikit_build_core.build"`；`requires = ["scikit-build-core>=0.10", "ninja>=1.11"]`（原生扩展追加 `"cmake>=3.26"`；FFI 子项目追加 `"apache-tvm-ffi"`；pybind11 模板追加 `"pybind11>=2.12"`）。
2. **`[project]`**：遵循 PEP 621，`requires-python = ">=3.14.6"`，`license`（`{file=...}` 或 `{text=...}`），按序排列 `name/version/description/readme/license/requires-python/dependencies/...`。
3. **`[tool.scikit-build]`** 必备键，统一顺序：`minimum-version = "0.10"`、`cmake.build-type = "Release"`、`wheel.packages`、FFI 项目 `wheel.install-dir`、`ninja.version`、`ninja.make-fallback = false`、`build-dir = "build"`、`build.verbose = true`、FFI 项目 `editable.rebuild = false` / `editable.verbose = true` / `logging.level = "INFO"`。
4. **纯 Python 项目（LANGUAGES NONE）**：`wheel.packages` 指向包根目录（如 `["src/okf"]` 或 `["src/xuan_core"]`），CMakeLists 用 `if(SKBUILD) install(DIRECTORY src/<pkg>/ DESTINATION <pkg>) endif()`。
5. **`sdist.include` / `sdist.exclude`**：FFI/原生项目显式列出 `/CMakeLists.txt`、`/CMakePresets.json`、`/pyproject.toml`、`/src/**`、`/include/**`、`/scripts/**`、`/python/<pkg>/**`、`/LICENSE`、`/tests/**`；sdist.exclude 统一 `**/.git`、`**/__pycache__`、`**/*.pyc`、`build`、`dist`。
6. **缩进**：统一 2 空格（Array 内联元素）或与原生项目现有风格一致（caffe-ffi 用 2 空格数组 + 2 空格 key）；**全仓库统一为 2 空格缩进、`key = value` 数组元素 2 空格偏移**。
7. **注释风格**：中文注释，`#` 后跟一空格，分节注释用 `# ── <节名> ──`。

## 转变与兼容

- **根目录是枢纽包**：不伪装成可安装库，`wheel.packages` 为空并通过 CMake 注释说明；`pip install -e ".[dev]"` 仍可安装全部 dev 工具链。
- **`[tool.pdm.workspace]` 保留**：monorepo 工作区语义依赖它，不受构建后端切换影响。
- **`xuan-core` / `xuan-ext-demo` 已从 README 索引移除**：本次仅统一其构建后端与格式，不重新登记索引、不改变其内容代码。

## ADDED Requirements

### Requirement: 统一构建后端为 scikit-build-core

全仓库每个 `pyproject.toml` 的 `build-backend` SHALL 为 `scikit_build_core.build`，`requires` SHALL 包含 `scikit-build-core>=0.10` 与 `ninja>=1.11`，并在 `[tool.scikit-build]` 中声明 CMake 配置。

#### Scenario: 新子项目按 canonical 风格生成
- **WHEN** 开发者使用 `xs new --type native|python|native-ffi` 从模板创建子项目
- **THEN** 生成的 `pyproject.toml` 遵循 Canonical Style，构建后端为 scikit-build-core，纯 Python 模板含 `LANGUAGES NONE` CMakeLists

#### Scenario: 已存在的子项目被校验
- **WHEN** 运行全仓库一致性校验脚本
- **THEN** 所有 `pyproject.toml` 均无 `setuptools`/`tool.setuptools`/`[tool.setuptools.packages.find]` 残留，且键排序/缩进符合规范

### Requirement: 根目录为纯 Python 枢纽包

根 `pyproject.toml` SHALL 切换到 scikit-build-core，根 `CMakeLists.txt` SHALL 声明 `LANGUAGES NONE`，安装 dev 依赖与 CLI 入口，无需 C++ 编译器。

#### Scenario: 在根目录安装开发环境
- **WHEN** 开发者执行 `pip install -e ".[dev]"`
- **THEN** 构建成功，`[tool.pdm.workspace]` 保留，全部 dev 依赖（含 xs/okf 工具链）可导入，无残留 setuptools 配置

### Requirement: python 项目模板同步更新

`tools/templates/python/` 模板 SHALL 使用 scikit-build-core，并新增 `LANGUAGES NONE` 的 `CMakeLists.txt`，与新生成子项目保持一致。

#### Scenario: 从 python 模板创建子项目
- **WHEN** 开发者用模板创建纯 Python 子项目
- **THEN** 生成项目构建后端为 scikit-build-core，含可工作于 pip install . 的 CMakeLists.txt

## MODIFIED Requirements

### Requirement: 现有 scikit-build-core 子项目格式校准

`tools/xs`、`tools/okf`、`libs/caffe-ffi`、`libs/demo-ffi`、`libs/npu-ffi`、`libs/xuan-ext-demo`、`tools/templates/native`（以及 FFI 模板）SHALL 按 Canonical Style 校准 `minimum-version`、键排序、缩进、`wheel.packages` 语义；`xs` 的 `wheel.packages = ["xs"]` 与 `okf` 的 `["src/okf"]` 需统一（纯 Python 项目统一为包根路径）。

### Requirement: 文档同步

`docs/build-system.md` SHALL 将 scikit-build-core 描述为仓库唯一构建后端，更新「纯 Python 项目配置」示例，移除或被弃用标注 setuptools 示例。

## REMOVED Requirements

### Requirement: setuptools 作为可用构建后端

**Reason**：统一构建链，降低双后端心智负担；使全仓库可被一致性校验覆盖。
**Migration**：所有 setuptools 后端项目切换为 scikit-build-core；纯 Python 项目按 `LANGUAGES NONE` + `install(DIRECTORY)` 模式改造。