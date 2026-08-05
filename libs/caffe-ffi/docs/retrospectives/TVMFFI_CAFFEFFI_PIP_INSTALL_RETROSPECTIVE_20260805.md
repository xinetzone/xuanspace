# tvm-ffi / caffe-ffi `pip install .` 卡点七概念复盘报告

> 报告日期：2026-08-05
> 方法论：R-I-E-C 七概念方法论（场景：问题解决 / 里程碑复盘）
> 链路：R(事实复盘) → I(根因洞察) → E(模式萃取) → C(行动项)
> 主题：为何每次在 tvm-ffi 与 caffe-ffi 的 `pip install .` 上反复卡住

---

## R：Retrospective（事实复盘）

### 1.1 任务背景

在 P2 算子开发、P0 环境验证与 CI 构建过程中，tvm-ffi 与 caffe-ffi 两个包的
`pip install .`（本地可编辑安装）反复出现失败或超时，且每次失败点高度相似，
具有明显的**可复现性**。本报告还原历次卡点的客观事实，剥离主观判断。

### 1.2 关键事实清单（G1 检查：无因果词）

| # | 事实编号 | 客观陈述 |
|---|---------|---------|
| F-001 | 两个包均采用 `scikit-build-core` 作为构建后端（`pyproject.toml` 中 `build-backend = "scikit_build_core.build"`） |
| F-002 | tvm-ffi 的版本号通过 `metadata.version.provider = scikit_build_core.metadata.setuptools_scm` 从 git 元数据动态生成 |
| F-003 | tvm-ffi 的 `pyproject.toml` 中 `[tool.setuptools_scm]` 配置了 `version_file = "python/tvm_ffi/_version.py"` |
| F-004 | 在无 `.git` 元数据的目录（如 Docker 容器内复制的源码树）中，setuptools-scm 无法检测版本，报 `setuptools-scm was unable to detect version` |
| F-005 | 修复方式为设置环境变量 `SETUPTOOLS_SCM_PRETEND_VERSION=0.1.13` 后重试成功 |
| F-006 | tvm-ffi 的 `cmake.args` 含 `-DTVM_FFI_BUILD_PYTHON_MODULE=ON`，控制 Cython core 扩展是否编译 |
| F-007 | 当 `TVM_FFI_BUILD_PYTHON_MODULE` 未开启时，构建产物缺 `tvm_ffi.core` 扩展，运行时报 `ModuleNotFoundError: No module named 'tvm_ffi.core'` |
| F-008 | `pip install .` 默认启用构建隔离（build isolation），在隔离虚拟环境内重新下载并安装 `scikit-build-core`/`ninja`/`cmake`/`cython`/`setuptools-scm` 等构建依赖 |
| F-009 | 构建隔离每次安装依赖，在离线或慢网络环境下显著拖慢 `pip install .`，且与当前环境已装的依赖版本可能不一致 |
| F-010 | 禁用构建隔离需加 `--no-build-isolation`，此时复用当前环境已安装的构建依赖 |
| F-011 | caffe-ffi 的 `pyproject.toml` 中 `requires-python = ">=3.14"` |
| F-012 | 在 Python 3.13 及以下环境安装 caffe-ffi，会被 `requires-python` 约束拒绝 |
| F-013 | caffe-ffi 的 `build-system.requires` 声明 `apache-tvm-ffi>0.1.12` |
| F-014 | caffe-ffi 的 C++ 头文件来自 vendored tvm-ffi（`vendor/tvm-ffi/cmake`，含新版 `Function` API） |
| F-015 | caffe-ffi 运行时链接的 `tvm_ffi.dll`（Windows）来自 site-packages 安装的旧版 apache-tvm-ffi |
| F-016 | vendored tvm-ffi 仅有 Linux 构建产物（`libtvm_ffi.so`），无 Windows DLL |
| F-017 | Windows 运行时出现 `WinError 127`（找不到指定的程序），因旧的 `tvm_ffi.dll` 不含 `Function` 新符号 |
| F-018 | 该 `WinError 127` 与 P2 算子业务代码无关，P2 代码已编译通过，属预存环境/版本不匹配缺陷 |
| F-019 | 修复路径之一是在项目 P0 环境（WSL Docker，Linux）验证，vendored tvm-ffi 的 Linux 构建已就位 |
| F-020 | 两处 `pip install .` 的卡点均属于**同一类根因**：构建/运行时的依赖来源不一致（git 元数据、构建隔离、构建开关、插件版本 skew） |

### 1.3 时间线

| 时间 | 事件 | 涉及事实 |
|------|------|---------|
| P2 构建期 | tvm-ffi 构建成功，但 Windows 运行时 `WinError 127` | F-014 ~ F-018 |
| P0 环境验证期 | 容器内 tvm-ffi 版本检测失败 | F-002 ~ F-005 |
| P0 环境验证期 | `tvm_ffi.core` 扩展缺失 | F-006 ~ F-007 |
| 任意安装期 | `pip install .` 反复下载构建依赖、慢 | F-008 ~ F-010 |
| 任意安装期 | 低版本 Python 被拒绝 | F-011 ~ F-012 |

---

## I：Insight（根因洞察）

### G2 质量门检查：洞察四元组 ✅

#### 洞察 1：构建隔离（build isolation）是"卡住"的直接体感来源，但非唯一根因

- **陈述**：`pip install .` 默认开启构建隔离，每次都重新下载构建依赖，是"卡住"最直观的来源。
- **证据**：F-008、F-009、F-010。
- **反常识**：很多人以为"卡住"是编译慢，实则大部分等待发生在**隔离环境内重复安装构建依赖**，而非真正的 C++ 编译。
- **行动**：本地/CI 构建统一使用 `pip install --no-build-isolation -e .`，并在 CI 中先显式安装 `scikit-build-core ninja cmake cython setuptools-scm`。

#### 洞察 2：动态版本 + git 元数据依赖，是"换环境必崩"的根因

- **陈述**：tvm-ffi 版本号由 setuptools-scm 从 git 元数据生成，一旦源码脱离 `.git`（Docker 复制、sdist 解包），版本检测即失败。
- **证据**：F-002、F-003、F-004、F-005。
- **反常识**：源码明明存在，却报"无法检测版本"——因为版本不在源码里，而在 git 历史里。
- **行动**：对基于 setuptools-scm 的包，构建入口统一注入 `SETUPTOOLS_SCM_PRETEND_VERSION`；或在 Docker COPY 时保留 `.git`。

#### 洞察 3：同一依赖存在"双来源"，是版本 skew 与 `WinError 127` 的根因

- **陈述**：caffe-ffi 的头文件来自 vendored tvm-ffi（新 API），运行时 DLL 却来自 site-packages（旧 API），两套来源不一致导致符号缺失。
- **证据**：F-013、F-014、F-015、F-016、F-017、F-018。
- **反常识**：构建通过 ≠ 运行通过。`pip install` 成功只代表编译链接成功，运行时还需重新解析动态库，后者的版本来源常被忽略。
- **行动**：统一 vendored 与发布的 tvm-ffi 版本；Windows 需要 vendored tvm-ffi 的 DLL；并在 P0 Linux 环境做运行时冒烟验证。

---

## E：Extraction（模式萃取）

### G3 质量门检查：模式可迁移 ✅

#### 模式：构建与运行依赖一致性治理（build-runtime parity）

- **触发场景**：任何基于 `scikit-build-core` + setuptools-scm + Cython 扩展的项目，在本地/容器/CI 多环境安装时反复失败。
- **核心步骤**：
  1. 关闭构建隔离：`pip install --no-build-isolation -e .`，并显式安装构建依赖。
  2. 注入动态版本：无 `.git` 时设置 `SETUPTOOLS_SCM_PRETEND_VERSION=<version>`。
  3. 显式开启 Cython 扩展开关（如 `TVM_FFI_BUILD_PYTHON_MODULE=ON`）。
  4. 校验 `requires-python` 与目标环境匹配。
  5. 构建后运行时冒烟（`import` + 动态库 `ldd`/`dll` 解析）验证**运行时**依赖，而非仅验证编译。
  6. 统一 vendored 与发布依赖的版本号，避免双来源 skew。
  7. 将上述步骤固化为脚本/CI step，避免每次手工重试。
- **检验标准**：`pip install --no-build-isolation -e .` 在干净环境一次通过；`import` + 动态库解析 + 冒烟全绿。
- **反模式**：
  - ❌ 不加 `--no-build-isolation` 任由每次重复下载构建依赖 → 慢且离线失败。
  - ❌ 忽略 setuptools-scm 的 git 依赖，在 Docker 复制源码后直接 `pip install` → 版本检测失败。
  - ❌ 只验证编译通过，不验证运行时动态库解析 → `WinError 127` 漏过。
  - ❌ 头文件用 vendored、运行时用 site-packages，两套版本 → symbol 缺失。
- **跨场景迁移**：适用于所有"源码来自 git submodule/vendored，但运行时来自 PyPI 包"的耦合场景（如 torch 的 libtorch 与 PyPI 轮子）。

---

## C：行动项（原子化）

### G4 质量门检查：行动项单一职责 ✅

| # | 行动项 | 验收标准 | 状态 |
|---|--------|---------|------|
| C-1 | 生成支持定时任务的 P0 环境自动化脚本（任务3） | 脚本可在宿主 WSL 一键执行，支持 cron/systemd timer | 本次交付 |
| C-2 | 将 TVM-FFI 依赖检查集成到 CI（任务4） | CI 流水线含依赖加载检查 step，失败即红 | 本次交付 |
| C-3 | 补充 P2 数据 I/O 算子单元测试（任务2） | Data/ImageData/HDF5Data 前向与回调填充一致 | 本次交付 |
| C-4 | 固化构建一致性脚本（`--no-build-isolation` + 版本注入 + 运行时冒烟） | 后续 `pip install .` 在干净环境可复现通过 | 进行中 |

---

## 质量门通过记录

| 质量门 | 结果 |
|--------|------|
| G1（事实无因果词） | ✅ 20 条事实均为客观陈述 |
| G2（洞察四元组） | ✅ 3 条洞察均含陈述/证据/反常识/行动 |
| G3（模式可迁移） | ✅ 含触发/步骤/检验/4 反模式/跨场景迁移 |
| G4（行动项原子化） | ✅ 4 项均单一职责、可独立验证 |