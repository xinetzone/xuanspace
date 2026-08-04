---
title: CMake 原子化重构里程碑复盘
date: 2026-08-04
category: caffe-ffi
task_type: retrospective
tags: [caffe-ffi, cmake, refactor, milestone, m6, wsl]
status: completed
source: "tasks.md#Task18"
based_on: "CMake_REFACTOR_WSL_REGRESSION_LOG_20260804.md"
---

# CMake 原子化重构里程碑复盘（Task 18 / M6）

> 本复盘覆盖 [tasks.md#Task18](caffe-ffi-tvm-integration/tasks.md)「独立项目萃取迁移」中 CMake 原子化重构的 WSL 验证闭环。链路：R→I→E→C。

## 一、事实清单（R 阶段，G1 通过）

| 编号 | 事实 |
|------|------|
| F-001 | 2026-07-30 Task 18 标记完成，状态"✅ 已完成（2026-07-30）" |
| F-002 | 2026-08-04 在 WSL docker 镜像（caffe-ffi-jupyter，conda env caffe-ffi）验证 CMake 原子化重构 |
| F-003 | 验证环境：Python 3.14.6、cmake 4.4.1、ninja 1.13.2、gcc 14.3.0 |
| F-004 | CMake 原子化重构产出 10 个模块化 .cmake 文件（Tests/WindowsDllCopy/TargetBuild/ProtoCompile/Options/Install/DetectOpenBLAS/DetectBLAS/Dependencies/CompilerConfig） |
| F-005 | scikit-build-core 构建 `_caffe_ffi.so` 成功 |
| F-006 | `import caffe_ffi` 正常，version 0.1.0 |
| F-007 | 2026-08-04 10:20 运行 `pytest tests/python -q`（COW + Phase3 宏：`CAFFE_FFI_ENABLE_COW=1 CAFFE_FFI_ENABLE_COW_PHASE3=1`） |
| F-008 | 回归结果 1646 passed, 1 skipped, 0 failures，耗时 10.52s |
| F-009 | 回归覆盖 43 个测试文件 |
| F-010 | `strings` 检查确认 `lazy_reshape=` 符号存在于 `_caffe_ffi.so`，验证 COW_PHASE3 宏已编译 |
| F-011 | editable install 路径下 `python/caffe_ffi/_caffe_ffi.so` 为 stale（缺 COW_PHASE3 符号） |
| F-012 | 将 `build/python/caffe_ffi/_caffe_ffi.so` 复制到源码树 `python/caffe_ffi/` 后，3 个 lazy allocation 测试通过 |
| F-013 | lazy allocation 触发条件为 N≥16 时 Split 层使用 `SetShapeOnly` |
| F-014 | `test_n16_boundary`/`test_split_n64_lazy_reshape`/`test_large_n_triggers_lazy_allocation` 通过 |
| F-015 | `import caffe_ffi._caffe_ffi`（显式子模块导入）触发 protobuf descriptor 重复注册崩溃（`File already exists in database: caffe/proto/caffe.proto`） |
| F-016 | 2026-08-04 生成详细回归日志 `docs/setup/CMake_REFACTOR_WSL_REGRESSION_LOG_20260804.md`（1646 用例逐条） |
| F-017 | 2026-08-04 生成里程碑总结报告 `docs/summaries/MILESTONE_SUMMARY_CMake_REFACTOR_WSL_REGRESSION_20260804.md` |
| F-018 | 2026-08-04 更新 spec.md M6 里程碑行，记录 CMake 重构 WSL 验证 |
| F-019 | 2026-08-04 更新 p4-roadmap.md 顶部，添加前置验证依据链接 |
| F-020 | 2026-08-04 更新 tasks.md Task 18 Post-optimization notes |
| F-021 | 主仓库变更：p4-roadmap.md、spec.md、tasks.md 三文件修改 |
| F-022 | submodule 变更：caffe-ffi 新增 2 个文档文件（回归日志 + 里程碑总结） |

## 二、核心洞察（I 阶段，G2 通过）

### 洞察 1：C++ 扩展的 editable install 不自动更新编译产物
- **陈述**：editable install（`pip install -e`）对 C++ 扩展只更新 Python 层，不自动重建/更新已编译的 `_caffe_ffi.so`，测试会静默加载 stale 库。
- **证据**：F-011、F-012
- **反常识**：多数开发者假设 editable install 会"即时反映源码变更"，但编译产物（.so）是例外。
- **行动**：editable install 场景下，构建 C++ 扩展后必须对比/复制新编译 .so 到源码树路径，或使用 `--force-reinstall`。

### 洞察 2：编译宏默认值与测试期望脱节会静默跑错路径
- **陈述**：`CAFFE_FFI_ENABLE_COW_PHASE3` 宏默认关闭，若测试期望启用该特性的代码而编译产物未含该宏，测试会静默通过"错误"路径。
- **证据**：F-010、F-011、F-013
- **反常识**：环境变量/宏开关看似"可配置"，但默认值若与测试期望不一致，回归无法发现该脱节。
- **行动**：回归前用 `strings` 验证编译产物确实包含测试所依赖的宏/符号，而非仅信任配置。

### 洞察 3：显式子模块导入触发 protobuf descriptor 重复注册崩溃
- **陈述**：`import caffe_ffi._caffe_ffi`（显式子模块导入）对已注册 protobuf descriptor 的 C++ 扩展会二次注册，触发 `File already exists in database` 崩溃。
- **证据**：F-015
- **反常识**：显式导入子模块用于诊断看似合理，但对扩展的 descriptor 注册机制是系统性陷阱。
- **行动**：诊断脚本避免 `import x._x` 子模块，改用 `import x` 后经 `x.__file__` 或模块属性定位。

## 三、可复用模式（E 阶段，G3 通过）

### 模式 1：editable-install-stale-so
| 项 | 内容 |
|----|------|
| **名称** | editable-install-stale-so（editable 安装 stale .so 处理） |
| **触发场景** | 适用于：C++/Cython 扩展 + editable install 开发；不适用于：纯 Python 包、正式 wheel 安装 |
| **核心步骤** | ① 触发构建（`pip install -e .` / scikit-build-core）；② 对比 `build/` 与源码树 editable 路径的 .so 时间戳与符号（`strings`）；③ 若不一致，复制新 .so 到源码树路径；④ 重跑测试验证 |
| **反模式** | ❌ 假设 editable install 自动更新编译产物；❌ 只重新构建不复制 .so；❌ 用 `import x._x` 子模块诊断替代符号检查 |
| **检验标准** | 测试加载的 .so 与 `build/` 一致（符号/时间戳匹配），相关测试通过 |
| **跨场景迁移** | 同类 MLOps 场景：共享库/模型文件在 editable 或 Jupyter 热加载环境下被缓存，需显式刷新而非信任自动重载 |

### 模式 2：cxx-build-regression-verification
| 项 | 内容 |
|----|------|
| **名称** | cxx-build-regression-verification（C++ 扩展构建回归验证） |
| **触发场景** | 适用于：跨平台 C++ 项目（CMake+scikit-build）构建/重构验证；不适用于：无编译产物的纯脚本项目 |
| **核心步骤** | ① 环境确认（容器/conda/Python/编译器版本）；② 宏与符号验证（`strings` 检查测试依赖宏）；③ 全量回归（`pytest -q`）；④ 日志归档（`-v` 逐用例输出转 markdown） |
| **反模式** | ❌ 在错误环境跑测试；❌ 跳过宏/符号验证；❌ 回归结果不归档、无追溯 |
| **检验标准** | 全量回归通过 + 详细日志已归档 + 里程碑文档已链接 |
| **跨场景迁移** | 同类 CI/CD 场景：任何"编译产物可配置特性"的回归，需先验证产物含预期特性再执行测试 |

## 四、里程碑状态与行动项

| 里程碑 | 状态 | 说明 |
|--------|------|------|
| Task 18（M6-CMake 重构） | ✅ 闭环 | 验收通过，1646 用例 WSL 回归全通过 |
| 文档产出 | ✅ 完成 | 回归日志 + 里程碑总结 + spec.md/tasks.md/p4-roadmap.md 更新 |
| 待提交 | ⏳ 待执行 | 本复盘需原子提交（C 阶段） |

## 五、关联文档
- 详细回归日志：[CMake_REFACTOR_WSL_REGRESSION_LOG_20260804.md](../setup/CMake_REFACTOR_WSL_REGRESSION_LOG_20260804.md)
- 里程碑总结：[MILESTONE_SUMMARY_CMake_REFACTOR_WSL_REGRESSION_20260804.md](../summaries/MILESTONE_SUMMARY_CMake_REFACTOR_WSL_REGRESSION_20260804.md)
- 任务记录：[tasks.md#Task18](../../../../../../.trae/specs/caffe-ffi-tvm-integration/tasks.md)