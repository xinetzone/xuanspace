# caffe-ffi 智能体路由表

> 本文件是 caffe-ffi 作为独立项目的 AI 智能体入口。完整规范体系请参考上游 SpecWeave 工作区。

## 项目概述

**caffe-ffi** — Caffe 深度学习框架的 tvm-ffi 原生 FFI 绑定库，提供 C++/Python 跨语言对象系统和零拷贝张量访问。

## 技术栈

- **语言**：C++17 / Python 3.14+
- **构建系统**：CMake 3.26+ + scikit-build-core + Ninja
- **核心依赖**：tvm-ffi (apache-tvm-ffi >= 0.3.0)、Protobuf >= 7、BLAS (OpenBLAS)
- **测试框架**：pytest (Python)、header-only 自研框架 (C++)

## 目录结构

```
caffe-ffi/
├── AGENTS.md              # 本文件（智能体入口）
├── .agents/               # 项目级智能体规范
├── .temp/                 # 临时文件目录（不提交内容，仅保留.gitkeep）
├── cmake/                 # CMake模块化配置
├── conda.recipe/          # Conda打包配置
├── docs/                  # 文档和报告
├── examples/              # 使用示例
├── include/caffe_ffi/     # C++头文件
├── proto/                 # Protobuf定义
├── python/caffe_ffi/      # Python绑定
├── scripts/               # 开发脚本（dev.sh/dev.ps1/conda_build等）
├── src/caffe_ffi/         # C++实现
├── tests/                 # 测试（cpp/ + python/）
├── CMakeLists.txt         # 顶层CMake
├── CMakePresets.json      # CMake预设
├── environment.yml        # Conda环境
├── pyproject.toml         # Python项目配置
└── README.md              # 项目说明
```

## 开发约定

### 构建与测试

- **WSL/Linux**：`bash scripts/conda_build.sh` 或 `bash scripts/dev.sh`
- **Windows**：`scripts\conda_build.bat` 或 `powershell scripts\dev.ps1`
- **CMake预设**：`cmake --preset default|debug|developer`
- **Python测试**：`python -m pytest tests/python/ -v`
- **C++测试**：`cmake --build build && cd build && ctest --output-on-failure`

### 代码规范

- C++：遵循现有代码风格，公共函数必须参数校验
- Python：遵循现有代码风格，使用 `_ffi_api` 桥接C++层
- CMake：模块化设计，新增模块放入cmake/，公共函数必须参数校验
- 提交规范：Conventional Commits（feat/fix/docs/refactor/test/chore），中文描述

### 临时文件

所有临时脚本、调试文件、测试输出必须放在 `.temp/` 目录下，不要散落在项目根目录或其他位置。

- `.temp/*.py` 用于临时测试脚本
- `.temp/*.log` 用于构建日志
- `.temp/` 下除 `.gitkeep` 外所有文件被 `.gitignore` 忽略

### CMake模块命名

禁止使用 `Find<Name>.cmake` 命名（避免与CMake内置模块冲突导致无限递归），使用 `Detect<Name>.cmake` 或项目前缀命名。

### 关键约束（来自项目记忆）

- 路径独立：`cmake/Dependencies.cmake` 默认使用 `find_package(tvm_ffi CONFIG REQUIRED)`，通过 `CAFFE_FFI_TVM_FFI_DIR` 选项指定本地路径
- Windows DLL：使用 `NPU_FFI_API` 宏导出符号，启用 `WINDOWS_EXPORT_ALL_SYMBOLS`
- 运行时依赖：pytest 是运行时依赖（tvm.testing 间接需要）
- 日志框架：5级日志（TRACE/DEBUG/INFO/WARN/ERROR），默认WARN级别

## 上游规范引用

- 完整方法论体系：SpecWeave `.agents/commands/`
- 原子提交规范：`atomic-commit-cmd`
- CI检查：`ci-check-cmd`
- 复盘/洞察/萃取：`retrospective-cmd` / `insight-cmd` / `extraction-cmd`

