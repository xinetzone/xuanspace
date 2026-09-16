# 构建系统

## 概述

Xuanspace **全仓库统一采用 `scikit-build-core` 构建后端**，按项目是否含原生代码分两种形态：

| 项目类型 | 构建形态 | 说明 |
|---|---|---|
| 纯 Python 子项目/根枢纽包 | scikit-build-core + `wheel.cmake=false`，**无 CMakeLists.txt** | 无需 CMake/Ninja/编译器，产出 `py3-none-any` 通用 wheel |
| C++ 原生扩展 / FFI | scikit-build-core + CMake + Ninja + CMakeLists.txt | C/C++ 扩展模块，产出平台相关 wheel |

> 历史形态 `LANGUAGES NONE` + 占位 CMakeLists.txt 已废止：它仍会触发 CMake 配置，
> 导致纯 Python 包被钉上平台相关 wheel 标签（`cp3xx-cp3xx-<os>_<arch>`），
> 且 sdist 构建被迫依赖 cmake/ninja。`scripts/check_pyproject_style.py` 会
> 强制校验两种形态的一致性（纯 Python 包不得携带 CMakeLists.txt、不得声明 ninja）。

## 构建工具链

### 必需工具

| 工具 | 最低版本 | 安装方式 |
|---|---|---|
| Python | 3.14.6 | [python.org](https://www.python.org/downloads/) |
| pip | 24+ | 随 Python 自带 |

### 原生扩展工具（按需）

| 工具 | 最低版本 | 安装方式 |
|---|---|---|
| CMake | 3.26 | `pip install cmake` 或系统包管理器 |
| Ninja | 1.11 | `pip install ninja` 或系统包管理器 |
| C++ 编译器 | - | 见下方平台说明 |

### 平台编译器

| 平台 | 编译器 | 安装方式 |
|---|---|---|
| Windows | MSVC (Visual Studio Build Tools) | `winget install Microsoft.VisualStudio.2022.BuildTools` |
| macOS | Clang (Xcode Command Line Tools) | `xcode-select --install` |
| Linux | GCC | `apt install build-essential` / `dnf install gcc-c++` |

## pyproject.toml 标准

所有子项目遵循 PEP 621 规范，使用 `pyproject.toml` 声明构建配置。

### 纯 Python 项目配置

纯 Python 项目使用 scikit-build-core 的纯 Python 模式：**不创建 CMakeLists.txt**，
在 `[tool.scikit-build.wheel]` 显式声明 `cmake = false`：

```toml
[build-system]
requires = ["scikit-build-core>=0.10"]
build-backend = "scikit_build_core.build"

[project]
name = "my-lib"
version = "0.1.0"
requires-python = ">=3.14.6"

[tool.scikit-build]
minimum-version = "0.10"

[tool.scikit-build.wheel]
cmake = false
packages = ["src/my_lib"]
```

特征：

- 构建全程不调用 CMake/Ninja，干净环境无需预装原生工具链；
- 产出 `py3-none-any`、`Root-Is-Purelib: true` 的通用 wheel；
- sdist 与从 sdist 重建 wheel 同样无需 cmake/ninja；
- 未来若挂载原生模块：恢复 CMakeLists.txt、移除 `wheel.cmake=false`，
  并在 `build-system.requires` 加回 `cmake`/`ninja`（即转为下一节的原生形态）。

### C++ 原生扩展项目配置

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "cmake>=3.26", "ninja"]
build-backend = "scikit_build_core.build"

[project]
name = "my-ext"
version = "0.1.0"
requires-python = ">=3.14.6"

[tool.scikit-build]
minimum-version = "0.10"
cmake.build-type = "Release"
wheel.packages = ["my_ext"]
ninja.make-fallback = false
```

## CMakePresets.json

用于跨平台构建预设，确保不同平台行为一致：

```json
{
  "version": 3,
  "configurePresets": [
    {
      "name": "default",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/${presetName}",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON"
      }
    }
  ]
}
```

## 构建命令

### 使用 xs CLI

```bash
# 构建所有项目
xs build

# 构建指定项目
xs build --project xuan-ext-demo

# 按类型构建
xs build --type native
```

### 手动构建

```bash
# 纯 Python 项目
cd libs/my-lib
python -m build

# C++ 原生扩展
cd libs/my-ext
pip install .
```

## 跨平台构建

### 路径处理

- 使用 `pathlib` 而非字符串拼接
- CMake 中使用 `${CMAKE_CURRENT_SOURCE_DIR}` 等变量
- 避免硬编码路径分隔符

### 编译器差异

| 差异点 | MSVC | GCC/Clang |
|---|---|---|
| 运行时库 | `/MD` (动态) | 默认动态 |
| C++ 标准 | `/std:c++17` | `-std=c++17` |
| 导出符号 | `__declspec(dllexport)` | `__attribute__((visibility("default")))` |
| 调试符号 | `.pdb` | DWARF |

### 验证跨平台一致性

```bash
# 检查工具链
xs toolchain check

# 构建验证
xs build --type native
```

## 包管理器对比

| 特性 | PDM | uv | pip |
|---|---|---|---|
| 安装速度 | 快 | 极快 | 中等 |
| Workspace 支持 | ✅ 原生 | ⚠️ 有限 | ❌ 无 |
| PEP 621 兼容 | ✅ | ✅ | ✅ |
| 锁文件 | `pdm.lock` | `uv.lock` | 无 |
| 额外安装 | 需要 | 需要 | 自带 |
| 推荐场景 | 日常开发 | 快速搭建 | CI/标准环境 |