# 构建系统

## 概述

Xuanspace 支持两种构建模式：

| 模式 | 构建后端 | 适用场景 |
|---|---|---|
| 纯 Python | setuptools | Python 库和应用 |
| C++ 原生扩展 | scikit-build-core + CMake + Ninja | C/C++ 扩展模块 |

## 构建工具链

### 必需工具

| 工具 | 最低版本 | 安装方式 |
|---|---|---|
| Python | 3.13 | [python.org](https://www.python.org/downloads/) |
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

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-lib"
version = "0.1.0"
requires-python = ">=3.13"
```

### C++ 原生扩展项目配置

```toml
[build-system]
requires = ["scikit-build-core>=0.10"]
build-backend = "scikit_build_core.build"

[project]
name = "my-ext"
version = "0.1.0"
requires-python = ">=3.13"

[tool.scikit-build]
cmake.args = ["-G", "Ninja"]
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