# {{name}}

{{name}} 是 Xuanspace（玄境）Python monorepo 中的 C/C++ 原生扩展子项目，基于 pybind11 + scikit-build-core 构建。

## 功能介绍

在此处描述原生扩展的主要功能和性能优势。

## 构建前置条件

- CMake 3.26 或更高版本
- Ninja 构建工具
- 支持 C++17 的 C++ 编译器：
  - Windows: MSVC (Visual Studio 2022 17.7+) 或 Clang
  - Linux: GCC 12+ 或 Clang 16+
  - macOS: Clang 15+
- Python 3.14.6 或更高版本（含开发头文件）

### Windows 环境准备

推荐使用 Visual Studio Installer 安装：
- 「使用 C++ 的桌面开发」工作负载
- MSVC v143 生成工具
- C++ CMake 工具 for Windows
- Windows SDK

### Linux 环境准备

```bash
# Ubuntu/Debian
sudo apt install cmake ninja-build build-essential python3.14-dev
```

### macOS 环境准备

```bash
brew install cmake ninja python@3.14
```

## 安装

### 开发模式安装

```bash
# PDM
pdm install -d

# pip
pip install -e ".[dev]"
```

### 发行版构建

```bash
pip install .
```

## 快速使用

```python
from {{package_name}} import add, add_f

# 整数加法
print(add(1, 2))  # 输出: 3

# 浮点数加法
print(add_f(1.5, 2.5))  # 输出: 4.0
```

## 维护状态

- **状态**: 开发中 (Alpha)
- **维护者**: Xuanspace Team
- **兼容性**: Python 3.14.6+、C++17
