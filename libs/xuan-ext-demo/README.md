# xuan-ext-demo

Xuanspace（玄境）C/C++ 原生扩展示例库，使用 scikit-build-core + CMake + Ninja 构建。

## 构建要求

- Python >= 3.14.6
- CMake >= 3.26
- Ninja 构建系统
- C++17 兼容编译器（MSVC 2022 / GCC 11+ / Clang 14+）
- （可选）pybind11 用于 C++ 绑定

## 项目结构

```
xuan-ext-demo/
├── CMakeLists.txt          # CMake 构建配置
├── pyproject.toml          # scikit-build-core 配置
├── src/
│   └── xuan_ext_demo/      # Python 包
│       └── __init__.py
└── README.md
```

## 功能示例

提供一个简单的 `add(a, b)` 函数，展示原生扩展的基本结构。

## 版本

0.1.0
