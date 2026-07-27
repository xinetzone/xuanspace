# {{name}}

基于 [tvm-ffi](https://github.com/tlc-pack/tvm-ffi) 的高性能 C++/Python FFI 扩展项目模板。

## 特性

- **四层目录结构**：include（公共头文件）、src（C++ 实现）、python（Python 绑定）、tests（测试）
- **Stub/Real 双运行时抽象**：Stub 模式支持无硬件 CPU 模拟运行，Real 模式对接真实硬件
- **CMake + Ninja + scikit-build-core**：跨平台高效构建系统
- **类型安全**：使用 enum class 进行类型枚举，编译期类型检查
- **前缀一致性检查**：内置脚本验证 C++ 注册前缀与 Python 初始化前缀匹配

## 快速开始

### 前置条件

- Python >= 3.13
- CMake >= 3.26
- Ninja >= 1.11
- C++17 兼容编译器（MSVC 2022、GCC 13+、Clang 17+）

### 1. 安装 tvm-ffi

首先安装 tvm-ffi 依赖：

```bash
# 从源码安装（推荐用于开发）
git clone <tvm-ffi-repo-url> vendor/tvm-ffi
pip install --no-build-isolation -e vendor/tvm-ffi
```

> ⚠️ **重要**：必须使用 `--no-build-isolation`，否则构建隔离环境看不到已安装的 editable 包。

### 2. 构建并安装项目

使用开发脚本一键构建：

**Windows (PowerShell)**:
```powershell
# 完整构建 + 安装 + 验证
.\scripts\dev.ps1

# 或仅构建 C++
.\scripts\dev.ps1 -Build

# 或运行测试
.\scripts\dev.ps1 -Test
```

**Linux/macOS (bash)**:
```bash
# 完整构建 + 安装 + 验证
bash scripts/dev.sh

# 或仅构建 C++
bash scripts/dev.sh -b

# 或运行测试
bash scripts/dev.sh -t
```

或手动构建：

```bash
# 配置 CMake（Stub 模式，无硬件也能运行）
cmake -B build -G Ninja -D{{package_name|upper}}_USE_STUB=ON

# 构建
cmake --build build --config Release

# 安装（editable 模式）
pip install --no-build-isolation -e .
```

### 3. 验证安装

```python
import {{package_name}}
from {{package_name}} import {{module_name}}

# 测试基础功能
handle = {{module_name}}.tls_command_handle()
print(f"Command handle: {handle}")
```

### 4. 运行测试

```bash
pytest tests/python/ -v
```

## Stub 模式 vs Real 模式

项目支持两种运行时模式：

| 模式 | 用途 | 硬件要求 | 配置选项 |
|------|------|---------|---------|
| **Stub 模式** | 开发、单元测试、CI | 无（纯 CPU 模拟） | `-D{{package_name|upper}}_USE_STUB=ON`（默认） |
| **Real 模式** | 生产环境、真实硬件执行 | 需要真实硬件 | `-D{{package_name|upper}}_USE_STUB=OFF` |

**Stub 先行开发模式**：所有单元测试（共 116+ 个）都可在 Stub 模式下 CPU 运行，FFI 层与硬件彻底解耦。开发流程：

1. 始终在 Stub 模式下开发和测试 FFI 层
2. 验证逻辑正确性后再切换到 Real 模式对接硬件
3. 真实硬件问题不会影响 FFI 层测试

切换模式：
```bash
# Stub 模式
cmake -B build -G Ninja -D{{package_name|upper}}_USE_STUB=ON

# Real 模式
cmake -B build -G Ninja -D{{package_name|upper}}_USE_STUB=OFF
```

## FFI 前缀一致性检查

C++ 端注册的函数前缀必须与 Python 端 `_FFI_INIT_FUNC` 的第一个参数完全一致，否则会导致运行时函数找不到。

使用内置检查脚本：

```bash
python scripts/check_ffi_prefix.py --verbose
```

输出示例：
```
[PASS] C++ registered functions: 8
[PASS] Python initialized prefixes: 1
[PASS] Prefix match: {{module_name}}
[PASS] All functions accessible: {{module_name}}.tls_command_handle, {{module_name}}.buffer_alloc, ...
```

**常见错误**：
- `MISSING IN PYTHON: xxx`：C++ 注册了前缀为 xxx 的函数，但 Python 端没有初始化
- `MISSING IN C++: xxx`：Python 端初始化了前缀 xxx，但 C++ 端没有注册对应函数
- `Function not accessible: xxx.yyy`：C++ 注册了函数，但 Python getattr 无法访问

## DLL 路径问题

### Windows

Windows 不会自动搜索 build 目录下的 DLL。`_ffi_api.py` 中已自动处理：

```python
# DLL 搜索路径自动配置
_EXTRA_LIB_PATHS = [
    _PROJECT_ROOT / "build" / "lib",
    _PROJECT_ROOT / "build" / "src" / "{{module_name}}" / "Release",
    _PROJECT_ROOT / "build" / "src" / "{{module_name}}",
]
```

如果遇到 DLL 加载错误，可以手动添加：

```python
import os
os.add_dll_directory(r"path\to\build\lib")
```

### Linux/macOS

开发脚本会自动设置 `LD_LIBRARY_PATH`（Linux）或 `DYLD_LIBRARY_PATH`（macOS）。

如果需要手动设置：

```bash
# Linux
export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH

# macOS
export DYLD_LIBRARY_PATH=$PWD/build/lib:$DYLD_LIBRARY_PATH
```

### OpenMP 多副本问题

Windows 环境下多个 DLL 可能各自携带 OpenMP 运行时，导致初始化冲突。开发脚本已设置：

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

## 构建隔离问题

**永远使用 `--no-build-isolation`**：

```bash
# ✅ 正确
pip install --no-build-isolation -e .

# ❌ 错误（build isolation 隔离环境看不到已安装的 editable 包）
pip install -e .
```

原因：scikit-build-core 的 build isolation 会创建一个干净的虚拟环境，该环境看不到你之前以 editable 模式安装的 tvm-ffi 和其他本地依赖。

## 项目结构

```
{{name}}/
├── CMakeLists.txt              # 根 CMake 配置
├── CMakePresets.json           # CMake 预设
├── pyproject.toml              # Python 包配置
├── include/
│   └── {{package_name}}/
│       └── types.h             # 公共类型定义
├── src/
│   ├── CMakeLists.txt
│   └── {{module_name}}/
│       ├── CMakeLists.txt
│       ├── stub_rt.cc          # Stub 运行时实现
│       ├── real_rt.cc          # Real 运行时实现
│       └── ffi_registry.cc     # FFI 函数注册
├── python/
│   └── {{package_name}}/
│       ├── __init__.py
│       ├── py.typed
│       └── {{module_name}}/
│           ├── __init__.py
│           └── _ffi_api.py     # FFI API 初始化
├── tests/
│   └── python/
│       ├── __init__.py
│       ├── conftest.py
│       └── test_basic.py       # 基础测试
└── scripts/
    ├── dev.ps1                 # Windows 开发脚本
    ├── dev.sh                  # Linux/macOS 开发脚本
    └── check_ffi_prefix.py     # FFI 前缀检查脚本
```

## 添加新函数

### 1. 在 C++ 端注册

编辑 `src/{{module_name}}/ffi_registry.cc`，在 `TVM_FFI_STATIC_INIT_BLOCK()` 中添加：

```cpp
refl::GlobalDef()
    .def("{{module_name}}.my_new_func", my_new_func_impl);
```

**重要**：函数名必须以 `{{module_name}}.` 为前缀。

### 2. 在 Python 端声明

编辑 `python/{{package_name}}/{{module_name}}/__init__.py`，添加函数类型注解：

```python
def my_new_func(param1: int, param2: str) -> int:
    """我的新函数文档字符串。"""
    return _LIB._api_call("{{module_name}}.my_new_func", param1, param2)
```

### 3. 运行前缀检查

```bash
python scripts/check_ffi_prefix.py
```

### 4. 添加测试

编辑 `tests/python/test_basic.py`，添加测试用例：

```python
def test_my_new_func():
    result = {{module_name}}.my_new_func(42, "hello")
    assert result == expected_value
```

## 开发脚本参考

### dev.ps1 (Windows)

| 参数 | 功能 |
|------|------|
| （无参数） | 构建 C++ + 安装 pip 包 + 快速验证 import |
| `-Build` | 仅构建 C++ |
| `-Install` | 仅安装 pip 包 |
| `-Test` | 运行 pytest |
| `-Clean` | 清理 build 目录 |
| `-Rebuild` | 清理 + 重新构建 + 安装 |

### dev.sh (Linux/macOS)

| 参数 | 功能 |
|------|------|
| （无参数） | 构建 C++ + 安装 pip 包 + 快速验证 import |
| `-b` | 仅构建 C++ |
| `-i` | 仅安装 pip 包 |
| `-t` | 运行 pytest |
| `-c` | 清理 build 目录 |
| `-r` | 清理 + 重新构建 + 安装 |

## 许可证

本项目模板基于 [Apache License 2.0](LICENSE) 发布。
