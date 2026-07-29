# caffe-ffi

Caffe FFI bindings using tvm-ffi native object system

<!-- badges: start -->
<!-- TODO: Add CI, PyPI, Conda badges here -->
<!-- badges: end -->

## 项目简介

caffe-ffi 是基于 tvm-ffi 原生对象系统的 Caffe 深度学习框架绑定，提供 Python 与 C++ 之间的零拷贝互操作能力，采用现代 C++17 特性实现类型安全的 FFI 接口。

## 特性

- 基于 tvm-ffi 的类型安全 FFI 绑定
- 零拷贝数据交互（Blob 数据共享）
- CPU-only 模式支持（无需 GPU）
- 现代化 C++17 实现
- Protobuf 模型定义序列化支持
- 丰富的层实现（Convolution, Pooling, InnerProduct, ReLU, Softmax 等）
- 跨平台支持（Windows/Linux/macOS/WSL）
- Conda 包管理支持
- Ninja 快速构建

## 系统要求

- **Python**: 3.14+
- **CMake**: >= 3.26
- **Ninja**: >= 1.11
- **编译器**: C++17 兼容（GCC 9+, Clang 12+, MSVC 2022）
- **BLAS**: OpenBLAS 或其他 BLAS 实现
- **可选**: Conda（推荐用于环境管理）

## 安装方式

### 开发模式安装（推荐）

```bash
# 步骤1：先安装 tvm-ffi (editable模式)
# 假设项目路径为 projects/xuanspace/libs
cd libs/tvm-ffi
pip install --no-build-isolation -e .

# 步骤2：安装 caffe-ffi (editable模式)
cd ../caffe-ffi
pip install --no-build-isolation -e .
```

> **注意**：必须使用 `--no-build-isolation` 参数，以确保构建时使用已安装的 editable 模式 tvm-ffi，而不是在隔离环境中重新下载依赖。两个包都需要此参数。

### Conda 环境安装

```bash
# 创建并激活 conda 环境
conda env create -f environment.yml
conda activate caffe-ffi-dev

# 本地开发模式安装 tvm-ffi（需要先克隆 tvm-ffi 到 libs/tvm-ffi）
# pip install --no-build-isolation -e ../tvm-ffi

# 安装 caffe-ffi
pip install --no-build-isolation -e .
```

## WSL 环境开发指南

WSL (Windows Subsystem for Linux) 是 Windows 上开发 caffe-ffi 的推荐环境，提供更好的编译性能和兼容性。

### WSL 环境设置

#### 1. 安装 WSL2

```powershell
# 在 Windows PowerShell（管理员）中执行
wsl --install -d Ubuntu-22.04
# 重启后设置用户名和密码
```

#### 2. 更新系统并安装基础依赖

```bash
# 在 WSL Ubuntu 中执行
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential cmake ninja-build python3-dev python3-pip git
sudo apt install -y libopenblas-dev libprotobuf-dev protobuf-compiler
```

#### 3. 安装 Miniconda（推荐）

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# 按照提示完成安装，重启终端生效
```

#### 4. 配置项目环境

```bash
# 克隆项目（如果尚未克隆）
cd ~
mkdir -p projects/xuanspace/libs
cd projects/xuanspace/libs
# git clone <tvm-ffi-repo> tvm-ffi
# git clone <caffe-ffi-repo> caffe-ffi

# 创建 conda 环境
cd caffe-ffi
conda env create -f environment.yml
conda activate caffe-ffi-dev

# 设置 KMP_DUPLICATE_LIB_OK（可选，Linux 一般不需要）
export KMP_DUPLICATE_LIB_OK=TRUE
```

### 使用 Docker（可选）

如果希望使用容器化开发环境：

```bash
# 构建 Docker 镜像
docker build -t caffe-ffi-dev -f Dockerfile.dev .

# 运行容器（挂载项目目录）
docker run -it --rm -v $(pwd):/workspace caffe-ffi-dev
```

### WSL 编译步骤

```bash
# 激活 conda 环境
conda activate caffe-ffi-dev

# 方式1：使用一键开发脚本（推荐）
chmod +x scripts/dev.sh
./scripts/dev.sh             # 构建 + 安装 + 验证
./scripts/dev.sh -b          # 仅构建 C++
./scripts/dev.sh -t          # 运行测试
./scripts/dev.sh -r          # 清理重建

# 方式2：手动构建
cmake --preset default       # 使用 CMakePresets 配置
cmake --build --preset default  # 构建
pip install --no-build-isolation -e .  # 安装
```

### WSL 文件系统性能提示

- 将项目放在 WSL 文件系统内（`~/projects/`）而非 Windows 挂载点（`/mnt/c/`）可获得显著的 I/O 性能提升
- 使用 VS Code 的 "Remote - WSL" 扩展获得原生开发体验

## 快速开始

### 基础 Blob 操作

```python
import numpy as np
from caffe_ffi.blob import Blob

# 创建 Blob
blob = Blob()
blob.reshape(1, 3, 224, 224)  # NCHW 格式

# 获取数据指针（零拷贝）
data = blob.data
print(f"Blob shape: {data.shape}")  # (1, 3, 224, 224)
print(f"Blob count: {blob.count()}")

# 写入数据
data[:] = np.random.randn(*data.shape).astype(np.float32)

# Reshape（自动管理内存）
blob.reshape(2, 3, 224, 224)
```

### 创建并运行简单网络

```python
from caffe_ffi.net import Net
from caffe_ffi.io import read_net_from_text_protobuf

# 从 prototxt 加载网络（示例）
# net_param = read_net_from_text_protobuf("model.prototxt")
# net = Net(net_param)
# net.forward()

print("caffe-ffi ready for network construction!")
```

### 内存诊断工具

```python
from caffe_ffi.tools.memory import memory_report, enable_logging, disable_logging

# 启用内存日志
enable_logging()

# 执行操作...

# 生成内存报告
report = memory_report()
print(report)

# 禁用日志
disable_logging()
```

## 项目结构

```
caffe-ffi/
├── CMakeLists.txt           # 根 CMake 配置
├── CMakePresets.json        # CMake 预设配置
├── pyproject.toml           # Python 构建配置
├── environment.yml          # Conda 环境配置
├── LICENSE                  # BSD-2-Clause 许可证
├── CHANGELOG.md             # 变更日志
├── include/                 # C++ 头文件
│   └── caffe_ffi/
│       ├── blob.hpp         # Blob 类定义
│       ├── net.hpp          # Net 类定义
│       ├── layer.hpp        # Layer 基类
│       ├── common.hpp       # 公共定义
│       ├── math_utils.hpp   # 数学工具
│       └── layers/          # 各层实现头文件
├── src/                     # C++ 源代码
│   └── caffe_ffi/
│       ├── blob.cpp
│       ├── net.cpp
│       ├── layer.cpp
│       └── layers/          # 各层实现
├── python/caffe_ffi/        # Python 包
│   ├── __init__.py
│   ├── blob.py              # Blob Python 封装
│   ├── net.py               # Net Python 封装
│   ├── layer.py             # Layer Python 封装
│   ├── _ffi_api.py          # FFI API 绑定
│   ├── io.py                # IO 工具
│   ├── caffe/               # Caffe 子模块
│   └── tools/               # 调试工具
├── proto/caffe/proto/       # Protobuf 定义
│   └── caffe.proto
├── cmake/                   # CMake 模块
├── tests/python/            # Python 单元测试
├── conda.recipe/            # Conda 构建配方
├── scripts/                 # 辅助脚本
│   ├── dev.sh               # Linux/WSL/macOS 开发脚本
│   ├── dev.ps1              # Windows 开发脚本
│   ├── check_ffi_prefix.py  # FFI 前缀一致性检查
│   ├── verify_install.py    # 安装验证脚本
│   └── gen_proto.py         # Protobuf 代码生成
├── docs/                    # 文档
└── examples/                # 示例代码
```

## 开发命令

### Linux/WSL/macOS (dev.sh)

```bash
# 完整构建流程（构建 + 安装 + 验证）
./scripts/dev.sh

# 仅构建 C++ 代码
./scripts/dev.sh -b
./scripts/dev.sh --build

# 仅安装 pip 包
./scripts/dev.sh -i
./scripts/dev.sh --install

# 运行测试
./scripts/dev.sh -t
./scripts/dev.sh --test

# 清理构建目录
./scripts/dev.sh -c
./scripts/dev.sh --clean

# 清理并重建
./scripts/dev.sh -r
./scripts/dev.sh --rebuild

# 查看帮助
./scripts/dev.sh -h
```

### Windows (dev.ps1)

```powershell
# 完整构建流程
.\scripts\dev.ps1

# 仅构建 C++
.\scripts\dev.ps1 -Build

# 仅安装
.\scripts\dev.ps1 -Install

# 运行测试
.\scripts\dev.ps1 -Test

# 清理构建
.\scripts\dev.ps1 -Clean

# 清理重建
.\scripts\dev.ps1 -Rebuild
```

## 运行测试

```bash
# 运行所有 Python 测试
pytest tests/python -v

# 使用开发脚本运行测试
./scripts/dev.sh -t          # Linux/WSL
.\scripts\dev.ps1 -Test      # Windows

# FFI 前缀一致性检查（修改 C++/Python FFI 代码后运行）
python scripts/check_ffi_prefix.py --verbose

# 安装验证
python scripts/verify_install.py
```

## CMake 预设

项目使用 CMakePresets.json 提供标准化构建配置：

```bash
# 配置并构建 Release 版本（默认）
cmake --preset default
cmake --build --preset default

# Debug 版本
cmake --preset debug
cmake --build --preset debug

# 开发者版本（Debug + 日志启用）
cmake --preset developer
cmake --build --preset developer
```

## Conda 包构建

```bash
# 安装 conda-build
conda install conda-build -c conda-forge

# 构建包
conda build conda.recipe -c conda-forge

# 安装本地构建的包
conda install --use-local caffe-ffi
```

## 许可证

BSD-2-Clause 许可证。详见 [LICENSE](LICENSE) 文件。

## 相关项目

- [tvm-ffi](https://github.com/tlc-pack/tvm-ffi) - Type-safe foreign function interface for TVM
- [Caffe](http://caffe.berkeleyvision.org/) - Original Caffe deep learning framework
