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

## 版本规划

| 规划版本 | 阶段定位 | 核心目标 | 关键里程碑 |
|---------|---------|---------|-----------|
| v0.2.0 (Beta) | 功能扩展期 | 扩展层覆盖与性能优化 | 补齐常用推理层至40+，性能benchmark体系，CI/CD流水线（GitHub Actions），自动测试覆盖Linux/Windows/macOS三平台 |
| v0.3.0 (RC) | 发布准备期 | 包发布与API稳定化 | PyPI/Conda正式发布，API冻结与稳定性保证，向后兼容策略，完整用户文档与API参考，模型兼容性验证（经典Caffe模型） |
| v1.0.0 (Stable) | 生产就绪期 | 稳定可用 | API稳定保证（SemVer），生产级性能与内存安全，完整示例与教程，与caffe-slim的互操作桥接层完成 |

> 当前v0.1.0 (Alpha)已完成独立库基础设施建设（CMake模块化、40 C++测试+65 Python测试、Docker开发环境、conda打包配置），详见 [CHANGELOG.md](CHANGELOG.md)。

## 系统要求

- **Python**: 3.14+
- **CMake**: >= 3.26
- **Ninja**: >= 1.11
- **编译器**: C++17 兼容（GCC 9+, Clang 12+, MSVC 2026）
- **BLAS**: OpenBLAS 或其他 BLAS 实现
- **Protobuf**: >= 7.4（Docker 镜像中已预装）
- **可选**: Conda（推荐用于环境管理）、Docker（推荐用于构建验证）

## 🐳 快速开始：Docker 构建验证（推荐）

> **Docker 是构建验证的黄金标准**——提供一致的编译器版本（GCC 14.3.0）、Protobuf 版本（7.x）和隔离的构建环境，完全规避 Windows MSVC 预览版 Bug、conda 环境穿透、跨平台路径冲突等问题。
>
> 详细方法论见 [Docker 作为规范构建环境](../../../../.agents/docs/retrospective/patterns/methodology-patterns/governance-strategy/docker-canonical-build-environment.md)。

### 前置条件

- 已安装 [caffe-ffi-jupyter Docker 镜像](../../../../apps/caffe-ffi-jupyter/README.md)
- WSL2 或 Linux 环境下 Docker 可用

### 一键构建并运行 C++ 测试

```bash
# 在 WSL/Linux 终端中，从 SpecWeave 根目录执行
cd /path/to/SpecWeave
docker run --rm \
  -v "$(pwd):/SpecWeave" \
  -v caffe-ffi-workspace:/workspace \
  caffe-ffi-jupyter:latest \
  bash -c "cp /SpecWeave/apps/caffe-ffi-jupyter/scripts/test-cpp-tests.sh /workspace/ && bash /workspace/test-cpp-tests.sh"
```

这会自动完成：环境检查 → CMake 配置 → C++ 编译 → 链接 → 运行全部测试套件，并输出每个测试套件的通过率。

### 使用已运行的容器（交互式开发）

```bash
# 如果容器已通过 docker compose 启动
docker exec -it caffe-ffi-jupyter bash

# 在容器内运行测试
test-cpp-tests.sh

# 或手动构建
source /opt/conda/etc/profile.d/conda.sh
conda activate caffe-ffi
cmake -S /SpecWeave/projects/xuanspace/libs/caffe-ffi \
      -B /workspace/caffe-ffi-cpp-build \
      -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCAFFE_FFI_BUILD_TESTS=ON -DTVM_FFI_USE_LIBBACKTRACE=OFF
cmake --build /workspace/caffe-ffi-cpp-build -j$(nproc)
cd /workspace/caffe-ffi-cpp-build
LD_LIBRARY_PATH=/opt/conda/envs/caffe-ffi/lib:$(pwd):$(pwd)/lib ./caffe_ffi_tests
```

### Docker 环境规格

| 组件 | 版本 |
|------|------|
| 操作系统 | Ubuntu (conda-forge gcc 14.3.0) |
| Python | 3.14 |
| GCC | 14.3.0 (conda-forge, 稳定版) |
| CMake | 4.4.1 |
| Ninja | 1.13.2 |
| Protobuf (libprotoc) | 35.1 |
| Python protobuf | 7.35.1 |
| numpy | 2.5.1 |

## 本地安装方式

> ⚠️ 本地安装适合日常开发，但**最终构建验证请使用 Docker**（黄金标准）。

### 开发模式安装

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

## Windows 原生 Conda 环境开发指南

Windows 原生环境下推荐使用 conda 管理依赖，无需 WSL。`environment.yml` 已包含所有必要依赖。

### Windows 环境设置

#### 1. 安装 Miniconda

从 [Miniconda 官网](https://docs.anaconda.com/miniconda/) 下载 Windows 安装包并安装。

#### 2. 创建并激活 conda 环境

```powershell
cd caffe-ffi
conda env create -f environment.yml
conda activate caffe-ffi-dev
```

`environment.yml` 已包含 `libopenblas` 依赖，无需手动安装。CMake 的 `DetectBLAS.cmake` 会自动从 conda 环境前缀中搜索 OpenBLAS 头文件和库文件（`$CONDA_PREFIX/Library/include` 和 `$CONDA_PREFIX/Library/lib`）。

#### 3. 构建与安装

```powershell
# 使用一键开发脚本（推荐）
.\scripts\dev.ps1

# 或手动构建
cmake --preset default
cmake --build --preset default
pip install --no-build-isolation -e .
```

#### 4. 构建失败排查指南

遇到构建失败时，**按 L0→L1→L2 三层顺序排查**，顺序不可颠倒（环境层问题30秒验证，工具链层2分钟，最后才深入项目代码）。

> **完整方法论文档**：[构建失败分层排查法](../../../../.agents/docs/retrospective/patterns/code-patterns/build-failure-layered-triage.md)
> **黄金标准**：本地环境问题无法快速定位时，**直接使用 Docker 验证**（5分钟内得到可信结果）。
> 参见 [Docker 作为规范构建环境](../../../../.agents/docs/retrospective/patterns/methodology-patterns/governance-strategy/docker-canonical-build-environment.md)。

**L0 环境层（30秒快速检查）**：

| 检查项 | 命令 | 判定标准 |
|--------|------|----------|
| 编译器版本是否稳定版 | MSVC: `cl 2>&1`；GCC: `gcc --version` | 含 Preview/Insiders/svn/trunk/RC → 换稳定版或 WSL |
| MSVC 环境变量初始化 | `echo $env:INCLUDE`（PowerShell） | 为空 → 使用 Developer PowerShell 或运行 `vcvarsall.bat` |
| 关键工具在 PATH 中 | `where cl.exe` / `which gcc && which protoc` | 命令不存在或路径错误 → 修复环境变量 |

**L1 工具链层（2分钟检查）**：

| 检查项 | 命令 | 判定标准 |
|--------|------|----------|
| protoc 与 libprotobuf 版本一致 | `protoc --version` + 检查 `Protobuf_VERSION` in CMakeCache | 版本不一致 → 统一版本，清理 build 目录 |
| 跨环境 build 目录隔离 | WSL中检查CMakeCache路径格式 | 含 `/mnt/d/` 在Windows构建，或 `D:/` 在WSL构建 → `rm -rf build` 新建独立目录 |
| Windows conda PATH 不穿透到 WSL | WSL中 `which protoc` 不应指向 `/mnt/c/Users/.../anaconda3/` | 穿透到Windows路径 → 使用干净shell或conda环境隔离 |

**L2 项目层（仅当 L0+L1 全通过后）**：
- 阅读完整 CMake configure 输出，查找 WARNING/NOT FOUND
- 只看编译器输出的**第一个** error，忽略后续级联错误
- 使用 `ninja -j1 -v` 单线程编译获取完整错误信息
- 对比已知可工作的环境/分支，使用 git bisect 定位引入问题的提交

**常见具体问题**：

**BLAS/OpenBLAS 未找到**：`DetectBLAS.cmake` 已针对 Windows conda 做平台适配（`Library/include/openblas` 路径、`libopenblas.lib` 库名）。若仍出现检测失败：
- 确认已激活正确的 conda 环境：`conda activate caffe-ffi-dev`
- 手动安装 OpenBLAS：`conda install -c conda-forge libopenblas`
- 若使用非默认环境名，设置 `CONDA_PREFIX` 或在 CMake 配置时指定 `-DCMAKE_PREFIX_PATH=<env_path>`

**Protobuf 版本不兼容（caffe.pb.h 编译错误）**：
- 确保 protoc 与链接的 libprotobuf 来自同一安装（推荐 conda-forge libprotobuf >= 7.0.0）
- WSL 中禁止使用 Windows conda 的 protoc.exe（会导致版本不匹配）
- 详细指南见 [WSL2 构建环境配置指南](docs/setup/WSL2_BUILD_SETUP_GUIDE.md)
- 变更说明见 [Protobuf 兼容性改动说明](docs/setup/PROTOBUF_COMPATIBILITY_AND_COMPILER_FLAGS.md)

**KMP_DUPLICATE_LIB_OK**：Windows 上多个组件可能各自包含 OpenMP 运行时，设置此环境变量避免冲突：

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```

**MSVC C1041 PDB 锁定错误**：这是 MSVC 预览版（19.50 Insiders）的已知 Bug，`/FS` 标志无法解决。**不要反复清理/重试**，直接使用 Docker（推荐）或切换到稳定版 MSVC / WSL 环境构建。

```bash
# 使用 Docker 一键规避所有 MSVC 环境问题
cd /path/to/SpecWeave
docker run --rm \
  -v "$(pwd):/SpecWeave" \
  -v caffe-ffi-workspace:/workspace \
  caffe-ffi-jupyter:latest \
  bash -c "bash /SpecWeave/apps/caffe-ffi-jupyter/scripts/test-cpp-tests.sh"
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

### 使用 Docker（⭐ 首选方案）

项目提供完整的 Docker 开发环境，基于 `apps/caffe-ffi-jupyter`，内置 SSH + Jupyter 双服务，支持 Python 3.14+，编译器和依赖版本完全固定。**这是构建验证的黄金标准**，推荐所有团队成员使用。

> **完整方法论**：[Docker 作为规范构建环境](../../../../.agents/docs/retrospective/patterns/methodology-patterns/governance-strategy/docker-canonical-build-environment.md)

#### 快速部署（WSL 环境）

```bash
# 在 WSL 中进入 apps/caffe-ffi-jupyter 目录
cd apps/caffe-ffi-jupyter

# 一键部署（构建镜像 + 启动容器 + 验证）
bash scripts/wsl-deploy.sh
```

或在 Windows PowerShell 中：

```powershell
# 自动检测 WSL 并调用 wsl-deploy.sh
.\scripts\deploy.ps1
```

#### Docker 环境特性

- **基础镜像**: `jupyter-ssh-base`（保留 SSH + Jupyter 双服务）
- **Python 版本**: 3.14+（Miniconda 环境，名为 `caffe-ffi`）
- **SSH 访问**: 容器暴露 22 端口，支持密钥/密码登录
- **Jupyter**: 容器暴露 8888 端口，已注册 "Python 3.14 (caffe-ffi)" 内核
- **自动挂载**: 项目源码以 editable 模式挂载，代码修改即时生效

#### 运行 C++/Python 单元测试

容器内置完整测试脚本，可直接运行：

```bash
# 进入容器（SSH 或 docker exec）
docker exec -it caffe-ffi-jupyter bash

# 运行 C++ 和 Python 单元测试（含耗时统计）
test-cpp-tests.sh
```

#### 手动构建参考

如果需要自定义镜像构建：

```bash
cd apps/caffe-ffi-jupyter
bash scripts/build.sh
```

> **注意**: 旧版 `Dockerfile.dev` 已废弃，请使用 `apps/caffe-ffi-jupyter/` 目录下的完整 Docker 环境。

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
├── AGENTS.md                # AI 智能体路由入口
├── .agents/                 # 项目级智能体规范
├── .temp/                   # 临时文件目录（不提交内容）
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
├── tests/cpp/               # C++ 单元测试
├── conda.recipe/            # Conda 构建配方
├── scripts/                 # 开发/构建脚本
│   ├── dev.sh               # Linux/WSL/macOS 一键开发脚本
│   ├── dev.ps1              # Windows 一键开发脚本
│   ├── conda_build.sh       # Linux/WSL Conda 环境构建脚本
│   ├── conda_build.bat      # Windows Conda 环境构建脚本
│   ├── check_ffi_prefix.py  # FFI 前缀一致性检查
│   ├── verify_install.py    # 安装验证脚本
│   └── gen_proto.py         # Protobuf 代码生成
├── docs/                    # 文档
└── examples/                # 示例代码
```

> **临时文件约定**：调试脚本、临时测试、构建日志等临时文件请统一放在 `.temp/` 目录下，不要散落在项目根目录或其他位置。`.temp/` 目录下除 `.gitkeep` 外的文件不会被 Git 追踪。

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

### Conda 环境一键构建

```bash
# Linux/WSL：在已激活的 conda 环境中执行
bash scripts/conda_build.sh       # 配置 + 构建 + 安装 + 测试

# Windows：在已激活的 conda 环境中执行
scripts\conda_build.bat           # 配置 + 构建 + 安装 + 测试
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

### 🐳 Docker 中运行测试（推荐）

```bash
# 一键运行 C++ 和 Python 测试（黄金标准环境）
cd /path/to/SpecWeave
docker run --rm \
  -v "$(pwd):/SpecWeave" \
  -v caffe-ffi-workspace:/workspace \
  caffe-ffi-jupyter:latest \
  bash -c "bash /SpecWeave/apps/caffe-ffi-jupyter/scripts/test-cpp-tests.sh"

# 或在已运行的容器中
docker exec -it caffe-ffi-jupyter test-cpp-tests.sh
```

### 本地运行测试

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

