# Conda 包构建指南

本文档说明如何使用 Conda 构建和安装 npu-ffi 包。

## 目录结构

```
npu-ffi/
├── environment.yml              # Conda 开发环境配置
├── conda.recipe/
│   ├── meta.yaml                # Conda 构建配方
│   ├── conda_build_config.yaml  # 构建配置（锁定 Python 版本）
│   ├── build.sh                 # Linux/macOS 构建脚本
│   ├── bld.bat                  # Windows 构建脚本
│   └── README.md                # 本文档
└── scripts/
    ├── setup_conda_dev.sh       # Linux/macOS 开发环境一键设置
    └── setup_conda_dev.ps1      # Windows PowerShell 开发环境一键设置
```

## 开发环境设置

### 方式一：使用辅助脚本（推荐）

**Linux/macOS:**
```bash
cd libs/npu-ffi
chmod +x scripts/setup_conda_dev.sh
./scripts/setup_conda_dev.sh              # 使用默认环境名 npu-ffi-dev
# 或指定环境名
./scripts/setup_conda_dev.sh my-env-name
```

**Windows PowerShell:**
```powershell
cd libs\npu-ffi
.\scripts\setup_conda_dev.ps1             # 使用默认环境名 npu-ffi-dev
# 或指定环境名
.\scripts\setup_conda_dev.ps1 -EnvName my-env-name
```

### 方式二：手动设置

```bash
# 1. 创建并激活 conda 环境
conda env create -f environment.yml
conda activate npu-ffi-dev

# 2. 安装 tvm-ffi
# 方式 A: 本地 vendor 目录（开发模式）
pip install --no-build-isolation -e ../../vendor/tvm-ffi
# 方式 B: 从 PyPI 安装
pip install apache-tvm-ffi

# 3. 安装 npu-ffi (editable，开发模式)
pip install --no-build-isolation -e .
```

### 运行测试

```bash
# Windows PowerShell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
pytest tests/python -v

# Linux/macOS
export KMP_DUPLICATE_LIB_OK=TRUE
pytest tests/python -v
```

## 构建 Conda 包

### 前置要求

```bash
# 安装 conda-build
conda install conda-build -c conda-forge
```

### 构建包

```bash
cd libs/npu-ffi

# 注意：apache-tvm-ffi 目前不在 conda-forge 上
# 构建前确保 tvm-ffi 已安装（在 base 或构建环境中）
pip install apache-tvm-ffi
# 或
pip install -e ../../vendor/tvm-ffi

# 执行构建
conda build conda.recipe -c conda-forge
```

### 安装本地构建的包

```bash
# 安装本地构建的包
conda install --use-local npu-ffi

# 或者创建新环境并安装
conda create -n npu-ffi-test npu-ffi -c local -c conda-forge
conda activate npu-ffi-test
```

## 国内镜像配置（可选）

### Conda 镜像（北外）

```bash
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/pkgs/msys2
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes
```

### pip 镜像（北外）

编辑 `~/.pip/pip.conf`（Linux/macOS）或 `%APPDATA%\pip\pip.ini`（Windows）：

```ini
[global]
index-url = https://mirrors.bfsu.edu.cn/pypi/web/simple
trusted-host = mirrors.bfsu.edu.cn
```

## 注意事项

### 依赖说明

- **apache-tvm-ffi**: 目前主要通过 PyPI 发布，不提供 Conda 包。构建脚本会自动通过 pip 安装。如需使用本地开发版本，确保 `../../vendor/tvm-ffi` 存在。
- **protobuf>=7.0.0**: 通过 conda-forge 安装。
- **C++ 编译器**:
  - Windows: Visual Studio 2022 或 Build Tools（需支持 C++17）
  - Linux: GCC 9+ 或 Clang 12+
  - macOS: Xcode Command Line Tools (Clang)

### Python 版本

- 项目要求 Python >= 3.13
- `conda_build_config.yaml` 锁定构建版本为 3.13

### 构建系统

- 使用 CMake + Ninja + scikit-build-core
- 构建脚本会自动设置 `CMAKE_GENERATOR=Ninja`

## 常见问题

### Q: 构建时找不到 tvm-ffi？

A: 确保在运行 `conda build` 之前，已在当前环境中通过 pip 安装了 apache-tvm-ffi：
```bash
pip install apache-tvm-ffi
```

### Q: Windows 上构建失败，提示找不到 MSVC 编译器？

A: 确保已安装 Visual Studio 2022 或 Build Tools，并且在 "Developer Command Prompt for VS 2022" 或 "Developer PowerShell for VS 2022" 中运行构建命令。

### Q: 测试时出现 KMP duplicate lib 错误？

A: 设置环境变量 `KMP_DUPLICATE_LIB_OK=TRUE`，这是由于 OpenMP 运行时库重复加载导致的，不影响功能正确性。
