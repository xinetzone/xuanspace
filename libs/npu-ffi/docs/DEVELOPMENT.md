# npu-ffi 开发环境配置指南

本文档详细说明如何在 Python 3.14 环境下配置 npu-ffi 的开发环境。

## 环境要求

| 依赖 | 最低版本 | 说明 |
|------|----------|------|
| Python | **3.14+** | 项目 `requires-python = ">=3.14"`，利用 PEP 649 延迟注解求值 |
| CMake | >= 3.26 | C++ 构建系统 |
| Ninja | >= 1.11 | 构建加速器 |
| C++ 编译器 | C++17 兼容 | MSVC 2022 / GCC 9+ / Clang 12+ |
| Conda | 推荐 | 环境管理 |

## 关键注意事项

在开始之前，请务必了解以下要点，避免常见陷阱：

1. **必须使用 `--no-build-isolation`**：安装 tvm-ffi 和 npu-ffi 时都必须加上此参数，确保构建时使用已安装的 editable 模式 tvm-ffi，而不是在隔离环境中重新下载 PyPI 版本导致 DLL 冲突。

2. **Windows 必须设置 `KMP_DUPLICATE_LIB_OK=TRUE`**：Windows 数据科学栈中 OpenMP 多副本共存是常态，未设置此环境变量会导致程序崩溃。

3. **安装顺序**：必须先安装 tvm-ffi，再安装 npu-ffi，且两者都使用 `--no-build-isolation`。

4. **每个 conda 环境需独立安装**：editable 安装是环境隔离的，如果使用多个 conda 环境（如 base、py313、py314），每个环境都需要单独安装 tvm-ffi 和 npu-ffi。

---

## 方式一：使用现有 py314 环境（推荐用于已有 conda 环境的开发者）

如果你已经有一个 Python 3.14 的 conda 环境（如名为 `py314`），按以下步骤配置：

### Windows PowerShell

```powershell
# 1. 激活 py314 环境
conda activate py314

# 2. 验证 Python 版本（应显示 3.14.x）
python --version

# 3. 安装编译依赖（如果尚未安装）
conda install -c conda-forge cmake>=3.26 ninja>=1.11 cxx-compiler pip
pip install "scikit-build-core>=0.10.0" "protobuf>=7.0.0" "pytest>=8.0"

# 4. 设置 Windows OpenMP 环境变量
#    永久设置（仅需执行一次，对所有新终端生效）
[Environment]::SetEnvironmentVariable("KMP_DUPLICATE_LIB_OK", "TRUE", "User")
#    当前会话立即生效
$env:KMP_DUPLICATE_LIB_OK="TRUE"

# 5. 安装 tvm-ffi（editable 模式）
#    假设当前在 projects/xuanspace/libs/npu-ffi 目录
cd ..\..\..
pip install --no-build-isolation -e projects/xuanspace/vendor/tvm-ffi

# 6. 安装 npu-ffi（editable 模式）
cd projects/xuanspace/libs/npu-ffi
pip install --no-build-isolation -e .

# 7. 运行安装验证（9 项检查应全部通过）
python scripts/verify_install.py

# 8. 运行测试（116 个测试应全部通过）
pytest tests/python -v
```

### Linux / macOS

```bash
# 1. 激活 py314 环境
conda activate py314

# 2. 验证 Python 版本
python --version

# 3. 安装编译依赖
conda install -c conda-forge cmake>=3.26 ninja>=1.11 cxx-compiler pip
pip install "scikit-build-core>=0.10.0" "protobuf>=7.0.0" "pytest>=8.0"

# 4. 设置 OpenMP 环境变量（当前会话）
export KMP_DUPLICATE_LIB_OK=TRUE
#    永久设置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export KMP_DUPLICATE_LIB_OK=TRUE' >> ~/.bashrc

# 5. 安装 tvm-ffi（editable 模式）
cd ../../..
pip install --no-build-isolation -e projects/xuanspace/vendor/tvm-ffi

# 6. 安装 npu-ffi（editable 模式）
cd projects/xuanspace/libs/npu-ffi
pip install --no-build-isolation -e .

# 7. 运行安装验证
python scripts/verify_install.py

# 8. 运行测试
KMP_DUPLICATE_LIB_OK=TRUE pytest tests/python -v
```

---

## 方式二：使用自动化脚本创建独立开发环境

项目提供了自动化脚本，一键创建名为 `npu-ffi-dev` 的独立 conda 环境，自动安装所有依赖：

### Windows PowerShell

```powershell
# 在 npu-ffi 项目根目录执行
.\scripts\setup_conda_dev.ps1

# 脚本完成后激活环境
conda activate npu-ffi-dev

# 设置环境变量并运行测试
$env:KMP_DUPLICATE_LIB_OK="TRUE"
pytest tests/python -v
```

脚本自动完成以下步骤：
1. 根据 [environment.yml](environment.yml) 创建 Python 3.14 环境
2. 安装 cmake、ninja、cxx-compiler、scikit-build-core、protobuf、pytest
3. 检测本地 vendor/tvm-ffi 并以 editable 模式安装（找不到则从 PyPI 安装）
4. 以 editable 模式安装 npu-ffi

### Linux / macOS

```bash
chmod +x scripts/setup_conda_dev.sh
./scripts/setup_conda_dev.sh
conda activate npu-ffi-dev
KMP_DUPLICATE_LIB_OK=TRUE pytest tests/python -v
```

如果需要自定义环境名称：

```powershell
.\scripts\setup_conda_dev.ps1 -EnvName my-dev-env
```

---

## 方式三：手动从 environment.yml 创建环境

```bash
# 创建环境
conda env create -f environment.yml

# 激活
conda activate npu-ffi-dev

# 手动安装 tvm-ffi
pip install --no-build-isolation -e ../../vendor/tvm-ffi

# 手动安装 npu-ffi
pip install --no-build-isolation -e .

# 验证
python scripts/verify_install.py
pytest tests/python -v
```

---

## 安装验证

安装完成后，运行验证脚本确认环境配置正确：

```bash
python scripts/verify_install.py
```

该脚本会检查以下 9 项：

| 检查项 | 说明 |
|--------|------|
| Python 版本 | 确认 >= 3.14 |
| KMP_DUPLICATE_LIB_OK | Windows 环境变量检查 |
| apache-tvm-ffi 包 | 确认 tvm-ffi 已安装 |
| tvm_ffi 导入 | 确认模块可导入 |
| npu-ffi 包 | 确认 npu-ffi 已安装 |
| npu_ffi 导入 | 确认模块可导入 |
| vta FFI 模块导入 | 确认核心 FFI 绑定可加载，获取版本信息 |
| Buffer 分配/释放 | 测试缓冲区 RAII 功能 |
| CommandContext | 测试上下文管理器功能 |

所有检查通过会显示 🎉 成功消息；失败时会给出具体的故障排除建议。

## 运行测试

```bash
# 详细模式
pytest tests/python -v

# 简洁模式
pytest tests/python -q

# 运行特定测试文件
pytest tests/python/test_buffer.py -v

# 带性能基准测试（需 pytest-benchmark）
pytest tests/python -v --benchmark-only
```

测试覆盖范围（共 116 个测试用例）：
- **Buffer 管理**：创建、RAII 自动释放、上下文管理器、CPU 指针、屏障、双重释放安全
- **CommandContext**：上下文管理、工作流、多上下文、调试模式、依赖链、prepare_call
- **VTAConfig**：默认配置、参数校验、序列化（dict/protobuf/json/text）、不可变性
- **枚举类型**：DebugFlag、MemcpyKind、MemoryType、ALUOpcode 值校验
- **FFI API**：命令句柄、缓冲区分配/拷贝/屏障、2D Load/Store、UOP 操作、GEMM/ALU、依赖操作、同步、调试模式、运行时关闭

## FFI 前缀一致性检查

修改 C++ FFI 注册或 Python 初始化代码后，运行前缀检查脚本验证一致性：

```bash
python scripts/check_ffi_prefix.py --verbose
```

该脚本自动扫描 C++ 中 `.def("prefix.func", ...)` 注册的函数与 Python 中 `init_ffi_api("prefix", ...)` 初始化的前缀是否匹配，防止因字符串不匹配导致运行时找不到函数。

## 常见问题

完整的常见问题解答（包括安装错误、DLL 加载失败、FFI 函数找不到、测试问题、环境管理等）请参见 **[docs/FAQ.md](FAQ.md)**。

### 快速故障排查

如果遇到问题，按以下顺序排查：

1. **运行验证脚本**：`python scripts/verify_install.py`，根据输出定位问题
2. **检查 Python 版本**：`python --version`，必须为 3.14+
3. **检查 FFI 前缀一致性**：`python scripts/check_ffi_prefix.py --verbose`
4. **确认环境变量**：Windows 下 `KMP_DUPLICATE_LIB_OK=TRUE`
5. **查看 FAQ**：[FAQ.md](FAQ.md) 中的对应章节

## 国内镜像配置（可选）

如果在国内网络环境下下载速度慢，可以配置 conda 和 pip 镜像：

### Conda 镜像

```powershell
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/pkgs/msys2
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes
```

### pip 镜像

创建或编辑 `%APPDATA%\pip\pip.ini`（Windows）或 `~/.pip/pip.conf`（Linux/macOS）：

```ini
[global]
index-url = https://mirrors.bfsu.edu.cn/pypi/web/simple
trusted-host = mirrors.bfsu.edu.cn
```
