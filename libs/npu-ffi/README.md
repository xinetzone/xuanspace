# npu-ffi

Type-safe FFI bindings for VTA NPU accelerator based on tvm-ffi

<!-- badges: start -->
<!-- TODO: Add CI, PyPI, Conda badges here -->
<!-- badges: end -->

## 特性

- 基于 tvm-ffi 的类型安全 FFI 绑定
- 支持 Stub 模式（无需硬件即可开发测试）
- RAII 内存管理（Buffer/CommandContext）
- Protobuf 硬件配置序列化
- Conda 包管理支持
- 跨平台（Windows/Linux/macOS）

## 前置要求

- Python 3.13+
- CMake &gt;= 3.26
- Ninja &gt;= 1.11
- C++17 兼容编译器（MSVC 2022/GCC 9+/Clang 12+）
- Conda（可选，推荐）

## 安装指南

### 开发模式安装（推荐）

```bash
# 步骤1：先安装 tvm-ffi (editable模式)
# 假设项目路径为 projects/xuanspace
cd vendor/tvm-ffi
pip install --no-build-isolation -e .

# 步骤2：安装 npu-ffi (editable模式)
cd ../../libs/npu-ffi
pip install --no-build-isolation -e .
```

> **注意**：必须使用 `--no-build-isolation` 参数，以确保构建时使用已安装的 editable 模式 tvm-ffi，而不是在隔离环境中重新下载依赖。两个包都需要此参数。

### 一键开发脚本（推荐）

项目提供了自动化开发脚本，一键完成依赖安装、构建、测试：

```powershell
# Windows PowerShell
.\scripts\dev.ps1            # 构建 + 安装 + 验证
.\scripts\dev.ps1 -Build     # 仅构建C++
.\scripts\dev.ps1 -Test      # 运行测试
.\scripts\dev.ps1 -Rebuild   # 清理重建

# Linux/macOS
chmod +x scripts/dev.sh
./scripts/dev.sh             # 构建 + 安装 + 验证
./scripts/dev.sh -t          # 运行测试
```

### Conda 环境快速设置

项目提供了自动化脚本用于快速配置 Conda 开发环境：

```bash
# Windows PowerShell
.\scripts\setup_conda_dev.ps1

# Linux/macOS
chmod +x scripts/setup_conda_dev.sh
./scripts/setup_conda_dev.sh
conda activate npu-ffi-dev
```

### 环境变量（Windows）

在 Windows 上运行时，建议设置以下环境变量以避免 OpenMP 重复初始化错误：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

## 快速开始示例

### 基础使用

```python
from npu_ffi import vta

# 获取线程本地命令句柄
cmd = vta.tls_command_handle()

# 分配缓冲区
buf = vta.buffer_alloc(1024)  # 分配1KB

# 设置调试模式
vta.set_debug_mode(cmd, vta.DebugFlag.DUMP_INSN | vta.DebugFlag.DUMP_UOP)

# 执行VTA操作...

# 释放缓冲区
vta.buffer_free(buf)

# 同步等待完成
vta.synchronize(cmd, 0)  # 0 = 无限等待
```

### Buffer RAII 管理

```python
from npu_ffi import vta
from npu_ffi.vta import Buffer, CommandContext

with CommandContext() as cmd:
    with Buffer(1024 * 1024) as buf:  # 1MB
        ptr = buf.cpu_ptr(cmd)
        # 使用缓冲区...
        buf.write_barrier(cmd, elem_bits=32, start=0, extent=buf.size // 4)
    # 离开with块自动释放
```

### CommandContext

```python
from npu_ffi import vta
from npu_ffi.vta import CommandContext

with CommandContext() as cmd:
    # cmd 是命令句柄（int类型）
    buf = vta.buffer_alloc(4096)
    # ... 推送命令 ...
# 离开with块自动同步并清理
```

### 枚举类型

```python
from npu_ffi.vta import DebugFlag, MemcpyKind, MemoryType, ALUOpcode

# 调试标志（支持位运算组合）
flags = DebugFlag.DUMP_INSN | DebugFlag.DUMP_UOP

# 内存拷贝类型
kind = MemcpyKind.H2D  # Host to Device

# 内存类型
mem = MemoryType.DRAM  # DRAM, SRAM, UOP, INP, WGT, ACC, OUT

# ALU操作码
op = ALUOpcode.ADD  # ADD, SUB, MUL, MIN, MAX, SHR, SHL
```

### 硬件配置（Protobuf）

```python
from npu_ffi.vta.config import get_default_config
from npu_ffi.vta import proto_io

# 获取默认配置
config = get_default_config('vta')  # 或 'vta_v3', 'vta_v4'
print(f"Block size: {config.block_in} x {config.block_out}")

# 序列化为JSON
proto_io.save_json(config, 'config.json')

# 从JSON加载
loaded = proto_io.load_json('config.json')

# 序列化为二进制Protobuf
proto_io.save_config(config, 'config.bin')

# 从二进制加载
loaded_bin = proto_io.load_config('config.bin')
```

## 项目结构

```
npu-ffi/
├── CMakeLists.txt           # 根CMake配置
├── pyproject.toml           # Python构建配置
├── environment.yml          # Conda环境配置
├── include/                 # C++头文件
│   └── npu_ffi/vta/        # VTA类型安全封装
├── src/                     # C++源代码
│   └── vta/                # VTA实现（stub+FFI注册）
├── python/npu_ffi/         # Python包
│   └── vta/                # VTA Python API
├── proto/                   # Protobuf Schema
├── tests/python/           # Python单元测试
├── conda.recipe/           # Conda构建配方
└── scripts/                 # 辅助脚本
```

## 构建模式

### Stub 模式（默认）

不依赖真实 VTA 硬件，模拟所有 API 调用，用于开发测试：

```bash
cmake -B build -DNPU_FFI_VTA_USE_STUB=ON
```

### 真实硬件模式

链接真实 VTA runtime 库（需指定 VTA_DIR）：

```bash
cmake -B build -DNPU_FFI_VTA_USE_STUB=OFF -DVTA_DIR=/path/to/vta
```

### 从源码构建 tvm-ffi

默认使用已安装的 `find_package` 模式。如需从源码构建 tvm-ffi：

```bash
cmake -B build -DNPU_FFI_FROM_SOURCE=ON
```

## 运行测试

```bash
# Windows
$env:KMP_DUPLICATE_LIB_OK="TRUE"
pytest tests/python -v

# Linux/macOS
KMP_DUPLICATE_LIB_OK=TRUE pytest tests/python -v
```

### FFI 前缀一致性检查

在修改C++ FFI注册或Python初始化代码后，运行前缀检查脚本验证一致性：

```bash
python scripts/check_ffi_prefix.py --verbose
```

该脚本自动扫描C++中 `.def("prefix.func", ...)` 注册的函数与Python中 `_FFI_INIT_FUNC("prefix", ...)` 初始化前缀是否匹配，防止因字符串不匹配导致运行时找不到函数的错误。

## Conda 包构建

```bash
conda install conda-build -c conda-forge
conda build conda.recipe -c conda-forge
conda install --use-local npu-ffi
```

详细参见 [conda.recipe/README.md](conda.recipe/README.md)。

## API 参考

### 模块结构

- `npu_ffi.vta` - 核心 VTA runtime API
- `npu_ffi.vta.buffer` - Buffer RAII 封装类
- `npu_ffi.vta.command` - CommandContext 上下文管理器
- `npu_ffi.vta.config` - VTAConfig 硬件配置数据类
- `npu_ffi.vta.proto_io` - Protobuf 序列化工具

### 核心函数

| 函数 | 说明 |
|------|------|
| `vta.tls_command_handle()` | 获取线程本地命令句柄 |
| `vta.buffer_alloc(size)` | 分配指定大小（字节）的缓冲区 |
| `vta.buffer_free(buf)` | 释放缓冲区 |
| `vta.buffer_copy(dst, src, size, kind)` | 缓冲区拷贝 |
| `vta.buffer_cpu_ptr(cmd, buf)` | 获取 CPU 可访问指针 |
| `vta.set_debug_mode(cmd, flags)` | 设置调试标志 |
| `vta.synchronize(cmd, wait_cycles)` | 同步等待命令完成 |
| `vta.write_barrier(cmd, buf, elem_bits, start, extent)` | 写屏障 |
| `vta.read_barrier(cmd, buf, elem_bits, start, extent)` | 读屏障 |
| `vta.uop_push(...)` | 推送微操作 |
| `vta.push_gemm_op(...)` | 推送 GEMM 操作 |
| `vta.push_alu_op(...)` | 推送 ALU 操作 |

### Buffer 类 API

| 方法/属性 | 说明 |
|-----------|------|
| `Buffer(size)` | 构造函数，分配指定大小缓冲区 |
| `Buffer(size, data, owns)` | 包装现有指针 |
| `.data` | 原始缓冲区指针（int） |
| `.size` | 缓冲区大小（字节） |
| `.owns_data` | 是否拥有数据所有权 |
| `.cpu_ptr(cmd)` | 获取 CPU 可访问指针 |
| `.write_barrier(cmd, elem_bits, start, extent)` | 写屏障 |
| `.read_barrier(cmd, elem_bits, start, extent)` | 读屏障 |
| `__enter__/__exit__` | 上下文管理器支持（with 语句） |

### CommandContext 类 API

| 方法/属性 | 说明 |
|-----------|------|
| `CommandContext(wait_cycles=0)` | 构造函数，指定同步等待周期 |
| `__enter__()` | 返回命令句柄（int） |
| `__exit__()` | 自动同步 |
| `.handle` | 获取命令句柄（未进入上下文时抛出异常） |

### 枚举类型

- **DebugFlag**: `DUMP_INSN`, `DUMP_UOP`, `SKIP_READ_BARRIER`, `SKIP_WRITE_BARRIER`, `FORCE_SERIAL`（支持位运算组合）
- **MemcpyKind**: `H2D` (Host→Device), `D2H` (Device→Host), `D2D` (Device→Device)
- **MemoryType**: `DRAM`, `SRAM`, `UOP`, `INP`, `WGT`, `ACC`, `OUT`
- **ALUOpcode**: `ADD`, `SUB`, `MUL`, `MIN`, `MAX`, `SHR`, `SHL`

### 配置 API

- `VTAConfig` - 不可变数据类（frozen dataclass），包含所有硬件参数
- `get_default_config(name)` - 获取预设配置（"vta", "vta_v3", "vta_v4"）
- `validate_config(config)` - 验证配置参数合法性
- `config.to_dict()` / `VTAConfig.from_dict(d)` - 字典序列化
- `config.replace(**changes)` - 创建修改后的新配置

## 许可证

Apache-2.0（与 tvm-ffi 和 VTA 一致）。详见 [LICENSE](LICENSE) 文件。

## 相关项目

- [tvm-ffi](https://github.com/tlc-pack/tvm-ffi) - Type-safe foreign function interface for TVM
- [VTA](https://github.com/apache/tvm-vta) - Versatile Tensor Accelerator
