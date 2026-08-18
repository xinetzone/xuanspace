# npu-ffi 常见问题解答 (FAQ)

本文档收集了 npu-ffi 开发和使用过程中的常见问题及解决方案。

---

## 目录

- [安装与环境](#安装与环境)
- [导入与运行时错误](#导入与运行时错误)
- [构建与编译](#构建与编译)
- [FFI 相关问题](#ffi-相关问题)
- [测试相关问题](#测试相关问题)
- [环境管理](#环境管理)

---

## 安装与环境

### Q: `ModuleNotFoundError: No module named 'npu_ffi'`

**症状**：导入 npu_ffi 时提示模块不存在。

**原因**：npu-ffi 未安装在当前激活的 conda/Python 环境中。editable 安装（`pip install -e`）是环境隔离的，每个 conda 环境需要独立安装。

**解决**：
```bash
# 确认当前环境
conda activate py314
python --version  # 应显示 3.14.x

# 在正确的环境中重新安装
cd path/to/libs/npu-ffi
pip install --no-build-isolation -e .
```

---

### Q: `NameError: name 'VTAConfig' is not defined`

**症状**：导入 npu_ffi 时在 config.py 报 `NameError: name 'VTAConfig' is not defined`。

**原因**：使用了 Python 3.14.5 或更低版本。项目要求 Python 3.14.6+，利用 PEP 649（Deferred Evaluation of Annotations）默认启用延迟注解求值，类体内自引用类型注解（如 `def from_dict(...) -> VTAConfig`）才不会在类定义完成前被求值。

**解决**：切换到 Python 3.14.6+ 环境：
```bash
conda activate py314
python --version  # 必须显示 3.14.6+
```

> **注意**：Python 3.7-3.14.5 可以通过 `from __future__ import annotations` 启用 PEP 563 字符串化注解，但本项目明确要求 3.14.6+，不支持旧版本。

---

### Q: 安装时 tvm-ffi 从 PyPI 下载而不是使用本地 vendor 版本

**症状**：`pip install -e .` 时自动下载了 PyPI 上的 apache-tvm-ffi，而不是使用 vendor/tvm-ffi 的本地源码。

**原因**：没有使用 `--no-build-isolation` 参数，导致 pip 在隔离的构建环境中重新解析依赖，从 PyPI 获取 tvm-ffi，造成版本不匹配和 DLL 冲突。

**解决**：始终使用 `--no-build-isolation`，并确保安装顺序正确：
```bash
# 步骤1：先安装本地 tvm-ffi（editable 模式）
pip install --no-build-isolation -e path/to/vendor/tvm-ffi

# 步骤2：再安装 npu-ffi
pip install --no-build-isolation -e path/to/libs/npu-ffi
```

---

## 导入与运行时错误

### Q: 导入时出现 OSError / DLL 加载失败（Windows）

**症状**：
- `ImportError: DLL load failed while importing _ffi_api: 找不到指定的模块`
- `OSError: [WinError 126] 找不到指定的模块`

**原因**：
1. 未设置 `KMP_DUPLICATE_LIB_OK=TRUE` 环境变量
2. tvm-ffi 版本冲突（PyPI 版本的 DLL 与本地构建的 DLL 不一致）
3. C++ 扩展未正确构建（editable 安装不自动复制 DLL）

**解决**：
```powershell
# 1. 设置 OpenMP 环境变量
$env:KMP_DUPLICATE_LIB_OK="TRUE"
# 永久设置（仅需执行一次）
[Environment]::SetEnvironmentVariable("KMP_DUPLICATE_LIB_OK", "TRUE", "User")

# 2. 强制重新安装 tvm-ffi 和 npu-ffi
pip install --no-build-isolation --force-reinstall -e path/to/vendor/tvm-ffi
pip install --no-build-isolation --force-reinstall -e .

# 3. 如仍有问题，清理重建
Remove-Item -Recurse -Force build
pip install --no-build-isolation -e .
```

---

### Q: Windows 下程序崩溃退出（无 Python traceback）

**症状**：程序运行时直接崩溃，没有 Python 异常信息，可能显示"程序已停止工作"。

**原因**：Windows 数据科学栈中 OpenMP 运行时多副本共存（Intel MKL、LLVM OpenMP、MSVC OpenMP 等），未设置 `KMP_DUPLICATE_LIB_OK=TRUE` 时 libiomp5md.dll 检测到重复初始化会直接 abort。

**解决**：
```powershell
# 永久设置
[Environment]::SetEnvironmentVariable("KMP_DUPLICATE_LIB_OK", "TRUE", "User")
# 当前会话
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

> 注意：运行 pytest 时 [tests/python/conftest.py](../tests/python/conftest.py) 会自动设置此环境变量，所以测试一般不会遇到此问题。

---

### Q: FFI 函数调用时提示 `AttributeError: module 'tvm_ffi._api' has no attribute 'vta.xxx'`

**症状**：调用 npu_ffi.vta 的函数时，提示 tvm_ffi 注册表中找不到对应的函数。

**原因**：
1. C++ 中注册的 FFI 函数字符串前缀与 Python 中 `init_ffi_api` 使用的前缀不一致
2. C++ 扩展未重新编译（修改了 C++ 代码后没有重新 build）
3. npu-ffi 的 C++ 共享库（DLL/SO）没有被正确加载

**解决**：
```bash
# 1. 运行 FFI 前缀一致性检查
python scripts/check_ffi_prefix.py --verbose

# 2. 如果有不匹配的前缀，修正 C++ 或 Python 中的字符串
#    C++:   TVM_FFI_REGISTER_GLOBAL("vta.buffer_alloc")
#    Python: _FFI_INIT_FUNC("vta", ...)
#    两边前缀必须完全一致！

# 3. 重新构建 C++ 扩展
pip install --no-build-isolation -e .
```

---

## 构建与编译

### Q: CMake 找不到 tvm-ffi（find_package 失败）

**症状**：构建时 CMake 报错 `Could not find a package configuration file provided by "tvm_ffi"`。

**原因**：tvm-ffi 未安装，或安装的版本不包含 CMake config 文件。

**解决**：确保 tvm-ffi 正确安装：
```bash
pip install --no-build-isolation -e path/to/vendor/tvm-ffi

# 验证 tvm-ffi 的 CMake config 存在
python -c "import tvm_ffi; print(tvm_ffi.__path__)"
# 检查 cmake 目录下是否有 tvm_ffiConfig.cmake
```

> **注意**：npu-ffi 使用 `find_package(tvm_ffi CONFIG REQUIRED)` 模式链接已安装的 tvm-ffi，**禁止**使用 `add_subdirectory` 方式 vendored 构建，否则会导致双份 DLL 冲突。

---

### Q: 构建时 protobuf 相关编译错误

**症状**：C++ 编译时出现 protobuf 头文件找不到或链接错误。

**原因**：
1. protobuf 版本过低（要求 >= 7.0.0）
2. libprotobuf C++ 版本与 Python protobuf 版本不匹配
3. C++ libprotobuf 22.x+ 需要 abseil-cpp 作为依赖

**解决**：
```bash
# 确认 protobuf 版本
pip show protobuf  # 应 >= 7.0.0

# 如果是系统级 C++ protobuf 问题，使用 conda 安装兼容版本
conda install -c conda-forge protobuf>=7 abseil-cpp
```

---

### Q: editable 安装后修改了 C++ 代码但 Python 端看不到变化

**症状**：修改了 C++ 源代码，重新运行 Python 但行为没有变化。

**原因**：pip editable 安装不会自动检测 C++ 源文件变化并重新编译。scikit-build-core 的 editable 模式需要手动重新触发构建。

**解决**：
```bash
# 重新安装（触发 C++ 构建）
pip install --no-build-isolation -e .

# 或者使用 dev 脚本（推荐）
.\scripts\dev.ps1 -Rebuild    # Windows
./scripts/dev.sh               # Linux/macOS
```

---

## FFI 相关问题

### Q: 如何确认 vta 的 FFI 函数是否正确注册？

**解决**：
```python
import tvm_ffi
from npu_ffi import vta

# 列出所有 vta 开头的注册函数
names = [n for n in tvm_ffi.list_global_func_names() if n.startswith("vta.")]
print(f"已注册 {len(names)} 个 vta FFI 函数:")
for name in sorted(names):
    print(f"  - {name}")

# 验证关键函数是否存在
required = [
    "vta.tls_command_handle", "vta.buffer_alloc", "vta.buffer_free",
    "vta.synchronize", "vta.write_barrier", "vta.read_barrier",
]
for func in required:
    assert tvm_ffi.get_global_func(func) is not None, f"缺失 FFI 函数: {func}"
```

也可以使用前缀检查脚本：
```bash
python scripts/check_ffi_prefix.py --verbose
```

---

### Q: Buffer 操作后结果不正确

**症状**：通过 buffer_cpu_ptr 获取指针写入数据后，VTA 计算结果不正确。

**原因**：缺少内存屏障（barrier）。CPU 写入数据后必须调用 `write_barrier` 确保数据对 VTA 可见；VTA 计算完成后必须调用 `read_barrier` 确保 CPU 能读到最新数据。

**解决**：
```python
from npu_ffi.vta import Buffer

with CommandContext() as cmd:
    buf = Buffer(1024)
    ptr = buf.cpu_ptr(cmd)

    # CPU 写入数据
    # ... 写入 ptr 指向的内存 ...

    # 关键：写屏障，确保 VTA 能看到 CPU 写入的数据
    buf.write_barrier(cmd, elem_bits=32, start=0, extent=buf.size // 4)

    # ... 执行 VTA 操作 ...

    # 关键：读屏障，确保 CPU 能看到 VTA 写入的结果
    buf.read_barrier(cmd, elem_bits=32, start=0, extent=buf.size // 4)

    # 现在可以安全读取结果
```

---

## 测试相关问题

### Q: pytest 显示 "collected 0 items"

**原因**：
1. 不在正确的目录下运行
2. 测试文件命名不符合 pytest 约定（应以 `test_` 开头）
3. conftest.py 中的导入错误导致测试收集失败

**解决**：
```bash
# 在 npu-ffi 项目根目录运行
cd path/to/libs/npu-ffi
pytest tests/python -v

# 如果收集失败，查看具体错误
pytest tests/python -v --tb=long
```

---

### Q: pytest 中所有 vta 相关测试都 SKIPPED

**症状**：测试运行但标记为 SKIPPED，提示 "C++ extension not available"。

**原因**：npu-ffi 的 C++ 扩展（_ffi_api 原生模块）未能正确加载，测试在纯 Python stub 模式下运行，跳过了需要真实 FFI 调用的测试。

**解决**：
1. 运行验证脚本确认 FFI 模块加载状态：
   ```bash
   python scripts/verify_install.py
   ```
2. 检查 "vta FFI 模块导入" 项是否通过
3. 如果失败，参考 [DLL 加载失败](#q-导入时出现-oserror--dll-加载失败windows) 的解决方案

---

### Q: 运行测试时出现 SIGABRT / 程序直接崩溃

**症状**：某个测试运行时 Python 直接崩溃，没有 traceback，可能显示 SIGABRT。

**原因**：通常是 C++ 层的 `CHECK` 宏断言失败（如参数校验失败），触发 `abort()`。C++ 的 abort 无法被 Python 的 try-except 捕获。

**解决**：
1. 检查传入 FFI 函数的参数是否合法（如 buffer 大小不能为 0，指针不能为 null）
2. 检查 Reshape 等操作的维度参数是否匹配
3. 在调用 C++ FFI 函数前，Python 层应做参数校验（参见 `npu_ffi.vta._ffi_api` 中的校验逻辑）

---

## 环境管理

### Q: `conda run -n py314` 中环境变量不生效

**症状**：使用 `conda run -n py314` 执行命令时，`KMP_DUPLICATE_LIB_OK` 等环境变量没有生效。

**原因**：`conda run` 默认在干净的子进程中执行，不继承当前 shell 的环境变量。

**解决**：
```powershell
# 方式1：使用 cmd /c 在子进程中设置环境变量
conda run -n py314 --no-capture-output cmd /c "set KMP_DUPLICATE_LIB_OK=TRUE && python scripts/verify_install.py"

# 方式2：先激活环境再执行（推荐用于交互式开发）
conda activate py314
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python scripts/verify_install.py
pytest tests/python -v
```

> 注意：pytest 通过 conftest.py 自动设置 `KMP_DUPLICATE_LIB_OK=TRUE`，所以运行测试时无需手动设置。

---

### Q: 如何确认当前使用的是哪个 Python 环境？

```bash
# 查看当前 Python 路径和版本
python -c "import sys; print(sys.executable); print(sys.version)"

# 查看已安装的 npu-ffi 位置和版本
pip show npu-ffi
pip show apache-tvm-ffi

# 列出所有 conda 环境
conda env list

# 查看当前环境安装的包
conda list | grep -E "tvm|npu|protobuf"
```

---

### Q: 如何在多个 conda 环境中切换使用 npu-ffi？

npu-ffi 的 editable 安装是环境隔离的，每个 conda 环境需要独立安装：

```bash
# 在 py314 环境中安装
conda activate py314
pip install --no-build-isolation -e path/to/vendor/tvm-ffi
pip install --no-build-isolation -e path/to/libs/npu-ffi

# 如果还有其他环境（如 npu-ffi-dev），也要重新安装
conda activate npu-ffi-dev
pip install --no-build-isolation -e path/to/vendor/tvm-ffi
pip install --no-build-isolation -e path/to/libs/npu-ffi
```

editable 模式下 C++ 扩展的构建产物（build 目录）是共享的，但每个环境的 Python 包元数据是独立的。

---

### Q: 如何完全卸载 npu-ffi 重新安装？

```bash
conda activate py314

# 卸载 npu-ffi
pip uninstall npu-ffi -y

# 卸载 tvm-ffi（如果需要）
pip uninstall apache-tvm-ffi -y

# 清理构建产物
cd path/to/libs/npu-ffi
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force *.egg-info -ErrorAction SilentlyContinue

# 重新安装（顺序很重要！）
pip install --no-build-isolation -e path/to/vendor/tvm-ffi
pip install --no-build-isolation -e .

# 验证
python scripts/verify_install.py
```

---

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
