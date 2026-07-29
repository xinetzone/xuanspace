# CMake 模块说明

本目录包含 caffe-ffi 项目的原子化 CMake 构建模块，每个模块职责单一，按固定顺序被根 `CMakeLists.txt` 引用。

## 模块清单

| 模块文件 | 职责 | 提供的变量/函数 | 依赖模块 |
|---------|------|----------------|---------|
| **Options.cmake** | 构建选项定义 | `CAFFE_FFI_ENABLE_DEBUG_LOG`、`CAFFE_FFI_ENABLE_BACKTRACE` 等选项变量 | 无 |
| **DetectBLAS.cmake** | BLAS/OpenBLAS 检测 | `BLAS_FOUND`、`BLAS_LIBRARIES`、`BLAS_INCLUDE_DIRS` | 无 |
| **Dependencies.cmake** | 第三方依赖查找 | tvm_ffi、Protobuf、Threads、Python 依赖配置 | DetectBLAS（内部include） |
| **CompilerConfig.cmake** | 公共编译配置函数 | `caffe_ffi_configure_target(target VISIBILITY <PUBLIC\|PRIVATE\|INTERFACE>)` | Dependencies（使用其查找的依赖变量） |
| **ProtoCompile.cmake** | Protobuf 文件编译 | `CAFFE_FFI_PROTO_SRCS`、`CAFFE_FFI_GEN_PROTO_DIR` | Dependencies |
| **TargetBuild.cmake** | 主库 `_caffe_ffi` 构建 | `_caffe_ffi` 共享库目标 | CompilerConfig、ProtoCompile |
| **WindowsDllCopy.cmake** | Windows DLL 复制 | `caffe_ffi_copy_runtime_dlls()`、`caffe_ffi_copy_target_dll()` 等函数；自动配置主库DLL复制 | TargetBuild（主库目标存在后） |
| **Tests.cmake** | C++ 单元测试配置 | `caffe_ffi_tests` 可执行目标、CTest 注册 | CompilerConfig、TargetBuild、WindowsDllCopy |
| **Install.cmake** | 安装规则配置 | install 目标 | TargetBuild |

## Include 顺序（根 CMakeLists.txt）

```cmake
include(Options)         # 1. 首先定义构建选项
include(Dependencies)    # 2. 查找第三方依赖（内部include DetectBLAS）
include(CompilerConfig)  # 3. 定义公共编译配置函数（必须在TargetBuild/Tests之前）
include(ProtoCompile)    # 4. 编译Proto文件
include(TargetBuild)     # 5. 构建主库目标
include(WindowsDllCopy)  # 6. Windows DLL复制配置（必须在TargetBuild之后）
include(Tests)           # 7. 构建测试目标（必须在TargetBuild之后）
include(Install)         # 8. 最后配置安装规则
```

**顺序约束原因**：
- `CompilerConfig` 必须在 `TargetBuild` 和 `Tests` 之前，因为这两个模块调用它提供的函数
- `WindowsDllCopy` 必须在 `TargetBuild` 之后，因为它需要为 `_caffe_ffi` 目标配置 POST_BUILD 命令
- `Tests` 必须在 `TargetBuild` 之后，因为测试目标链接 `_caffe_ffi`

## 公共函数使用说明

### caffe_ffi_configure_target

统一设置目标的 include 目录、编译定义、编译选项、链接库：

```cmake
caffe_ffi_configure_target(&lt;target_name&gt; VISIBILITY &lt;PUBLIC|PRIVATE|INTERFACE&gt;)
```

**参数**：
- `target_name`: 要配置的 CMake 目标名称（必须已通过 add_library/add_executable 创建）
- `VISIBILITY`: 目标属性的可见性
  - `PUBLIC`：主库使用，属性传递给链接者
  - `PRIVATE`：测试/可执行文件使用，属性不传递
  - `INTERFACE`：仅INTERFACE目标使用

**自动配置的内容**：
- Include 目录：`CAFFE_FFI_INCLUDE_DIR`、`CAFFE_FFI_GEN_PROTO_DIR`、`Protobuf_INCLUDE_DIRS`、条件添加 `BLAS_INCLUDE_DIRS`
- 编译定义：`CPU_ONLY`（条件）、`CAFFE_FFI_VERSION`，条件添加调试/回溯/BLAS相关宏
- 编译选项：MSVC `/W3`，GCC/Clang `-Wall -Wextra -Wno-unused-parameter`
- 链接库：`protobuf::libprotobuf`、`Threads::Threads`，条件添加 BLAS 和 DbgHelp.lib

**错误提示**：函数内置参数校验，以下情况会给出友好的 FATAL_ERROR：
- 未提供 `target_name`
- `VISIBILITY` 参数值无效（不是 PUBLIC/PRIVATE/INTERFACE）
- 目标不存在（未提前创建）

### DLL 复制函数（仅 MSVC）

```cmake
# 复制单个DLL（如果存在）
caffe_ffi_copy_dll_if_exists(&lt;target&gt; &lt;dll_path&gt;)

# 复制 tvm_ffi 共享库
caffe_ffi_copy_tvm_ffi_dll(&lt;target&gt;)

# 复制 OpenBLAS DLLs
caffe_ffi_copy_openblas_dlls(&lt;target&gt;)

# 复制 Protobuf DLLs
caffe_ffi_copy_protobuf_dlls(&lt;target&gt;)

# 复制 abseil DLLs
caffe_ffi_copy_abseil_dlls(&lt;target&gt;)

# 复制 utf8_range DLLs
caffe_ffi_copy_utf8_dlls(&lt;target&gt;)

# 聚合函数：复制所有运行时依赖 DLLs
caffe_ffi_copy_runtime_dlls(&lt;target&gt;)

# 复制依赖目标的DLL（如_caffe_ffi.dll复制到测试目录）
caffe_ffi_copy_target_dll(&lt;target&gt; &lt;dependency_target&gt;)
```

## 扩展指南

添加新模块时：
1. 确保模块职责单一，文件不超过 80 行（特殊情况如 WindowsDllCopy.cmake 因平台适配可放宽）
2. 在本 README 中添加模块说明
3. 在根 `CMakeLists.txt` 中按正确顺序添加 `include()`
4. 如果提供公共函数，在"公共函数使用说明"章节添加文档
5. **公共函数必须添加参数校验**：
   - 检查必需参数是否提供
   - 检查枚举参数值合法性（如 VISIBILITY 必须是 PUBLIC/PRIVATE/INTERFACE）
   - 检查 TARGET 是否存在（如果函数操作已定义的目标）
   - 使用 `message(FATAL_ERROR ...)` 给出清晰的错误信息，包含函数名、用法示例

## 原子化原则

每个模块遵循以下原则：
- **单一职责**：一个模块只做一件事
- **显式依赖**：通过 include 顺序和函数参数明确依赖关系
- **消除重复**：公共逻辑抽象为函数供多个模块复用
- **自包含注释**：文件头说明模块用途和提供的函数/变量
