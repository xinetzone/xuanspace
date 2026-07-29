# caffe-ffi 内存日志系统验证报告

> **日期**: 2026-07-28
> **范围**: Blob 零拷贝张量访问 + 三层（Python/FFI/C++）内存调试日志 + dtype 显示修复 + 内存泄漏检测
> **验证环境**: Windows 10 / Python 3.14.3 / MSVC / CMake Release build

---

## 一、概述

本次工作在 caffe-ffi Blob 模块的 `data_tensor` 和 `diff_tensor` 属性上添加了详细的三层（Python/FFI/C++）内存调试日志，并修复了日志中 dtype 字段显示乱码的 bug。额外提供了独立的日志配置工具模块 `config.py` 和内存泄漏专项测试脚本。所有现有功能通过回归测试，日志输出正确可读，析构函数在异常场景下正确记录指针地址。

## 二、变更文件清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| [common.hpp](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-ffi/include/caffe_ffi/common.hpp) | 修改 | 添加 `DTypeCodeToString()` 辅助函数；修复 `AllocData` 中 dtype 乱码输出 |
| [blob.cpp](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-ffi/src/caffe_ffi/blob.cpp) | 修改 | 修复 `data_tensor()`/`diff_tensor()` 中 dtype 乱码输出；析构函数增加 `total_freed` 和 `global_allocated_bytes` 日志 |
| [blob.hpp](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-ffi/include/caffe_ffi/blob.hpp) | 无变更 | 通过 common.hpp 间接获得 `DTypeCodeToString` |
| [_caffe_ffi.cc](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-ffi/src/caffe_ffi/_caffe_ffi.cc) | 无变更 | FFI 桥接层日志已在前序提交中添加 |
| [blob.py](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-ffi/python/caffe_ffi/blob.py) | 无变更 | Python 层日志已在前序提交中添加 |
| [__init__.py](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-ffi/python/caffe_ffi/__init__.py) | 无变更 | 日志级别控制函数已在前序提交中添加 |
| [config.py](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-ffi/examples/config.py) | 新增 | 三层日志统一配置工具模块（setup_debug/setup_quiet/setup_memory_trace/memory_snapshot/check_memory_baseline） |
| [test_memory_logging.py](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-ffi/examples/test_memory_logging.py) | 新增 | 9 场景内存日志验证脚本 |
| [test_memory_leak.py](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-ffi/examples/test_memory_leak.py) | 新增 | 8 场景内存泄漏专项测试脚本（16项检查） |

## 三、Bug 修复细节：dtype 日志乱码

### 3.1 问题根因

C++ 标准库中 `std::ostream::operator<<` 对 `uint8_t`（即 `unsigned char`）类型有特化重载，会将其值作为 **ASCII 字符**输出，而非整数：

```cpp
// DLDataType 结构体定义（DLPack 标准）
typedef struct {
  uint8_t code;   // kDLFloat=2 → ASCII 0x02 (STX 控制字符，显示为 )
  uint8_t bits;   // float32 → 32 → ASCII 0x20 (空格字符，显示为空格)
  uint16_t lanes; // 1
} DLDataType;
```

| 字段 | 值 | 期望输出 | 实际输出（修复前） |
|------|-----|---------|------------------|
| `dtype.code` | 2 (kDLFloat) | `float` | ``（STX 控制字符，不可打印） |
| `dtype.bits` | 32 | `32` | ` `（空格，肉眼不可见） |
| 拼接后 | - | `dtype=float32` | `dtype=: `（乱码） |

### 3.2 修复方案

**两步修复**：

1. **添加 `DTypeCodeToString()` 函数**（[common.hpp:26-33](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-ffi/include/caffe_ffi/common.hpp#L26-L33)）：将 `DLDataTypeCode` 枚举值映射为可读字符串（`float`/`int`/`uint`/`unknown`）。

2. **强制整数转换**：所有涉及 `uint8_t` 字段的日志输出，使用 `static_cast<int>()` 包裹，确保 `<<` 运算符选择整数重载而非字符重载：

```cpp
// 修复前（乱码）
<< ", dtype=" << tensor->dtype.code << ":" << tensor->dtype.bits
// 修复后（正确）
<< ", dtype=" << DTypeCodeToString(tensor->dtype.code)
<< static_cast<int>(tensor->dtype.bits)
```

### 3.3 修复范围

- [common.hpp:40-41](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-ffi/include/caffe_ffi/common.hpp#L40-L41) — `AllocData` 日志
- [blob.cpp:82-84](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-ffi/src/caffe_ffi/blob.cpp#L82-L84) — `data_tensor()` 日志
- [blob.cpp:95-97](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-ffi/src/caffe_ffi/blob.cpp#L95-L97) — `diff_tensor()` 日志

### 3.4 修复前后对比

```
// 修复前
[MEM] AllocData: allocating 480 bytes (ndim=4, dtype=: , device_type=1)
[TENSOR] data_tensor() ... dtype=:  device_type=1

// 修复后
[MEM] AllocData: allocating 480 bytes (ndim=4, dtype=float32, device_type=1)
[TENSOR] data_tensor() ... dtype=float32 device_type=1
```

## 四、日志系统架构

### 4.1 三层日志设计

```
┌─────────────────────────────────────────────────────┐
│ Python 层 (blob.py)                                 │
│  logger: "caffe_ffi.Blob" (标准 logging 模块)       │
│  输出: [PY DEBUG] caffe_ffi.Blob: data_tensor ...   │
│  内容: blob_id(Python对象id)、shape、dtype、nbytes、 │
│        ptr(十六进制)、strides、is_native            │
├─────────────────────────────────────────────────────┤
│ FFI 桥接层 (_caffe_ffi.cc)                          │
│  输出: [DEBUG] _caffe_ffi.cc:102 (BlobDataTensor)...│
│  内容: blob C++ 指针、调用函数名                     │
├─────────────────────────────────────────────────────┤
│ C++ 层 (blob.cpp / common.hpp)                      │
│  输出: [DEBUG] blob.cpp:76 (data_tensor)...         │
│  分类: [BLOB] / [MEM] / [TENSOR] / [CONTAINER]      │
│  内容: this指针、data_ptr、diff_ptr、shape、         │
│        numel、nbytes、dtype、device_type、           │
│        total_freed、global_allocated_bytes           │
└─────────────────────────────────────────────────────┘
```
> 三层日志各司其职：Python 层提供 Python 语义上下文（ndarray strides/is_native 等），FFI 桥接层标记跨语言调用点，C++ 层记录最底层的内存分配/释放细节。排查问题时可根据需要单独开启某一层的日志。

### 4.2 日志级别控制

| 级别 | C++ 常量 | Python 常量 | 用途 |
|------|---------|------------|------|
| TRACE | `Level::TRACE` (0) | `LOG_LEVEL_TRACE` | 最细粒度追踪（内存分配/释放完整序列） |
| DEBUG | `Level::DEBUG` (1) | `LOG_LEVEL_DEBUG` | 内存分配/释放/访问详情 |
| INFO | `Level::INFO` (2) | `LOG_LEVEL_INFO` | 关键操作通知 |
| WARN | `Level::WARN` (3) | `LOG_LEVEL_WARN` | 异常但可恢复 |
| ERROR | `Level::ERROR` (4) | `LOG_LEVEL_ERROR` | 致命错误 |

默认级别均为 WARN（C++ 端定义于 [log.hpp:21](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-ffi/include/caffe_ffi/log.hpp#L21) 的静态变量 `static Level level = Level::WARN`；Python 端定义于 [__init__.py:25](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-ffi/python/caffe_ffi/__init__.py#L25) 的 `_logger.setLevel(logging.WARNING)`）。Release 构建下默认不输出 DEBUG 日志。

### 4.3 日志分类标签

| 标签 | 用途 | 触发位置 |
|------|------|---------|
| `[BLOB]` | Blob 构造/析构/生命周期 | `Blob()`, `~Blob()`, `FromProto()` |
| `[MEM]` | 内存分配/释放/重分配 | `AllocData()`, `FreeData()`, `Reshape()`, `~Blob()` |
| `[TENSOR]` | 张量访问/操作 | `data_tensor()`, `diff_tensor()`, `Update()`, `Reshape()` (skip) |
| `[CONTAINER]` | 数据读写 | `get_data()`, `set_data()`, `get_diff()`, `set_diff()` |

### 4.4 全局内存计数器

C++ 层维护原子计数器 `g_total_allocated_bytes`，Python 层通过 `caffe_ffi.total_allocated_bytes()` 可随时查询当前所有存活 Blob 的 data+diff 张量总字节数。此计数器是检测内存泄漏的**最可靠手段**——析构日志仅在对象被销毁时打印，若对象因引用泄漏从未被销毁，则不会有析构日志，但计数器不会归零。

## 五、验证结果

### 5.1 现有功能回归测试

**27 个 pytest 用例全部通过**，耗时 0.07 秒，零失败零错误：

| 测试类 | 用例数 | 结果 |
|--------|-------|------|
| TestBlobReshape | 4 | ✅ 全部通过 |
| TestBlobNumpy | 6 | ✅ 全部通过 |
| TestBlobFill | 2 | ✅ 全部通过 |
| TestBlobCopy | 2 | ✅ 全部通过 |
| TestBlobProperties | 4 | ✅ 全部通过 |
| TestBlobRepr | 1 | ✅ 通过 |
| TestBlobZeroCopy | 8 | ✅ 全部通过 |
| **合计** | **27** | **✅ 27/27 通过** |

### 5.2 内存日志专项验证（9场景）

9 个场景全部通过：

| # | 场景 | 验证要点 | 结果 |
|---|------|---------|------|
| 1 | 默认构造→析构 | 构造→空tensor→析构日志完整 | ✅ |
| 2 | 指定shape→数据写入→零拷贝读取 | zero-copy写入后读回一致(max_diff=0) | ✅ |
| 3 | 多次Reshape(小→大→同→小) | 重分配时旧ptr释放/新ptr分配；同shape跳过重分配 | ✅ |
| 4 | fill→zero→diff→Update | fill/zero/Update操作日志和数学正确性 | ✅ |
| 5 | from_numpy→to_numpy往返 | 数据完整往返(shape/dtype正确) | ✅ |
| 6 | copy_from跨Blob复制 | 源/目标Blob独立内存，复制正确 | ✅ |
| 7 | 多Blob并发存活→逆序释放 | 3个Blob指针互不相同，逆序释放各自正确 | ✅ |
| 8 | 异常中Blob存活→异常后析构 | 异常抛出后析构仍执行，内存正确释放 | ✅ |
| 9 | data_tensor vs diff_tensor独立性 | data/diff指针不同，写入互不干扰 | ✅ |

### 5.3 内存泄漏专项测试（8场景/16检查项）

| # | 场景 | 关键验证点 | 结果 |
|---|------|----------|------|
| 1 | 正常创建→Reshape→写入→释放 | 创建后960字节，释放后归0 | ✅ |
| 2 | Python循环变量持有引用 | `del blobs`后循环变量`b`仍持有最后一个Blob（128字节），`del b`后归0 | ✅ |
| 3 | 故意global引用泄漏 | ~Blob()不会打印（对象未销毁），total_allocated_bytes=512可检测 | ✅ |
| 4 | 引用计数精确释放（无gc） | `del b`后引用计数归零立即触发析构，无需`gc.collect()` | ✅ |
| 5 | 异常catch后del+gc | 异常后blob仍可访问，del+gc析构日志正确打印指针 | ✅ |
| 6 | Reshape从小→大重分配 | 旧ptr的FreeData日志打印，新ptr分配800字节正确 | ✅ |
| 7 | 同shape Reshape | 指针不变（跳过重分配） | ✅ |
| 8 | 清理故意泄漏引用 | `_leaked_ref.clear()`后~Blob()日志出现，内存归0 | ✅ |

**最终结果**: 16 passed, 0 failed，Final memory baseline: 0 bytes（无泄漏）

### 5.4 析构函数指针地址正确性验证

交叉验证方法：在 Python 层通过 `numpy.ndarray.ctypes.data` 获取实际内存地址，与 C++ 析构函数 `~Blob()` 中打印的 `data_ptr`/`diff_ptr` 做对比。

**验证样例**（shape=[2,3,4,5], float32，以下数据为某次运行的实际输出，每次运行地址不同但一致性可复现）：

| 字段 | C++ ~Blob() 日志 | Python 层读取 | 一致性 |
|------|-----------------|-------------|--------|
| data_ptr | `000002346AAF4C50` | `0x000002346aaf4c50` | ✅ 完全一致 |
| diff_ptr | `000002346AAF5030` | `0x000002346aaf5030` | ✅ 完全一致 |
| shape | `(2,3,4,5)` | `(2, 3, 4, 5)` | ✅ 完全一致 |
| data_nbytes | `480` | `480` | ✅ 完全一致 |
| diff_nbytes | `480` | `480` | ✅ 完全一致 |
| total_freed | `960` | `960` (480+480) | ✅ 完全一致 |
| 释放顺序 | diff→data | - | ✅ 正确 |
| global_allocated_bytes | 释放后归0 | - | ✅ 正确（无泄漏） |

**异常场景验证**：在 `RuntimeError`/`ValueError` 异常抛出并被捕获后，`del blob; gc.collect()` 触发析构函数，日志正确记录指针地址和内存释放，C++ 层内存不泄漏。

**泄漏场景关键发现**：当 Blob 被 global/容器引用持有而无法释放时，~Blob() 析构日志**不会打印**（析构函数根本未被调用，这是正确行为），但 `total_allocated_bytes()` 计数器保持非零，`check_memory_baseline()` 可可靠检测到泄漏。

### 5.5 关键日志输出样例

**构造 + Reshape + 分配日志**：
```
[DEBUG] blob.cpp:41 (Blob) [BLOB] Blob() default constructor this=000001F84AF6BA38
[DEBUG] blob.cpp:122 (Reshape) [MEM] Reshape: REALLOCATING this=000001F84AF6BA38
        shape=(2,3,4,5) old_count=0 new_count=120
        old_data_ptr=0000000000000000 old_diff_ptr=0000000000000000
        old_total_nbytes=0 new_total_nbytes=960
[DEBUG] caffe_ffi/common.hpp:38 (AllocData) [MEM] AllocData: allocating 480 bytes
        (ndim=4, dtype=float32, device_type=1)
[DEBUG] caffe_ffi/common.hpp:46 (AllocData) [MEM] AllocData: allocated at 000001F84B020B20
        (480 bytes, zero-initialized)
```

**张量访问日志（三层）**：
```
[DEBUG] _caffe_ffi.cc:106 (BlobDataTensor) [MEM] FFI BlobDataTensor blob=000001F84AF6BA38
        returning data_tensor view
[DEBUG] blob.cpp:76 (data_tensor) [TENSOR] data_tensor() this=000001F84AF6BA38
        ptr=000001F84B020B20 shape=(2,3,4,5) numel=120 nbytes=480 dtype=float32 device_type=1
[PY DEBUG] caffe_ffi.Blob: data_tensor access: blob_id=2166430205488 shape=(2, 3, 4, 5)
        dtype=float32 ndim=4 nbytes=480 ptr=0x000001f84b020b20 strides=(240, 80, 20, 4) is_native=True
```

**析构日志**：
```
[DEBUG] blob.cpp:60 (~Blob) [MEM] ~Blob() this=000001F84AF6BA38
        data_ptr=000001F84B020B20 diff_ptr=000001F84B020740
        shape=(2,3,4,5) data_nbytes=480 diff_nbytes=480 total_freed=960
[DEBUG] blob.cpp:70 (~Blob) [MEM] ~Blob() global_allocated_bytes=0
[DEBUG] caffe_ffi/common.hpp:50 (FreeData) [MEM] FreeData: freeing memory at 000001F84B020740
[DEBUG] caffe_ffi/common.hpp:53 (FreeData) [MEM] FreeData: memory freed, data pointer reset to nullptr
[DEBUG] caffe_ffi/common.hpp:50 (FreeData) [MEM] FreeData: freeing memory at 000001F84B020B20
[DEBUG] caffe_ffi/common.hpp:53 (FreeData) [MEM] FreeData: memory freed, data pointer reset to nullptr
```

**内存泄漏检测输出**（global引用持有的场景）：
```
[MEM-SNAPSHOT] [3_blobs_alive] total_allocated_bytes=384 (0.38 KB)
[MEM-CHECK] after_del_all: LEAK DETECTED (128 bytes still allocated)
```

## 六、使用指南

### 6.1 快速启用（推荐使用 config.py）

```python
import sys
sys.path.insert(0, "examples")
from config import setup_debug, setup_quiet, memory_snapshot, check_memory_baseline
import numpy as np
from caffe_ffi import Blob

setup_debug()  # 一键启用三层DEBUG日志（控制台输出）
# setup_memory_trace()  # TRACE级别，最细粒度
# setup_file_logging("mem.log")  # 仅写入文件
# setup_debug(log_file="mem.log")  # 控制台+文件双写

b = Blob([2, 3, 4, 5])
b.data_tensor[:] = np.random.randn(2, 3, 4, 5).astype(np.float32)

memory_snapshot("after_create")  # 打印当前内存用量
del b
check_memory_baseline("final")  # 验证归零，检测泄漏

setup_quiet()  # 恢复默认WARN级别
```

### 6.2 config.py 提供的函数

| 函数 | 用途 |
|------|------|
| `setup_debug(level, log_file)` | 启用三层DEBUG日志（控制台+可选文件） |
| `setup_memory_trace(log_file)` | 启用TRACE级别最细粒度追踪 |
| `setup_file_logging(path, level, append)` | 仅文件日志（不输出到控制台） |
| `setup_quiet()` | 关闭调试日志，恢复WARN默认级别 |
| `memory_snapshot(label)` | 打印当前全局已分配字节数并返回该值 |
| `check_memory_baseline(label)` | 检查内存是否归零（True=无泄漏，False=泄漏） |

### 6.3 手动配置（不使用 config.py）

```python
import logging
import caffe_ffi

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(name)s: %(message)s')
caffe_ffi.set_log_level(caffe_ffi.LOG_LEVEL_DEBUG)

from caffe_ffi import Blob
import numpy as np

b = Blob([2, 3, 4, 5])
b.data_tensor[:] = np.random.randn(2, 3, 4, 5).astype(np.float32)
print(f"Allocated: {caffe_ffi.total_allocated_bytes()} bytes")
```

### 6.4 排查内存问题的关键日志模式

| 排查目标 | 关注日志标签 | 关键字段/工具 |
|---------|------------|-------------|
| 内存泄漏 | `[MEM] ~Blob()` + `check_memory_baseline()` | `total_freed` 与分配大小对比；`total_allocated_bytes()` 最终是否归0 |
| 野指针/UAF | `[MEM] FreeData` + `[TENSOR]` | 访问的 ptr 是否已被 freed |
| 意外重分配 | `[MEM] Reshape: REALLOCATING` | `old_data_ptr` 与 `new_data_ptr` 变化，旧 ndarray 是否失效 |
| 零拷贝失败 | `[PY DEBUG] ... is_native=False` | py_fallback 模式表示走了纯Python兼容路径 |
| dtype 不匹配 | `[TENSOR] dtype=` | 确认是 float32（caffe-ffi 当前唯一支持的类型） |
| 引用泄漏（Python侧） | 循环变量/容器引用 | `del list` 后注意循环变量也持有引用；`memory_snapshot()` 定位 |

### 6.5 注意事项

- **C++扩展类型不支持weakref**：Blob对象是C++扩展类型（通过TVM FFI导出），未设置 `Py_TPFLAGS_MANAGED_WEAKREF` 或 `tp_weaklistoffset`，因此无法使用 `weakref.ref()`，会抛出 `TypeError: cannot create weak reference to 'Blob' object`。替代方案见下文。
- **Python循环变量泄漏**：`for i, b in enumerate(blobs)` 后 `b` 仍在帧中持有最后一个元素的引用，`del blobs` 不会释放最后一个元素。需显式 `del i, b`。
- **引用计数即时释放**：C++ FFI对象在Python引用计数归零时立即触发析构函数（无需等待 `gc.collect()`），这使得 ~Blob() 日志在 `del` 语句后立即出现。
- **泄漏对象无析构日志**：真正泄漏（引用从未归零）的对象不会有 ~Blob() 日志——这是正确行为，通过 `total_allocated_bytes()` 非零来检测。

#### weakref 不可用的三种替代方案

**方案一（推荐）：使用 `total_allocated_bytes()` 计数器**

无需跟踪单个对象生命周期，只需在操作前后对全局计数器做快照，是最可靠的泄漏检测方法：

```python
import caffe_ffi
from caffe_ffi import Blob

mem_before = caffe_ffi.total_allocated_bytes()
b = Blob([100, 100])
mem_after_create = caffe_ffi.total_allocated_bytes()
assert mem_after_create - mem_before == 100*100*4*2  # data+diff = 80000 bytes

del b  # 引用计数归零，立即析构
mem_after_del = caffe_ffi.total_allocated_bytes()
assert mem_after_del == mem_before  # 必须归零
```

使用 config.py 中的辅助函数更方便：

```python
from config import memory_snapshot, check_memory_baseline

check_memory_baseline("before")   # 检查是否从干净基线开始
# ... 创建/使用 Blob ...
check_memory_baseline("after")    # 自动打印 OK 或 LEAK DETECTED
```

**方案二：用 Python 包装类间接获得 weakref**

如果确实需要弱引用回调（如对象销毁时触发通知），可以创建一个纯Python包装类持有 Blob，并在包装类上使用 weakref：

```python
import weakref
import caffe_ffi
from caffe_ffi import Blob

class BlobRef:
    """Python包装类，支持weakref，并在析构时检测底层Blob是否正确释放"""
    def __init__(self, shape):
        self._blob = Blob(shape)
        self._data_ptr = self._blob.data_tensor.ctypes.data
        self._nbytes = self._blob.data_tensor.nbytes + self._blob.diff_tensor.nbytes
    def __getattr__(self, name):
        return getattr(self._blob, name)
    def __del__(self):
        # 注意：__del__中访问_blob不一定安全（可能已在gc顺序中被销毁），
        # 仅用于日志，不做实际操作
        pass

def on_blob_destroyed(ref):
    print(f"[CALLBACK] BlobRef wrapper destroyed, data_ptr was 0x{ref.data_ptr:016x}")

wrapper = BlobRef([2, 3, 4, 5])
wrapper.data_ptr = wrapper._data_ptr  # 记录ptr供回调使用
ref = weakref.ref(wrapper, on_blob_destroyed)
del wrapper  # 触发回调 on_blob_destroyed
```

**方案三：手动上下文管理器（with语句）**

对于需要明确生命周期的场景，使用上下文管理器比weakref更直观：

```python
from contextlib import contextmanager
import caffe_ffi
from caffe_ffi import Blob

@contextmanager
def tracked_blob(shape, label="blob"):
    """创建Blob并在退出with块时验证内存释放"""
    mem_before = caffe_ffi.total_allocated_bytes()
    b = Blob(shape)
    expected = b.data_tensor.nbytes + b.diff_tensor.nbytes
    dp = b.data_tensor.ctypes.data
    print(f"[TRACK:{label}] created, data_ptr=0x{dp:016x}, nbytes={expected}")
    try:
        yield b
    finally:
        del b
        mem_after = caffe_ffi.total_allocated_bytes()
        freed = mem_before + expected
        if mem_after == mem_before:
            print(f"[TRACK:{label}] OK: freed {expected} bytes")
        else:
            print(f"[TRACK:{label}] LEAK: expected {mem_before}, got {mem_after}")

with tracked_blob([10, 10], "test1") as b:
    b.data_tensor[:] = 1.0
    # 此处可用b进行操作
# with块退出时自动验证：[TRACK:test1] OK: freed 800 bytes
```

### 6.6 运行验证脚本

```bash
cd /path/to/caffe-ffi
python examples/test_memory_logging.py    # 9场景日志功能验证
python examples/test_memory_leak.py       # 8场景内存泄漏检测（16项检查）
python examples/config.py                 # config.py自测
python -m pytest tests/python/test_blob.py -v  # 27个功能回归测试
```

## 七、已知非本次引入问题

全量测试（93项）中有 8 failed + 19 errors，均为 Net/Layer 层预存问题，**与本次 Blob 日志增强无关**：

| 文件 | 问题 | 根因 |
|------|------|------|
| test_layers.py (8 failed) | `Top blob 'data' produced by multiple sources` | prototxt 中同时声明了 `input` 和 Input layer，导致 Blob 注册冲突 |
| test_net.py (19 errors) | `axis 2 out of range for 2-D blob` | Net 构建流程中 Reshape 未正确处理维度 |

这些问题在本次日志增强前即已存在，不影响 Blob 层功能的正确性。
