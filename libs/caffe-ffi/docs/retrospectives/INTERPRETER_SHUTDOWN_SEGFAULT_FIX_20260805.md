---
source: "segfault at interpreter shutdown when caffe-ffi static callback registries are populated"
date: "2026-08-05"
status: "completed"
tags: ["segfault", "static-destructor", "atexit", "python-runtime-lifecycle", "tdm-ffi", "caffe-ffi"]
---

# Caffe-FFI 解释器退出 SIGSEGV 修复复盘

> 场景：问题解决（场景 2，I→F→V→C 链路）
> 质量门：G4 通过（原子化交付，已提交至 xuanspace@c705ea5 + SpecWeave@80f07f55）

## 1. 现象描述

在 P0 环境（WSL Docker 容器 `caffe-ffi-jupyter`）中运行 P2 算子测试或自定义脚本，只要通过 Python 侧向 `data_io` 或 `python_layer` 任一静态注册中心注册了回调，**Python 解释器退出时即触发 `Segmentation fault (core dumped)`**，进程以 exit code 139 终止。

### 复现条件

1. 启动 Python 解释器
2. `import caffe_ffi`（导入 package，此时尚未触发问题）
3. 通过 `_ffi_api.get_global_func("caffe_ffi.data_io.register")` 注册任意回调
4. 让解释器自然退出（或 `sys.exit(0)`）
5. 触发 SIGSEGV

```
$ python .temp/repro_segfault.py
Registered callbacks into data_io + python_layer registries.
Exiting interpreter (triggers atexit cleanup + static destruction)...
Segmentation fault (core dumped)
```

## 2. 根因分析（第一性原理）

### 2.1 静态对象析构时序问题

Caffe-FFI 的两个静态回调注册中心使用以下惯用法：

```cpp
// data_io_bridge.cpp / python_layer.cpp
std::unordered_map<std::string, Function>& DataIOCallbackRegistry() {
  static std::unordered_map<std::string, Function> registry;  // 函数内静态
  return registry;
}
```

这种 Meyer's Singleton 模式在函数首次调用时构造，生命周期与进程相同。存储在 `unordered_map` 中的 `Function` 对象是 **TVM FFI 句柄**，其内部持有对 Python 可调用对象的引用。

### 2.2 析构顺序的因果倒置

C++ 标准保证**函数内静态对象的析构按构造的逆序**进行，但**跨翻译单元 / 跨动态库**的析构顺序不受保证（FIFO 队列未规定全局顺序）。关键冲突：

| 销毁方 | 销毁时间点 | 后果 |
|--------|-----------|------|
| Python 解释器运行时 | `Py_Finalize()` 先执行 | 所有 Python 对象已被回收 |
| 静态 `registry` 析构 | 进程退出阶段（`atexit` 之后） | `Function` 句柄的析构触发 `PyGILState_Ensure()` 或访问已回收的 Python 对象 |

**核心矛盾**：当 `registry` 的析构晚于 `Py_Finalize` 时，`Function` 句柄的析构函数尝试对已销毁的 Python 运行时进行操作，访问非法内存导致 SIGSEGV。

### 2.3 触发条件

只有当 `registry` 非空时才会触发——如果从未注册过回调，map 为空，析构只是空操作，不会访问任何 Python 对象。这解释了为什么：
- P1 测试（未使用 data-io / python_layer 回调）从不触发
- P2 测试（注册了回调）在解释器退出时必现

## 3. 修复方案

### 3.1 方案对比

| 方案 | 描述 | 优势 | 劣势 |
|------|------|------|------|
| A. 显式清理（采纳） | 在 `Py_Finalize` 前通过 `atexit` 清空 registry | 简单、可验证、无侵入 | 需 Python 侧配合 |
| B. 惰性持有（`std::shared_ptr` + 自定义 deleter） | 将析构责任移到 Python 对象生命周期内 | 零配置 | 复杂度高、易引入新 bug |
| C. 禁用静态（改为实例级） | 将 registry 改为 Net 级成员 | 从根本消除静态析构 | 需重构接口、破坏现有 API |

**结论**：方案 A（显式清理）最稳妥，风险最低。

### 3.2 具体实现

#### Step 1：C++ 层新增清理函数

**`data_io_bridge.cpp`**
```cpp
void ClearDataIOCallback() {
  DataIOCallbackRegistry().clear();
  CAFFE_FFI_LAYER_LOG << "Cleared all data-io callbacks (registry size -> 0)";
}
```

**`python_layer.cpp`**
```cpp
void ClearPythonLayerCallback() {
  PythonCallbackRegistry().clear();
  CAFFE_FFI_LAYER_LOG << "Cleared all python_layer callbacks (registry size -> 0)";
}
```

#### Step 2：在头文件中声明

`data_io_bridge.hpp` / `python_layer.hpp` 各新增 `ClearXxxCallback()` 声明，附带文档说明：
> Call this from the Python side via atexit before the Python interpreter shuts down.

#### Step 3：注册 FFI 接口

`_caffe_ffi.cc` 新增两个 FFI 函数并注册：

```cpp
void ClearDataIO()       { ClearDataIOCallback(); }
void ClearPythonLayer()  { ClearPythonLayerCallback(); }

// In the registration block:
.def("caffe_ffi.python_layer.clear", ClearPythonLayer,
     "Clear all python_layer callbacks (call before interpreter shutdown)")
.def("caffe_ffi.data_io.clear", ClearDataIO,
     "Clear all data_io callbacks (call before interpreter shutdown)")
```

#### Step 4：Python 侧注册 `atexit` 钩子

`__init__.py` 在导入时注册清理钩子：

```python
import atexit
from . import _ffi_api

def _cleanup_callbacks():
    """Release Python Function objects held by C++ static registries before
    interpreter shutdown to prevent segfault on exit."""
    if not _ffi_api.is_available():
        return
    for name in ("caffe_ffi.data_io.clear", "caffe_ffi.python_layer.clear"):
        fn = _ffi_api.get_global_func(name)
        if fn is not None:
            try:
                fn()
            except Exception:
                pass  # best-effort during shutdown

atexit.register(_cleanup_callbacks)
```

钩子具有以下特性：
- **幂等安全**：多次调用不会出错（`unordered_map::clear()` 对空 map 是 no-op）
- **防御式**：`try/except` 保证清理失败不阻塞进程退出
- **可用性检查**：`_ffi_api.is_available()` 短路，避免 C++ 扩展未加载时的空指针

## 4. 验证步骤

### 4.1 构建

在 P0 容器中重建 caffe-ffi：

```bash
docker exec caffe-ffi-jupyter bash -lc '
source /opt/conda/etc/profile.d/conda.sh && conda activate caffe-ffi
cd /SpecWeave/projects/xuanspace/libs/caffe-ffi

# 修复 Windows 挂载 CRLF 问题
for f in src/caffe_ffi/_caffe_ffi.cc src/caffe_ffi/layers/data_io_bridge.cpp \
         src/caffe_ffi/layers/python_layer.cpp include/caffe_ffi/layers/data_io_bridge.hpp \
         include/caffe_ffi/layers/python_layer.hpp python/caffe_ffi/__init__.py; do
  [ -f "$f" ] && grep -q $'\r' "$f" && sed -i "s/\r$//" "$f" && echo "Fixed $f"
done

rm -rf build
rm -f python/caffe_ffi/_caffe_ffi*.so

TVM_FFI_CMAKE_DIR=$(python -c "import tvm_ffi, os; print(os.path.dirname(tvm_ffi.__file__))")
ARGS="-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON;-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON"
ARGS="${ARGS};-DCMAKE_PREFIX_PATH=$CONDA_PREFIX;-Dcaffe-ffi_DIR=$TVM_FFI_CMAKE_DIR"
SKBUILD_CMAKE_ARGS="$ARGS" pip install --no-cache-dir --no-build-isolation -e . 2>&1 | tail -30
'
```

> **坑点**：Windows 编辑工具会将文件写入 CRLF 行尾。在 WSL/Linux 环境下使用该 `.so` 时，GCC/链接器会报 `file format not recognized` / `invalid ELF header`。必须在重建前执行 `sed -i 's/\r$//'` 归一化为 LF。

### 4.2 复现脚本验证

**修复前**（旧 `.so`）：
```
$ python .temp/repro_segfault.py
Segmentation fault (core dumped)    ← exit 139
```

**修复后**（重建 `.so`）：
```
$ python .temp/repro_segfault.py
Registered callbacks into data_io + python_layer registries.
Exiting interpreter...              ← exit 0，干净退出
```

### 4.3 回归测试

`tests/python/test_callback_registry_cleanup.py`（5 例）：

| 测试用例 | 内容 | 结果 |
|---------|------|------|
| `test_clear_ffi_functions_exist` | 验证两个 `clear` FFI 函数已注册 | ✅ |
| `test_atexit_cleanup_hook_registered` | 验证 `_cleanup_callbacks` 已注册到 `atexit` | ✅ |
| `test_register_then_clear_roundtrip[data_io]` | register→clear→re-register 往返 | ✅ |
| `test_register_then_clear_roundtrip[python_layer]` | register→clear→re-register 往返 | ✅ |
| `test_cleanup_callbacks_runs_without_error` | 钩子可被多次安全调用（幂等） | ✅ |

### 4.4 全量 P2 测试回归

```
$ python -m pytest tests/python/test_p2*.py -q --tb=short
============================== 41 passed in 2.3s ==============================
```

修复未破坏任何现有功能。

## 5. CI  nightly job 补充

### 5.1 新增步骤

在 `.github/workflows/ci.yml` 的 `nightly` job 中补充：

```yaml
- name: TVM-FFI dependency loading check (nightly)
  env:
    KMP_DUPLICATE_LIB_OK: TRUE
  run: |
    python scripts/ci_check_tvmffi.py

- name: Run P2 data-IO operator tests (nightly)
  env:
    KMP_DUPLICATE_LIB_OK: TRUE
  run: |
    python -m pytest tests/python/test_p2_data_io_ops.py -v --tb=short
```

### 5.2 覆盖的根因

这两个步骤直接覆盖了过往反复阻塞 P0 环境的根因：
- **TVM-FFI 依赖检查**：捕获 `tvm_ffi.core` 扩展缺失、版本 skew、动态链接库解析失败（WinError 127）
- **P2 数据-IO 测试**：捕获 `data_io` / `python_layer` 回调桥接功能回退，以及（间接）解释器退出 segfault

## 6. 预防措施

| 预防措施类型 | 具体内容 |
|-------------|---------|
| `[prevent: test-case]` | 新增 `test_callback_registry_cleanup.py` 回归测试（5 例），每次 CI 均执行 |
| `[prevent: ci-step]` | nightly job 新增 TVM-FFI 依赖检查与 P2 测试步骤 |
| `[prevent: code-review]` | 后续引入静态 `std::unordered_map` 持有 Python 对象的代码需强制执行"清理钩子"审查 |

## 7. 萃取的可复用模式

**模式名称**：`static-registry-python-lifecycle-guard`

**触发场景**：C++ 静态容器（函数内静态 `std::unordered_map` / `std::vector`）持有 TVM FFI `Function` 句柄，且句柄指向 Python 可调用对象。

**核心步骤**：
1. 在容器所在 `.cpp` 中新增 `ClearXxx()` 函数，调用容器的 `clear()`
2. 在对应 `.hpp` 中声明清理函数，附文档说明调用时机
3. 在 FFI 注册文件中注册 `caffe_ffi.xxx.clear` 接口
4. 在 Python 包 `__init__.py` 中注册 `atexit.register(_cleanup)`，在 `Py_Finalize` 前清空

**反模式**：在 `Py_Finalize` 之后依赖静态对象析构自动清理（析构时序不受保证）。

**迁移验证**：该模式已在 caffe-ffi 的 `data_io_bridge` 与 `python_layer` 两个注册中心成功应用。

## 8. 提交记录

| 仓库 | 提交哈希 | 说明 |
|------|---------|------|
| xuanspace (caffe-ffi) | `c705ea5` | fix(caffe-ffi): 修复解释器退出时静态 registry 析构导致 SIGSEGV |
| SpecWeave | `80f07f55` | chore(submodule): 更新 xuanspace 子模块至 c705ea5 |

## 9. 后续行动

- [ ] 确认 nightly workflow 在 GitHub Actions 上生效（触发一次手动验证）
- [ ] 在团队内部分享该模式，将 `static-registry-python-lifecycle-guard` 纳入代码审查清单
- [ ] 在 `docs/retrospective/patterns/` 中归档该模式（下次遇到静态+Python 交互问题时可直接复用）
