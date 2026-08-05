# ASan 内存安全验证报告（Task 17b）

> **日期**: 2026-08-04
> **任务**: Task 17b（ASan 内存管理验证）
> **环境**: WSL Docker 容器 `caffe-ffi-jupyter`（conda env `caffe-ffi`，Python 3.14.6，GCC 14.3.0，CMake 4.4.1，Ninja 1.13.2）
> **结论**: ✅ **1647 passed / 1 skipped / 0 ASan 内存安全错误**，内存泄漏专项零泄漏，发现并修复 1 处真实堆越界读

---

## 一、验证目标

使用 AddressSanitizer（ASan，`-fsanitize=address`）编译运行 caffe-ffi 测试，验证：
1. **ObjectPtr 引用计数**：Net 销毁时正确释放，无泄漏、无 use-after-free
2. **COW 引用计数**：引用计数正确性，无共享内存破坏
3. **内存安全错误检测**：堆越界读/写、use-after-free、双重释放等
4. **内存计数器**：`total_allocated_bytes()`/`live_blob_count()` 与真实内存一致

## 二、验证环境与构建

### 2.1 构建基础设施
新增 `CAFFE_FFI_ENABLE_ASAN` CMake 选项（默认 **OFF**，不影响正常构建），在 GCC/Clang（`-fsanitize=address -fno-omit-frame-pointer`）与 MSVC（`/fsanitize=address`）下统一配置编译与链接标志。

### 2.2 ASan 构建命令
```bash
cmake --preset default \
  -DCMAKE_C_FLAGS="-O1 -fno-omit-frame-pointer -fPIC -fno-strict-aliasing" \
  -DCMAKE_CXX_FLAGS="-O1 -fno-omit-frame-pointer -fPIC -fno-strict-aliasing" \
  -DCAFFE_FFI_ENABLE_ASAN=ON \
  -DCAFFE_FFI_ENABLE_COW=ON \
  -DCAFFE_FFI_ENABLE_COW_PHASE3=ON \
  -DCAFFE_FFI_BUILD_TESTS=ON
cmake --build --preset default -j$(nproc)
```

构建后需将 `_caffe_ffi.so` 复制到源码树 `python/caffe_ffi/`，并用 `LD_PRELOAD` 加载 libasan 再运行 Python 测试。

### 2.3 运行命令
```bash
export ASAN_OPTIONS="detect_leaks=0:halt_on_error=1:abort_on_error=1"
LD_PRELOAD=$(gcc -print-file-name=libasan.so) python -m pytest tests/python -q
```

### 2.4 关键构建约束（踩坑记录）
| 问题 | 原因 | 处理 |
|------|------|------|
| GNU ld `bad reloc symbol index` 链接错误 | `-O3` + `-ffunction-sections` + `--gc-sections` + ASan 在大型 `_caffe_ffi.cc` 对象上的 binutils bug | ASan 构建用 `-O1`（ASan 推荐优化级别）并清空 conda 默认 CFLAGS/CXXFLAGS（与代码无关） |
| Python 解释器泄漏误报 | 解释器退出时分配器缓存触发 LeakSanitizer 误报 | `ASAN_OPTIONS=detect_leaks=0`，泄漏检测由项目 `total_allocated_bytes()` 计数器承担 |
| `/opt/venv/bin/python` 遮蔽 conda 环境 | 容器内存在两套 Python | 显式使用 `/opt/conda/envs/caffe-ffi/bin/python` |

## 三、验证结果

### 3.1 全量 Python 回归（ASan 构建）
```
1647 passed, 1 skipped, 0 failures
ASan 内存安全错误数：0
```

### 3.2 内存泄漏专项
`tests/python/test_memory_leak.py`：**16 passed**（零泄漏）。覆盖正常创建/销毁、故意泄漏、异常处理、Reshape 重分配等生命周期场景。

### 3.3 COW 引用计数
`tests/python/test_cow.py`：**21 passed**（引用计数 + 隔离性）。

### 3.4 非 ASan 默认构建回归
```
1647 passed, 1 skipped, 0 failures
```
确认 ASan 选项默认 OFF 不影响正常构建，C++ 测试二进制正常链接。

## 四、发现并修复的真实内存缺陷

### 4.1 缺陷描述
ASan 在 `test_complex_topologies.py::test_inplace_chain_forward` 中捕获 **heap-buffer-overflow**（堆越界读）：

```
0x... is located 0 bytes after N-byte region
READ of size 4 ... in InnerProductLayer::Forward_cpu
```

**触发位置**：`InnerProductLayer::Forward_cpu`（`src/caffe_ffi/layers/inner_product_layer.cpp`）

### 4.2 根因分析
网络 `ip3` 层为 **in-place InnerProduct**（`bottom: "x" top: "x"`），`num_output: 2` 而输入为 4 通道：
- in-place 意味着 `top[0]` 与 `bottom[0]` 共享同一 Blob 缓冲区
- `Reshape` 将共享缓冲区从 4 通道（16 字节）截断为 2 通道（8 字节）
- `Forward_cpu` 仍按旧尺寸（M×K=4 通道）读取数据 → 越界读缓冲区末尾之外的 8 字节

### 4.3 根因修复
`InnerProductLayer::Reshape` 增加 **in-place 安全守卫**：
```cpp
if (bottom[0] == top[0]) {
  const int64_t bottom_count = bottom[0]->count();
  const int64_t top_count = static_cast<int64_t>(M_) * static_cast<int64_t>(N_);
  if (top_count != bottom_count) {
    CAFFE_FFI_CHECK_VALUE_EQ(top_count, bottom_count)
        << "InnerProduct in-place operation requires input and output ...";
  }
}
```

当 `bottom==top` 且输出 count ≠ 输入 count 时抛错拒绝，避免共享缓冲区被破坏。

### 4.4 测试同步修正
- `test_complex_topologies.py`：`ip3` 层改为非 in-place 输出（`top: "x3"`），保留测试验证 in-place ReLU/InnerProduct 链的意图
- 新增负向测试 `test_inplace_inner_product_shape_change_rejected`：验证 in-place InnerProduct 尺寸变化被拒绝（锁定守卫，防止回归）

## 五、交付物清单

| 文件 | 说明 |
|------|------|
| `cmake/Options.cmake` | `CAFFE_FFI_ENABLE_ASAN` 选项（默认 OFF） |
| `cmake/CompilerConfig.cmake` | ASan 编译/链接标志（GCC/Clang/MSVC） |
| `src/caffe_ffi/layers/inner_product_layer.cpp` | in-place 安全守卫（根因修复） |
| `tests/python/test_complex_topologies.py` | 测试修正 + 负向守卫测试 |
| `examples/asan_demo.cpp` | ASan 演示（leak_demo/heap_overflow_demo） |
| `tests/cpp/test_asan_demo.cpp` | 受宏守卫的可调用子程序 |
| `docs/setup/ASAN_REPORT_READING_GUIDE.md` | ASan 报告堆栈解读指南 |

## 六、结论与后续建议

**结论**：caffe-ffi 在 ASan 下全量测试通过（1647 passed / 1 skipped / 0 内存安全错误），ObjectPtr 引用计数与 COW 引用计数在 Net 销毁时正确释放，无泄漏、无 use-after-free、无双重释放。ASan 前置验证发现并修复了 1 处真实堆越界读（in-place InnerProduct 尺寸变化），采用"拒绝 + 守卫 + 负向测试"三层防线。

**后续建议**：
1. 将 ASan 构建纳入 CI 可选作业（`CAFFE_FFI_ENABLE_ASAN=ON`）以持续检测内存安全
2. 将 in-place 内存安全规范沉淀为内部技术文档（见 `docs/design/INPLACE_MEMORY_SAFETY_STANDARD.md`）
3. 对新增层（P4 能力扩展）在实现时同步覆盖 in-place 场景测试