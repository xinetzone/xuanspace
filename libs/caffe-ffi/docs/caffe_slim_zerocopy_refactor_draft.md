# caffe-slim 零拷贝架构改造代码草案

> **生成日期**: 2026-07-29
> **目标模块**: [caffe-slim](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-slim)
> **参考实现**: [caffe-ffi](file:///d:/spaces/SpecWeave/projects/xuanspace/vendor/caffe/caffe-ffi)（已完成零拷贝优化）
> **改造范围**: P0 优先级 — Blob 写入零拷贝 + 三层日志 + 错误增强 + @register_object
> **状态**: 代码草案（draft），需 review 后合入 caffe-slim

---

## 一、改造要点总览

| 改造项 | 文件 | 说明 |
|---|---|---|
| **新增三层日志头文件** | `include/caffe/ffi_log.hpp` | RAII Logger + 编译期门控 + 组件标签 |
| **重构 FFI 绑定层** | `src/caffe/_caffe.cpp` | 双类模型 + 写入零拷贝 + 详细内存日志 |
| **Python 绑定重构** | `python/caffe/__init__.py` | @register_object + 零拷贝方法 + 日志控制 |
| **新增内存追踪** | `src/caffe/_caffe.cpp` | 记录 Tensor 地址、Blob 指针、引用计数变化 |

---

## 二、新增文件：`include/caffe/ffi_log.hpp`（三层日志核心）

```cpp
#ifndef CAFFE_FFI_LOG_HPP_
#define CAFFE_FFI_LOG_HPP_

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <sstream>

namespace caffe {
namespace ffi_log {

// ---------------------------------------------------------------------------
// 日志级别（与 caffe-ffi 保持一致）
// ---------------------------------------------------------------------------
enum class LogLevel : int {
    TRACE = 0,   // 最详细：内存地址、引用计数、Tensor 绑定/解绑
    DEBUG = 1,   // 调试：形状变化、层配置
    INFO  = 2,   // 信息：网络初始化、前向开始/结束
    WARN  = 3,   // 警告
    ERROR = 4,   // 错误
    SILENT = 5   // 静默
};

// ---------------------------------------------------------------------------
// 全局状态（通过 FFI 函数从 Python 层控制）
// ---------------------------------------------------------------------------
inline LogLevel& GlobalLogLevel() {
    static LogLevel level = LogLevel::WARN;
    return level;
}

inline std::mutex& LogMutex() {
    static std::mutex mtx;
    return mtx;
}

inline void SetLogLevel(int level) {
    std::lock_guard<std::mutex> lock(LogMutex());
    GlobalLogLevel() = static_cast<LogLevel>(level);
}

inline int GetLogLevel() {
    return static_cast<int>(GlobalLogLevel());
}

inline bool LogEnabled(LogLevel level) {
    return static_cast<int>(level) >= static_cast<int>(GlobalLogLevel());
}

// ---------------------------------------------------------------------------
// RAII Logger：析构时输出（流式 API）
// ---------------------------------------------------------------------------
class Logger {
public:
    explicit Logger(LogLevel level, const char* tag, const char* func, int line)
        : level_(level), tag_(tag), func_(func), line_(line) {}

    ~Logger() {
        if (!LogEnabled(level_)) return;
        std::lock_guard<std::mutex> lock(LogMutex());
        std::fprintf(stderr, "[caffe-slim][%s] %s", tag_, stream_.str().c_str());
        std::fprintf(stderr, "  [%s:%d]\n", func_, line_);
        std::fflush(stderr);
    }

    template <typename T>
    Logger& operator<<(const T& val) {
        if (LogEnabled(level_)) {
            stream_ << val;
        }
        return *this;
    }

private:
    LogLevel level_;
    const char* tag_;
    const char* func_;
    int line_;
    std::ostringstream stream_;
};

// ---------------------------------------------------------------------------
// 编译期门控：设置 CAFFE_FFI_LOG_LEVEL=0 编译得到全部日志
// 设置 CAFFE_FFI_LOG_LEVEL=5 编译完全消除日志开销
// ---------------------------------------------------------------------------
#ifndef CAFFE_FFI_LOG_LEVEL
#define CAFFE_FFI_LOG_LEVEL 1  // 默认 DEBUG 级别以上
#endif

#define CAFFE_FFI_LOG(level, tag) \
    if (static_cast<int>(::caffe::ffi_log::LogLevel::level) >= CAFFE_FFI_LOG_LEVEL) \
        ::caffe::ffi_log::Logger(::caffe::ffi_log::LogLevel::level, tag, __func__, __LINE__)

// 便捷宏
#define FFI_TRACE(tag)   CAFFE_FFI_LOG(TRACE, tag)
#define FFI_DEBUG(tag)   CAFFE_FFI_LOG(DEBUG, tag)
#define FFI_INFO(tag)    CAFFE_FFI_LOG(INFO,  tag)
#define FFI_WARN(tag)    CAFFE_FFI_LOG(WARN,  tag)
#define FFI_ERROR(tag)   CAFFE_FFI_LOG(ERROR, tag)

// 日志标签约定
#define TAG_TENSOR  "TENSOR"   // Tensor 内存绑定、地址、引用计数
#define TAG_BLOB    "BLOB"     // Blob 创建/重塑/销毁
#define TAG_NET     "NET"      // 网络初始化/前向/反向
#define TAG_MEM     "MEM"      // 内存分配/释放/拷贝
#define TAG_FFI     "FFI"      // FFI 调用入口/出口

// ---------------------------------------------------------------------------
// 内存地址格式化工具（统一格式：0x%p 带前缀）
// ---------------------------------------------------------------------------
inline std::string PtrStr(const void* p) {
    std::ostringstream oss;
    oss << "0x" << std::hex << reinterpret_cast<uintptr_t>(p);
    return oss.str();
}

}  // namespace ffi_log
}  // namespace caffe

#endif  // CAFFE_FFI_LOG_HPP_
```

---

## 三、重构文件：`src/caffe/_caffe.cpp`（完整草案）

> **改造重点**：
> 1. Blob 读取零拷贝已存在（`Blob_GetData` 使用 `Tensor::FromNDAlloc`），**增加详细日志**
> 2. **Blob 写入零拷贝**：`Blob_SetData` 从 `memcpy` 改为直接指针共享（当张量兼容时）
> 3. 添加 `Tensor` 生命周期追踪：记录每个 Tensor 的创建地址、绑定的 Blob 指针、引用计数
> 4. 新增全局对象计数器：`LiveNetCount`、`LiveTensorCount`
> 5. 使用 `TVM_FFI_ICHECK` 替代 `TVM_FFI_CHECK` 提供更丰富的错误上下文

```cpp
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/stl.h>

#include <atomic>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <mutex>

#include "caffe/caffe.hpp"
#include "caffe/ffi_log.hpp"  // 新增：三层日志

namespace caffe {

using tvm::ffi::Tensor;
using tvm::ffi::TensorView;
using tvm::ffi::ShapeView;
using tvm::ffi::ObjectRef;

typedef float Dtype;

// ===========================================================================
// 全局计数器（内存追踪、泄漏检测）
// ===========================================================================
static std::atomic<int64_t> g_live_net_count{0};
static std::atomic<int64_t> g_live_tensor_count{0};
static std::atomic<int64_t> g_total_tensor_allocs{0};
static std::atomic<int64_t> g_total_memcpy_bytes{0};
static std::atomic<int64_t> g_zero_copy_hits{0};
static std::atomic<int64_t> g_memcpy_fallbacks{0};

// Tensor 追踪表：tensor 数据指针 → 描述字符串
static std::mutex g_tensor_map_mtx;
static std::unordered_map<const void*, std::string> g_tensor_map;

static void LogTensorCreated(const void* ptr, const std::string& desc) {
    g_live_tensor_count.fetch_add(1);
    g_total_tensor_allocs.fetch_add(1);
    std::lock_guard<std::mutex> lock(g_tensor_map_mtx);
    g_tensor_map[ptr] = desc;
    FFI_TRACE(TAG_TENSOR) << "CREATE tensor=" << ffi_log::PtrStr(ptr)
                          << " live=" << g_live_tensor_count.load()
                          << " total=" << g_total_tensor_allocs.load()
                          << " desc=[" << desc << "]";
}

static void LogTensorDestroyed(const void* ptr) {
    g_live_tensor_count.fetch_sub(1);
    std::string desc;
    {
        std::lock_guard<std::mutex> lock(g_tensor_map_mtx);
        auto it = g_tensor_map.find(ptr);
        if (it != g_tensor_map.end()) {
            desc = it->second;
            g_tensor_map.erase(it);
        }
    }
    FFI_TRACE(TAG_TENSOR) << "DESTROY tensor=" << ffi_log::PtrStr(ptr)
                          << " live=" << g_live_tensor_count.load()
                          << " desc=[" << desc << "]";
}

// ===========================================================================
// 文件校验
// ===========================================================================
static void CheckFile(const std::string& filename) {
    std::ifstream f(filename.c_str());
    if (!f.good()) {
        f.close();
        TVM_FFI_THROW(tvm::ffi::RuntimeError)
            << "Cannot open file: " << filename;
    }
    f.close();
}

// ===========================================================================
// 零拷贝 Tensor Allocator（带日志）
// CpuBlobDataAllocator 让 numpy 数组直接指向 Caffe Blob 的 CPU 内存
// 通过持有 net_keep_alive (shared_ptr<Net>) 确保 Blob 内存不会提前释放
// ===========================================================================
struct CpuBlobDataAllocator {
    Dtype* data;
    std::shared_ptr<Net<Dtype>> net_keep_alive;
    std::string blob_name;
    bool is_diff;  // true=diff梯度, false=data数据

    void AllocData(DLTensor* tensor) {
        tensor->data = data;
        // 记录：numpy Tensor 已绑定到 C++ Blob 内存
        const void* tensor_ptr = tensor->data;
        const void* blob_ptr = static_cast<const void*>(data);
        std::string desc = std::string("Blob.") + (is_diff ? "diff:" : "data:") + blob_name;
        LogTensorCreated(tensor_ptr, desc);

        FFI_DEBUG(TAG_TENSOR) << "BIND zero-copy tensor=" << ffi_log::PtrStr(tensor_ptr)
                              << " -> blob[" << blob_name << "]."
                              << (is_diff ? "diff" : "data")
                              << " ptr=" << ffi_log::PtrStr(blob_ptr)
                              << " net_refcount=" << net_keep_alive.use_count();
    }

    void FreeData(DLTensor* tensor) {
        const void* tensor_ptr = tensor->data;
        long use_count_before = net_keep_alive.use_count();
        LogTensorDestroyed(tensor_ptr);

        FFI_DEBUG(TAG_TENSOR) << "UNBIND zero-copy tensor=" << ffi_log::PtrStr(tensor_ptr)
                              << " from blob[" << blob_name << "]."
                              << (is_diff ? "diff" : "data")
                              << " net_refcount_before=" << use_count_before;
        net_keep_alive.reset();
        FFI_TRACE(TAG_TENSOR) << "UNBIND done, net_refcount_after=0";
    }
};

// ===========================================================================
// Net 生命周期
// ===========================================================================
uintptr_t Net_Init(const std::string& network_file, int phase) {
    CheckFile(network_file);
    g_live_net_count.fetch_add(1);
    FFI_INFO(TAG_NET) << "INIT network_file=" << network_file
                      << " phase=" << phase
                      << " live_nets=" << g_live_net_count.load();
    auto* net_handle = new std::shared_ptr<Net<Dtype>>(
        new Net<Dtype>(network_file, static_cast<Phase>(phase)));
    FFI_DEBUG(TAG_NET) << "INIT handle=" << ffi_log::PtrStr(net_handle)
                       << " blob_count=" << (*net_handle)->blobs().size()
                       << " layer_count=" << (*net_handle)->layers().size();
    return reinterpret_cast<uintptr_t>(net_handle);
}

uintptr_t Net_Init_Load(const std::string& param_file,
                         const std::string& pretrained_param_file,
                         int phase) {
    CheckFile(param_file);
    CheckFile(pretrained_param_file);
    g_live_net_count.fetch_add(1);
    FFI_INFO(TAG_NET) << "INIT_LOAD param=" << param_file
                      << " weights=" << pretrained_param_file
                      << " phase=" << phase
                      << " live_nets=" << g_live_net_count.load();
    auto* net_handle = new std::shared_ptr<Net<Dtype>>(
        new Net<Dtype>(param_file, static_cast<Phase>(phase)));
    (*net_handle)->CopyTrainedLayersFrom(pretrained_param_file);
    FFI_DEBUG(TAG_NET) << "INIT_LOAD handle=" << ffi_log::PtrStr(net_handle)
                       << " weights loaded successfully";
    return reinterpret_cast<uintptr_t>(net_handle);
}

void Net_Destroy(uintptr_t handle) {
    auto* net_handle = reinterpret_cast<std::shared_ptr<Net<Dtype>>*>(handle);
    FFI_INFO(TAG_NET) << "DESTROY handle=" << ffi_log::PtrStr(net_handle)
                      << " live_nets_before=" << g_live_net_count.load()
                      << " remaining_tensors=" << g_live_tensor_count.load();

    if (g_live_tensor_count.load() > 0) {
        FFI_WARN(TAG_MEM) << "DESTROYing net while "
                          << g_live_tensor_count.load()
                          << " zero-copy tensors still alive (potential leak!)";
        std::lock_guard<std::mutex> lock(g_tensor_map_mtx);
        for (const auto& kv : g_tensor_map) {
            FFI_WARN(TAG_MEM) << "  leaked tensor=" << ffi_log::PtrStr(kv.first)
                              << " desc=[" << kv.second << "]";
        }
    }

    delete net_handle;
    g_live_net_count.fetch_sub(1);
    FFI_DEBUG(TAG_NET) << "DESTROY done, live_nets=" << g_live_net_count.load();
}

void Net_CopyTrainedLayersFrom(uintptr_t handle, const std::string& weights_file) {
    auto* net_handle = reinterpret_cast<std::shared_ptr<Net<Dtype>>*>(handle);
    CheckFile(weights_file);
    FFI_INFO(TAG_NET) << "COPY_WEIGHTS from=" << weights_file;
    (*net_handle)->CopyTrainedLayersFrom(weights_file);
}

void Net_Forward(uintptr_t handle) {
    auto* net_handle = reinterpret_cast<std::shared_ptr<Net<Dtype>>*>(handle);
    FFI_DEBUG(TAG_NET) << "FORWARD start handle=" << ffi_log::PtrStr(net_handle)
                       << " input_blobs=" << (*net_handle)->input_blobs().size()
                       << " output_blobs=" << (*net_handle)->output_blobs().size();
    (*net_handle)->ForwardPrefilled();
    FFI_DEBUG(TAG_NET) << "FORWARD end handle=" << ffi_log::PtrStr(net_handle);
}

void Net_Reshape(uintptr_t handle) {
    auto* net_handle = reinterpret_cast<std::shared_ptr<Net<Dtype>>*>(handle);
    FFI_DEBUG(TAG_NET) << "RESHAPE handle=" << ffi_log::PtrStr(net_handle);
    (*net_handle)->Reshape();
}

std::vector<std::string> Net_BlobNames(uintptr_t handle) {
    auto* net_handle = reinterpret_cast<std::shared_ptr<Net<Dtype>>*>(handle);
    return (*net_handle)->blob_names();
}

std::vector<int> Blob_GetShape(uintptr_t net_handle, const std::string& blob_name) {
    auto& net = *reinterpret_cast<std::shared_ptr<Net<Dtype>>*>(net_handle);
    auto blob = net->blob_by_name(blob_name);
    TVM_FFI_ICHECK(blob != nullptr, tvm::ffi::ValueError)
        << "Unknown blob name: '" << blob_name << "'"
        << ". Available blobs: [" << [&](){
            std::string s;
            for (const auto& n : net->blob_names()) s += n + ",";
            return s;
        }() << "]";
    return blob->shape();
}

// ===========================================================================
// Blob 读取：零拷贝（已存在，增加日志）
// ===========================================================================
Tensor Blob_GetData(uintptr_t net_handle, const std::string& blob_name) {
    auto& net = *reinterpret_cast<std::shared_ptr<Net<Dtype>>*>(net_handle);
    auto blob = net->blob_by_name(blob_name);
    TVM_FFI_ICHECK(blob != nullptr, tvm::ffi::ValueError)
        << "Unknown blob name: '" << blob_name << "'";

    Dtype* data_ptr = blob->mutable_cpu_data();
    DLDevice cpu_device{static_cast<DLDeviceType>(kDLCPU), 0};
    DLDataType dtype{static_cast<uint8_t>(kDLFloat), 32, 1};

    FFI_DEBUG(TAG_BLOB) << "GET_DATA blob[" << blob_name << "]"
                        << " ptr=" << ffi_log::PtrStr(data_ptr)
                        << " shape=" << blob->shape_string()
                        << " bytes=" << blob->count() * sizeof(Dtype);

    return Tensor::FromNDAlloc(
        CpuBlobDataAllocator{data_ptr, net, blob_name, /*is_diff=*/false},
        blob->shape_view(),
        dtype, cpu_device);
}

Tensor Blob_GetDiff(uintptr_t net_handle, const std::string& blob_name) {
    auto& net = *reinterpret_cast<std::shared_ptr<Net<Dtype>>*>(net_handle);
    auto blob = net->blob_by_name(blob_name);
    TVM_FFI_ICHECK(blob != nullptr, tvm::ffi::ValueError)
        << "Unknown blob name: '" << blob_name << "'";

    Dtype* diff_ptr = blob->mutable_cpu_diff();
    DLDevice cpu_device{static_cast<DLDeviceType>(kDLCPU), 0};
    DLDataType dtype{static_cast<uint8_t>(kDLFloat), 32, 1};

    FFI_DEBUG(TAG_BLOB) << "GET_DIFF blob[" << blob_name << "]"
                        << " ptr=" << ffi_log::PtrStr(diff_ptr)
                        << " shape=" << blob->shape_string();

    return Tensor::FromNDAlloc(
        CpuBlobDataAllocator{diff_ptr, net, blob_name, /*is_diff=*/true},
        blob->shape_view(),
        dtype, cpu_device);
}

// ===========================================================================
// Blob 写入：零拷贝优先，memcpy 兜底（新增！）
// 核心改进：当输入 Tensor 满足以下条件时，直接共享指针而不拷贝：
//   1. dtype 为 float32
//   2. 在 CPU 上
//   3. 连续内存（compact）
//   4. 形状完全匹配
//   5. 调用方通过 zero_copy=True 参数明确要求零拷贝
// 否则回退到 memcpy（安全路径）
// ===========================================================================

// 零拷贝写入 Allocator：让 Blob 直接使用外部 numpy 数组的内存
// 风险：调用方必须保证 numpy 数组在 Blob 使用期间不被释放！
struct ExternalDataAllocator {
    const Dtype* ext_data;       // 外部 numpy 数据指针（不持有所有权）
    std::shared_ptr<Net<Dtype>> net_keep_alive;  // 仅保持 net 存活
    std::string blob_name;

    void AllocData(DLTensor* tensor) {
        // 注意：这里不会真正"分配"，而是指向外部数据
        // Blob 的 data_ 需要被替换为 SyncedMemory 包装外部指针
        // 由于 caffe-slim Blob 使用 shared_ptr<SyncedMemory>，
        // 零拷贝写入需要先 set_cpu_data() 替换内部指针
        tensor->data = const_cast<Dtype*>(ext_data);
        FFI_DEBUG(TAG_TENSOR) << "EXT_BIND tensor=" << ffi_log::PtrStr(tensor->data)
                              << " -> blob[" << blob_name << "] (external zero-copy write)";
    }
    void FreeData(DLTensor* tensor) {
        FFI_DEBUG(TAG_TENSOR) << "EXT_UNBIND tensor=" << ffi_log::PtrStr(tensor->data)
                              << " from blob[" << blob_name << "] (external, not freeing)";
        // 外部数据不归我们管理，不释放
    }
};

void Blob_SetData(uintptr_t net_handle, const std::string& blob_name,
                  TensorView data, bool zero_copy = false) {
    auto& net = *reinterpret_cast<std::shared_ptr<Net<Dtype>>*>(net_handle);
    auto blob = net->blob_by_name(blob_name);
    TVM_FFI_ICHECK(blob != nullptr, tvm::ffi::ValueError)
        << "Unknown blob name: '" << blob_name << "'";

    DLDataType f32_dtype{static_cast<uint8_t>(kDLFloat), 32, 1};
    TVM_FFI_ICHECK(data.dtype() == f32_dtype, tvm::ffi::TypeError)
        << "Blob '" << blob_name << "' expects float32 data, got dtype_code="
        << static_cast<int>(data.dtype().code)
        << " bits=" << data.dtype().bits;

    TVM_FFI_ICHECK(data.IsContiguous(), tvm::ffi::ValueError)
        << "Blob '" << blob_name << "' input must be contiguous";

    ShapeView expected_shape = blob->shape_view();
    ShapeView data_shape = data.shape();
    TVM_FFI_ICHECK(data_shape.size() == expected_shape.size(),
                   tvm::ffi::ValueError)
        << "Blob '" << blob_name << "' shape dimension mismatch: expected "
        << expected_shape.size() << "D, got " << data_shape.size() << "D";

    for (size_t i = 0; i < expected_shape.size(); ++i) {
        TVM_FFI_ICHECK(data_shape[i] == expected_shape[i],
                       tvm::ffi::ValueError)
            << "Blob '" << blob_name << "' shape mismatch at dim " << i
            << ": expected " << expected_shape[i]
            << ", got " << data_shape[i];
    }

    Dtype* dst = blob->mutable_cpu_data();
    const Dtype* src = static_cast<const Dtype*>(data.data_ptr());
    size_t nbytes = static_cast<size_t>(expected_shape.Product()) * sizeof(Dtype);

    // ===== 零拷贝路径（新增）=====
    // 当 zero_copy=true 且数据在 CPU 上时，直接替换 Blob 内部指针
    // 注意：这是"写入端零拷贝"，要求调用方保证 src 在网络前向传播期间存活
    if (zero_copy && data->device.device_type == kDLCPU) {
        FFI_INFO(TAG_MEM) << "SET_DATA zero-copy blob[" << blob_name << "]"
                          << " src=" << ffi_log::PtrStr(src)
                          << " dst=" << ffi_log::PtrStr(dst)
                          << " nbytes=" << nbytes
                          << " (REPLACING internal pointer, NO memcpy)";
        // 通过 set_cpu_data 替换 Blob 的内部 CPU 数据指针
        // SyncedMemory 不会释放这个指针（因为不是它自己分配的）
        blob->set_cpu_data(const_cast<Dtype*>(src));
        g_zero_copy_hits.fetch_add(1);

        FFI_DEBUG(TAG_TENSOR) << "SET_DATA zero-copy bound:"
                              << " blob[" << blob_name << "].cpu_data()="
                              << ffi_log::PtrStr(blob->cpu_data())
                              << " (now points to external numpy buffer)";
        return;
    }

    // ===== memcpy 兜底路径（原有逻辑，增加日志）=====
    FFI_DEBUG(TAG_MEM) << "SET_DATA memcpy blob[" << blob_name << "]"
                       << " src=" << ffi_log::PtrStr(src)
                       << " dst=" << ffi_log::PtrStr(dst)
                       << " nbytes=" << nbytes
                       << (zero_copy ? " (zero_copy requested but falling back)" : "");
    std::memcpy(dst, src, nbytes);
    g_total_memcpy_bytes.fetch_add(static_cast<int64_t>(nbytes));
    g_memcpy_fallbacks.fetch_add(1);

    FFI_TRACE(TAG_MEM) << "SET_DATA memcpy done, verified first=" << dst[0]
                       << " last=" << dst[expected_shape.Product() - 1];
}

// ===========================================================================
// Blob 名称查询
// ===========================================================================
std::vector<std::string> Net_InputBlobNames(uintptr_t handle) {
    auto& net = *reinterpret_cast<std::shared_ptr<Net<Dtype>>*>(handle);
    return net->input_blob_names();
}

std::vector<std::string> Net_OutputBlobNames(uintptr_t handle) {
    auto& net = *reinterpret_cast<std::shared_ptr<Net<Dtype>>*>(handle);
    return net->output_blob_names();
}

std::vector<std::string> LayerTypeList() {
    return LayerRegistry<Dtype>::LayerTypeList();
}

// ===========================================================================
// 全局统计/日志控制函数（导出给 Python）
// ===========================================================================
void SetLogLevel(int level) {
    ffi_log::SetLogLevel(level);
}

int GetLogLevel() {
    return ffi_log::GetLogLevel();
}

// 内存统计
int64_t LiveNetCount() { return g_live_net_count.load(); }
int64_t LiveTensorCount() { return g_live_tensor_count.load(); }
int64_t TotalTensorAllocs() { return g_total_tensor_allocs.load(); }
int64_t TotalMemcpyBytes() { return g_total_memcpy_bytes.load(); }
int64_t ZeroCopyHits() { return g_zero_copy_hits.load(); }
int64_t MemcpyFallbacks() { return g_memcpy_fallbacks.load(); }

void MemoryStats() {
    FFI_INFO(TAG_MEM) << "STATS live_nets=" << g_live_net_count.load()
                      << " live_tensors=" << g_live_tensor_count.load()
                      << " total_tensor_allocs=" << g_total_tensor_allocs.load()
                      << " zero_copy_hits=" << g_zero_copy_hits.load()
                      << " memcpy_fallbacks=" << g_memcpy_fallbacks.load()
                      << " total_memcpy_bytes=" << g_total_memcpy_bytes.load();
}

void SetModeCPU() {
    Caffe::set_mode(Caffe::CPU);
}

void SetRandomSeed(unsigned int seed) {
    Caffe::set_random_seed(seed);
}

const char* Version() {
    return CAFFE_VERSION;
}

// ===========================================================================
// TVM FFI 注册
// ===========================================================================
TVM_FFI_DLL_EXPORT_TYPED_FUNC(SetModeCPU, SetModeCPU)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(SetRandomSeed, SetRandomSeed)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(Version, Version)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(LayerTypeList, LayerTypeList)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(Net_Init, Net_Init)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(Net_Init_Load, Net_Init_Load)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(Net_CopyTrainedLayersFrom, Net_CopyTrainedLayersFrom)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(Net_Destroy, Net_Destroy)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(Net_Forward, Net_Forward)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(Net_Reshape, Net_Reshape)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(Net_BlobNames, Net_BlobNames)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(Net_InputBlobNames, Net_InputBlobNames)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(Net_OutputBlobNames, Net_OutputBlobNames)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(Blob_GetShape, Blob_GetShape)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(Blob_GetData, Blob_GetData)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(Blob_GetDiff, Blob_GetDiff)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(Blob_SetData, Blob_SetData)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(SetLogLevel, SetLogLevel)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(GetLogLevel, GetLogLevel)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(LiveNetCount, LiveNetCount)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(LiveTensorCount, LiveTensorCount)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(TotalTensorAllocs, TotalTensorAllocs)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(TotalMemcpyBytes, TotalMemcpyBytes)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ZeroCopyHits, ZeroCopyHits)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(MemcpyFallbacks, MemcpyFallbacks)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(MemoryStats, MemoryStats)

}  // namespace caffe
```

---

## 四、重构文件：`python/caffe/__init__.py`（Python 层草案）

```python
"""Caffe Python inference package (tvm-ffi slimmed version, zero-copy enabled)."""

from __future__ import annotations

import gc
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

__version__ = "1.1.0-slim-zerocopy"

_PACKAGE_DIR = Path(__file__).resolve().parent

_logger = logging.getLogger("caffe_slim")
_logger.setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# 日志级别常量（与 C++ 层对齐）
# ---------------------------------------------------------------------------
LOG_TRACE = 0
LOG_DEBUG = 1
LOG_INFO = 2
LOG_WARN = 3
LOG_ERROR = 4
LOG_SILENT = 5


def _setup_library_paths():
    lib_dir = str(_PACKAGE_DIR)
    if sys.platform.startswith("win32"):
        os.environ["PATH"] = lib_dir + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(lib_dir)
            except (OSError, FileNotFoundError):
                pass
    elif sys.platform.startswith("darwin"):
        os.environ["DYLD_LIBRARY_PATH"] = lib_dir + os.pathsep + os.environ.get("DYLD_LIBRARY_PATH", "")
    else:
        os.environ["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")


_setup_library_paths()

try:
    import tvm_ffi
except ImportError:
    tvm_ffi = None

_LIB_PATH = None
_mod = None


def _find_lib():
    global _LIB_PATH, _mod
    if _mod is not None:
        return _mod
    if tvm_ffi is None:
        raise ImportError("tvm_ffi is required. Please install tvm-ffi.")
    current_dir = _PACKAGE_DIR
    search_paths = [
        current_dir,
        current_dir.parent.parent.parent / "build" / "python" / "caffe",
        current_dir.parent.parent.parent / "build",
    ]
    lib_names = ["_caffe.dll", "_caffe.so", "_caffe.dylib", "lib_caffe.so"]
    for search_path in search_paths:
        if not search_path.exists():
            continue
        for lib_name in lib_names:
            lib_path = search_path / lib_name
            if lib_path.exists():
                _LIB_PATH = str(lib_path)
                _mod = tvm_ffi.load_module(_LIB_PATH)
                return _mod
    raise ImportError("Cannot find _caffe shared library. Build the project first.")


TRAIN = 0
TEST = 1


class Net:
    """Caffe Net for inference (CPU-only, zero-copy enabled)."""

    def __init__(
        self,
        network_file: str,
        phase: int = TEST,
        weights: Optional[str] = None,
    ):
        self._mod = _find_lib()
        self._handle = None
        if weights is not None:
            self._handle = self._mod.Net_Init_Load(network_file, weights, phase)
        else:
            self._handle = self._mod.Net_Init(network_file, phase)
        self._blob_names = None
        self._input_names = None
        self._output_names = None
        self._zero_copy_tensors: Dict[str, np.ndarray] = {}  # 持有零拷贝引用
        _logger.debug("Net created: handle=%s blobs=%d", hex(self._handle), len(self.blob_names))

    def __del__(self):
        if self._handle is not None and _mod is not None:
            # 释放所有零拷贝引用，防止泄漏
            self._zero_copy_tensors.clear()
            gc.collect()
            try:
                self._mod.Net_Destroy(self._handle)
            except Exception:
                pass
            self._handle = None

    def reshape(self):
        self._mod.Net_Reshape(self._handle)

    def forward(self):
        self._mod.Net_Forward(self._handle)

    @property
    def blob_names(self) -> List[str]:
        if self._blob_names is None:
            self._blob_names = list(self._mod.Net_BlobNames(self._handle))
        return self._blob_names

    @property
    def inputs(self) -> List[str]:
        if self._input_names is None:
            self._input_names = list(self._mod.Net_InputBlobNames(self._handle))
        return self._input_names

    @property
    def outputs(self) -> List[str]:
        if self._output_names is None:
            self._output_names = list(self._mod.Net_OutputBlobNames(self._handle))
        return self._output_names

    def blob_shape(self, blob_name: str) -> tuple:
        return tuple(self._mod.Blob_GetShape(self._handle, blob_name))

    # ------------------------------------------------------------------
    # 零拷贝读取（与之前一致，增加引用追踪日志）
    # ------------------------------------------------------------------
    def blob_data(self, blob_name: str) -> np.ndarray:
        """Get blob data as numpy array (zero-copy view).

        The returned array directly aliases C++ memory. Hold a reference to it
        only as long as needed; the Net must outlive all views.
        """
        tensor = self._mod.Blob_GetData(self._handle, blob_name)
        arr = np.from_dlpack(tensor)
        _logger.debug("blob_data('%s'): ptr=0x%x shape=%s dtype=%s",
                      blob_name, arr.ctypes.data, arr.shape, arr.dtype)
        return arr

    def blob_diff(self, blob_name: str) -> np.ndarray:
        """Get blob diff as numpy array (zero-copy view)."""
        tensor = self._mod.Blob_GetDiff(self._handle, blob_name)
        arr = np.from_dlpack(tensor)
        _logger.debug("blob_diff('%s'): ptr=0x%x shape=%s", blob_name, arr.ctypes.data, arr.shape)
        return arr

    # ------------------------------------------------------------------
    # 写入：新增 zero_copy 参数
    # ------------------------------------------------------------------
    def set_input_data(self, input_name: str, data: np.ndarray,
                       zero_copy: bool = False):
        """Set input blob data from numpy array.

        Args:
            input_name: Name of the input blob.
            data: Numpy array (float32, contiguous).
            zero_copy: If True, share the numpy buffer directly with C++
                (NO memcpy). The data array MUST remain alive for the entire
                lifetime of the Net or until the next set_input_data call.
                If False (default), safely copy data into Blob storage.
        """
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        if zero_copy:
            # 零拷贝路径：持有引用防止 numpy 数组被 GC
            self._zero_copy_tensors[input_name] = data
            _logger.debug("set_input_data('%s') ZERO-COPY: numpy_ptr=0x%x "
                          "nbytes=%d (holding reference)",
                          input_name, data.ctypes.data, data.nbytes)
        else:
            _logger.debug("set_input_data('%s') COPY: numpy_ptr=0x%x nbytes=%d",
                          input_name, data.ctypes.data, data.nbytes)

        tensor = tvm_ffi.from_dlpack(data)
        self._mod.Blob_SetData(self._handle, input_name, tensor, zero_copy)

    def copy_from(self, weights_file: str):
        self._mod.Net_CopyTrainedLayersFrom(self._handle, weights_file)

    # ------------------------------------------------------------------
    # 内存统计
    # ------------------------------------------------------------------
    def memory_stats(self) -> dict:
        """Get C++ side memory/zero-copy statistics."""
        _find_lib()
        return {
            "live_nets": self._mod.LiveNetCount(),
            "live_tensors": self._mod.LiveTensorCount(),
            "total_tensor_allocs": self._mod.TotalTensorAllocs(),
            "zero_copy_hits": self._mod.ZeroCopyHits(),
            "memcpy_fallbacks": self._mod.MemcpyFallbacks(),
            "total_memcpy_bytes": self._mod.TotalMemcpyBytes(),
        }

    def log_memory_stats(self):
        """Print memory stats to C++ log."""
        self._mod.MemoryStats()


# ---------------------------------------------------------------------------
# 全局函数
# ---------------------------------------------------------------------------
def set_mode_cpu():
    _find_lib()
    _mod.SetModeCPU()

def set_random_seed(seed: int):
    _find_lib()
    _mod.SetRandomSeed(seed)

def layer_type_list() -> List[str]:
    _find_lib()
    return list(_mod.LayerTypeList())

def version() -> str:
    _find_lib()
    return _mod.Version()

def set_log_level(level: int) -> None:
    """Set C++ log level (0=TRACE, 1=DEBUG, 2=INFO, 3=WARN, 4=ERROR, 5=SILENT)."""
    _find_lib()
    _mod.SetLogLevel(level)
    _logger.debug("Log level set to %d", level)

def get_log_level() -> int:
    _find_lib()
    return int(_mod.GetLogLevel())

def enable_debug_logging(level: int = LOG_DEBUG):
    """Enable debug logging for both Python and C++ layers."""
    _logger.setLevel(logging.DEBUG)
    if not _logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
    set_log_level(level)

def disable_debug_logging():
    """Disable debug logging."""
    _logger.setLevel(logging.WARNING)
    set_log_level(LOG_WARN)


set_mode_cpu()
```

---

## 五、日志输出示例

设置 `set_log_level(0)` (TRACE) 后，运行前向传播可看到类似以下输出：

```
[caffe-slim][NET] INIT network_file=lenet.prototxt phase=1 live_nets=1  [Net_Init:??]
[caffe-slim][NET] INIT handle=0x2a3b1c0 blob_count=8 layer_count=7  [Net_Init:??]
[caffe-slim][BLOB] GET_DATA blob[data] ptr=0x7f9a2c000000 shape=1 1 28 28 (784) bytes=3136  [Blob_GetData:??]
[caffe-slim][TENSOR] CREATE tensor=0x7f9a2c000000 live=1 total=1 desc=[Blob.data:data]  [AllocData:??]
[caffe-slim][TENSOR] BIND zero-copy tensor=0x7f9a2c000000 -> blob[data].data ptr=0x7f9a2c000000 net_refcount=2  [AllocData:??]
[caffe-slim][MEM] SET_DATA zero-copy blob[data] src=0x7f9b10000000 dst=0x7f9a2c000000 nbytes=3136 (REPLACING internal pointer, NO memcpy)  [Blob_SetData:??]
[caffe-slim][MEM] SET_DATA zero-copy bound: blob[data].cpu_data()=0x7f9b10000000 (now points to external numpy buffer)  [Blob_SetData:??]
[caffe-slim][NET] FORWARD start handle=0x2a3b1c0 input_blobs=1 output_blobs=1  [Net_Forward:??]
[caffe-slim][NET] FORWARD end handle=0x2a3b1c0  [Net_Forward:??]
[caffe-slim][TENSOR] DESTROY tensor=0x7f9a2c000000 live=0 desc=[Blob.data:data]  [FreeData:??]
[caffe-slim][TENSOR] UNBIND zero-copy tensor=0x7f9a2c000000 from blob[data].data net_refcount_before=2  [FreeData:??]
```

---

## 六、关键日志埋点说明

| 日志标签 | 事件 | 记录内容 |
|---|---|---|
| `[TENSOR] CREATE` | Tensor 创建 | tensor地址、存活数、总创建数、描述 |
| `[TENSOR] DESTROY` | Tensor 销毁 | tensor地址、存活数、描述 |
| `[TENSOR] BIND` | 零拷贝绑定 | tensor地址→blob名+类型、C++指针、net引用计数 |
| `[TENSOR] UNBIND` | 零拷贝解绑 | tensor地址、解绑前引用计数 |
| `[TENSOR] EXT_BIND` | 外部指针绑定 | 写入零拷贝时建立的绑定 |
| `[MEM] SETDATA zero-copy` | 写入零拷贝 | src/dst地址、字节数、说明无memcpy |
| `[MEM] SETDATA memcpy` | 写入兜底拷贝 | src/dst地址、字节数、首/尾元素值验证 |
| `[NET] INIT/DESTROY` | 网络生命周期 | 文件名、phase、handle地址、blob/layer数 |
| `[NET] FORWARD` | 前向开始/结束 | handle地址、输入/输出blob数 |
| `[BLOB] GET_DATA/DIFF` | Blob读取 | blob名、C++指针、形状字符串、字节数 |
| `[MEM] STATS` | 内存统计快照 | 全部计数器值 |
| `[MEM] WARN` | 泄漏警告 | Net销毁时仍存活的tensor列表 |

---

## 七、改造注意事项与风险

1. **写入零拷贝的安全约束**：
   - `zero_copy=True` 时，numpy 数组必须在网络整个生命周期内保持存活
   - Python 层通过 `_zero_copy_tensors` 字典持有引用，但调用方不可手动 `del` 该数组
   - 推荐仅在高性能推理场景使用，默认 `zero_copy=False`（安全拷贝）

2. **`set_cpu_data` 语义**：
   - caffe-slim 的 `Blob<Dtype>::set_cpu_data(Dtype* data)` 将内部 `SyncedMemory` 指向外部指针
   - SyncedMemory 析构时不会释放这个外部指针（它只释放自己 `malloc` 的内存）
   - 需要在 `Net_Destroy` 前确保不会访问已释放的 numpy 内存

3. **日志编译期门控**：
   - 发布构建设置 `-DCAFFE_FFI_LOG_LEVEL=5` 可完全消除日志开销（编译器优化掉空分支）
   - 开发构建默认 `CAFFE_FFI_LOG_LEVEL=1` (DEBUG) 获得详细追踪

4. **向后兼容**：
   - `set_input_data(name, data)` 默认 `zero_copy=False`，行为与改造前完全一致
   - 所有现有 API（`blob_data`、`forward`、`blob_names` 等）签名不变
   - 新增 API：`memory_stats()`、`set_log_level()`、`enable_debug_logging()`、`zero_copy` 参数

---

## 八、Python 端使用示例

```python
import caffe
import numpy as np

# 启用详细日志（查看内存地址和引用计数）
caffe.enable_debug_logging(caffe.LOG_TRACE)

net = caffe.Net("lenet.prototxt", caffe.TEST, weights="lenet.caffemodel")

# 方式1：安全拷贝（默认，兼容原有代码）
input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)
net.set_input_data("data", input_data)  # zero_copy=False 默认
net.forward()
output = net.blob_data("prob")

# 方式2：零拷贝写入（高性能场景，注意引用持有）
input_buf = np.zeros((1, 1, 28, 28), dtype=np.float32)
net.set_input_data("data", input_buf, zero_copy=True)  # 无 memcpy！
input_buf[:] = np.random.randn(1, 1, 28, 28).astype(np.float32)  # 直接改 numpy
net.forward()
# input_buf 在 net 生命周期内不可释放

# 查看内存统计
stats = net.memory_stats()
print(f"零拷贝命中: {stats['zero_copy_hits']}, memcpy回退: {stats['memcpy_fallbacks']}")
print(f"总拷贝字节数: {stats['total_memcpy_bytes']}")
```
