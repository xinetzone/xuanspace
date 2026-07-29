#ifndef CAFFE_FFI_COMMON_HPP_
#define CAFFE_FFI_COMMON_HPP_

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <tvm/ffi/tvm_ffi.h>

#include "caffe_ffi/log.hpp"

namespace caffe_ffi {

/** Global atomic counter tracking total bytes allocated across all Blobs. */
extern std::atomic<int64_t> g_total_allocated_bytes;

using namespace tvm::ffi;

class Blob;
class Layer;

/** Convenience alias: vector of Blob smart pointers. */
using BlobVec = std::vector<ObjectPtr<Blob>>;
/** Convenience alias: vector of Layer smart pointers. */
using LayerVec = std::vector<ObjectPtr<Layer>>;

/** Maximum number of axes (dimensions) a Blob can have. */
constexpr int kMaxBlobAxes = 32;

/**
 * @brief Convert a DLPack dtype code to a human-readable string.
 * @param code DLPack type code (kDLFloat, kDLInt, kDLUInt).
 * @return Static string name ("float", "int", "uint", or "unknown").
 */
inline const char* DTypeCodeToString(uint8_t code) {
  switch (code) {
    case kDLFloat:   return "float";
    case kDLInt:     return "int";
    case kDLUInt:    return "uint";
    default:         return "unknown";
  }
}

/**
 * @brief CPU memory allocator for TVM FFI Tensors.
 *
 * Allocates zero-initialized memory via std::malloc/std::free, tracks total
 * allocation bytes in g_total_allocated_bytes, and logs allocation/deallocation
 * events for debugging. Used by NewCPUTensor() to create Blob data/diff tensors.
 */
struct CPUMemAlloc {
  /**
   * @brief Allocate and zero-initialize CPU memory for a DLTensor.
   * @param tensor DLTensor whose shape/dtype determine allocation size; data pointer is set on success.
   */
  void AllocData(DLTensor* tensor) {
    size_t nbytes = tvm::ffi::GetDataSize(*tensor);
    CAFFE_FFI_MEM_LOG << "AllocData: allocating " << nbytes << " bytes"
                      << " (ndim=" << tensor->ndim
                      << ", dtype=" << DTypeCodeToString(tensor->dtype.code)
                      << static_cast<int>(tensor->dtype.bits)
                      << ", device_type=" << static_cast<int>(tensor->device.device_type) << ")";
    tensor->data = std::malloc(nbytes);
    TVM_FFI_ICHECK(tensor->data != nullptr) << "Failed to allocate CPU memory of size " << nbytes;
    std::memset(tensor->data, 0, nbytes);
    if (nbytes > 0) {
      g_total_allocated_bytes.fetch_add(static_cast<int64_t>(nbytes), std::memory_order_relaxed);
    }
    CAFFE_FFI_MEM_LOG << "AllocData: allocated at " << tensor->data << " (" << nbytes << " bytes, zero-initialized)"
                      << " global_total=" << g_total_allocated_bytes.load(std::memory_order_relaxed) << "B";
  }
  /**
   * @brief Free CPU memory previously allocated by AllocData.
   * @param tensor DLTensor whose data pointer will be freed and reset to nullptr.
   */
  void FreeData(DLTensor* tensor) {
    if (tensor->data) {
      size_t nbytes = tvm::ffi::GetDataSize(*tensor);
      CAFFE_FFI_MEM_LOG << "FreeData: freeing memory at " << tensor->data << " (" << nbytes << " bytes)";
      std::free(tensor->data);
      tensor->data = nullptr;
      if (nbytes > 0) {
        g_total_allocated_bytes.fetch_sub(static_cast<int64_t>(nbytes), std::memory_order_relaxed);
      }
      CAFFE_FFI_MEM_LOG << "FreeData: memory freed, data pointer reset to nullptr"
                        << " global_total=" << g_total_allocated_bytes.load(std::memory_order_relaxed) << "B";
    } else {
      CAFFE_FFI_MEM_LOG << "FreeData: data is already nullptr, skipping";
    }
  }
};

/** @return DLDevice descriptor for CPU (device_type=kDLCPU, device_id=0). */
inline DLDevice CPU() {
  DLDevice dev;
  dev.device_type = kDLCPU;
  dev.device_id = 0;
  return dev;
}

/** @return DLDataType descriptor for float32 (kDLFloat, 32 bits, 1 lane). */
inline DLDataType Float32() {
  DLDataType dtype;
  dtype.code = kDLFloat;
  dtype.bits = 32;
  dtype.lanes = 1;
  return dtype;
}

/**
 * @brief Create a new float32 CPU Tensor with the given shape.
 *
 * Allocates zero-initialized memory via CPUMemAlloc. The returned Tensor owns
 * its memory and will free it when the last reference is released.
 *
 * @param shape Tensor shape (dimensions).
 * @return New TVM FFI Tensor on CPU with float32 dtype.
 */
inline Tensor NewCPUTensor(ShapeView shape) {
  return Tensor::FromNDAlloc(CPUMemAlloc(), shape, Float32(), CPU());
}

}  // namespace caffe_ffi

#endif  // CAFFE_FFI_COMMON_HPP_
