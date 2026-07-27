// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "npu_ffi/vta/runtime.h"

#include <tvm/ffi/memory.h>

#include <cstring>
#include <new>
#include <unordered_map>
#include <atomic>
#include <mutex>

namespace npu_ffi {
namespace vta {
namespace stub {

namespace {

thread_local void* current_cmd_handle = nullptr;
thread_local unsigned current_debug_flags = 0;
static std::unordered_map<void*, size_t> buffer_tracker;
static std::mutex buffer_tracker_mutex;
static std::atomic<uint64_t> next_handle_id{1};

}  // namespace

extern "C" {

void* npu_ffi_vta_buffer_alloc(size_t size) {
  void* ptr = nullptr;
  try {
    ptr = tvm::ffi::details::AlignedAlloc(size, kAllocAlignment);
  } catch (const std::bad_alloc&) {
    return nullptr;
  }
  if (ptr != nullptr) {
    std::lock_guard<std::mutex> lock(buffer_tracker_mutex);
    buffer_tracker[ptr] = size;
  }
  return ptr;
}

void npu_ffi_vta_buffer_free(void* buffer) {
  if (buffer == nullptr) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(buffer_tracker_mutex);
    buffer_tracker.erase(buffer);
  }
  tvm::ffi::details::AlignedFree(buffer);
}

void npu_ffi_vta_buffer_copy(const void* from, size_t from_offset, void* to,
                             size_t to_offset, size_t size, int kind_mask) {
  (void)kind_mask;
  if (from == nullptr || to == nullptr || size == 0) {
    return;
  }
  const char* src = static_cast<const char*>(from) + from_offset;
  char* dst = static_cast<char*>(to) + to_offset;
  std::memcpy(dst, src, size);
}

void* npu_ffi_vta_buffer_cpu_ptr(void* cmd, void* buffer) {
  (void)cmd;
  return buffer;
}

void* npu_ffi_vta_tls_command_handle() {
  uint64_t id = next_handle_id.fetch_add(1, std::memory_order_relaxed);
  current_cmd_handle = reinterpret_cast<void*>(id);
  return current_cmd_handle;
}

void npu_ffi_vta_runtime_shutdown() {
  std::lock_guard<std::mutex> lock(buffer_tracker_mutex);
#ifndef NDEBUG
  if (!buffer_tracker.empty()) {
  }
#endif
  buffer_tracker.clear();
}

void npu_ffi_vta_set_debug_mode(void* cmd, int debug_flag) {
  (void)cmd;
  current_debug_flags = static_cast<unsigned>(debug_flag);
}

void npu_ffi_vta_load_buffer_2d(void* cmd, void* src_dram_addr,
                                 uint32_t src_elem_offset, uint32_t x_size,
                                 uint32_t y_size, uint32_t x_stride,
                                 uint32_t x_pad_before, uint32_t y_pad_before,
                                 uint32_t x_pad_after, uint32_t y_pad_after,
                                 uint32_t dst_sram_index,
                                 uint32_t dst_memory_type) {
  (void)cmd;
  (void)src_dram_addr;
  (void)src_elem_offset;
  (void)x_size;
  (void)y_size;
  (void)x_stride;
  (void)x_pad_before;
  (void)y_pad_before;
  (void)x_pad_after;
  (void)y_pad_after;
  (void)dst_sram_index;
  (void)dst_memory_type;
}

void npu_ffi_vta_store_buffer_2d(void* cmd, uint32_t src_sram_index,
                                  uint32_t src_memory_type, void* dst_dram_addr,
                                  uint32_t dst_elem_offset, uint32_t x_size,
                                  uint32_t y_size, uint32_t x_stride) {
  (void)cmd;
  (void)src_sram_index;
  (void)src_memory_type;
  (void)dst_dram_addr;
  (void)dst_elem_offset;
  (void)x_size;
  (void)y_size;
  (void)x_stride;
}

void npu_ffi_vta_uop_push(uint32_t mode, uint32_t reset_out, uint32_t dst_index,
                           uint32_t src_index, uint32_t wgt_index,
                           uint32_t opcode, uint32_t use_imm, int32_t imm_val) {
  (void)mode;
  (void)reset_out;
  (void)dst_index;
  (void)src_index;
  (void)wgt_index;
  (void)opcode;
  (void)use_imm;
  (void)imm_val;
}

void npu_ffi_vta_uop_loop_begin(uint32_t extent, uint32_t dst_factor,
                                 uint32_t src_factor, uint32_t wgt_factor) {
  (void)extent;
  (void)dst_factor;
  (void)src_factor;
  (void)wgt_factor;
}

void npu_ffi_vta_uop_loop_end() {
}

int npu_ffi_vta_push_gemm_op(void** uop_handle, int (*finit)(void*),
                              void* signature, int nbytes) {
  (void)uop_handle;
  (void)finit;
  (void)signature;
  (void)nbytes;
  return 0;
}

int npu_ffi_vta_push_alu_op(void** uop_handle, int (*finit)(void*),
                             void* signature, int nbytes) {
  (void)uop_handle;
  (void)finit;
  (void)signature;
  (void)nbytes;
  return 0;
}

int npu_ffi_vta_dep_push(void* cmd, int from_qid, int to_qid) {
  (void)cmd;
  (void)from_qid;
  (void)to_qid;
  return 0;
}

int npu_ffi_vta_dep_pop(void* cmd, int from_qid, int to_qid) {
  (void)cmd;
  (void)from_qid;
  (void)to_qid;
  return 0;
}

void npu_ffi_vta_synchronize(void* cmd, uint32_t wait_cycles) {
  (void)cmd;
  (void)wait_cycles;
}

void npu_ffi_vta_write_barrier(void* cmd, void* buffer, uint32_t elem_bits,
                                uint32_t start, uint32_t extent) {
  (void)cmd;
  (void)buffer;
  (void)elem_bits;
  (void)start;
  (void)extent;
}

void npu_ffi_vta_read_barrier(void* cmd, void* buffer, uint32_t elem_bits,
                               uint32_t start, uint32_t extent) {
  (void)cmd;
  (void)buffer;
  (void)elem_bits;
  (void)start;
  (void)extent;
}

void npu_ffi_vta_prepare_call_func(void* cmd, const char* name) {
  (void)cmd;
  (void)name;
}

}  // extern "C"

}  // namespace stub

Buffer::Buffer(size_t size)
    : data_(stub::npu_ffi_vta_buffer_alloc(size)), size_(size), owns_(true) {}

Buffer::Buffer(void* data, size_t size, bool owns)
    : data_(data), size_(size), owns_(owns) {}

Buffer::~Buffer() { reset(); }

Buffer::Buffer(Buffer&& other) noexcept
    : data_(other.data_), size_(other.size_), owns_(other.owns_) {
  other.data_ = nullptr;
  other.size_ = 0;
  other.owns_ = false;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  reset();
  data_ = other.data_;
  size_ = other.size_;
  owns_ = other.owns_;
  other.data_ = nullptr;
  other.size_ = 0;
  other.owns_ = false;
  return *this;
}

void* Buffer::cpu_ptr(CommandHandle cmd) const {
  return stub::npu_ffi_vta_buffer_cpu_ptr(cmd.get(), data_);
}

void Buffer::reset() {
  if (owns_ && data_) {
    stub::npu_ffi_vta_buffer_free(data_);
  }
  data_ = nullptr;
  size_ = 0;
  owns_ = false;
}

CommandHandle tls_command_handle() {
  return CommandHandle(stub::npu_ffi_vta_tls_command_handle());
}

void runtime_shutdown() { stub::npu_ffi_vta_runtime_shutdown(); }

void set_debug_mode(CommandHandle cmd, DebugFlag flags) {
  stub::npu_ffi_vta_set_debug_mode(
      cmd.get(), static_cast<int>(static_cast<std::underlying_type_t<DebugFlag>>(flags)));
}

void synchronize(CommandHandle cmd, uint32_t wait_cycles) {
  stub::npu_ffi_vta_synchronize(cmd.get(), wait_cycles);
}

void load_buffer_2d(CommandHandle cmd, const Buffer& src, uint32_t src_elem_offset,
                    uint32_t x_size, uint32_t y_size, uint32_t x_stride,
                    uint32_t x_pad_before, uint32_t y_pad_before,
                    uint32_t x_pad_after, uint32_t y_pad_after,
                    uint32_t dst_sram_index, MemoryType dst_memory_type) {
  stub::npu_ffi_vta_load_buffer_2d(
      cmd.get(), src.get(), src_elem_offset, x_size, y_size, x_stride,
      x_pad_before, y_pad_before, x_pad_after, y_pad_after, dst_sram_index,
      static_cast<uint32_t>(dst_memory_type));
}

void store_buffer_2d(CommandHandle cmd, uint32_t src_sram_index,
                     MemoryType src_memory_type, Buffer& dst,
                     uint32_t dst_elem_offset, uint32_t x_size, uint32_t y_size,
                     uint32_t x_stride) {
  stub::npu_ffi_vta_store_buffer_2d(cmd.get(), src_sram_index,
                                     static_cast<uint32_t>(src_memory_type),
                                     dst.get(), dst_elem_offset, x_size, y_size,
                                     x_stride);
}

void uop_push(uint32_t mode, uint32_t reset_out, uint32_t dst_index,
              uint32_t src_index, uint32_t wgt_index, ALUOpcode opcode,
              bool use_imm, int32_t imm_val) {
  stub::npu_ffi_vta_uop_push(mode, reset_out, dst_index, src_index, wgt_index,
                              static_cast<uint32_t>(opcode),
                              use_imm ? 1U : 0U, imm_val);
}

void uop_loop_begin(uint32_t extent, uint32_t dst_factor, uint32_t src_factor,
                    uint32_t wgt_factor) {
  stub::npu_ffi_vta_uop_loop_begin(extent, dst_factor, src_factor, wgt_factor);
}

void uop_loop_end() { stub::npu_ffi_vta_uop_loop_end(); }

int dep_push(CommandHandle cmd, int from_qid, int to_qid) {
  return stub::npu_ffi_vta_dep_push(cmd.get(), from_qid, to_qid);
}

int dep_pop(CommandHandle cmd, int from_qid, int to_qid) {
  return stub::npu_ffi_vta_dep_pop(cmd.get(), from_qid, to_qid);
}

void write_barrier(CommandHandle cmd, Buffer& buffer, uint32_t elem_bits,
                   uint32_t start, uint32_t extent) {
  stub::npu_ffi_vta_write_barrier(cmd.get(), buffer.get(), elem_bits, start,
                                   extent);
}

void read_barrier(CommandHandle cmd, Buffer& buffer, uint32_t elem_bits,
                  uint32_t start, uint32_t extent) {
  stub::npu_ffi_vta_read_barrier(cmd.get(), buffer.get(), elem_bits, start,
                                  extent);
}

void buffer_copy(const Buffer& from, size_t from_offset, Buffer& to,
                 size_t to_offset, size_t size, MemcpyKind kind) {
  stub::npu_ffi_vta_buffer_copy(from.get(), from_offset, to.get(), to_offset,
                                 size, static_cast<int>(kind));
}

}  // namespace vta
}  // namespace npu_ffi
