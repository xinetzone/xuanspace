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

#include <tvm/ffi/memory.h>

#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <atomic>
#include <mutex>

#include "npu_ffi/vta/types.h"

namespace {

thread_local void* current_cmd_handle = nullptr;
thread_local unsigned current_debug_flags = 0;
std::unordered_map<void*, size_t> buffer_tracker;
std::mutex buffer_tracker_mutex;
std::atomic<uint64_t> next_handle_id{1};

}  // namespace

extern "C" {

void* npu_ffi_vta_buffer_alloc(size_t size) {
  void* ptr = nullptr;
  try {
    ptr = tvm::ffi::details::AlignedAlloc(size, npu_ffi::vta::kAllocAlignment);
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
  if (!buffer_tracker.empty()) {
    fprintf(stderr, "Warning: npu_ffi VTA stub runtime detected %zu leaked buffer(s) at shutdown\n",
            buffer_tracker.size());
  }
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
