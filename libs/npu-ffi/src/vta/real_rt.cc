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

#include <vta/runtime/runtime.h>

namespace npu_ffi {
namespace vta {
namespace real {

extern "C" {

void* npu_ffi_vta_buffer_alloc(size_t size) {
  return VTABufferAlloc(size);
}

void npu_ffi_vta_buffer_free(void* buffer) {
  VTABufferFree(buffer);
}

void npu_ffi_vta_buffer_copy(const void* from, size_t from_offset, void* to,
                             size_t to_offset, size_t size, int kind_mask) {
  VTABufferCopy(from, from_offset, to, to_offset, size, kind_mask);
}

void* npu_ffi_vta_buffer_cpu_ptr(void* cmd, void* buffer) {
  return VTABufferCPUPtr(cmd, buffer);
}

void* npu_ffi_vta_tls_command_handle() {
  return VTATLSCommandHandle();
}

void npu_ffi_vta_runtime_shutdown() {
  VTARuntimeShutdown();
}

void npu_ffi_vta_set_debug_mode(void* cmd, int debug_flag) {
  VTASetDebugMode(cmd, debug_flag);
}

void npu_ffi_vta_load_buffer_2d(void* cmd, void* src_dram_addr,
                                 uint32_t src_elem_offset, uint32_t x_size,
                                 uint32_t y_size, uint32_t x_stride,
                                 uint32_t x_pad_before, uint32_t y_pad_before,
                                 uint32_t x_pad_after, uint32_t y_pad_after,
                                 uint32_t dst_sram_index,
                                 uint32_t dst_memory_type) {
  VTALoadBuffer2D(cmd, src_dram_addr, src_elem_offset, x_size, y_size, x_stride,
                  x_pad_before, y_pad_before, x_pad_after, y_pad_after,
                  dst_sram_index, dst_memory_type);
}

void npu_ffi_vta_store_buffer_2d(void* cmd, uint32_t src_sram_index,
                                  uint32_t src_memory_type, void* dst_dram_addr,
                                  uint32_t dst_elem_offset, uint32_t x_size,
                                  uint32_t y_size, uint32_t x_stride) {
  VTAStoreBuffer2D(cmd, src_sram_index, src_memory_type, dst_dram_addr,
                   dst_elem_offset, x_size, y_size, x_stride);
}

void npu_ffi_vta_uop_push(uint32_t mode, uint32_t reset_out, uint32_t dst_index,
                           uint32_t src_index, uint32_t wgt_index,
                           uint32_t opcode, uint32_t use_imm, int32_t imm_val) {
  VTAUopPush(mode, reset_out, dst_index, src_index, wgt_index, opcode, use_imm, imm_val);
}

void npu_ffi_vta_uop_loop_begin(uint32_t extent, uint32_t dst_factor,
                                 uint32_t src_factor, uint32_t wgt_factor) {
  VTAUopLoopBegin(extent, dst_factor, src_factor, wgt_factor);
}

void npu_ffi_vta_uop_loop_end() {
  VTAUopLoopEnd();
}

int npu_ffi_vta_push_gemm_op(void** uop_handle, int (*finit)(void*),
                              void* signature, int nbytes) {
  return VTAPushGEMMOp(uop_handle, finit, signature, nbytes);
}

int npu_ffi_vta_push_alu_op(void** uop_handle, int (*finit)(void*),
                             void* signature, int nbytes) {
  return VTAPushALUOp(uop_handle, finit, signature, nbytes);
}

int npu_ffi_vta_dep_push(void* cmd, int from_qid, int to_qid) {
  return VTADepPush(cmd, from_qid, to_qid);
}

int npu_ffi_vta_dep_pop(void* cmd, int from_qid, int to_qid) {
  return VTADepPop(cmd, from_qid, to_qid);
}

void npu_ffi_vta_synchronize(void* cmd, uint32_t wait_cycles) {
  VTASynchronize(cmd, wait_cycles);
}

void npu_ffi_vta_write_barrier(void* cmd, void* buffer, uint32_t elem_bits,
                                uint32_t start, uint32_t extent) {
  VTAWriteBarrier(cmd, buffer, elem_bits, start, extent);
}

void npu_ffi_vta_read_barrier(void* cmd, void* buffer, uint32_t elem_bits,
                               uint32_t start, uint32_t extent) {
  VTAReadBarrier(cmd, buffer, elem_bits, start, extent);
}

void npu_ffi_vta_prepare_call_func(void* cmd, const char* name) {
  VTAPrepareCallFunc(cmd, name);
}

}  // extern "C"

}  // namespace real

Buffer::Buffer(size_t size)
    : data_(real::npu_ffi_vta_buffer_alloc(size)), size_(size), owns_(true) {}

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
  return real::npu_ffi_vta_buffer_cpu_ptr(cmd.get(), data_);
}

void Buffer::reset() {
  if (owns_ && data_) {
    real::npu_ffi_vta_buffer_free(data_);
  }
  data_ = nullptr;
  size_ = 0;
  owns_ = false;
}

CommandHandle tls_command_handle() {
  return CommandHandle(real::npu_ffi_vta_tls_command_handle());
}

void runtime_shutdown() { real::npu_ffi_vta_runtime_shutdown(); }

void set_debug_mode(CommandHandle cmd, DebugFlag flags) {
  real::npu_ffi_vta_set_debug_mode(
      cmd.get(), static_cast<int>(static_cast<std::underlying_type_t<DebugFlag>>(flags)));
}

void synchronize(CommandHandle cmd, uint32_t wait_cycles) {
  real::npu_ffi_vta_synchronize(cmd.get(), wait_cycles);
}

void load_buffer_2d(CommandHandle cmd, const Buffer& src, uint32_t src_elem_offset,
                    uint32_t x_size, uint32_t y_size, uint32_t x_stride,
                    uint32_t x_pad_before, uint32_t y_pad_before,
                    uint32_t x_pad_after, uint32_t y_pad_after,
                    uint32_t dst_sram_index, MemoryType dst_memory_type) {
  real::npu_ffi_vta_load_buffer_2d(
      cmd.get(), src.get(), src_elem_offset, x_size, y_size, x_stride,
      x_pad_before, y_pad_before, x_pad_after, y_pad_after, dst_sram_index,
      static_cast<uint32_t>(dst_memory_type));
}

void store_buffer_2d(CommandHandle cmd, uint32_t src_sram_index,
                     MemoryType src_memory_type, Buffer& dst,
                     uint32_t dst_elem_offset, uint32_t x_size, uint32_t y_size,
                     uint32_t x_stride) {
  real::npu_ffi_vta_store_buffer_2d(cmd.get(), src_sram_index,
                                     static_cast<uint32_t>(src_memory_type),
                                     dst.get(), dst_elem_offset, x_size, y_size,
                                     x_stride);
}

void uop_push(uint32_t mode, uint32_t reset_out, uint32_t dst_index,
              uint32_t src_index, uint32_t wgt_index, ALUOpcode opcode,
              bool use_imm, int32_t imm_val) {
  real::npu_ffi_vta_uop_push(mode, reset_out, dst_index, src_index, wgt_index,
                              static_cast<uint32_t>(opcode),
                              use_imm ? 1U : 0U, imm_val);
}

void uop_loop_begin(uint32_t extent, uint32_t dst_factor, uint32_t src_factor,
                    uint32_t wgt_factor) {
  real::npu_ffi_vta_uop_loop_begin(extent, dst_factor, src_factor, wgt_factor);
}

void uop_loop_end() { real::npu_ffi_vta_uop_loop_end(); }

int dep_push(CommandHandle cmd, int from_qid, int to_qid) {
  return real::npu_ffi_vta_dep_push(cmd.get(), from_qid, to_qid);
}

int dep_pop(CommandHandle cmd, int from_qid, int to_qid) {
  return real::npu_ffi_vta_dep_pop(cmd.get(), from_qid, to_qid);
}

void write_barrier(CommandHandle cmd, Buffer& buffer, uint32_t elem_bits,
                   uint32_t start, uint32_t extent) {
  real::npu_ffi_vta_write_barrier(cmd.get(), buffer.get(), elem_bits, start,
                                   extent);
}

void read_barrier(CommandHandle cmd, Buffer& buffer, uint32_t elem_bits,
                  uint32_t start, uint32_t extent) {
  real::npu_ffi_vta_read_barrier(cmd.get(), buffer.get(), elem_bits, start,
                                  extent);
}

void buffer_copy(const Buffer& from, size_t from_offset, Buffer& to,
                 size_t to_offset, size_t size, MemcpyKind kind) {
  real::npu_ffi_vta_buffer_copy(from.get(), from_offset, to.get(), to_offset,
                                 size, static_cast<int>(kind));
}

}  // namespace vta
}  // namespace npu_ffi
