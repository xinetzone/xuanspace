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

#include <vta/runtime/runtime.h>

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
