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

#include <tvm/ffi/tvm_ffi.h>

#include <cstddef>
#include <cstdint>

extern "C" {

void* npu_ffi_vta_buffer_alloc(size_t size);
void npu_ffi_vta_buffer_free(void* buffer);
void npu_ffi_vta_buffer_copy(const void* from, size_t from_offset, void* to,
                             size_t to_offset, size_t size, int kind_mask);
void* npu_ffi_vta_buffer_cpu_ptr(void* cmd, void* buffer);
void* npu_ffi_vta_tls_command_handle();
void npu_ffi_vta_runtime_shutdown();
void npu_ffi_vta_set_debug_mode(void* cmd, int debug_flag);
void npu_ffi_vta_load_buffer_2d(void* cmd, void* src_dram_addr,
                                 uint32_t src_elem_offset, uint32_t x_size,
                                 uint32_t y_size, uint32_t x_stride,
                                 uint32_t x_pad_before, uint32_t y_pad_before,
                                 uint32_t x_pad_after, uint32_t y_pad_after,
                                 uint32_t dst_sram_index,
                                 uint32_t dst_memory_type);
void npu_ffi_vta_store_buffer_2d(void* cmd, uint32_t src_sram_index,
                                  uint32_t src_memory_type, void* dst_dram_addr,
                                  uint32_t dst_elem_offset, uint32_t x_size,
                                  uint32_t y_size, uint32_t x_stride);
void npu_ffi_vta_uop_push(uint32_t mode, uint32_t reset_out, uint32_t dst_index,
                           uint32_t src_index, uint32_t wgt_index,
                           uint32_t opcode, uint32_t use_imm, int32_t imm_val);
void npu_ffi_vta_uop_loop_begin(uint32_t extent, uint32_t dst_factor,
                                 uint32_t src_factor, uint32_t wgt_factor);
void npu_ffi_vta_uop_loop_end();
int npu_ffi_vta_push_gemm_op(void** uop_handle, int (*finit)(void*),
                              void* signature, int nbytes);
int npu_ffi_vta_push_alu_op(void** uop_handle, int (*finit)(void*),
                             void* signature, int nbytes);
int npu_ffi_vta_dep_push(void* cmd, int from_qid, int to_qid);
int npu_ffi_vta_dep_pop(void* cmd, int from_qid, int to_qid);
void npu_ffi_vta_synchronize(void* cmd, uint32_t wait_cycles);
void npu_ffi_vta_write_barrier(void* cmd, void* buffer, uint32_t elem_bits,
                                uint32_t start, uint32_t extent);
void npu_ffi_vta_read_barrier(void* cmd, void* buffer, uint32_t elem_bits,
                               uint32_t start, uint32_t extent);
void npu_ffi_vta_prepare_call_func(void* cmd, const char* name);

}  // extern "C"

namespace npu_ffi {
namespace vta {
namespace ffi_registry {

int64_t buffer_alloc(int64_t size) {
  return reinterpret_cast<int64_t>(npu_ffi_vta_buffer_alloc(static_cast<size_t>(size)));
}

void buffer_free(int64_t ptr) {
  npu_ffi_vta_buffer_free(reinterpret_cast<void*>(static_cast<intptr_t>(ptr)));
}

void buffer_copy(int64_t from, int64_t from_offset, int64_t to,
                 int64_t to_offset, int64_t size, int kind_mask) {
  npu_ffi_vta_buffer_copy(
      reinterpret_cast<const void*>(static_cast<intptr_t>(from)),
      static_cast<size_t>(from_offset),
      reinterpret_cast<void*>(static_cast<intptr_t>(to)),
      static_cast<size_t>(to_offset),
      static_cast<size_t>(size),
      kind_mask);
}

int64_t buffer_cpu_ptr(int64_t cmd, int64_t buffer) {
  return reinterpret_cast<int64_t>(npu_ffi_vta_buffer_cpu_ptr(
      reinterpret_cast<void*>(static_cast<intptr_t>(cmd)),
      reinterpret_cast<void*>(static_cast<intptr_t>(buffer))));
}

int64_t tls_command_handle() {
  return reinterpret_cast<int64_t>(npu_ffi_vta_tls_command_handle());
}

void runtime_shutdown() {
  npu_ffi_vta_runtime_shutdown();
}

void set_debug_mode(int64_t cmd, int debug_flag) {
  npu_ffi_vta_set_debug_mode(reinterpret_cast<void*>(static_cast<intptr_t>(cmd)), debug_flag);
}

void load_buffer_2d(int64_t cmd, int64_t src_dram_addr, int src_elem_offset,
                    int x_size, int y_size, int x_stride,
                    int x_pad_before, int y_pad_before,
                    int x_pad_after, int y_pad_after,
                    int dst_sram_index, int dst_memory_type) {
  npu_ffi_vta_load_buffer_2d(
      reinterpret_cast<void*>(static_cast<intptr_t>(cmd)),
      reinterpret_cast<void*>(static_cast<intptr_t>(src_dram_addr)),
      static_cast<uint32_t>(src_elem_offset),
      static_cast<uint32_t>(x_size),
      static_cast<uint32_t>(y_size),
      static_cast<uint32_t>(x_stride),
      static_cast<uint32_t>(x_pad_before),
      static_cast<uint32_t>(y_pad_before),
      static_cast<uint32_t>(x_pad_after),
      static_cast<uint32_t>(y_pad_after),
      static_cast<uint32_t>(dst_sram_index),
      static_cast<uint32_t>(dst_memory_type));
}

void store_buffer_2d(int64_t cmd, int src_sram_index, int src_memory_type,
                     int64_t dst_dram_addr, int dst_elem_offset,
                     int x_size, int y_size, int x_stride) {
  npu_ffi_vta_store_buffer_2d(
      reinterpret_cast<void*>(static_cast<intptr_t>(cmd)),
      static_cast<uint32_t>(src_sram_index),
      static_cast<uint32_t>(src_memory_type),
      reinterpret_cast<void*>(static_cast<intptr_t>(dst_dram_addr)),
      static_cast<uint32_t>(dst_elem_offset),
      static_cast<uint32_t>(x_size),
      static_cast<uint32_t>(y_size),
      static_cast<uint32_t>(x_stride));
}

void uop_push(int mode, int reset_out, int dst_index,
              int src_index, int wgt_index,
              int opcode, int use_imm, int imm_val) {
  npu_ffi_vta_uop_push(
      static_cast<uint32_t>(mode),
      static_cast<uint32_t>(reset_out),
      static_cast<uint32_t>(dst_index),
      static_cast<uint32_t>(src_index),
      static_cast<uint32_t>(wgt_index),
      static_cast<uint32_t>(opcode),
      static_cast<uint32_t>(use_imm),
      static_cast<int32_t>(imm_val));
}

void uop_loop_begin(int extent, int dst_factor, int src_factor, int wgt_factor) {
  npu_ffi_vta_uop_loop_begin(
      static_cast<uint32_t>(extent),
      static_cast<uint32_t>(dst_factor),
      static_cast<uint32_t>(src_factor),
      static_cast<uint32_t>(wgt_factor));
}

void uop_loop_end() {
  npu_ffi_vta_uop_loop_end();
}

int push_gemm_op() {
  return 0;
}

int push_alu_op() {
  return 0;
}

int dep_push(int64_t cmd, int from_qid, int to_qid) {
  return npu_ffi_vta_dep_push(reinterpret_cast<void*>(static_cast<intptr_t>(cmd)), from_qid, to_qid);
}

int dep_pop(int64_t cmd, int from_qid, int to_qid) {
  return npu_ffi_vta_dep_pop(reinterpret_cast<void*>(static_cast<intptr_t>(cmd)), from_qid, to_qid);
}

void synchronize(int64_t cmd, int wait_cycles) {
  npu_ffi_vta_synchronize(reinterpret_cast<void*>(static_cast<intptr_t>(cmd)),
                          static_cast<uint32_t>(wait_cycles));
}

void write_barrier(int64_t cmd, int64_t buffer, int elem_bits,
                   int start, int extent) {
  npu_ffi_vta_write_barrier(
      reinterpret_cast<void*>(static_cast<intptr_t>(cmd)),
      reinterpret_cast<void*>(static_cast<intptr_t>(buffer)),
      static_cast<uint32_t>(elem_bits),
      static_cast<uint32_t>(start),
      static_cast<uint32_t>(extent));
}

void read_barrier(int64_t cmd, int64_t buffer, int elem_bits,
                  int start, int extent) {
  npu_ffi_vta_read_barrier(
      reinterpret_cast<void*>(static_cast<intptr_t>(cmd)),
      reinterpret_cast<void*>(static_cast<intptr_t>(buffer)),
      static_cast<uint32_t>(elem_bits),
      static_cast<uint32_t>(start),
      static_cast<uint32_t>(extent));
}

void prepare_call_func(int64_t cmd, const tvm::ffi::String& name) {
  npu_ffi_vta_prepare_call_func(reinterpret_cast<void*>(static_cast<intptr_t>(cmd)), name.c_str());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("vta.buffer_alloc", buffer_alloc)
      .def("vta.buffer_free", buffer_free)
      .def("vta.buffer_copy", buffer_copy)
      .def("vta.buffer_cpu_ptr", buffer_cpu_ptr)
      .def("vta.tls_command_handle", tls_command_handle)
      .def("vta.runtime_shutdown", runtime_shutdown)
      .def("vta.set_debug_mode", set_debug_mode)
      .def("vta.load_buffer_2d", load_buffer_2d)
      .def("vta.store_buffer_2d", store_buffer_2d)
      .def("vta.uop_push", uop_push)
      .def("vta.uop_loop_begin", uop_loop_begin)
      .def("vta.uop_loop_end", uop_loop_end)
      .def("vta.push_gemm_op", push_gemm_op)
      .def("vta.push_alu_op", push_alu_op)
      .def("vta.dep_push", dep_push)
      .def("vta.dep_pop", dep_pop)
      .def("vta.synchronize", synchronize)
      .def("vta.write_barrier", write_barrier)
      .def("vta.read_barrier", read_barrier)
      .def("vta.prepare_call_func", prepare_call_func);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(vta_buffer_alloc, buffer_alloc);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vta_buffer_free, buffer_free);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vta_buffer_copy, buffer_copy);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vta_buffer_cpu_ptr, buffer_cpu_ptr);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vta_tls_command_handle, tls_command_handle);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vta_runtime_shutdown, runtime_shutdown);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vta_set_debug_mode, set_debug_mode);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vta_load_buffer_2d, load_buffer_2d);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vta_store_buffer_2d, store_buffer_2d);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vta_uop_push, uop_push);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vta_uop_loop_begin, uop_loop_begin);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vta_uop_loop_end, uop_loop_end);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vta_push_gemm_op, push_gemm_op);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vta_push_alu_op, push_alu_op);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vta_dep_push, dep_push);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vta_dep_pop, dep_pop);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vta_synchronize, synchronize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vta_write_barrier, write_barrier);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vta_read_barrier, read_barrier);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vta_prepare_call_func, prepare_call_func);

}  // namespace ffi_registry
}  // namespace vta
}  // namespace npu_ffi
