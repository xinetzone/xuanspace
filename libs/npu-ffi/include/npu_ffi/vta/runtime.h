/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#pragma once

/*!
 * \file npu_ffi/vta/runtime.h
 * \brief Type-safe C++ wrapper for VTA runtime API.
 */

#include <cstddef>
#include <cstdint>

#include "npu_ffi/vta/buffer.h"
#include "npu_ffi/vta/command_context.h"
#include "npu_ffi/vta/handle.h"
#include "npu_ffi/vta/types.h"

namespace npu_ffi {
namespace vta {

/*!
 * \brief Get thread-local command handle.
 * \return Thread-local command handle.
 */
NPU_FFI_API CommandHandle tls_command_handle();

/*!
 * \brief Shutdown VTA runtime and cleanup resources.
 */
NPU_FFI_API void runtime_shutdown();

/*!
 * \brief Set debug mode on command handle.
 * \param cmd Command handle.
 * \param flags Debug flags (can be combined with operator|).
 */
NPU_FFI_API void set_debug_mode(CommandHandle cmd, DebugFlag flags);

/*!
 * \brief Synchronize command handle - commit instructions and wait for completion.
 * \param cmd Command handle.
 * \param wait_cycles Maximum poll cycles to wait (0 = wait indefinitely).
 */
NPU_FFI_API void synchronize(CommandHandle cmd, uint32_t wait_cycles = 0);

/*!
 * \brief Perform 2D data load from DRAM buffer to SRAM.
 *
 * Sizes are measured in units of vector elements.
 *
 * \param cmd Command handle.
 * \param src Source DRAM buffer.
 * \param src_elem_offset Source DRAM offset in number of unit elements.
 * \param x_size Lowest dimension (x axis) size in unit elements.
 * \param y_size Number of rows (y axis).
 * \param x_stride X axis stride.
 * \param x_pad_before Start padding on x axis.
 * \param y_pad_before Start padding on y axis.
 * \param x_pad_after End padding on x axis.
 * \param y_pad_after End padding on y axis.
 * \param dst_sram_index Destination SRAM index.
 * \param dst_memory_type Destination memory type.
 */
NPU_FFI_API void load_buffer_2d(CommandHandle cmd, const Buffer& src, uint32_t src_elem_offset,
                    uint32_t x_size, uint32_t y_size, uint32_t x_stride,
                    uint32_t x_pad_before, uint32_t y_pad_before,
                    uint32_t x_pad_after, uint32_t y_pad_after,
                    uint32_t dst_sram_index, MemoryType dst_memory_type);

/*!
 * \brief Perform 2D data store from SRAM to DRAM buffer.
 *
 * Sizes are measured in units of vector elements.
 *
 * \param cmd Command handle.
 * \param src_sram_index Source SRAM index.
 * \param src_memory_type Source memory type.
 * \param dst Destination DRAM buffer.
 * \param dst_elem_offset Destination DRAM offset in unit elements.
 * \param x_size Lowest dimension (x axis) size in unit elements.
 * \param y_size Number of rows.
 * \param x_stride X axis stride.
 */
NPU_FFI_API void store_buffer_2d(CommandHandle cmd, uint32_t src_sram_index, MemoryType src_memory_type,
                     Buffer& dst, uint32_t dst_elem_offset,
                     uint32_t x_size, uint32_t y_size, uint32_t x_stride);

/*!
 * \brief Push micro-op into kernel buffer.
 *
 * In GEMM mode (mode=0), does a blocked GEMM with 2D access pattern.
 * In ALU mode (mode=1), does a vectorized ALU operation with 2D access pattern.
 *
 * \param mode 0=GEMM mode, 1=ALU mode.
 * \param reset_out If 1, resets accumulator to 0 first.
 * \param dst_index Accumulator memory index.
 * \param src_index Input memory index (GEMM) / accumulator memory index (ALU).
 * \param wgt_index Weight memory index.
 * \param opcode ALU operation code (used in ALU mode).
 * \param use_imm If true, use immediate value in ALU mode.
 * \param imm_val Immediate value for ALU mode.
 */
NPU_FFI_API void uop_push(uint32_t mode, uint32_t reset_out, uint32_t dst_index, uint32_t src_index,
              uint32_t wgt_index, ALUOpcode opcode, bool use_imm, int32_t imm_val);

/*!
 * \brief Mark start of a micro-op loop.
 * \param extent Loop extent.
 * \param dst_factor Accumulator factor.
 * \param src_factor Input factor.
 * \param wgt_factor Weight factor.
 */
NPU_FFI_API void uop_loop_begin(uint32_t extent, uint32_t dst_factor = 0, uint32_t src_factor = 0,
                    uint32_t wgt_factor = 0);

/*!
 * \brief Mark end of a micro-op loop.
 */
NPU_FFI_API void uop_loop_end();

/*!
 * \brief Push dependence token.
 * \param cmd Command handle.
 * \param from_qid Source queue ID.
 * \param to_qid Destination queue ID.
 * \return 0 on success.
 */
NPU_FFI_API int dep_push(CommandHandle cmd, int from_qid, int to_qid);

/*!
 * \brief Pop dependence signal.
 * \param cmd Command handle.
 * \param from_qid Source queue ID.
 * \param to_qid Destination queue ID.
 * \return 0 on success.
 */
NPU_FFI_API int dep_pop(CommandHandle cmd, int from_qid, int to_qid);

/*!
 * \brief Perform write barrier to make memory region visible to CPU.
 * \param cmd Command handle.
 * \param buffer Head buffer pointer.
 * \param elem_bits Size in bits of each element.
 * \param start Start of the region (in elements).
 * \param extent End of the region (in elements).
 */
NPU_FFI_API void write_barrier(CommandHandle cmd, Buffer& buffer, uint32_t elem_bits, uint32_t start,
                   uint32_t extent);

/*!
 * \brief Perform read barrier to make memory region visible to VTA.
 * \param cmd Command handle.
 * \param buffer Head buffer pointer.
 * \param elem_bits Unit bits of each element.
 * \param start Start of the region (in elements).
 * \param extent End of the region (in elements).
 */
NPU_FFI_API void read_barrier(CommandHandle cmd, Buffer& buffer, uint32_t elem_bits, uint32_t start,
                  uint32_t extent);

/*!
 * \brief Copy data between buffers.
 * \param from Source buffer.
 * \param from_offset Offset in source buffer (bytes).
 * \param to Destination buffer.
 * \param to_offset Offset in destination buffer (bytes).
 * \param size Number of bytes to copy.
 * \param kind Copy direction (H2D/D2H/D2D).
 */
NPU_FFI_API void buffer_copy(const Buffer& from, size_t from_offset, Buffer& to, size_t to_offset,
                 size_t size, MemcpyKind kind);

}  // namespace vta
}  // namespace npu_ffi
