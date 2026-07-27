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

/*!
 * \file npu_ffi/vta/types.h
 * \brief VTA type-safe enumerations and constants.
 */

#ifndef NPU_FFI_VTA_TYPES_H_
#define NPU_FFI_VTA_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace npu_ffi {
namespace vta {

/*!
 * \brief Memory copy kind for buffer copy operations.
 */
enum class MemcpyKind : int {
  /*! \brief Host to device copy */
  H2D = 1,
  /*! \brief Device to host copy */
  D2H = 2,
  /*! \brief Device to device copy */
  D2D = 3
};

/*!
 * \brief Debug flags for VTA command handle.
 *
 * These flags can be combined using bitwise OR operator.
 */
enum class DebugFlag : unsigned {
  /*! \brief Dump instructions */
  DUMP_INSN = 1U << 1,
  /*! \brief Dump micro-operations */
  DUMP_UOP = 1U << 2,
  /*! \brief Skip read barrier */
  SKIP_READ_BARRIER = 1U << 3,
  /*! \brief Skip write barrier */
  SKIP_WRITE_BARRIER = 1U << 4,
  /*! \brief Force serial execution */
  FORCE_SERIAL = 1U << 5,
  /*! \brief Sort force serial */
  SORT_FORCE_SERIAL = 1U << 6,
  /*! \brief Dump profiler information */
  DUMP_PROFILER = 1U << 7
};

/*!
 * \brief Bitwise OR operator for DebugFlag.
 * \param lhs Left-hand side flag.
 * \param rhs Right-hand side flag.
 * \return Combined flags.
 */
inline DebugFlag operator|(DebugFlag lhs, DebugFlag rhs) {
  return static_cast<DebugFlag>(
      static_cast<std::underlying_type_t<DebugFlag>>(lhs) |
      static_cast<std::underlying_type_t<DebugFlag>>(rhs));
}

/*!
 * \brief Bitwise AND operator for DebugFlag.
 * \param lhs Left-hand side flag.
 * \param rhs Right-hand side flag.
 * \return Intersection of flags.
 */
inline DebugFlag operator&(DebugFlag lhs, DebugFlag rhs) {
  return static_cast<DebugFlag>(
      static_cast<std::underlying_type_t<DebugFlag>>(lhs) &
      static_cast<std::underlying_type_t<DebugFlag>>(rhs));
}

/*!
 * \brief Bitwise OR assignment operator for DebugFlag.
 * \param lhs Left-hand side flag (modified in place).
 * \param rhs Right-hand side flag.
 * \return Reference to lhs.
 */
inline DebugFlag& operator|=(DebugFlag& lhs, DebugFlag rhs) {
  lhs = lhs | rhs;
  return lhs;
}

/*!
 * \brief Bitwise AND assignment operator for DebugFlag.
 * \param lhs Left-hand side flag (modified in place).
 * \param rhs Right-hand side flag.
 * \return Reference to lhs.
 */
inline DebugFlag& operator&=(DebugFlag& lhs, DebugFlag rhs) {
  lhs = lhs & rhs;
  return lhs;
}

/*!
 * \brief Check if any flag bit is set.
 * \param flags The flags to check.
 * \return True if flags is non-zero.
 */
inline bool any(DebugFlag flags) {
  return static_cast<std::underlying_type_t<DebugFlag>>(flags) != 0;
}

/*!
 * \brief VTA memory types.
 */
enum class MemoryType : uint32_t {
  /*! \brief DRAM memory */
  DRAM = 0,
  /*! \brief SRAM memory */
  SRAM = 1,
  /*! \brief Micro-op memory */
  UOP = 2,
  /*! \brief Input memory */
  INP = 3,
  /*! \brief Weight memory */
  WGT = 4,
  /*! \brief Accumulator memory */
  ACC = 5,
  /*! \brief Output memory */
  OUT = 6
};

/*!
 * \brief ALU operation codes.
 */
enum class ALUOpcode : uint32_t {
  /*! \brief Addition */
  ADD = 0,
  /*! \brief Subtraction */
  SUB = 1,
  /*! \brief Multiplication */
  MUL = 2,
  /*! \brief Minimum */
  MIN = 3,
  /*! \brief Maximum */
  MAX = 4,
  /*! \brief Shift right */
  SHR = 5,
  /*! \brief Shift left */
  SHL = 6
};

/*!
 * \brief Buffer allocation alignment requirement in bytes.
 */
constexpr size_t kAllocAlignment = 64;

}  // namespace vta
}  // namespace npu_ffi

#endif  // NPU_FFI_VTA_TYPES_H_
