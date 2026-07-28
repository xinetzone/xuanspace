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
 * \file npu_ffi/logging.h
 * \brief Lightweight compile-time controllable logging for npu-ffi.
 *
 * Logging is disabled by default for zero overhead in release builds.
 * Enable by defining NPU_FFI_ENABLE_LOG before including this header
 * or via CMake option NPU_FFI_ENABLE_LOG=ON.
 *
 * Log levels:
 *   NPU_FFI_LOG_DEBUG  - Verbose lifecycle tracing (alloc/free/sync)
 *   NPU_FFI_LOG_INFO   - Important lifecycle events
 *   NPU_FFI_LOG_WARN   - Warning conditions (leaks, errors handled)
 *   NPU_FFI_LOG_ERROR  - Error conditions
 *
 * All logs go to stderr with consistent "[npu-ffi][LEVEL]" prefix.
 */

#include <cstdio>
#include <cstdlib>

#if defined(NPU_FFI_ENABLE_LOG)
  /*!
   * \brief Internal log macro - do not use directly.
   * \param level Log level string (DEBUG/INFO/WARN/ERROR).
   * \param fmt Printf-style format string.
   */
  #define NPU_FFI_LOG_INTERNAL(level, fmt, ...) \
    std::fprintf(stderr, "[npu-ffi][%s] %s:%d: " fmt "\n", \
                 level, __FILE__, __LINE__, ##__VA_ARGS__)
#else
  #define NPU_FFI_LOG_INTERNAL(level, fmt, ...) ((void)0)
#endif

/*!
 * \brief Debug-level logging. Verbose lifecycle tracing.
 * Enabled only when NPU_FFI_ENABLE_LOG is defined.
 */
#define NPU_FFI_LOG_DEBUG(fmt, ...) \
  NPU_FFI_LOG_INTERNAL("DEBUG", fmt, ##__VA_ARGS__)

/*!
 * \brief Info-level logging. Important lifecycle events.
 * Enabled when NPU_FFI_ENABLE_LOG is defined (less verbose than DEBUG).
 */
#define NPU_FFI_LOG_INFO(fmt, ...) \
  NPU_FFI_LOG_INTERNAL("INFO", fmt, ##__VA_ARGS__)

/*!
 * \brief Warning-level logging. Always compiled in.
 * Used for recoverable issues like leaks, double-free attempts.
 */
#define NPU_FFI_LOG_WARN(fmt, ...) \
  std::fprintf(stderr, "[npu-ffi][WARN] %s:%d: " fmt "\n", \
               __FILE__, __LINE__, ##__VA_ARGS__)

/*!
 * \brief Error-level logging. Always compiled in.
 * Used for unrecoverable errors before throwing/aborting.
 */
#define NPU_FFI_LOG_ERROR(fmt, ...) \
  std::fprintf(stderr, "[npu-ffi][ERROR] %s:%d: " fmt "\n", \
               __FILE__, __LINE__, ##__VA_ARGS__)
