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
 * \brief Lightweight compile-time controllable logging for npu-ffi and
 *        reusable by other FFI binding libraries.
 *
 * Logging is disabled by default for zero overhead in release builds.
 * Enable by defining NPU_FFI_ENABLE_LOG before including this header
 * or via CMake option NPU_FFI_ENABLE_LOG=ON.
 *
 * Features:
 *   - Four log levels: DEBUG, INFO, WARN, ERROR
 *   - Configurable module tag (prefix in [tag][LEVEL] format)
 *   - Zero-cost when disabled (macros expand to ((void)0))
 *   - Thread-safe via fprintf (no buffering)
 *   - Header-only, no linking required
 *
 * Log levels:
 *   NPU_FFI_LOG_DEBUG  - Verbose lifecycle tracing (alloc/free/sync); enabled only in debug builds
 *   NPU_FFI_LOG_INFO   - Important lifecycle events; enabled in debug builds
 *   NPU_FFI_LOG_WARN   - Warning conditions; always compiled in
 *   NPU_FFI_LOG_ERROR  - Error conditions; always compiled in
 *
 * Custom tag usage (for other projects reusing this header):
 *   // In your project's header, after including npu_ffi/logging.h:
 *   #define MYPROJ_LOG_DEBUG(fmt, ...)  NPU_FFI_LOG_TAG_DEBUG("myproj", fmt, ##__VA_ARGS__)
 *   #define MYPROJ_LOG_INFO(fmt, ...)   NPU_FFI_LOG_TAG_INFO("myproj", fmt, ##__VA_ARGS__)
 *   #define MYPROJ_LOG_WARN(fmt, ...)   NPU_FFI_LOG_TAG_WARN("myproj", fmt, ##__VA_ARGS__)
 *   #define MYPROJ_LOG_ERROR(fmt, ...)  NPU_FFI_LOG_TAG_ERROR("myproj", fmt, ##__VA_ARGS__)
 *
 * All logs go to stderr with consistent "[tag][LEVEL]" prefix.
 */

#include <cstdio>
#include <cstdlib>

/*!
 * \brief Internal tag-based debug/info log macro.
 *
 * Only emits output when NPU_FFI_ENABLE_LOG is defined.
 * WARN and ERROR are always emitted regardless of NPU_FFI_ENABLE_LOG.
 *
 * \param tag Module tag string (e.g., "npu-ffi").
 * \param level Log level string ("DEBUG", "INFO", "WARN", "ERROR").
 * \param fmt Printf-style format string.
 */
#if defined(NPU_FFI_ENABLE_LOG)
  #define NPU_FFI_LOG_TAG_INTERNAL(tag, level, fmt, ...) \
    std::fprintf(stderr, "[%s][%s] %s:%d: " fmt "\n", \
                 tag, level, __FILE__, __LINE__, ##__VA_ARGS__)
#else
  #define NPU_FFI_LOG_TAG_INTERNAL(tag, level, fmt, ...) ((void)0)
#endif

/*!
 * \brief Internal always-on log macro (for WARN/ERROR levels).
 */
#define NPU_FFI_LOG_TAG_ALWAYS(tag, level, fmt, ...) \
  std::fprintf(stderr, "[%s][%s] %s:%d: " fmt "\n", \
               tag, level, __FILE__, __LINE__, ##__VA_ARGS__)

/*!
 * \defgroup GenericTagLogs Generic tag-based logging macros
 * \brief These accept a custom tag parameter for use by other modules.
 *
 * Usage: NPU_FFI_LOG_TAG_DEBUG("mymodule", "value=%d", x);
 * @{
 */
#define NPU_FFI_LOG_TAG_DEBUG(tag, fmt, ...) \
  NPU_FFI_LOG_TAG_INTERNAL(tag, "DEBUG", fmt, ##__VA_ARGS__)

#define NPU_FFI_LOG_TAG_INFO(tag, fmt, ...) \
  NPU_FFI_LOG_TAG_INTERNAL(tag, "INFO", fmt, ##__VA_ARGS__)

#define NPU_FFI_LOG_TAG_WARN(tag, fmt, ...) \
  NPU_FFI_LOG_TAG_ALWAYS(tag, "WARN", fmt, ##__VA_ARGS__)

#define NPU_FFI_LOG_TAG_ERROR(tag, fmt, ...) \
  NPU_FFI_LOG_TAG_ALWAYS(tag, "ERROR", fmt, ##__VA_ARGS__)
/*! @} */

/*!
 * \defgroup NpuFfiLogs Default npu-ffi logging macros
 * \brief These use "npu-ffi" as the tag and are used throughout the library.
 * @{
 */

/*!
 * \brief Debug-level logging. Verbose lifecycle tracing.
 * Enabled only when NPU_FFI_ENABLE_LOG is defined.
 */
#define NPU_FFI_LOG_DEBUG(fmt, ...) \
  NPU_FFI_LOG_TAG_DEBUG("npu-ffi", fmt, ##__VA_ARGS__)

/*!
 * \brief Info-level logging. Important lifecycle events.
 * Enabled when NPU_FFI_ENABLE_LOG is defined (less verbose than DEBUG).
 */
#define NPU_FFI_LOG_INFO(fmt, ...) \
  NPU_FFI_LOG_TAG_INFO("npu-ffi", fmt, ##__VA_ARGS__)

/*!
 * \brief Warning-level logging. Always compiled in.
 * Used for recoverable issues like leaks, double-free attempts.
 */
#define NPU_FFI_LOG_WARN(fmt, ...) \
  NPU_FFI_LOG_TAG_WARN("npu-ffi", fmt, ##__VA_ARGS__)

/*!
 * \brief Error-level logging. Always compiled in.
 * Used for unrecoverable errors before throwing/aborting.
 */
#define NPU_FFI_LOG_ERROR(fmt, ...) \
  NPU_FFI_LOG_TAG_ERROR("npu-ffi", fmt, ##__VA_ARGS__)

/*! @} */
