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
 * \file npu_ffi/npu_ffi.h
 * \brief Main header file for NPU FFI - include this to get all VTA APIs.
 */

/*!
 * \brief NPU FFI version major number.
 */
#define NPU_FFI_VERSION_MAJOR 0

/*!
 * \brief NPU FFI version minor number.
 */
#define NPU_FFI_VERSION_MINOR 1

/*!
 * \brief NPU FFI version patch number.
 */
#define NPU_FFI_VERSION_PATCH 0

/*!
 * \brief Helper macro for stringifying version numbers.
 */
#define NPU_FFI_STR_HELPER(x) #x

/*!
 * \brief Stringify macro.
 */
#define NPU_FFI_STR(x) NPU_FFI_STR_HELPER(x)

/*!
 * \brief NPU FFI version string in "major.minor.patch" format.
 */
#define NPU_FFI_VERSION_STRING \
  NPU_FFI_STR(NPU_FFI_VERSION_MAJOR) "." \
  NPU_FFI_STR(NPU_FFI_VERSION_MINOR) "." \
  NPU_FFI_STR(NPU_FFI_VERSION_PATCH)

/*!
 * \brief Platform detection macros.
 */
#if defined(_WIN32) || defined(_WIN64)
  #define NPU_FFI_PLATFORM_WINDOWS 1
  #define NPU_FFI_PLATFORM_POSIX 0
#elif defined(__linux__) || defined(__APPLE__) || defined(__unix__)
  #define NPU_FFI_PLATFORM_WINDOWS 0
  #define NPU_FFI_PLATFORM_POSIX 1
#else
  #define NPU_FFI_PLATFORM_WINDOWS 0
  #define NPU_FFI_PLATFORM_POSIX 0
#endif

/*!
 * \brief Compiler detection macros.
 */
#if defined(_MSC_VER)
  #define NPU_FFI_COMPILER_MSVC 1
  #define NPU_FFI_COMPILER_GCC 0
  #define NPU_FFI_COMPILER_CLANG 0
#elif defined(__clang__)
  #define NPU_FFI_COMPILER_MSVC 0
  #define NPU_FFI_COMPILER_GCC 0
  #define NPU_FFI_COMPILER_CLANG 1
#elif defined(__GNUC__)
  #define NPU_FFI_COMPILER_MSVC 0
  #define NPU_FFI_COMPILER_GCC 1
  #define NPU_FFI_COMPILER_CLANG 0
#else
  #define NPU_FFI_COMPILER_MSVC 0
  #define NPU_FFI_COMPILER_GCC 0
  #define NPU_FFI_COMPILER_CLANG 0
#endif

/*!
 * \brief DLL export/import macros for shared library linkage.
 *
 * Use NPU_FFI_DLL_EXPORT when building the library,
 * NPU_FFI_DLL_IMPORT when using the library.
 * The unified NPU_FFI_API macro is automatically defined based on
 * NPU_FFI_EXPORTS (set by CMake when building the shared library).
 */
#if NPU_FFI_PLATFORM_WINDOWS
  #define NPU_FFI_DLL_EXPORT __declspec(dllexport)
  #define NPU_FFI_DLL_IMPORT __declspec(dllimport)
#elif NPU_FFI_COMPILER_GCC || NPU_FFI_COMPILER_CLANG
  #define NPU_FFI_DLL_EXPORT __attribute__((visibility("default")))
  #define NPU_FFI_DLL_IMPORT __attribute__((visibility("default")))
#else
  #define NPU_FFI_DLL_EXPORT
  #define NPU_FFI_DLL_IMPORT
#endif

/*!
 * \brief Unified API export/import macro.
 *
 * Defined as NPU_FFI_DLL_EXPORT when NPU_FFI_EXPORTS is set
 * (during library build), NPU_FFI_DLL_IMPORT otherwise.
 */
#if defined(NPU_FFI_EXPORTS)
  #define NPU_FFI_API NPU_FFI_DLL_EXPORT
#else
  #define NPU_FFI_API NPU_FFI_DLL_IMPORT
#endif

#include "npu_ffi/vta/runtime.h"
