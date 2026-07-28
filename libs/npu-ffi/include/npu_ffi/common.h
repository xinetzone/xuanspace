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
 * \file npu_ffi/common.h
 * \brief Common platform utilities, DLL export macros, compiler abstractions,
 *        and reusable RAII utilities for the NPU FFI library.
 *
 * This header is the foundational include for all npu-ffi modules.
 * It provides:
 *   - Platform detection (Windows/POSIX)
 *   - Compiler detection (MSVC/GCC/Clang)
 *   - DLL export/import macros (cross-platform)
 *   - Alignment constants for memory allocation
 *   - Copy/move semantics macros
 *   - Generic ScopeGuard for RAII resource cleanup
 *
 * Other FFI binding libraries in the xuanspace ecosystem can reuse this
 * header by including npu_ffi/common.h directly.
 */

#include <cstddef>
#include <utility>

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
 * Usage:
 *   - When building the library: define NPU_FFI_EXPORTS via CMake,
 *     NPU_FFI_API expands to __declspec(dllexport) or __attribute__((visibility("default")))
 *   - When using the library: NPU_FFI_API expands to __declspec(dllimport)
 *     or __attribute__((visibility("default")))
 *
 * For sub-module libraries that need their own API macro (e.g., NPU_XXX_API),
 * use NPU_FFI_DEFINE_API_MACRO to generate it consistently:
 *
 *   // In your module's common header:
 *   #ifndef MY_MODULE_API
 *   #define MY_MODULE_API NPU_FFI_DLL_EXPORT  // default for static build
 *   #endif
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
 * \brief Pattern for defining project-specific API macros.
 *
 * For sub-module libraries that need their own DLL export macro
 * (e.g., MY_MODULE_API), follow this pattern in your module's common header:
 *
 * \code
 *   #ifndef MY_MODULE_API
 *     #if defined(MY_MODULE_EXPORTS)
 *       #define MY_MODULE_API NPU_FFI_DLL_EXPORT
 *     #else
 *       #define MY_MODULE_API NPU_FFI_DLL_IMPORT
 *     #endif
 *   #endif
 * \endcode
 *
 * Then define MY_MODULE_EXPORTS as a PRIVATE compile definition in CMake:
 *   target_compile_definitions(my_module PRIVATE MY_MODULE_EXPORTS)
 */

/*!
 * \brief The unified NPU FFI API export/import macro.
 *
 * Automatically resolves to dllexport when NPU_FFI_EXPORTS is defined
 * (during library build), dllimport otherwise.
 */
#ifndef NPU_FFI_API
  #if defined(NPU_FFI_EXPORTS)
    #define NPU_FFI_API NPU_FFI_DLL_EXPORT
  #else
    #define NPU_FFI_API NPU_FFI_DLL_IMPORT
  #endif
#endif

/*!
 * \brief Default memory alignment for NPU buffers (64 bytes for cache line alignment).
 */
constexpr size_t kAllocAlignment = 64;

/*!
 * \brief Macros to disallow copy and assignment.
 *
 * Usage in a class declaration:
 *   class MyClass {
 *    public:
 *     NPU_FFI_DISALLOW_COPY(MyClass);
 *   };
 */
#define NPU_FFI_DISALLOW_COPY(ClassName) \
  ClassName(const ClassName&) = delete; \
  ClassName& operator=(const ClassName&) = delete

/*!
 * \brief Declare default move operations.
 */
#define NPU_FFI_DEFAULT_MOVE(ClassName) \
  ClassName(ClassName&&) noexcept = default; \
  ClassName& operator=(ClassName&&) noexcept = default

namespace npu_ffi {

/*!
 * \brief A generic RAII scope guard that executes a cleanup function on destruction.
 *
 * Useful for ensuring resource cleanup in C-style APIs that don't provide
 * their own RAII wrappers. Move-only; cannot be copied.
 *
 * Usage:
 *   void* raw_ptr = some_c_api_allocate(size);
 *   auto guard = MakeScopeGuard([&]() { some_c_api_free(raw_ptr); });
 *   // ... use raw_ptr ...
 *   // guard automatically calls some_c_api_free when it goes out of scope
 *
 * To dismiss the guard (prevent cleanup):
 *   guard.dismiss();
 *
 * \tparam Fn A callable type (lambda, function pointer, functor) with void() signature.
 */
template <typename Fn>
class ScopeGuard {
 public:
  explicit ScopeGuard(Fn fn) noexcept : fn_(std::move(fn)), active_(true) {}
  ~ScopeGuard() { if (active_) fn_(); }

  ScopeGuard(ScopeGuard&& other) noexcept
      : fn_(std::move(other.fn_)), active_(other.active_) {
    other.active_ = false;
  }

  ScopeGuard& operator=(ScopeGuard&& other) noexcept {
    if (this != &other) {
      if (active_) fn_();
      fn_ = std::move(other.fn_);
      active_ = other.active_;
      other.active_ = false;
    }
    return *this;
  }

  NPU_FFI_DISALLOW_COPY(ScopeGuard);

  /*!
   * \brief Dismiss the guard, preventing the cleanup function from running.
   */
  void dismiss() noexcept { active_ = false; }

  /*!
   * \brief Check if the guard is active.
   */
  bool active() const noexcept { return active_; }

 private:
  Fn fn_;
  bool active_;
};

/*!
 * \brief Factory function to create a ScopeGuard (deduces Fn type).
 * \tparam Fn Callable type.
 * \param fn Cleanup function to execute on scope exit.
 * \return A ScopeGuard that calls fn on destruction.
 */
template <typename Fn>
ScopeGuard<Fn> MakeScopeGuard(Fn fn) noexcept {
  return ScopeGuard<Fn>(std::move(fn));
}

}  // namespace npu_ffi
