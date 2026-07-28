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
 * \file npu_ffi/vta/handle.h
 * \brief Type-safe wrapper for VTA command handle.
 */

#include <cstddef>

namespace npu_ffi {
namespace vta {

// Forward declaration
class Buffer;

/*!
 * \brief Type-safe wrapper for VTA command handle (VTACommandHandle).
 *
 * This class provides a type-safe wrapper around the raw void* command handle
 * used by the VTA C API. It prevents implicit conversions from arbitrary pointers
 * or integers and provides explicit access to the underlying handle for C API calls.
 */
class CommandHandle {
 public:
  /*!
   * \brief Default constructor, creates a null handle.
   */
  CommandHandle() : handle_(nullptr) {}

  /*!
   * \brief Construct from nullptr.
   *
   * Not marked explicit to allow natural nullptr comparison and assignment
   * (e.g., `if (cmd == nullptr)`, `cmd = nullptr`).
   * Arbitrary pointer conversions are still blocked by the explicit void* constructor.
   */
  CommandHandle(std::nullptr_t) : handle_(nullptr) {}

  /*!
   * \brief Explicit constructor from raw void* handle.
   * \param h Raw C API handle.
   */
  explicit CommandHandle(void* h) : handle_(h) {}

  /*! \brief Copy constructor. */
  CommandHandle(const CommandHandle&) = default;

  /*! \brief Copy assignment operator. */
  CommandHandle& operator=(const CommandHandle&) = default;

  /*! \brief Move constructor. */
  CommandHandle(CommandHandle&&) = default;

  /*! \brief Move assignment operator. */
  CommandHandle& operator=(CommandHandle&&) = default;

  /*!
   * \brief Assign nullptr to reset the handle.
   * \return Reference to this.
   */
  CommandHandle& operator=(std::nullptr_t) {
    handle_ = nullptr;
    return *this;
  }

  /*!
   * \brief Get the underlying raw void* handle for C API calls.
   * \return Raw handle pointer.
   */
  void* get() const { return handle_; }

  /*!
   * \brief Check if the handle is non-null.
   * \return True if handle is not null.
   */
  explicit operator bool() const { return handle_ != nullptr; }

  /*!
   * \brief Equality comparison.
   * \param other Other handle to compare.
   * \return True if handles are equal.
   */
  bool operator==(const CommandHandle& other) const { return handle_ == other.handle_; }

  /*!
   * \brief Inequality comparison.
   * \param other Other handle to compare.
   * \return True if handles are not equal.
   */
  bool operator!=(const CommandHandle& other) const { return handle_ != other.handle_; }

  /*!
   * \brief Equality comparison with nullptr.
   * \return True if handle is null.
   */
  bool operator==(std::nullptr_t) const { return handle_ == nullptr; }

  /*!
   * \brief Inequality comparison with nullptr.
   * \return True if handle is non-null.
   */
  bool operator!=(std::nullptr_t) const { return handle_ != nullptr; }

 private:
  /*! \brief Raw C API handle. */
  void* handle_;
};

}  // namespace vta
}  // namespace npu_ffi
