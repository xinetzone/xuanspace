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
 * \file npu_ffi/vta/buffer.h
 * \brief Type-safe VTA buffer wrapper with RAII semantics.
 */

#ifndef NPU_FFI_VTA_BUFFER_H_
#define NPU_FFI_VTA_BUFFER_H_

#include <cstddef>

#include "npu_ffi/vta/handle.h"

namespace npu_ffi {
namespace vta {

/*!
 * \brief RAII wrapper for VTA device buffers.
 *
 * This class manages VTA buffer allocation and deallocation automatically.
 * It supports both owning (allocates and frees memory) and non-owning
 * (wraps an existing pointer) modes. Copy is disabled, move is supported.
 */
class Buffer {
 public:
  /*!
   * \brief Allocate a new VTA buffer of given size.
   * \param size Buffer size in bytes.
   * \note The buffer will be automatically freed when destroyed.
   */
  explicit Buffer(size_t size);

  /*!
   * \brief Wrap an existing buffer pointer without taking ownership.
   * \param data Existing buffer pointer.
   * \param size Buffer size in bytes.
   * \param owns If true, take ownership and free on destruction.
   */
  Buffer(void* data, size_t size, bool owns = false);

  /*!
   * \brief Destructor. Frees buffer if owns_data() is true.
   */
  ~Buffer();

  /*! \brief Copy constructor is deleted. */
  Buffer(const Buffer&) = delete;

  /*! \brief Copy assignment is deleted. */
  Buffer& operator=(const Buffer&) = delete;

  /*!
   * \brief Move constructor. Transfers ownership from other.
   * \param other Source buffer to move from.
   */
  Buffer(Buffer&& other) noexcept;

  /*!
   * \brief Move assignment operator. Transfers ownership from other.
   * \param other Source buffer to move from.
   * \return Reference to this.
   */
  Buffer& operator=(Buffer&& other) noexcept;

  /*!
   * \brief Get the underlying raw data pointer.
   * \return Raw buffer pointer.
   */
  void* get() const { return data_; }

  /*!
   * \brief Get buffer size in bytes.
   * \return Buffer size.
   */
  size_t size() const { return size_; }

  /*!
   * \brief Check if this buffer owns the underlying data.
   * \return True if this buffer will free the data on destruction.
   */
  bool owns_data() const { return owns_; }

  /*!
   * \brief Get CPU-accessible pointer for this buffer.
   * \param cmd Command handle for the operation.
   * \return CPU-accessible pointer.
   */
  void* cpu_ptr(CommandHandle cmd) const;

  /*!
   * \brief Manually release/free the buffer.
   *
   * If the buffer owns data, calls VTABufferFree and resets to null.
   * Safe to call multiple times.
   */
  void reset();

 private:
  /*! \brief Raw buffer data pointer. */
  void* data_;
  /*! \brief Buffer size in bytes. */
  size_t size_;
  /*! \brief Whether this buffer owns the data and should free it. */
  bool owns_;
};

}  // namespace vta
}  // namespace npu_ffi

#endif  // NPU_FFI_VTA_BUFFER_H_
