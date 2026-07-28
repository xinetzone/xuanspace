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
 * \file npu_ffi/vta/command_context.h
 * \brief RAII command context for automatic synchronization.
 */

#include <cstdint>
#include <utility>

#include "npu_ffi/vta/handle.h"

namespace npu_ffi {
namespace vta {

/*!
 * \brief Forward declaration of synchronize function.
 * \param cmd Command handle.
 * \param wait_cycles Maximum poll cycles to wait (0 = wait indefinitely).
 */
NPU_FFI_API void synchronize(CommandHandle cmd, uint32_t wait_cycles);

/*!
 * \brief RAII wrapper for VTA command execution context.
 *
 * This class provides automatic synchronization on destruction,
 * similar to Python's CommandContext context manager.
 * It acquires the thread-local command handle on construction
 * and calls synchronize() on destruction.
 *
 * Copy is disabled, move is supported.
 *
 * Usage example:
 * \code
 *   {
 *     CommandContext ctx;  // acquires tls command handle
 *     CommandHandle cmd = ctx.handle();
 *     // ... push commands using cmd ...
 *   }  // automatically synchronizes when ctx goes out of scope
 * \endcode
 */
class NPU_FFI_API CommandContext {
 public:
  /*!
   * \brief Construct a command context and acquire the thread-local command handle.
   * \param wait_cycles Maximum poll cycles to wait on synchronize (0 = wait indefinitely).
   */
  explicit CommandContext(uint32_t wait_cycles = 0);

  /*!
   * \brief Destructor. Automatically calls synchronize() if the context is active.
   */
  ~CommandContext();

  /*! \brief Copy constructor is deleted. */
  CommandContext(const CommandContext&) = delete;

  /*! \brief Copy assignment is deleted. */
  CommandContext& operator=(const CommandContext&) = delete;

  /*!
   * \brief Move constructor. Transfers ownership from other.
   * \param other Source context to move from.
   */
  CommandContext(CommandContext&& other) noexcept
      : cmd_(other.cmd_), wait_cycles_(other.wait_cycles_), active_(other.active_) {
    other.active_ = false;
  }

  /*!
   * \brief Move assignment operator. Transfers ownership from other.
   * \param other Source context to move from.
   * \return Reference to this.
   */
  CommandContext& operator=(CommandContext&& other) noexcept {
    if (this != &other) {
      if (active_) {
        ::npu_ffi::vta::synchronize(cmd_, wait_cycles_);
      }
      cmd_ = other.cmd_;
      wait_cycles_ = other.wait_cycles_;
      active_ = other.active_;
      other.active_ = false;
    }
    return *this;
  }

  /*!
   * \brief Get the command handle for this context.
   * \return The command handle.
   * \note Only valid while the context is active.
   */
  CommandHandle handle() const { return cmd_; }

  /*!
   * \brief Dereference operator to get the command handle.
   * \return The command handle.
   * \note Only valid while the context is active.
   */
  CommandHandle operator*() const { return cmd_; }

  /*!
   * \brief Check if the context is active (has not been moved from or synchronized).
   * \return True if the context will synchronize on destruction.
   */
  bool active() const { return active_; }

  /*!
   * \brief Explicitly synchronize early and deactivate the context.
   *
   * After calling this, the destructor will not call synchronize again.
   * Safe to call multiple times; subsequent calls are no-ops.
   */
  void synchronize();

 private:
  /*! \brief The command handle. */
  CommandHandle cmd_;
  /*! \brief Wait cycles for synchronization. */
  uint32_t wait_cycles_;
  /*! \brief Whether this context is active and will synchronize on destruction. */
  bool active_;
};

}  // namespace vta
}  // namespace npu_ffi
