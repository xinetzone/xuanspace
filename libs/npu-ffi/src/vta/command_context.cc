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

#include "npu_ffi/vta/runtime.h"

#include <cinttypes>
#include <cstdint>

#include "npu_ffi/logging.h"

namespace npu_ffi {
namespace vta {

CommandContext::CommandContext(uint32_t wait_cycles)
    : cmd_(tls_command_handle()), wait_cycles_(wait_cycles), active_(true) {
  NPU_FFI_LOG_DEBUG("CommandContext created: cmd=%p, wait_cycles=%" PRIu32,
                    cmd_.get(), wait_cycles_);
}

CommandContext::~CommandContext() {
  if (active_) {
    NPU_FFI_LOG_DEBUG("CommandContext destructor triggering auto-sync: cmd=%p", cmd_.get());
    synchronize();
  } else {
    NPU_FFI_LOG_DEBUG("CommandContext destructor (already synced): cmd=%p", cmd_.get());
  }
}

void CommandContext::synchronize() {
  if (active_) {
    NPU_FFI_LOG_DEBUG("CommandContext::synchronize: cmd=%p, wait_cycles=%" PRIu32,
                      cmd_.get(), wait_cycles_);
    ::npu_ffi::vta::synchronize(cmd_, wait_cycles_);
    NPU_FFI_LOG_DEBUG("CommandContext::synchronize complete: cmd=%p", cmd_.get());
    active_ = false;
  }
}

}  // namespace vta
}  // namespace npu_ffi
