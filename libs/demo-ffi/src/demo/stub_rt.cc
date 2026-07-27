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

#include "demo_ffi/types.h"

#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <atomic>
#include <mutex>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace demo_ffi {
namespace demo {
namespace stub {

namespace {

thread_local void* current_cmd_handle = nullptr;
static std::unordered_map<void*, size_t> buffer_tracker;
static std::mutex buffer_tracker_mutex;
static std::atomic<uint64_t> next_handle_id{1};

}  // namespace

extern "C" {

void* demo_ffi_demo_buffer_alloc(size_t size) {
  void* ptr = nullptr;
#ifdef _WIN32
  ptr = _aligned_malloc(size, kAllocAlignment);
#else
  int ret = posix_memalign(&ptr, kAllocAlignment, size);
  if (ret != 0) {
    ptr = nullptr;
  }
#endif
  if (ptr != nullptr) {
    std::lock_guard<std::mutex> lock(buffer_tracker_mutex);
    buffer_tracker[ptr] = size;
  }
  return ptr;
}

void demo_ffi_demo_buffer_free(void* buffer) {
  if (buffer == nullptr) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(buffer_tracker_mutex);
    buffer_tracker.erase(buffer);
  }
#ifdef _WIN32
  _aligned_free(buffer);
#else
  free(buffer);
#endif
}

void* demo_ffi_demo_tls_command_handle() {
  uint64_t id = next_handle_id.fetch_add(1, std::memory_order_relaxed);
  current_cmd_handle = reinterpret_cast<void*>(id);
  return current_cmd_handle;
}

void demo_ffi_demo_runtime_shutdown() {
  std::lock_guard<std::mutex> lock(buffer_tracker_mutex);
  buffer_tracker.clear();
}

}  // extern "C"

}  // namespace stub
}  // namespace demo
}  // namespace demo_ffi
