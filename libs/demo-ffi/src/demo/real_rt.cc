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

// Real runtime implementation - forward to actual hardware API
// Replace the extern "C" declarations and function bodies with
// calls to your real hardware runtime library.

namespace demo_ffi {
namespace demo {
namespace real {

extern "C" {

// Forward declarations - replace with actual hardware API includes
// #include <demo/runtime.h>

void* demo_ffi_demo_buffer_alloc(size_t size) {
  // return DEMOBufferAlloc(size);
  return nullptr;  // Replace with real implementation
}

void demo_ffi_demo_buffer_free(void* buffer) {
  // DEMOBufferFree(buffer);
}

void* demo_ffi_demo_tls_command_handle() {
  // return DEMOTLSCommandHandle();
  return nullptr;
}

void demo_ffi_demo_runtime_shutdown() {
  // DEMORuntimeShutdown();
}

}  // extern "C"

}  // namespace real
}  // namespace demo
}  // namespace demo_ffi
