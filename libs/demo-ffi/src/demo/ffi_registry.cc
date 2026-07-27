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

#include <tvm/ffi/tvm_ffi.h>

#include <cstddef>
#include <cstdint>

extern "C" {

void* demo_ffi_demo_buffer_alloc(size_t size);
void demo_ffi_demo_buffer_free(void* buffer);
void* demo_ffi_demo_tls_command_handle();
void demo_ffi_demo_runtime_shutdown();

}  // extern "C"

namespace demo_ffi {
namespace demo {
namespace ffi_registry {

int64_t buffer_alloc(int64_t size) {
  return reinterpret_cast<int64_t>(demo_ffi_demo_buffer_alloc(static_cast<size_t>(size)));
}

void buffer_free(int64_t ptr) {
  demo_ffi_demo_buffer_free(reinterpret_cast<void*>(static_cast<intptr_t>(ptr)));
}

int64_t tls_command_handle() {
  return reinterpret_cast<int64_t>(demo_ffi_demo_tls_command_handle());
}

void runtime_shutdown() {
  demo_ffi_demo_runtime_shutdown();
}

// IMPORTANT: The prefix used here (demo) MUST exactly match
// the first argument to _FFI_INIT_FUNC() in python/demo_ffi/demo/_ffi_api.py
// Use scripts/check_ffi_prefix.py to verify consistency.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("demo.buffer_alloc", buffer_alloc)
      .def("demo.buffer_free", buffer_free)
      .def("demo.tls_command_handle", tls_command_handle)
      .def("demo.runtime_shutdown", runtime_shutdown);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(demo_buffer_alloc, buffer_alloc);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(demo_buffer_free, buffer_free);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(demo_tls_command_handle, tls_command_handle);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(demo_runtime_shutdown, runtime_shutdown);

}  // namespace ffi_registry
}  // namespace demo
}  // namespace demo_ffi
