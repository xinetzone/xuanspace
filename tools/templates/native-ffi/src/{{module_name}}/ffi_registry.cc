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

void* {{package_name}}_{{module_name}}_buffer_alloc(size_t size);
void {{package_name}}_{{module_name}}_buffer_free(void* buffer);
void* {{package_name}}_{{module_name}}_tls_command_handle();
void {{package_name}}_{{module_name}}_runtime_shutdown();

}  // extern "C"

namespace {{package_name}} {
namespace {{module_name}} {
namespace ffi_registry {

int64_t buffer_alloc(int64_t size) {
  return reinterpret_cast<int64_t>({{package_name}}_{{module_name}}_buffer_alloc(static_cast<size_t>(size)));
}

void buffer_free(int64_t ptr) {
  {{package_name}}_{{module_name}}_buffer_free(reinterpret_cast<void*>(static_cast<intptr_t>(ptr)));
}

int64_t tls_command_handle() {
  return reinterpret_cast<int64_t>({{package_name}}_{{module_name}}_tls_command_handle());
}

void runtime_shutdown() {
  {{package_name}}_{{module_name}}_runtime_shutdown();
}

// IMPORTANT: The prefix used here ({{module_name}}) MUST exactly match
// the first argument to _FFI_INIT_FUNC() in python/{{package_name}}/{{module_name}}/_ffi_api.py
// Use scripts/check_ffi_prefix.py to verify consistency.
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("{{module_name}}.buffer_alloc", buffer_alloc)
      .def("{{module_name}}.buffer_free", buffer_free)
      .def("{{module_name}}.tls_command_handle", tls_command_handle)
      .def("{{module_name}}.runtime_shutdown", runtime_shutdown);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC({{module_name}}_buffer_alloc, buffer_alloc);
TVM_FFI_DLL_EXPORT_TYPED_FUNC({{module_name}}_buffer_free, buffer_free);
TVM_FFI_DLL_EXPORT_TYPED_FUNC({{module_name}}_tls_command_handle, tls_command_handle);
TVM_FFI_DLL_EXPORT_TYPED_FUNC({{module_name}}_runtime_shutdown, runtime_shutdown);

}  // namespace ffi_registry
}  // namespace {{module_name}}
}  // namespace {{package_name}}
