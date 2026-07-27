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

namespace {{package_name}} {
namespace {{module_name}} {
namespace real {

extern "C" {

// Forward declarations - replace with actual hardware API includes
// #include <{{module_name}}/runtime.h>

void* {{package_name}}_{{module_name}}_buffer_alloc(size_t size) {
  // return {{module_name|upper}}BufferAlloc(size);
  return nullptr;  // Replace with real implementation
}

void {{package_name}}_{{module_name}}_buffer_free(void* buffer) {
  // {{module_name|upper}}BufferFree(buffer);
}

void* {{package_name}}_{{module_name}}_tls_command_handle() {
  // return {{module_name|upper}}TLSCommandHandle();
  return nullptr;
}

void {{package_name}}_{{module_name}}_runtime_shutdown() {
  // {{module_name|upper}}RuntimeShutdown();
}

}  // extern "C"

}  // namespace real
}  // namespace {{module_name}}
}  // namespace {{package_name}}
