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

/*!
 * \file basic_usage.cc
 * \brief Basic usage example for npu-ffi VTA C++ API.
 *
 * This example demonstrates:
 * - Buffer RAII usage
 * - CommandContext for automatic synchronization
 * - Basic VTA operation flow
 */

#include <cstdio>

#include "npu_ffi/vta/buffer.h"
#include "npu_ffi/vta/command_context.h"
#include "npu_ffi/vta/runtime.h"
#include "npu_ffi/vta/types.h"

int main() {
  printf("npu-ffi VTA Basic Usage Example\n");
  printf("================================\n\n");

  using namespace npu_ffi::vta;

  printf("1. Testing Buffer RAII...\n");
  {
    Buffer buf(1024);
    printf("   Allocated buffer: size=%zu, data=%p, owns=%d\n",
           buf.size(), buf.data(), buf.owns_data());

    CommandHandle cmd0 = tls_command_handle();
    void* cpu_ptr = buf.cpu_ptr(cmd0);
    printf("   CPU-accessible pointer: %p\n", cpu_ptr);
  }
  printf("   Buffer automatically freed when going out of scope.\n\n");

  printf("2. Testing Buffer move semantics...\n");
  {
    Buffer buf1(512);
    printf("   buf1: size=%zu, data=%p\n", buf1.size(), buf1.data());

    Buffer buf2(std::move(buf1));
    printf("   After move:\n");
    printf("   buf2: size=%zu, data=%p\n", buf2.size(), buf2.data());
    printf("   buf1: size=%zu, data=%p (moved-from)\n", buf1.size(), buf1.data());
  }
  printf("\n");

  printf("3. Testing non-owning Buffer (foreign pointer)...\n");
  {
    Buffer owned_buf(256);
    printf("   Owned buffer: data=%p\n", owned_buf.data());

    {
      Buffer foreign(owned_buf.data(), owned_buf.size(), false);
      printf("   Foreign wrapper: data=%p, owns=%d\n",
             foreign.data(), foreign.owns_data());
    }
    printf("   Foreign wrapper destroyed without freeing.\n");
    printf("   Owned buffer still valid: data=%p\n", owned_buf.data());
  }
  printf("\n");

  printf("4. Testing CommandContext...\n");
  {
    CommandContext ctx(0);
    CommandHandle cmd = ctx.handle();
    printf("   CommandContext active: handle=%p\n", cmd.get());

    Buffer inp(1024);
    Buffer wgt(1024);
    Buffer acc(1024);
    printf("   Allocated 3 buffers: inp=%p, wgt=%p, acc=%p\n",
           inp.data(), wgt.data(), acc.data());

    printf("   Pushing commands (demo)...\n");
    uop_push(0, 1, 0, 0, 0, ALUOpcode::ADD, false, 0);

    printf("   Explicit synchronize...\n");
    ctx.synchronize();
    printf("   Context deactivated after synchronize.\n");
  }
  printf("   CommandContext automatically synchronizes on destruction.\n\n");

  printf("5. Testing runtime shutdown (idempotent)...\n");
  runtime_shutdown();
  runtime_shutdown();
  printf("   Runtime shutdown called twice safely.\n\n");

  printf("================================\n");
  printf("Example completed successfully!\n");

  return 0;
}
