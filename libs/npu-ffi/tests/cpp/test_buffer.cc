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

#include <cassert>
#include <cstdio>
#include <utility>

#include "npu_ffi/vta/buffer.h"
#include "npu_ffi/vta/command_context.h"
#include "npu_ffi/vta/handle.h"
#include "npu_ffi/vta/runtime.h"

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_CASE(name) \
  do { \
    printf("  TEST: %s ... ", #name); \
    bool test_passed = true; \
    try { \
      test_##name(); \
    } catch (...) { \
      test_passed = false; \
    } \
    if (test_passed) { \
      printf("PASSED\n"); \
      g_tests_passed++; \
    } else { \
      printf("FAILED\n"); \
      g_tests_failed++; \
    } \
  } while (0)

#define ASSERT_TRUE(cond) \
  do { \
    if (!(cond)) { \
      printf("\n    ASSERT_TRUE failed: %s (line %d)\n", #cond, __LINE__); \
      throw 1; \
    } \
  } while (0)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))
#define ASSERT_EQ(a, b) ASSERT_TRUE((a) == (b))
#define ASSERT_NE(a, b) ASSERT_TRUE((a) != (b))
#define ASSERT_NOT_NULL(ptr) ASSERT_TRUE((ptr) != nullptr)
#define ASSERT_NULL(ptr) ASSERT_TRUE((ptr) == nullptr)

void test_buffer_raii_alloc() {
  using namespace npu_ffi::vta;
  Buffer buf(1024);
  ASSERT_EQ(buf.size(), static_cast<size_t>(1024));
  ASSERT_NOT_NULL(buf.data());
  ASSERT_TRUE(buf.owns_data());
}

void test_buffer_size_verification() {
  using namespace npu_ffi::vta;
  Buffer buf(4096);
  ASSERT_EQ(buf.size(), static_cast<size_t>(4096));
  ASSERT_NOT_NULL(buf.get());
  ASSERT_EQ(buf.data(), buf.get());
}

void test_buffer_move_constructor() {
  using namespace npu_ffi::vta;
  Buffer buf1(2048);
  void* original_data = buf1.data();
  size_t original_size = buf1.size();

  Buffer buf2(std::move(buf1));
  ASSERT_EQ(buf2.size(), original_size);
  ASSERT_EQ(buf2.data(), original_data);
  ASSERT_TRUE(buf2.owns_data());
  ASSERT_NULL(buf1.data());
  ASSERT_EQ(buf1.size(), static_cast<size_t>(0));
  ASSERT_FALSE(buf1.owns_data());
}

void test_buffer_move_assignment() {
  using namespace npu_ffi::vta;
  Buffer buf1(1024);
  Buffer buf2(2048);
  void* original_data = buf1.data();
  size_t original_size = buf1.size();

  buf2 = std::move(buf1);
  ASSERT_EQ(buf2.size(), original_size);
  ASSERT_EQ(buf2.data(), original_data);
  ASSERT_TRUE(buf2.owns_data());
  ASSERT_NULL(buf1.data());
  ASSERT_EQ(buf1.size(), static_cast<size_t>(0));
}

void test_buffer_empty_constructor() {
  using namespace npu_ffi::vta;
  Buffer buf(0);
  ASSERT_EQ(buf.size(), static_cast<size_t>(0));
}

void test_buffer_reset() {
  using namespace npu_ffi::vta;
  Buffer buf(1024);
  ASSERT_NOT_NULL(buf.data());
  ASSERT_TRUE(buf.owns_data());

  buf.reset();
  ASSERT_NULL(buf.data());
  ASSERT_EQ(buf.size(), static_cast<size_t>(0));
  ASSERT_FALSE(buf.owns_data());

  buf.reset();
  ASSERT_NULL(buf.data());
}

void test_buffer_nonowning() {
  using namespace npu_ffi::vta;
  Buffer owned_buf(1024);
  void* raw_ptr = owned_buf.data();

  {
    Buffer nonowning(raw_ptr, 1024, false);
    ASSERT_EQ(nonowning.data(), raw_ptr);
    ASSERT_EQ(nonowning.size(), static_cast<size_t>(1024));
    ASSERT_FALSE(nonowning.owns_data());
  }
  ASSERT_NOT_NULL(owned_buf.data());
}

void test_command_context_basic() {
  using namespace npu_ffi::vta;
  CommandContext ctx(0);
  ASSERT_TRUE(ctx.active());
  ASSERT_TRUE(ctx.handle() != nullptr);
  ASSERT_NE(*ctx, nullptr);

  ctx.synchronize();
  ASSERT_FALSE(ctx.active());
}

void test_command_context_double_sync() {
  using namespace npu_ffi::vta;
  CommandContext ctx(0);
  ASSERT_TRUE(ctx.active());

  ctx.synchronize();
  ASSERT_FALSE(ctx.active());

  ctx.synchronize();
  ASSERT_FALSE(ctx.active());
}

void test_command_context_move() {
  using namespace npu_ffi::vta;
  CommandContext ctx1(0);
  ASSERT_TRUE(ctx1.active());
  CommandHandle original_handle = ctx1.handle();

  CommandContext ctx2(std::move(ctx1));
  ASSERT_TRUE(ctx2.active());
  ASSERT_EQ(ctx2.handle(), original_handle);
  ASSERT_FALSE(ctx1.active());
}

void test_handle_implicit_conversion() {
  using namespace npu_ffi::vta;
  CommandHandle h1;
  ASSERT_EQ(h1, nullptr);
  ASSERT_FALSE(h1);

  CommandContext ctx(0);
  CommandHandle h2 = ctx.handle();
  ASSERT_NE(h2, nullptr);
  ASSERT_TRUE(h2);
  ASSERT_NOT_NULL(h2.get());
}

void test_tls_command_handle() {
  using namespace npu_ffi::vta;
  CommandHandle cmd = tls_command_handle();
  ASSERT_NE(cmd, nullptr);
  ASSERT_NOT_NULL(cmd.get());
}

void test_runtime_shutdown() {
  using namespace npu_ffi::vta;
  runtime_shutdown();
  runtime_shutdown();
}

int main() {
  printf("Running npu-ffi C++ tests...\n\n");

  printf("Buffer tests:\n");
  TEST_CASE(buffer_raii_alloc);
  TEST_CASE(buffer_size_verification);
  TEST_CASE(buffer_move_constructor);
  TEST_CASE(buffer_move_assignment);
  TEST_CASE(buffer_empty_constructor);
  TEST_CASE(buffer_reset);
  TEST_CASE(buffer_nonowning);

  printf("\nCommandContext tests:\n");
  TEST_CASE(command_context_basic);
  TEST_CASE(command_context_double_sync);
  TEST_CASE(command_context_move);

  printf("\nHandle tests:\n");
  TEST_CASE(handle_implicit_conversion);
  TEST_CASE(tls_command_handle);

  printf("\nRuntime tests:\n");
  TEST_CASE(runtime_shutdown);

  printf("\n========================================\n");
  printf("Results: %d passed, %d failed\n", g_tests_passed, g_tests_failed);
  printf("========================================\n");

  return g_tests_failed > 0 ? 1 : 0;
}
