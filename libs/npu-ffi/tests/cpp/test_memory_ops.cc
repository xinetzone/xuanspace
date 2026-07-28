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
 * \file test_memory_ops.cc
 * \brief Tests for VTA memory operations: buffer_copy, load_buffer_2d,
 *        store_buffer_2d, read_barrier, write_barrier, cpu_ptr.
 *
 * These tests verify the C++ type-safe API layer (layer 4 of the FFI pattern).
 * When extending with new memory instructions, add test_<instruction_name>()
 * functions and register them in main().
 */

#include "npu_ffi/testing/test_utils.h"

using namespace npu_ffi::vta;
using namespace npu_ffi::vta::testing;

// ============================================================================
// buffer_copy tests
// ============================================================================

void test_buffer_copy_d2d() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    ScopedBuffer src(256);
    ScopedBuffer dst(256);
    fill_pattern(src.get(), 0xAB);
    fill_pattern(dst.get(), 0x00);

    buffer_copy(src.get(), 0, dst.get(), 0, 256, MemcpyKind::D2D);
    TEST_ASSERT_PATTERN(dst.get(), 0xAB);
  } TEST_INSTRUCTION_END
}

void test_buffer_copy_with_offset() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    ScopedBuffer src(512);
    ScopedBuffer dst(512);
    fill_pattern(src.get(), 0xCD);
    fill_pattern(dst.get(), 0x00);

    buffer_copy(src.get(), 64, dst.get(), 128, 256, MemcpyKind::D2D);
    const uint8_t* dst_ptr = static_cast<const uint8_t*>(dst.data());
    TEST_ASSERT_EQ(dst_ptr[0], 0x00);
    TEST_ASSERT_EQ(dst_ptr[127], 0x00);
    TEST_ASSERT_EQ(dst_ptr[128], 0xCD);
    TEST_ASSERT_EQ(dst_ptr[383], 0xCD);
    TEST_ASSERT_EQ(dst_ptr[384], 0x00);
  } TEST_INSTRUCTION_END
}

void test_buffer_copy_h2d_d2h() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    ScopedBuffer src(128);
    ScopedBuffer dst(128);
    fill_pattern(src.get(), 0x42);

    buffer_copy(src.get(), 0, dst.get(), 0, 128, MemcpyKind::H2D);
    buffer_copy(dst.get(), 0, src.get(), 0, 128, MemcpyKind::D2H);
    TEST_ASSERT_PATTERN(src.get(), 0x42);
  } TEST_INSTRUCTION_END
}

void test_buffer_copy_zero_size() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    ScopedBuffer src(64);
    ScopedBuffer dst(64);
    fill_pattern(dst.get(), 0xFF);
    buffer_copy(src.get(), 0, dst.get(), 0, 0, MemcpyKind::D2D);
    TEST_ASSERT_PATTERN(dst.get(), 0xFF);
  } TEST_INSTRUCTION_END
}

// ============================================================================
// cpu_ptr tests
// ============================================================================

void test_cpu_ptr_returns_buffer() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    ScopedBuffer buf(256);
    fill_pattern(buf.get(), 0x55);
    void* cpu = buf.get().cpu_ptr(ctx.cmd());
    TEST_ASSERT_NOT_NULL(cpu);
    TEST_ASSERT_EQ(cpu, buf.data());
  } TEST_INSTRUCTION_END
}

void test_cpu_ptr_null_buffer() {
  Buffer empty(0);
  void* cpu = empty.cpu_ptr(CommandHandle(nullptr));
  TEST_ASSERT_NULL(cpu);
}

// ============================================================================
// Barrier tests (stub backend: no-ops, but API should not crash)
// ============================================================================

void test_write_barrier_no_crash() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    ScopedBuffer buf(128);
    write_barrier(ctx.cmd(), buf.get(), 8, 0, 16);
  } TEST_INSTRUCTION_END
}

void test_read_barrier_no_crash() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    ScopedBuffer buf(128);
    read_barrier(ctx.cmd(), buf.get(), 8, 0, 16);
  } TEST_INSTRUCTION_END
}

// ============================================================================
// load/store buffer 2D tests (stub backend: no-ops, API should not crash)
// ============================================================================

void test_load_buffer_2d_no_crash() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    ScopedBuffer src(1024);
    load_buffer_2d(ctx.cmd(), src.get(), 0, 16, 1, 16, 0, 0, 0, 0, 0, MemoryType::ACC);
  } TEST_INSTRUCTION_END
}

void test_store_buffer_2d_no_crash() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    ScopedBuffer dst(1024);
    store_buffer_2d(ctx.cmd(), 0, MemoryType::ACC, dst.get(), 0, 16, 1, 16);
  } TEST_INSTRUCTION_END
}

// ============================================================================
// Test main
// ============================================================================

int main() {
  printf("Running npu-ffi memory operations tests...\n\n");

  TestRunner runner;

  runner.add_test("memory/buffer_copy_d2d", test_buffer_copy_d2d);
  runner.add_test("memory/buffer_copy_with_offset", test_buffer_copy_with_offset);
  runner.add_test("memory/buffer_copy_h2d_d2h", test_buffer_copy_h2d_d2h);
  runner.add_test("memory/buffer_copy_zero_size", test_buffer_copy_zero_size);
  runner.add_test("memory/cpu_ptr_returns_buffer", test_cpu_ptr_returns_buffer);
  runner.add_test("memory/cpu_ptr_null_buffer", test_cpu_ptr_null_buffer);
  runner.add_test("memory/write_barrier_no_crash", test_write_barrier_no_crash);
  runner.add_test("memory/read_barrier_no_crash", test_read_barrier_no_crash);
  runner.add_test("memory/load_buffer_2d_no_crash", test_load_buffer_2d_no_crash);
  runner.add_test("memory/store_buffer_2d_no_crash", test_store_buffer_2d_no_crash);

  return runner.run_all() ? 0 : 1;
}
