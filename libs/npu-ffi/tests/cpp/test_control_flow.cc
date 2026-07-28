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
 * \file test_control_flow.cc
 * \brief Tests for VTA control flow operations: uop_loop_begin/end,
 *        dep_push/dep_pop, synchronize, set_debug_mode, runtime_shutdown.
 *
 * Control flow instructions manage execution ordering and dependency
 * between VTA compute queues. When extending with new control flow
 * operations (e.g., conditional execution, wait events, pipeline barriers),
 * add tests following the arrange-act-assert pattern.
 */

#include "npu_ffi/testing/test_utils.h"

using namespace npu_ffi::vta;
using namespace npu_ffi::vta::testing;

// ============================================================================
// Synchronize tests
// ============================================================================

void test_synchronize_basic() {
  CommandContext ctx(0);
  ctx.synchronize();
  TEST_ASSERT_FALSE(ctx.active());
}

void test_synchronize_with_wait_cycles() {
  CommandContext ctx(100);
  ctx.synchronize();
  TEST_ASSERT_FALSE(ctx.active());
}

void test_synchronize_idempotent() {
  CommandContext ctx(0);
  ctx.synchronize();
  ctx.synchronize();
  TEST_ASSERT_FALSE(ctx.active());
}

// ============================================================================
// Micro-op loop tests
// ============================================================================

void test_uop_loop_basic() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    uop_loop_begin(4);
    uop_push(1, 0, 0, 0, 0, ALUOpcode::ADD, false, 0);
    uop_loop_end();
  } TEST_INSTRUCTION_END
}

void test_uop_loop_with_factors() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    uop_loop_begin(8, 1, 1, 1);
    uop_push(0, 0, 0, 0, 0, ALUOpcode::ADD, false, 0);
    uop_loop_end();
  } TEST_INSTRUCTION_END
}

void test_uop_nested_loops() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    uop_loop_begin(2);
    uop_loop_begin(4);
    uop_push(1, 0, 0, 0, 0, ALUOpcode::ADD, false, 0);
    uop_loop_end();
    uop_loop_end();
  } TEST_INSTRUCTION_END
}

// ============================================================================
// Dependency token tests
// ============================================================================

void test_dep_push_pop() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    int r1 = dep_push(ctx.cmd(), 0, 1);
    int r2 = dep_pop(ctx.cmd(), 0, 1);
    TEST_ASSERT_EQ(r1, 0);
    TEST_ASSERT_EQ(r2, 0);
  } TEST_INSTRUCTION_END
}

// ============================================================================
// Debug mode tests
// ============================================================================

void test_set_debug_mode_dump_insn() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    set_debug_mode(ctx.cmd(), DebugFlag::DUMP_INSN);
  } TEST_INSTRUCTION_END
}

void test_set_debug_mode_combined_flags() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    set_debug_mode(ctx.cmd(), DebugFlag::DUMP_INSN | DebugFlag::DUMP_UOP);
  } TEST_INSTRUCTION_END
}

void test_debug_flag_bitwise_ops() {
  using utype = std::underlying_type_t<DebugFlag>;
  DebugFlag flags = DebugFlag::DUMP_INSN | DebugFlag::DUMP_UOP;
  TEST_ASSERT_TRUE(any(flags));
  TEST_ASSERT_TRUE(any(flags & DebugFlag::DUMP_INSN));
  flags = flags & static_cast<DebugFlag>(~static_cast<utype>(DebugFlag::DUMP_INSN));
  TEST_ASSERT_FALSE(any(flags & DebugFlag::DUMP_INSN));
  TEST_ASSERT_TRUE(any(flags & DebugFlag::DUMP_UOP));
}

// ============================================================================
// Runtime shutdown tests
// ============================================================================

void test_runtime_shutdown_idempotent() {
  runtime_shutdown();
  runtime_shutdown();
}

// ============================================================================
// GEMM/ALU pipeline integration test (end-to-end with stub)
// ============================================================================

void test_full_pipeline_stub() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    ScopedBuffer inp(4096);
    ScopedBuffer wgt(4096);
    ScopedBuffer acc(4096);
    fill_pattern(inp.get(), 0x01);
    fill_pattern(wgt.get(), 0x02);
    fill_pattern(acc.get(), 0x00);

    set_debug_mode(ctx.cmd(), DebugFlag::FORCE_SERIAL);

    uop_loop_begin(1);
    uop_push(0, 1, 0, 0, 0, ALUOpcode::ADD, false, 0);
    uop_loop_end();
  } TEST_INSTRUCTION_END
}

// ============================================================================
// TLS command handle tests
// ============================================================================

void test_tls_command_handle_incrementing() {
  CommandHandle h1 = tls_command_handle();
  CommandHandle h2 = tls_command_handle();
  TEST_ASSERT_NE(h1, h2);
}

// ============================================================================
// Test main
// ============================================================================

int main() {
  printf("Running npu-ffi control flow tests...\n\n");

  TestRunner runner;

  runner.add_test("control/synchronize_basic", test_synchronize_basic);
  runner.add_test("control/synchronize_wait_cycles", test_synchronize_with_wait_cycles);
  runner.add_test("control/synchronize_idempotent", test_synchronize_idempotent);
  runner.add_test("control/uop_loop_basic", test_uop_loop_basic);
  runner.add_test("control/uop_loop_factors", test_uop_loop_with_factors);
  runner.add_test("control/uop_nested_loops", test_uop_nested_loops);
  runner.add_test("control/dep_push_pop", test_dep_push_pop);
  runner.add_test("control/set_debug_mode", test_set_debug_mode_dump_insn);
  runner.add_test("control/debug_combined_flags", test_set_debug_mode_combined_flags);
  runner.add_test("control/debug_flag_bitwise", test_debug_flag_bitwise_ops);
  runner.add_test("control/runtime_shutdown", test_runtime_shutdown_idempotent);
  runner.add_test("control/full_pipeline_stub", test_full_pipeline_stub);
  runner.add_test("control/tls_handle_incrementing", test_tls_command_handle_incrementing);

  return runner.run_all() ? 0 : 1;
}
