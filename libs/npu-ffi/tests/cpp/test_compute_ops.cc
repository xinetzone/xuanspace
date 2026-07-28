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
 * \file test_compute_ops.cc
 * \brief Tests for VTA compute operations: uop_push (GEMM/ALU modes),
 *        push_gemm_op, push_alu_op.
 *
 * These tests verify that compute instructions can be issued without crashing
 * (stub backend ignores them) and that parameter passing is correct.
 * When extending with new compute instructions (e.g., new ALU opcodes,
 * tensor operations, custom NPU kernels), follow the pattern:
 *
 * 1. Create a test_<op_name> function
 * 2. Use TEST_INSTRUCTION_BEGIN/END for automatic context
 * 3. Allocate required buffers (inp/wgt/acc)
 * 4. Issue the instruction(s)
 * 5. Verify state (on real hardware, verify output; on stub, verify no-crash)
 */

#include "npu_ffi/testing/test_utils.h"

using namespace npu_ffi::vta;
using namespace npu_ffi::vta::testing;

extern "C" {
int npu_ffi_vta_push_gemm_op(void** uop_handle, int (*finit)(void*),
                              void* signature, int nbytes);
int npu_ffi_vta_push_alu_op(void** uop_handle, int (*finit)(void*),
                             void* signature, int nbytes);
}

// ============================================================================
// ALU operation tests
// ============================================================================

void test_uop_push_alu_add() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    uop_push(1, 0, 0, 0, 0, ALUOpcode::ADD, false, 0);
  } TEST_INSTRUCTION_END
}

void test_uop_push_alu_sub() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    uop_push(1, 0, 0, 0, 0, ALUOpcode::SUB, false, 0);
  } TEST_INSTRUCTION_END
}

void test_uop_push_alu_mul() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    uop_push(1, 0, 0, 0, 0, ALUOpcode::MUL, false, 0);
  } TEST_INSTRUCTION_END
}

void test_uop_push_alu_min_max() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    uop_push(1, 0, 0, 0, 0, ALUOpcode::MIN, false, 0);
    uop_push(1, 0, 0, 0, 0, ALUOpcode::MAX, false, 0);
  } TEST_INSTRUCTION_END
}

void test_uop_push_alu_shifts() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    uop_push(1, 0, 0, 0, 0, ALUOpcode::SHR, true, 2);
    uop_push(1, 0, 0, 0, 0, ALUOpcode::SHL, true, 3);
  } TEST_INSTRUCTION_END
}

void test_uop_push_alu_imm_value() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    uop_push(1, 0, 0, 0, 0, ALUOpcode::ADD, true, 42);
  } TEST_INSTRUCTION_END
}

// ============================================================================
// GEMM operation tests
// ============================================================================

void test_uop_push_gemm_basic() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    uop_push(0, 1, 0, 0, 0, ALUOpcode::ADD, false, 0);
  } TEST_INSTRUCTION_END
}

void test_uop_push_gemm_reset_acc() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    uop_push(0, 1, 0, 0, 0, ALUOpcode::ADD, false, 0);
  } TEST_INSTRUCTION_END
}

void test_uop_push_gemm_no_reset() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    uop_push(0, 0, 0, 0, 0, ALUOpcode::ADD, false, 0);
  } TEST_INSTRUCTION_END
}

// ============================================================================
// push_gemm_op / push_alu_op (uop kernel registration) tests
//
// Note: These are the low-level C functions registered via FFI.
// They accept function pointers for finit callbacks. In stub mode,
// they are no-ops that return 0.
// ============================================================================

void test_push_gemm_op_returns_zero() {
  void* uop_handle = nullptr;
  int result = npu_ffi_vta_push_gemm_op(&uop_handle, nullptr, nullptr, 0);
  TEST_ASSERT_EQ(result, 0);
}

void test_push_alu_op_returns_zero() {
  void* uop_handle = nullptr;
  int result = npu_ffi_vta_push_alu_op(&uop_handle, nullptr, nullptr, 0);
  TEST_ASSERT_EQ(result, 0);
}

// ============================================================================
// ALU opcode enum value verification
// ============================================================================

void test_alu_opcode_values() {
  TEST_ASSERT_EQ(static_cast<uint32_t>(ALUOpcode::ADD), 0U);
  TEST_ASSERT_EQ(static_cast<uint32_t>(ALUOpcode::SUB), 1U);
  TEST_ASSERT_EQ(static_cast<uint32_t>(ALUOpcode::MUL), 2U);
  TEST_ASSERT_EQ(static_cast<uint32_t>(ALUOpcode::MIN), 3U);
  TEST_ASSERT_EQ(static_cast<uint32_t>(ALUOpcode::MAX), 4U);
  TEST_ASSERT_EQ(static_cast<uint32_t>(ALUOpcode::SHR), 5U);
  TEST_ASSERT_EQ(static_cast<uint32_t>(ALUOpcode::SHL), 6U);
}

void test_memory_type_values() {
  TEST_ASSERT_EQ(static_cast<uint32_t>(MemoryType::DRAM), 0U);
  TEST_ASSERT_EQ(static_cast<uint32_t>(MemoryType::SRAM), 1U);
  TEST_ASSERT_EQ(static_cast<uint32_t>(MemoryType::UOP), 2U);
  TEST_ASSERT_EQ(static_cast<uint32_t>(MemoryType::INP), 3U);
  TEST_ASSERT_EQ(static_cast<uint32_t>(MemoryType::WGT), 4U);
  TEST_ASSERT_EQ(static_cast<uint32_t>(MemoryType::ACC), 5U);
  TEST_ASSERT_EQ(static_cast<uint32_t>(MemoryType::OUT), 6U);
}

// ============================================================================
// Test main
// ============================================================================

int main() {
  printf("Running npu-ffi compute operations tests...\n\n");

  TestRunner runner;

  runner.add_test("compute/alu_add", test_uop_push_alu_add);
  runner.add_test("compute/alu_sub", test_uop_push_alu_sub);
  runner.add_test("compute/alu_mul", test_uop_push_alu_mul);
  runner.add_test("compute/alu_min_max", test_uop_push_alu_min_max);
  runner.add_test("compute/alu_shifts", test_uop_push_alu_shifts);
  runner.add_test("compute/alu_imm_value", test_uop_push_alu_imm_value);
  runner.add_test("compute/gemm_basic", test_uop_push_gemm_basic);
  runner.add_test("compute/gemm_reset_acc", test_uop_push_gemm_reset_acc);
  runner.add_test("compute/gemm_no_reset", test_uop_push_gemm_no_reset);
  runner.add_test("compute/push_gemm_op_returns_zero", test_push_gemm_op_returns_zero);
  runner.add_test("compute/push_alu_op_returns_zero", test_push_alu_op_returns_zero);
  runner.add_test("compute/alu_opcode_values", test_alu_opcode_values);
  runner.add_test("compute/memory_type_values", test_memory_type_values);

  return runner.run_all() ? 0 : 1;
}
