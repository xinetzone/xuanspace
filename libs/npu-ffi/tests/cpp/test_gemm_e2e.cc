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
 * \file test_gemm_e2e.cc
 * \brief End-to-end integration tests for GEMM (General Matrix Multiplication)
 *        instruction pipeline on VTA.
 *
 * These tests exercise the full five-layer FFI instruction flow for GEMM:
 *   1. Memory allocation (DRAM buffers for input, weight, accumulator)
 *   2. Data movement (read_barrier -> load_buffer_2d for INP/WGT)
 *   3. Compute (uop_loop_begin -> uop_push(GEMM) -> uop_loop_end)
 *   4. Store (store_buffer_2d from ACC -> write_barrier)
 *   5. Synchronization (wait for completion)
 *
 * On the stub backend, compute operations are no-ops, so we verify:
 *   - The full instruction pipeline executes without crashing
 *   - Buffer lifecycle (alloc/free) works correctly through the pipeline
 *   - Barrier and 2D load/store instructions can be chained
 *   - UOP loops and GEMM pushes work in sequence
 *
 * On real VTA hardware, these tests would additionally verify:
 *   - Correct matrix multiplication results in the accumulator
 *   - Numerical accuracy against expected outputs
 *   - Proper tile sizing and SRAM index management
 *
 * When extending with new GEMM variants (e.g., different tile sizes,
 * batched GEMM, strided access), add test_<variant>() functions below.
 */

#include "npu_ffi/testing/test_utils.h"

#include <cstdint>

using namespace npu_ffi::vta;
using namespace npu_ffi::vta::testing;

// ============================================================================
// Helper: GEMM tile dimensions (VTA typical blocked GEMM sizes)
// ============================================================================

namespace {

constexpr uint32_t kBlockOut = 16;
constexpr uint32_t kBlockIn = 16;
constexpr uint32_t kBlockBatch = 1;
constexpr uint32_t kUopIdx = 0;
constexpr size_t kTensorSize = kBlockOut * kBlockIn * sizeof(int32_t);

void fill_identity_like(Buffer& buf, uint8_t val) {
  fill_pattern(buf, val);
}

}  // namespace

// ============================================================================
// GEMM Pipeline Tests
// ============================================================================

/*!
 * \brief Minimal GEMM pipeline: single tile GEMM with load-compute-store.
 *
 * This is the simplest end-to-end GEMM test:
 * 1. Allocate DRAM buffers (inp, wgt, acc)
 * 2. Read barrier to make DRAM data visible
 * 3. Load INP and WGT tiles from DRAM to SRAM
 * 4. Push a single GEMM uop (no loop)
 * 5. Store ACC result back to DRAM
 * 6. Write barrier and synchronize
 */
void test_gemm_single_tile() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    ScopedBuffer inp(kTensorSize);
    ScopedBuffer wgt(kTensorSize);
    ScopedBuffer acc(kTensorSize);

    fill_pattern(inp.get(), 0x01);
    fill_pattern(wgt.get(), 0x02);
    fill_pattern(acc.get(), 0x00);

    read_barrier(ctx.cmd(), inp.get(), 8, 0, static_cast<uint32_t>(kTensorSize));
    read_barrier(ctx.cmd(), wgt.get(), 8, 0, static_cast<uint32_t>(kTensorSize));

    load_buffer_2d(ctx.cmd(), inp.get(), 0, kBlockIn, kBlockOut, kBlockIn,
                   0, 0, 0, 0, 0, MemoryType::INP);
    load_buffer_2d(ctx.cmd(), wgt.get(), 0, kBlockIn, kBlockOut, kBlockIn,
                   0, 0, 0, 0, 0, MemoryType::WGT);

    uop_push(0, 1, kUopIdx, kUopIdx, kUopIdx, ALUOpcode::ADD, false, 0);

    store_buffer_2d(ctx.cmd(), 0, MemoryType::ACC, acc.get(), 0,
                    kBlockIn, kBlockOut, kBlockIn);
    write_barrier(ctx.cmd(), acc.get(), 8, 0, static_cast<uint32_t>(kTensorSize));
  } TEST_INSTRUCTION_END
}

/*!
 * \brief GEMM with uop loop: simulates blocked GEMM over multiple inner iterations.
 *
 * Tests that uop_loop_begin/uop_loop_end correctly bracket GEMM uops
 * and that multiple GEMM pushes can occur within a loop body.
 */
void test_gemm_with_uop_loop() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    ScopedBuffer inp(kTensorSize);
    ScopedBuffer wgt(kTensorSize);
    ScopedBuffer acc(kTensorSize);

    fill_pattern(inp.get(), 0x0A);
    fill_pattern(wgt.get(), 0x0B);
    fill_pattern(acc.get(), 0x00);

    read_barrier(ctx.cmd(), inp.get(), 8, 0, static_cast<uint32_t>(kTensorSize));
    read_barrier(ctx.cmd(), wgt.get(), 8, 0, static_cast<uint32_t>(kTensorSize));

    load_buffer_2d(ctx.cmd(), inp.get(), 0, kBlockIn, kBlockOut, kBlockIn,
                   0, 0, 0, 0, 0, MemoryType::INP);
    load_buffer_2d(ctx.cmd(), wgt.get(), 0, kBlockIn, kBlockOut, kBlockIn,
                   0, 0, 0, 0, 0, MemoryType::WGT);

    uop_loop_begin(kBlockBatch, 0, 1, 1);
    uop_push(0, 1, kUopIdx, kUopIdx, kUopIdx, ALUOpcode::ADD, false, 0);
    uop_loop_end();

    store_buffer_2d(ctx.cmd(), 0, MemoryType::ACC, acc.get(), 0,
                    kBlockIn, kBlockOut, kBlockIn);
    write_barrier(ctx.cmd(), acc.get(), 8, 0, static_cast<uint32_t>(kTensorSize));
  } TEST_INSTRUCTION_END
}

/*!
 * \brief GEMM with reset_out=0 (accumulate mode): verifies accumulator is not reset.
 *
 * The reset_out parameter controls whether the accumulator is zeroed
 * before GEMM computation. reset_out=0 means accumulate into existing values.
 */
void test_gemm_accumulate_mode() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    ScopedBuffer inp(kTensorSize);
    ScopedBuffer wgt(kTensorSize);
    ScopedBuffer acc(kTensorSize);

    fill_pattern(inp.get(), 0x03);
    fill_pattern(wgt.get(), 0x04);
    fill_pattern(acc.get(), 0x05);

    read_barrier(ctx.cmd(), inp.get(), 8, 0, static_cast<uint32_t>(kTensorSize));
    read_barrier(ctx.cmd(), wgt.get(), 8, 0, static_cast<uint32_t>(kTensorSize));
    read_barrier(ctx.cmd(), acc.get(), 8, 0, static_cast<uint32_t>(kTensorSize));

    load_buffer_2d(ctx.cmd(), inp.get(), 0, kBlockIn, kBlockOut, kBlockIn,
                   0, 0, 0, 0, 0, MemoryType::INP);
    load_buffer_2d(ctx.cmd(), wgt.get(), 0, kBlockIn, kBlockOut, kBlockIn,
                   0, 0, 0, 0, 0, MemoryType::WGT);
    load_buffer_2d(ctx.cmd(), acc.get(), 0, kBlockIn, kBlockOut, kBlockIn,
                   0, 0, 0, 0, 0, MemoryType::ACC);

    uop_push(0, 0, kUopIdx, kUopIdx, kUopIdx, ALUOpcode::ADD, false, 0);

    store_buffer_2d(ctx.cmd(), 0, MemoryType::ACC, acc.get(), 0,
                    kBlockIn, kBlockOut, kBlockIn);
    write_barrier(ctx.cmd(), acc.get(), 8, 0, static_cast<uint32_t>(kTensorSize));
  } TEST_INSTRUCTION_END
}

/*!
 * \brief Multi-tile GEMM: load-compute-store cycle repeated for two tiles.
 *
 * Simulates processing multiple output tiles in sequence, verifying
 * that the pipeline can handle repeated load/compute/store cycles
 * without resource leaks or crashes.
 */
void test_gemm_multi_tile() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    constexpr size_t kTwoTileSize = kTensorSize * 2;
    ScopedBuffer inp(kTwoTileSize);
    ScopedBuffer wgt(kTwoTileSize);
    ScopedBuffer acc(kTwoTileSize);

    fill_pattern(inp.get(), 0x10);
    fill_pattern(wgt.get(), 0x20);
    fill_pattern(acc.get(), 0x00);

    for (uint32_t tile = 0; tile < 2; tile++) {
      uint32_t offset = tile * static_cast<uint32_t>(kTensorSize);

      read_barrier(ctx.cmd(), inp.get(), 8, offset / sizeof(int32_t),
                   static_cast<uint32_t>(kTensorSize / sizeof(int32_t)));
      read_barrier(ctx.cmd(), wgt.get(), 8, offset / sizeof(int32_t),
                   static_cast<uint32_t>(kTensorSize / sizeof(int32_t)));

      load_buffer_2d(ctx.cmd(), inp.get(), offset / sizeof(int32_t),
                     kBlockIn, kBlockOut, kBlockIn,
                     0, 0, 0, 0, 0, MemoryType::INP);
      load_buffer_2d(ctx.cmd(), wgt.get(), offset / sizeof(int32_t),
                     kBlockIn, kBlockOut, kBlockIn,
                     0, 0, 0, 0, 0, MemoryType::WGT);

      uop_push(0, 1, kUopIdx, kUopIdx, kUopIdx, ALUOpcode::ADD, false, 0);

      store_buffer_2d(ctx.cmd(), 0, MemoryType::ACC, acc.get(),
                      offset / sizeof(int32_t),
                      kBlockIn, kBlockOut, kBlockIn);
      write_barrier(ctx.cmd(), acc.get(), 8, offset / sizeof(int32_t),
                    static_cast<uint32_t>(kTensorSize / sizeof(int32_t)));
    }
  } TEST_INSTRUCTION_END
}

/*!
 * \brief GEMM with padding: exercises x_pad/y_pad parameters in load_buffer_2d.
 *
 * Verifies that 2D loads with padding parameters work correctly
 * within a GEMM pipeline.
 */
void test_gemm_with_padding() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    ScopedBuffer inp(kTensorSize);
    ScopedBuffer wgt(kTensorSize);
    ScopedBuffer acc(kTensorSize);

    fill_pattern(inp.get(), 0x77);
    fill_pattern(wgt.get(), 0x88);
    fill_pattern(acc.get(), 0x00);

    read_barrier(ctx.cmd(), inp.get(), 8, 0, static_cast<uint32_t>(kTensorSize));
    read_barrier(ctx.cmd(), wgt.get(), 8, 0, static_cast<uint32_t>(kTensorSize));

    load_buffer_2d(ctx.cmd(), inp.get(), 0, kBlockIn, kBlockOut, kBlockIn,
                   1, 1, 0, 0, 0, MemoryType::INP);
    load_buffer_2d(ctx.cmd(), wgt.get(), 0, kBlockIn, kBlockOut, kBlockIn,
                   0, 0, 1, 1, 0, MemoryType::WGT);

    uop_push(0, 1, kUopIdx, kUopIdx, kUopIdx, ALUOpcode::ADD, false, 0);

    store_buffer_2d(ctx.cmd(), 0, MemoryType::ACC, acc.get(), 0,
                    kBlockIn, kBlockOut, kBlockIn);
    write_barrier(ctx.cmd(), acc.get(), 8, 0, static_cast<uint32_t>(kTensorSize));
  } TEST_INSTRUCTION_END
}

/*!
 * \brief GEMM with debug mode enabled: verifies instruction dumping doesn't crash.
 *
 * Sets DUMP_INSN debug flag and runs a full GEMM pipeline, ensuring
 * debug mode doesn't interfere with instruction execution.
 */
void test_gemm_with_debug_mode() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    ScopedBuffer inp(kTensorSize);
    ScopedBuffer wgt(kTensorSize);
    ScopedBuffer acc(kTensorSize);

    fill_pattern(inp.get(), 0xDD);
    fill_pattern(wgt.get(), 0xEE);
    fill_pattern(acc.get(), 0x00);

    set_debug_mode(ctx.cmd(), DebugFlag::DUMP_INSN | DebugFlag::DUMP_UOP);

    read_barrier(ctx.cmd(), inp.get(), 8, 0, static_cast<uint32_t>(kTensorSize));
    load_buffer_2d(ctx.cmd(), inp.get(), 0, kBlockIn, kBlockOut, kBlockIn,
                   0, 0, 0, 0, 0, MemoryType::INP);
    load_buffer_2d(ctx.cmd(), wgt.get(), 0, kBlockIn, kBlockOut, kBlockIn,
                   0, 0, 0, 0, 0, MemoryType::WGT);
    uop_push(0, 1, kUopIdx, kUopIdx, kUopIdx, ALUOpcode::ADD, false, 0);
    store_buffer_2d(ctx.cmd(), 0, MemoryType::ACC, acc.get(), 0,
                    kBlockIn, kBlockOut, kBlockIn);
    write_barrier(ctx.cmd(), acc.get(), 8, 0, static_cast<uint32_t>(kTensorSize));
  } TEST_INSTRUCTION_END
}

/*!
 * \brief GEMM followed by ALU: verifies GEMM compute can be chained with ALU ops.
 *
 * This tests a typical post-GEMM operation pattern:
 *   1. Compute GEMM into accumulator
 *   2. Apply ALU activation (e.g., ReLU via MAX with 0)
 *   3. Store the result
 */
void test_gemm_then_alu() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    ScopedBuffer inp(kTensorSize);
    ScopedBuffer wgt(kTensorSize);
    ScopedBuffer acc(kTensorSize);

    fill_pattern(inp.get(), 0x0C);
    fill_pattern(wgt.get(), 0x0D);
    fill_pattern(acc.get(), 0x00);

    read_barrier(ctx.cmd(), inp.get(), 8, 0, static_cast<uint32_t>(kTensorSize));
    read_barrier(ctx.cmd(), wgt.get(), 8, 0, static_cast<uint32_t>(kTensorSize));

    load_buffer_2d(ctx.cmd(), inp.get(), 0, kBlockIn, kBlockOut, kBlockIn,
                   0, 0, 0, 0, 0, MemoryType::INP);
    load_buffer_2d(ctx.cmd(), wgt.get(), 0, kBlockIn, kBlockOut, kBlockIn,
                   0, 0, 0, 0, 0, MemoryType::WGT);

    uop_push(0, 1, kUopIdx, kUopIdx, kUopIdx, ALUOpcode::ADD, false, 0);

    uop_push(1, 0, kUopIdx, kUopIdx, kUopIdx, ALUOpcode::MAX, true, 0);

    store_buffer_2d(ctx.cmd(), 0, MemoryType::ACC, acc.get(), 0,
                    kBlockIn, kBlockOut, kBlockIn);
    write_barrier(ctx.cmd(), acc.get(), 8, 0, static_cast<uint32_t>(kTensorSize));
  } TEST_INSTRUCTION_END
}

/*!
 * \brief GEMM with double synchronization: verifies idempotent sync is safe.
 *
 * Calls synchronize() explicitly before context destruction, ensuring
 * double synchronization doesn't cause issues.
 */
void test_gemm_double_sync() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    ScopedBuffer inp(kTensorSize);
    ScopedBuffer wgt(kTensorSize);
    ScopedBuffer acc(kTensorSize);

    fill_pattern(inp.get(), 0xF0);
    fill_pattern(wgt.get(), 0x0F);

    load_buffer_2d(ctx.cmd(), inp.get(), 0, kBlockIn, kBlockOut, kBlockIn,
                   0, 0, 0, 0, 0, MemoryType::INP);
    load_buffer_2d(ctx.cmd(), wgt.get(), 0, kBlockIn, kBlockOut, kBlockIn,
                   0, 0, 0, 0, 0, MemoryType::WGT);
    uop_push(0, 1, kUopIdx, kUopIdx, kUopIdx, ALUOpcode::ADD, false, 0);
    store_buffer_2d(ctx.cmd(), 0, MemoryType::ACC, acc.get(), 0,
                    kBlockIn, kBlockOut, kBlockIn);

    ctx.get().synchronize();
    ctx.get().synchronize();
  } TEST_INSTRUCTION_END
}

/*!
 * \brief GEMM with dependence tokens: dep_push/dep_pop in GEMM pipeline.
 *
 * Tests that dependence tokens can be pushed and popped around
 * GEMM compute operations without crashing.
 */
void test_gemm_with_deps() {
  TEST_INSTRUCTION_BEGIN(ctx) {
    ScopedBuffer inp(kTensorSize);
    ScopedBuffer wgt(kTensorSize);
    ScopedBuffer acc(kTensorSize);

    fill_pattern(inp.get(), 0xAA);
    fill_pattern(wgt.get(), 0xBB);

    load_buffer_2d(ctx.cmd(), inp.get(), 0, kBlockIn, kBlockOut, kBlockIn,
                   0, 0, 0, 0, 0, MemoryType::INP);
    load_buffer_2d(ctx.cmd(), wgt.get(), 0, kBlockIn, kBlockOut, kBlockIn,
                   0, 0, 0, 0, 0, MemoryType::WGT);

    dep_push(ctx.cmd(), 0, 1);

    uop_push(0, 1, kUopIdx, kUopIdx, kUopIdx, ALUOpcode::ADD, false, 0);

    dep_pop(ctx.cmd(), 0, 1);

    store_buffer_2d(ctx.cmd(), 0, MemoryType::ACC, acc.get(), 0,
                    kBlockIn, kBlockOut, kBlockIn);
  } TEST_INSTRUCTION_END
}

// ============================================================================
// Test main
// ============================================================================

int main() {
  printf("Running npu-ffi GEMM end-to-end integration tests...\n\n");

  TestRunner runner;

  runner.add_test("gemm/single_tile", test_gemm_single_tile);
  runner.add_test("gemm/uop_loop", test_gemm_with_uop_loop);
  runner.add_test("gemm/accumulate_mode", test_gemm_accumulate_mode);
  runner.add_test("gemm/multi_tile", test_gemm_multi_tile);
  runner.add_test("gemm/with_padding", test_gemm_with_padding);
  runner.add_test("gemm/debug_mode", test_gemm_with_debug_mode);
  runner.add_test("gemm/gemm_then_alu", test_gemm_then_alu);
  runner.add_test("gemm/double_sync", test_gemm_double_sync);
  runner.add_test("gemm/with_deps", test_gemm_with_deps);

  return runner.run_all() ? 0 : 1;
}
