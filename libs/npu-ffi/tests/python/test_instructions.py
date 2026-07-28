#!/usr/bin/env python3
"""NPU-FFI VTA instruction tests (Python layer).

This module demonstrates the recommended test structure for NPU instruction set
extensions at the Python FFI binding layer. It mirrors the C++ test structure
(memory ops, compute ops, control flow) and serves as a template for adding
tests when new instructions are introduced.

Usage:
    python -m pytest tests/python/test_instructions.py -v
    or: python tests/python/test_instructions.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from npu_ffi import vta


# ============================================================================
# Memory operation tests
# ============================================================================

class TestMemoryOps:
    """Tests for memory-related VTA instructions."""

    def test_buffer_alloc_free(self):
        """Test buffer allocation and deallocation."""
        buf = vta.Buffer(1024)
        assert buf.data != 0
        assert buf.size == 1024
        assert buf.owns_data is True
        buf.reset()

    def test_buffer_context_manager(self):
        """Test buffer as context manager (RAII)."""
        with vta.Buffer(256) as buf:
            assert buf.data != 0
            assert len(buf) == 256

    def test_buffer_copy_d2d(self):
        """Test device-to-device buffer copy."""
        with vta.CommandContext() as cmd:
            src = vta.Buffer(256)
            dst = vta.Buffer(256)
            vta.buffer_copy(src.data, 0, dst.data, 0, 256, int(vta.MemcpyKind.D2D))

    def test_buffer_copy_h2d_d2h(self):
        """Test host-to-device and device-to-host copy round-trip."""
        with vta.CommandContext() as cmd:
            src = vta.Buffer(128)
            dst = vta.Buffer(128)
            vta.buffer_copy(src.data, 0, dst.data, 0, 128, int(vta.MemcpyKind.H2D))
            vta.buffer_copy(dst.data, 0, src.data, 0, 128, int(vta.MemcpyKind.D2H))

    def test_load_store_buffer_2d(self):
        """Test 2D buffer load (DRAM->SRAM) and store (SRAM->DRAM)."""
        with vta.CommandContext() as cmd:
            dram_buf = vta.Buffer(4096)
            sram_idx = 0
            mem_type = int(vta.MemoryType.ACC)

            vta.load_buffer_2d(
                cmd, dram_buf.data, 0,
                16, 1, 16,
                0, 0, 0, 0,
                sram_idx, mem_type
            )
            vta.store_buffer_2d(
                cmd, sram_idx, mem_type,
                dram_buf.data, 0,
                16, 1, 16
            )

    def test_write_read_barriers(self):
        """Test write barrier (device->CPU) and read barrier (CPU->device)."""
        with vta.CommandContext() as cmd:
            buf = vta.Buffer(512)
            elem_bits = 8
            buf.write_barrier(cmd, elem_bits, 0, 512)
            buf.read_barrier(cmd, elem_bits, 0, 512)

    def test_buffer_cpu_ptr(self):
        """Test getting CPU-accessible pointer from buffer."""
        with vta.CommandContext() as cmd:
            buf = vta.Buffer(256)
            cpu_p = buf.cpu_ptr(cmd)
            assert cpu_p != 0


# ============================================================================
# Compute operation tests
# ============================================================================

class TestComputeOps:
    """Tests for compute-related VTA instructions (GEMM/ALU)."""

    def test_gemm_basic(self):
        """Test basic GEMM micro-op with accumulator reset."""
        with vta.CommandContext() as cmd:
            vta.uop_push(0, 1, 0, 0, 0, int(vta.ALUOpcode.ADD), 0, 0)

    def test_gemm_no_accumulator_reset(self):
        """Test GEMM without resetting accumulator."""
        with vta.CommandContext() as cmd:
            vta.uop_push(0, 0, 0, 0, 0, int(vta.ALUOpcode.ADD), 0, 0)

    def test_alu_all_opcodes(self):
        """Test all ALU opcodes can be pushed without errors."""
        with vta.CommandContext() as cmd:
            for opcode in vta.ALUOpcode:
                vta.uop_push(1, 0, 0, 0, 0, int(opcode), 0, 0)

    def test_alu_with_immediate(self):
        """Test ALU operation using immediate value."""
        with vta.CommandContext() as cmd:
            vta.uop_push(1, 0, 0, 0, 0, int(vta.ALUOpcode.ADD), 1, 42)


# ============================================================================
# Control flow tests
# ============================================================================

class TestControlFlow:
    """Tests for VTA control flow instructions."""

    def test_uop_loop_basic(self):
        """Test basic micro-op loop."""
        with vta.CommandContext() as cmd:
            vta.uop_loop_begin(4, 0, 0, 0)
            vta.uop_push(1, 0, 0, 0, 0, int(vta.ALUOpcode.ADD), 0, 0)
            vta.uop_loop_end()

    def test_uop_loop_with_factors(self):
        """Test micro-op loop with memory index factors."""
        with vta.CommandContext() as cmd:
            vta.uop_loop_begin(8, 1, 1, 1)
            vta.uop_push(0, 0, 0, 0, 0, int(vta.ALUOpcode.ADD), 0, 0)
            vta.uop_loop_end()

    def test_uop_nested_loops(self):
        """Test nested micro-op loops."""
        with vta.CommandContext() as cmd:
            vta.uop_loop_begin(2, 0, 0, 0)
            vta.uop_loop_begin(4, 0, 0, 0)
            vta.uop_push(1, 0, 0, 0, 0, int(vta.ALUOpcode.ADD), 0, 0)
            vta.uop_loop_end()
            vta.uop_loop_end()

    def test_dep_push_pop(self):
        """Test dependency token push/pop between compute queues."""
        with vta.CommandContext() as cmd:
            r1 = vta.dep_push(cmd, 0, 1)
            r2 = vta.dep_pop(cmd, 0, 1)
            assert r1 == 0
            assert r2 == 0

    def test_set_debug_mode(self):
        """Test setting debug flags."""
        with vta.CommandContext() as cmd:
            vta.set_debug_mode(cmd, int(vta.DebugFlag.DUMP_INSN))

    def test_set_debug_mode_combined(self):
        """Test setting combined debug flags."""
        with vta.CommandContext() as cmd:
            flags = int(vta.DebugFlag.DUMP_INSN) | int(vta.DebugFlag.DUMP_UOP)
            vta.set_debug_mode(cmd, flags)

    def test_context_manager_auto_sync(self):
        """Test that CommandContext auto-synchronizes on exit."""
        ctx = vta.CommandContext()
        with ctx as cmd:
            assert cmd != 0
            assert ctx.handle == cmd
        assert ctx._cmd is None


# ============================================================================
# Template for new instruction tests
# ============================================================================

# When adding a new NPU instruction, follow this pattern:
#
# 1. Determine the instruction category (memory / compute / control_flow / new_category)
# 2. Add test methods to the corresponding Test* class (or create a new class)
# 3. Follow arrange-act-assert:
#    - Arrange: allocate buffers, set up context
#    - Act: call the instruction function with valid parameters
#    - Assert: verify expected outcomes (return values, buffer state, etc.)
# 4. For instructions that need specific hardware features, use appropriate guards
# 5. Test boundary conditions (zero sizes, edge offsets, invalid parameters)
#
# Example template for a new "conv2d" instruction:
#
# class TestConv2DOps:
#     """Tests for convolution 2D instructions."""
#
#     def test_conv2d_basic(self):
#         with vta.CommandContext() as cmd:
#             inp = vta.Buffer(inp_size)
#             wgt = vta.Buffer(wgt_size)
#             acc = vta.Buffer(acc_size)
#             # Arrange: fill patterns, load weights...
#             # Act: call conv2d instruction
#             vta.conv2d(cmd, inp.data, wgt.data, acc.data, ...)
#             # Assert: verify results after sync


if __name__ == '__main__':
    print("=" * 60)
    print("Running npu-ffi Python instruction tests")
    print("=" * 60)

    test_classes = [TestMemoryOps, TestComputeOps, TestControlFlow]
    total = 0
    passed = 0
    failed = 0

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        print(f"\n--- {cls.__name__} ({len(methods)} tests) ---")
        for method_name in sorted(methods):
            total += 1
            try:
                getattr(instance, method_name)()
                print(f"  PASS: {method_name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL: {method_name}: {e}")
                failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
