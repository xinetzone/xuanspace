"""Tests for low-level VTA FFI API bindings."""
from __future__ import annotations

import gc
import pytest

from npu_ffi import vta


class TestCommandHandle:
    """Tests for thread-local command handle."""

    def test_tls_command_handle_returns_nonzero(self):
        cmd = vta.tls_command_handle()
        assert isinstance(cmd, int)
        assert cmd != 0

    def test_tls_command_handle_returns_incrementing_ids(self):
        cmd1 = vta.tls_command_handle()
        cmd2 = vta.tls_command_handle()
        assert cmd2 > cmd1


class TestBufferAllocFree:
    """Tests for buffer allocation and deallocation."""

    def test_buffer_alloc_returns_nonzero(self):
        ptr = vta.buffer_alloc(1024)
        assert isinstance(ptr, int)
        assert ptr != 0
        vta.buffer_free(ptr)

    def test_buffer_alloc_small_size(self):
        ptr = vta.buffer_alloc(64)
        assert ptr != 0
        vta.buffer_free(ptr)

    def test_buffer_alloc_large_size(self):
        ptr = vta.buffer_alloc(1024 * 1024)
        assert ptr != 0
        vta.buffer_free(ptr)

    def test_buffer_free_null_does_not_crash(self):
        vta.buffer_free(0)

    def test_buffer_alloc_free_cycle(self):
        ptrs = []
        for _ in range(10):
            ptr = vta.buffer_alloc(256)
            assert ptr != 0
            ptrs.append(ptr)
        for ptr in ptrs:
            vta.buffer_free(ptr)

    def test_buffer_cpu_ptr_returns_same_pointer(self, cmd_handle, buffer_1k):
        cpu_ptr = vta.buffer_cpu_ptr(cmd_handle, buffer_1k)
        assert cpu_ptr == buffer_1k


class TestBufferCopy:
    """Tests for buffer copy operations."""

    def test_buffer_copy_basic(self):
        src = vta.buffer_alloc(256)
        dst = vta.buffer_alloc(256)
        assert src != 0
        assert dst != 0
        vta.buffer_copy(src, 0, dst, 0, 256, int(vta.MemcpyKind.D2D))
        vta.buffer_free(src)
        vta.buffer_free(dst)

    def test_buffer_copy_with_offset(self):
        src = vta.buffer_alloc(512)
        dst = vta.buffer_alloc(512)
        vta.buffer_copy(src, 64, dst, 128, 128, int(vta.MemcpyKind.H2D))
        vta.buffer_free(src)
        vta.buffer_free(dst)

    def test_buffer_copy_zero_size(self):
        src = vta.buffer_alloc(256)
        dst = vta.buffer_alloc(256)
        vta.buffer_copy(src, 0, dst, 0, 0, int(vta.MemcpyKind.D2H))
        vta.buffer_free(src)
        vta.buffer_free(dst)


class TestBarrierOperations:
    """Tests for read/write barrier operations."""

    def test_write_barrier(self, cmd_handle, buffer_1k):
        vta.write_barrier(cmd_handle, buffer_1k, 32, 0, 16)

    def test_read_barrier(self, cmd_handle, buffer_1k):
        vta.read_barrier(cmd_handle, buffer_1k, 32, 0, 16)

    def test_write_barrier_full_buffer(self, cmd_handle, buffer_1k):
        vta.write_barrier(cmd_handle, buffer_1k, 8, 0, 1024 // 8)

    def test_read_barrier_full_buffer(self, cmd_handle, buffer_1k):
        vta.read_barrier(cmd_handle, buffer_1k, 8, 0, 1024 // 8)


class Test2DLoadStore:
    """Tests for 2D buffer load/store operations."""

    def test_load_buffer_2d(self, cmd_handle, buffer_1k):
        vta.load_buffer_2d(
            cmd_handle, buffer_1k, 0,
            16, 8, 32,
            0, 0, 0, 0,
            0, int(vta.MemoryType.INP)
        )

    def test_store_buffer_2d(self, cmd_handle, buffer_1k):
        vta.store_buffer_2d(
            cmd_handle, 0, int(vta.MemoryType.ACC),
            buffer_1k, 0,
            16, 8, 32
        )

    def test_load_buffer_2d_with_padding(self, cmd_handle, buffer_1k):
        vta.load_buffer_2d(
            cmd_handle, buffer_1k, 0,
            8, 4, 16,
            1, 1, 1, 1,
            1, int(vta.MemoryType.WGT)
        )


class TestUopOperations:
    """Tests for micro-op push operations."""

    def test_uop_push_gemm_mode(self):
        vta.uop_push(
            0, 1, 0, 0, 0,
            int(vta.ALUOpcode.ADD), 0, 0
        )

    def test_uop_push_alu_mode(self):
        vta.uop_push(
            1, 0, 0, 0, 0,
            int(vta.ALUOpcode.ADD), 0, 0
        )

    def test_uop_push_with_immediate(self):
        vta.uop_push(
            1, 0, 0, 0, 0,
            int(vta.ALUOpcode.ADD), 1, 42
        )

    def test_uop_loop_begin_end(self):
        vta.uop_loop_begin(4, 1, 1, 1)
        vta.uop_loop_end()

    def test_uop_nested_loops(self):
        vta.uop_loop_begin(2, 0, 0, 0)
        vta.uop_loop_begin(4, 1, 1, 1)
        vta.uop_loop_end()
        vta.uop_loop_end()


class TestPushOps:
    """Tests for push_gemm_op and push_alu_op."""

    def test_push_gemm_op(self):
        result = vta.push_gemm_op()
        assert result == 0

    def test_push_alu_op(self):
        result = vta.push_alu_op()
        assert result == 0


class TestDependencyOps:
    """Tests for dependency push/pop operations."""

    def test_dep_push(self, cmd_handle):
        result = vta.dep_push(cmd_handle, 0, 1)
        assert result == 0

    def test_dep_pop(self, cmd_handle):
        result = vta.dep_pop(cmd_handle, 0, 1)
        assert result == 0

    def test_dep_push_pop_chain(self, cmd_handle):
        assert vta.dep_push(cmd_handle, 0, 1) == 0
        assert vta.dep_pop(cmd_handle, 0, 1) == 0


class TestSynchronize:
    """Tests for synchronize operation."""

    def test_synchronize_default(self, cmd_handle):
        vta.synchronize(cmd_handle, 0)

    def test_synchronize_with_timeout(self, cmd_handle):
        vta.synchronize(cmd_handle, 1000)


class TestDebugMode:
    """Tests for debug mode setting."""

    def test_set_debug_mode_none(self, cmd_handle):
        vta.set_debug_mode(cmd_handle, 0)

    def test_set_debug_mode_dump_insn(self, cmd_handle):
        vta.set_debug_mode(cmd_handle, int(vta.DebugFlag.DUMP_INSN))

    def test_set_debug_mode_all_flags(self, cmd_handle):
        all_flags = (
            int(vta.DebugFlag.DUMP_INSN) |
            int(vta.DebugFlag.DUMP_UOP)
        )
        vta.set_debug_mode(cmd_handle, all_flags)


class TestRuntimeShutdown:
    """Tests for runtime shutdown."""

    def test_runtime_shutdown(self):
        vta.runtime_shutdown()

    def test_runtime_shutdown_idempotent(self):
        vta.runtime_shutdown()
        vta.runtime_shutdown()


class TestPrepareCallFunc:
    """Tests for prepare_call_func."""

    def test_prepare_call_func(self, cmd_handle):
        vta.prepare_call_func(cmd_handle, "test_function")
