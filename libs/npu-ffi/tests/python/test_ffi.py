"""Tests for type-safe VTA FFI API bindings."""

import pytest

from npu_ffi import vta
from npu_ffi.vta import Buffer, MemcpyKind


class TestTlsCommandHandle:
    """Tests for thread-local command handle."""

    def test_tls_command_handle_returns_nonzero(self):
        cmd = vta.tls_command_handle()
        assert isinstance(cmd, int)
        assert cmd != 0

    def test_tls_command_handle_returns_incrementing_ids(self):
        cmd1 = vta.tls_command_handle()
        cmd2 = vta.tls_command_handle()
        assert cmd2 > cmd1


class TestRuntimeFunctions:
    """Tests for runtime functions."""

    def test_set_debug_mode(self):
        cmd = vta.tls_command_handle()
        vta.set_debug_mode(cmd, 0)

    def test_runtime_shutdown(self):
        vta.runtime_shutdown()

    def test_runtime_shutdown_idempotent(self):
        vta.runtime_shutdown()
        vta.runtime_shutdown()


class TestMemcpyKind:
    """Tests for MemcpyKind enum values."""

    def test_memcpy_kind_values(self):
        assert int(MemcpyKind.H2D) == 1
        assert int(MemcpyKind.D2H) == 2
        assert int(MemcpyKind.D2D) == 3

    def test_memcpy_kind_is_intenum(self):
        assert isinstance(MemcpyKind.H2D, int)
        assert MemcpyKind.H2D == 1


class TestBufferCopySafe:
    """Tests for type-safe buffer_copy_safe function."""

    def test_buffer_copy_safe_with_int_pointers(self):
        src = vta.buffer_alloc(256)
        dst = vta.buffer_alloc(256)
        assert src != 0
        assert dst != 0
        vta.buffer_copy_safe(src, 0, dst, 0, 256, MemcpyKind.D2D)
        vta.buffer_free(src)
        vta.buffer_free(dst)

    def test_buffer_copy_safe_with_buffer_objects(self):
        with Buffer(256) as src, Buffer(256) as dst:
            assert src.data != 0
            assert dst.data != 0
            vta.buffer_copy_safe(src, 0, dst, 0, 256, MemcpyKind.D2D)

    def test_buffer_copy_safe_with_mixed_pointers(self):
        src = vta.buffer_alloc(256)
        with Buffer(256) as dst:
            vta.buffer_copy_safe(src, 0, dst, 0, 128, MemcpyKind.D2D)
            vta.buffer_copy_safe(dst, 0, src, 128, 128, MemcpyKind.D2D)
        vta.buffer_free(src)

    def test_buffer_copy_safe_with_int_kind(self):
        src = vta.buffer_alloc(128)
        dst = vta.buffer_alloc(128)
        vta.buffer_copy_safe(src, 0, dst, 0, 128, int(MemcpyKind.H2D))
        vta.buffer_free(src)
        vta.buffer_free(dst)

    def test_buffer_copy_safe_with_offset(self):
        src = vta.buffer_alloc(512)
        dst = vta.buffer_alloc(512)
        vta.buffer_copy_safe(src, 64, dst, 128, 256, MemcpyKind.D2H)
        vta.buffer_free(src)
        vta.buffer_free(dst)


class TestGetCommandHandle:
    """Tests for get_command_handle convenience function."""

    def test_get_command_handle_returns_int(self):
        cmd = vta.get_command_handle()
        assert isinstance(cmd, int)
        assert cmd != 0

    def test_get_command_handle_equals_command_handle(self):
        cmd1 = vta.command_handle()
        cmd2 = vta.get_command_handle()
        assert cmd2 >= cmd1
