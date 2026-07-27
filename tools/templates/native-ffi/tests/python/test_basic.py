"""Basic tests for {{module_name|upper}} FFI API bindings."""
from __future__ import annotations

from {{package_name}} import {{module_name}}


class TestCommandHandle:
    """Tests for thread-local command handle."""

    def test_tls_command_handle_returns_nonzero(self):
        cmd = {{module_name}}.tls_command_handle()
        assert isinstance(cmd, int)
        assert cmd != 0

    def test_tls_command_handle_returns_incrementing_ids(self):
        cmd1 = {{module_name}}.tls_command_handle()
        cmd2 = {{module_name}}.tls_command_handle()
        assert cmd2 > cmd1


class TestBufferAllocFree:
    """Tests for buffer allocation and deallocation."""

    def test_buffer_alloc_returns_nonzero(self):
        ptr = {{module_name}}.buffer_alloc(1024)
        assert isinstance(ptr, int)
        assert ptr != 0
        {{module_name}}.buffer_free(ptr)

    def test_buffer_alloc_small_size(self):
        ptr = {{module_name}}.buffer_alloc(64)
        assert ptr != 0
        {{module_name}}.buffer_free(ptr)

    def test_buffer_free_null_does_not_crash(self):
        {{module_name}}.buffer_free(0)


class TestRuntimeShutdown:
    """Tests for runtime shutdown."""

    def test_runtime_shutdown(self):
        {{module_name}}.runtime_shutdown()

    def test_runtime_shutdown_idempotent(self):
        {{module_name}}.runtime_shutdown()
        {{module_name}}.runtime_shutdown()
