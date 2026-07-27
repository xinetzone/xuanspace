"""Basic tests for DEMO FFI API bindings."""

from demo_ffi import demo


class TestCommandHandle:
    """Tests for thread-local command handle."""

    def test_tls_command_handle_returns_nonzero(self):
        cmd = demo.tls_command_handle()
        assert isinstance(cmd, int)
        assert cmd != 0

    def test_tls_command_handle_returns_incrementing_ids(self):
        cmd1 = demo.tls_command_handle()
        cmd2 = demo.tls_command_handle()
        assert cmd2 > cmd1


class TestBufferAllocFree:
    """Tests for buffer allocation and deallocation."""

    def test_buffer_alloc_returns_nonzero(self):
        ptr = demo.buffer_alloc(1024)
        assert isinstance(ptr, int)
        assert ptr != 0
        demo.buffer_free(ptr)

    def test_buffer_alloc_small_size(self):
        ptr = demo.buffer_alloc(64)
        assert ptr != 0
        demo.buffer_free(ptr)

    def test_buffer_free_null_does_not_crash(self):
        demo.buffer_free(0)


class TestRuntimeShutdown:
    """Tests for runtime shutdown."""

    def test_runtime_shutdown(self):
        demo.runtime_shutdown()

    def test_runtime_shutdown_idempotent(self):
        demo.runtime_shutdown()
        demo.runtime_shutdown()
