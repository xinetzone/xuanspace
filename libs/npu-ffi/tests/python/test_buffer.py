"""Tests for VTA Buffer wrapper class."""
from __future__ import annotations

import gc
import pytest

from npu_ffi import vta
from npu_ffi.vta import Buffer


class TestBufferCreation:
    """Tests for Buffer creation and basic properties."""

    def test_buffer_create_valid_size(self):
        buf = Buffer(1024)
        assert buf.data != 0
        assert buf.size == 1024
        assert buf.owns_data is True
        assert len(buf) == 1024

    def test_buffer_create_small_size(self):
        buf = Buffer(64)
        assert buf.data != 0
        assert buf.size == 64

    def test_buffer_data_property_returns_int(self):
        buf = Buffer(256)
        assert isinstance(buf.data, int)
        assert buf.data > 0

    def test_buffer_wrap_existing_pointer(self):
        raw_ptr = vta.buffer_alloc(512)
        assert raw_ptr != 0
        buf = Buffer(512, data=raw_ptr, owns=False)
        assert buf.data == raw_ptr
        assert buf.size == 512
        assert buf.owns_data is False
        vta.buffer_free(raw_ptr)

    def test_buffer_wrap_existing_pointer_owns(self):
        raw_ptr = vta.buffer_alloc(512)
        buf = Buffer(512, data=raw_ptr, owns=True)
        assert buf.data == raw_ptr
        assert buf.owns_data is True


class TestBufferRAII:
    """Tests for RAII automatic deallocation."""

    def test_buffer_auto_free_on_del(self):
        buf = Buffer(256)
        ptr = buf.data
        assert ptr != 0
        del buf
        gc.collect()

    def test_buffer_context_manager(self):
        with Buffer(512) as buf:
            assert buf.data != 0
            assert buf.size == 512
            ptr = buf.data
        assert buf.data == 0

    def test_buffer_multiple_create_destroy(self):
        buffers = []
        for i in range(5):
            buf = Buffer(128 * (i + 1))
            assert buf.data != 0
            buffers.append(buf)
        for buf in buffers:
            assert buf.data != 0
        del buffers
        gc.collect()

    def test_buffer_nested_context_managers(self):
        with Buffer(256) as buf1:
            assert buf1.data != 0
            with Buffer(512) as buf2:
                assert buf2.data != 0
                assert buf1.data != buf2.data
            assert buf2.data == 0
            assert buf1.data != 0
        assert buf1.data == 0


class TestBufferCpuPtr:
    """Tests for buffer CPU pointer access."""

    def test_cpu_ptr_returns_same_address(self):
        with vta.CommandContext() as cmd:
            buf = Buffer(1024)
            cpu_ptr = buf.cpu_ptr(cmd)
            assert cpu_ptr == buf.data


class TestBufferBarriers:
    """Tests for buffer read/write barrier methods."""

    def test_write_barrier_method(self):
        with vta.CommandContext() as cmd:
            buf = Buffer(1024)
            buf.write_barrier(cmd, 32, 0, 8)

    def test_read_barrier_method(self):
        with vta.CommandContext() as cmd:
            buf = Buffer(1024)
            buf.read_barrier(cmd, 32, 0, 8)


class TestBufferLifecycle:
    """Tests for buffer lifecycle and edge cases."""

    def test_buffer_double_free_safe(self):
        buf = Buffer(256)
        ptr = buf.data
        buf.__del__()
        buf.__del__()
        gc.collect()

    def test_buffer_non_owning_not_freed(self):
        raw_ptr = vta.buffer_alloc(256)
        buf = Buffer(256, data=raw_ptr, owns=False)
        del buf
        gc.collect()
        vta.buffer_free(raw_ptr)
