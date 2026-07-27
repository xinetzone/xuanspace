"""Pytest configuration and shared fixtures for demo_ffi tests."""
from __future__ import annotations

import os
import sys
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PYTHON_ROOT = os.path.join(_PROJECT_ROOT, "python")

if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)


@pytest.fixture
def cmd_handle():
    """Provide a valid command handle."""
    from demo_ffi import demo
    return demo.tls_command_handle()


@pytest.fixture
def buffer_1k():
    """Provide a pre-allocated 1KB buffer, auto-freed after test."""
    from demo_ffi import demo
    ptr = demo.buffer_alloc(1024)
    assert ptr != 0
    yield ptr
    demo.buffer_free(ptr)
