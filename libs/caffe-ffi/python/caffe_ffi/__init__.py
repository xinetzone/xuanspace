from __future__ import annotations

import logging

__version__ = "0.1.0"

from . import _ffi_api
from . import caffe_pb2

from ._core import Blob, Layer, Net
from . import blob
from . import layer
from . import net
from . import io

from .io import (
    read_net,
    read_net_from_prototxt,
    read_net_from_binary,
    net_from_param,
    net_param_from_string,
)

_logger = logging.getLogger("caffe_ffi")
_logger.setLevel(logging.WARNING)


def version() -> str:
    """Get caffe-ffi version string."""
    if _ffi_api.is_available():
        v = _ffi_api.get_global_func("caffe_ffi.Version")
        if v is not None:
            return v()
    return __version__


LOG_LEVEL_TRACE = 0
LOG_LEVEL_DEBUG = 1
LOG_LEVEL_INFO = 2
LOG_LEVEL_WARN = 3
LOG_LEVEL_ERROR = 4


def set_log_level(level: int) -> None:
    """Set C++ native log level (0=TRACE, 1=DEBUG, 2=INFO, 3=WARN, 4=ERROR).

    Note: This controls C++ native logging only. To also control Python logging,
    use enable_debug_logging() or configure the 'caffe_ffi' logger directly.
    """
    if _ffi_api.is_available():
        fn = _ffi_api.get_global_func("caffe_ffi.SetLogLevel")
        if fn is not None:
            fn(level)


def get_log_level() -> int:
    """Get current C++ native log level."""
    if _ffi_api.is_available():
        fn = _ffi_api.get_global_func("caffe_ffi.GetLogLevel")
        if fn is not None:
            return int(fn())
    return LOG_LEVEL_WARN


def total_allocated_bytes() -> int:
    """Get total bytes allocated by caffe-ffi Blob tensors across all instances.

    Returns the global memory usage counter tracked by the C++ layer.
    This counts both data and diff tensors for all live Blob objects.
    Returns 0 if FFI is not available (Python-only mode).
    """
    if _ffi_api.is_available():
        fn = _ffi_api.get_global_func("caffe_ffi.TotalAllocatedBytes")
        if fn is not None:
            return int(fn())
    return 0


def live_blob_count() -> int:
    """Get the number of currently live Blob objects tracked by the C++ layer.

    This is a leak-detection counter: if this is non-zero when you expect
    all Blobs to be destroyed, you have a memory leak. Each Blob constructor
    increments this and each destructor decrements it.
    Returns 0 if FFI is not available (Python-only mode).
    """
    if _ffi_api.is_available():
        fn = _ffi_api.get_global_func("caffe_ffi.LiveBlobCount")
        if fn is not None:
            return int(fn())
    return 0


def memory_info() -> dict:
    """Get a dict with current memory statistics.

    Returns:
        Dict with 'total_allocated_bytes' and 'live_blob_count' keys.
        Values are 0 if FFI is not available.
    """
    return {
        "total_allocated_bytes": total_allocated_bytes(),
        "live_blob_count": live_blob_count(),
    }


def get_backtrace(skip_frames: int = 0, max_frames: int = 32) -> str:
    """Get a C++ stack backtrace string for debugging.

    Captures the current call stack from C++ perspective, with symbol
    resolution and source line info on supported platforms (Windows/MSVC, Linux).

    Args:
        skip_frames: Number of top frames to skip (default: 0).
        max_frames: Maximum number of frames to capture (default: 32).

    Returns:
        Formatted backtrace string, or a message if backtrace is unavailable
        (build without CAFFE_FFI_ENABLE_BACKTRACE).
    """
    if _ffi_api.is_available():
        fn = _ffi_api.get_global_func("caffe_ffi.GetBacktrace")
        if fn is not None:
            return str(fn(skip_frames, max_frames))
    return "(backtrace not available: C++ extension missing or build without CAFFE_FFI_ENABLE_BACKTRACE)"


def enable_debug_logging(level: int = LOG_LEVEL_DEBUG) -> None:
    """Enable debug logging for both Python and C++ layers.

    This is a convenience function that:
    1. Sets the Python 'caffe_ffi' logger to DEBUG level
    2. Adds a StreamHandler if no handler is configured
    3. Sets the C++ native log level to the specified level

    Args:
        level: C++ log level to set (default: LOG_LEVEL_DEBUG=1).
               Use LOG_LEVEL_TRACE=0 for most verbose output.
    """
    _logger.setLevel(logging.DEBUG)
    if not _logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
    set_log_level(level)
    _logger.debug("Debug logging enabled (C++ level=%d)", level)


def disable_debug_logging() -> None:
    """Disable debug logging, restoring WARNING as the default level."""
    _logger.setLevel(logging.WARNING)
    set_log_level(LOG_LEVEL_WARN)


__all__ = [
    "__version__",
    "version",
    "Blob",
    "Layer",
    "Net",
    "caffe_pb2",
    "read_net",
    "read_net_from_prototxt",
    "read_net_from_binary",
    "net_from_param",
    "net_param_from_string",
    "set_log_level",
    "get_log_level",
    "total_allocated_bytes",
    "live_blob_count",
    "memory_info",
    "get_backtrace",
    "enable_debug_logging",
    "disable_debug_logging",
    "LOG_LEVEL_TRACE",
    "LOG_LEVEL_DEBUG",
    "LOG_LEVEL_INFO",
    "LOG_LEVEL_WARN",
    "LOG_LEVEL_ERROR",
    "tools",
]

from . import tools  # noqa: E402
