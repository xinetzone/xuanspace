"""Logging configuration utilities for caffe-ffi.

Provides unified setup functions for the three-layer (Python/FFI/C++) logging system.

Usage:
    from caffe_ffi.tools import setup_debug, setup_quiet, setup_memory_trace

    setup_debug()                        # Enable DEBUG logging on all three layers
    setup_debug(log_file="mem.log")      # Also write to file
    setup_memory_trace()                 # Finest-grained TRACE level
    setup_quiet()                        # Restore default WARN level
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from .. import (
    LOG_LEVEL_TRACE,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_WARN,
    set_log_level,
)

_PY_LOGGER_NAME = "caffe_ffi"
_PY_LOGGER = logging.getLogger(_PY_LOGGER_NAME)

_CONFIGURED_HANDLERS: list[logging.Handler] = []


def _clear_handlers() -> None:
    for h in _CONFIGURED_HANDLERS:
        _PY_LOGGER.removeHandler(h)
        h.close()
    _CONFIGURED_HANDLERS.clear()


def _add_handler(handler: logging.Handler, level: int, fmt: str, datefmt: str) -> None:
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    _PY_LOGGER.addHandler(handler)
    _CONFIGURED_HANDLERS.append(handler)


def setup_debug(
    level: int = LOG_LEVEL_DEBUG,
    log_file: Optional[str] = None,
    python_level: int = logging.DEBUG,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt: str = "%H:%M:%S",
) -> None:
    """Enable three-layer debug logging.

    Turns on Python logging module output and C++ native DEBUG/Trace output simultaneously.

    Args:
        level: C++ native log level, default LOG_LEVEL_DEBUG(1).
               Use LOG_LEVEL_TRACE(0) for finest-grained output.
        log_file: Optional file path; logs will also be written to this file.
        python_level: Python logging module level, default logging.DEBUG.
        fmt: Log format string.
        datefmt: Time format string.
    """
    _clear_handlers()

    _PY_LOGGER.setLevel(python_level)
    if not _PY_LOGGER.handlers or not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in _PY_LOGGER.handlers
    ):
        _add_handler(logging.StreamHandler(sys.stdout), python_level, fmt, datefmt)

    if log_file:
        _add_handler(logging.FileHandler(log_file, encoding="utf-8"), python_level, fmt, datefmt)

    set_log_level(level)
    _PY_LOGGER.debug(
        "Debug logging enabled (C++ level=%d, Python level=%d, file=%s)",
        level, python_level, log_file,
    )


def setup_memory_trace(log_file: Optional[str] = None) -> None:
    """Enable finest-grained memory tracing logs (TRACE level).

    Outputs all memory allocation/free/access details for diagnosing leaks,
    dangling pointers, etc.
    """
    setup_debug(level=LOG_LEVEL_TRACE, log_file=log_file)
    _PY_LOGGER.debug("Memory trace mode enabled (TRACE level)")


def setup_quiet() -> None:
    """Disable debug logging and restore default WARN level.

    Both Python and C++ layers return to WARNING level; handlers added by
    this tool are removed.
    """
    _clear_handlers()
    _PY_LOGGER.setLevel(logging.WARNING)
    set_log_level(LOG_LEVEL_WARN)


def setup_file_logging(
    log_file: str,
    level: int = LOG_LEVEL_DEBUG,
    append: bool = False,
) -> None:
    """Enable file-only logging (no console output).

    Suitable for long-running background recording to avoid excessive console output.

    Args:
        log_file: Log file path.
        level: C++ log level, default DEBUG.
        append: True=append, False=overwrite (default overwrite).
    """
    _clear_handlers()
    _PY_LOGGER.setLevel(logging.DEBUG)
    mode = "a" if append else "w"
    fh = logging.FileHandler(log_file, mode=mode, encoding="utf-8")
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    _add_handler(fh, logging.DEBUG, fmt, "%H:%M:%S")
    set_log_level(level)
    _PY_LOGGER.debug("File logging enabled -> %s (append=%s)", log_file, append)
