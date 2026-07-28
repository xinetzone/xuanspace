"""npu_ffi.vta - VTA runtime Python API."""

import enum
from typing import Any, Union

from . import _ffi_api
from .buffer import Buffer
from .command import CommandContext, command_handle
from .config import VTAConfig, get_default_config, validate_config, DEFAULT_CONFIGS
from . import proto_io

__version__ = "0.1.0"


class DebugFlag(enum.IntFlag):
    DUMP_INSN = 1 << 1
    DUMP_UOP = 1 << 2
    SKIP_READ_BARRIER = 1 << 3
    SKIP_WRITE_BARRIER = 1 << 4
    FORCE_SERIAL = 1 << 5


class MemcpyKind(enum.IntEnum):
    H2D = 1
    D2H = 2
    D2D = 3


class MemoryType(enum.IntEnum):
    DRAM = 0
    SRAM = 1
    UOP = 2
    INP = 3
    WGT = 4
    ACC = 5
    OUT = 6


class ALUOpcode(enum.IntEnum):
    ADD = 0
    SUB = 1
    MUL = 2
    MIN = 3
    MAX = 4
    SHR = 5
    SHL = 6


def buffer_copy_safe(
    from_buf: Union[int, Buffer],
    from_offset: int,
    to_buf: Union[int, Buffer],
    to_offset: int,
    size: int,
    kind: Union[int, MemcpyKind],
) -> None:
    """Type-safe buffer copy accepting MemcpyKind enum.

    Args:
        from_buf: Source buffer (int pointer or Buffer object).
        from_offset: Offset in source buffer (bytes).
        to_buf: Destination buffer (int pointer or Buffer object).
        to_offset: Offset in destination buffer (bytes).
        size: Number of bytes to copy.
        kind: Copy direction (MemcpyKind enum or int).
    """
    if isinstance(kind, MemcpyKind):
        kind = int(kind)
    from_ptr = int(from_buf.data) if hasattr(from_buf, 'data') else int(from_buf)
    to_ptr = int(to_buf.data) if hasattr(to_buf, 'data') else int(to_buf)
    _ffi_api.buffer_copy(
        from_ptr, int(from_offset), to_ptr, int(to_offset), int(size), int(kind)
    )


def get_command_handle() -> int:
    """Convenience alias for command_handle().

    Returns:
        Thread-local command handle as integer pointer.
    """
    return command_handle()


buffer_alloc = _ffi_api.buffer_alloc
buffer_free = _ffi_api.buffer_free
buffer_copy = _ffi_api.buffer_copy
buffer_cpu_ptr = _ffi_api.buffer_cpu_ptr
tls_command_handle = _ffi_api.tls_command_handle
runtime_shutdown = _ffi_api.runtime_shutdown
set_debug_mode = _ffi_api.set_debug_mode
load_buffer_2d = _ffi_api.load_buffer_2d
store_buffer_2d = _ffi_api.store_buffer_2d
uop_push = _ffi_api.uop_push
uop_loop_begin = _ffi_api.uop_loop_begin
uop_loop_end = _ffi_api.uop_loop_end
push_gemm_op = _ffi_api.push_gemm_op
push_alu_op = _ffi_api.push_alu_op
dep_push = _ffi_api.dep_push
dep_pop = _ffi_api.dep_pop
synchronize = _ffi_api.synchronize
write_barrier = _ffi_api.write_barrier
read_barrier = _ffi_api.read_barrier
prepare_call_func = _ffi_api.prepare_call_func

__all__ = [
    "__version__",
    "DebugFlag", "MemcpyKind", "MemoryType", "ALUOpcode",
    "Buffer", "CommandContext", "command_handle", "get_command_handle",
    "VTAConfig", "get_default_config", "validate_config", "DEFAULT_CONFIGS",
    "proto_io",
    "buffer_alloc", "buffer_free", "buffer_copy", "buffer_copy_safe", "buffer_cpu_ptr",
    "tls_command_handle", "runtime_shutdown", "set_debug_mode",
    "load_buffer_2d", "store_buffer_2d",
    "uop_push", "uop_loop_begin", "uop_loop_end",
    "push_gemm_op", "push_alu_op",
    "dep_push", "dep_pop",
    "synchronize", "write_barrier", "read_barrier",
    "prepare_call_func",
]
