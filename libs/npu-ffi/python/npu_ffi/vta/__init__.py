"""npu_ffi.vta - VTA runtime Python API."""
from __future__ import annotations
import enum
from typing import Any

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
    "Buffer", "CommandContext", "command_handle",
    "VTAConfig", "get_default_config", "validate_config", "DEFAULT_CONFIGS",
    "proto_io",
    "buffer_alloc", "buffer_free", "buffer_copy", "buffer_cpu_ptr",
    "tls_command_handle", "runtime_shutdown", "set_debug_mode",
    "load_buffer_2d", "store_buffer_2d",
    "uop_push", "uop_loop_begin", "uop_loop_end",
    "push_gemm_op", "push_alu_op",
    "dep_push", "dep_pop",
    "synchronize", "write_barrier", "read_barrier",
    "prepare_call_func",
]
