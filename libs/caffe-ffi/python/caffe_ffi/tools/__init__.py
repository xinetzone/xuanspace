"""caffe-ffi debugging and memory tracking tools.

This subpackage provides:
- debug: Logging configuration (setup_debug/setup_quiet/setup_file_logging)
- memory: Memory lifecycle tracking (BlobRef, tracked_blob, mem_check, blob_snapshot)
"""

from .debug import (
    setup_debug,
    setup_file_logging,
    setup_memory_trace,
    setup_quiet,
)
from .memory import (
    BlobRef,
    MemoryTrace,
    blob_snapshot,
    mem_check,
    tracked_blob,
)

__all__ = [
    "setup_debug",
    "setup_quiet",
    "setup_memory_trace",
    "setup_file_logging",
    "BlobRef",
    "tracked_blob",
    "blob_snapshot",
    "mem_check",
    "MemoryTrace",
]
