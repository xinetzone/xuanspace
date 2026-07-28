"""Command handle and context management for VTA."""

from typing import Optional
from . import _ffi_api

def command_handle() -> int:
    """Get thread-local command handle.
    
    Returns:
        Command handle as integer pointer.
    """
    return int(_ffi_api.tls_command_handle())

class CommandContext:
    """Context manager for VTA command execution.
    
    Usage:
        with CommandContext() as cmd:
            # cmd is the command handle (int)
            vta.uop_push(...)
            # automatically synchronizes on exit
    """
    
    def __init__(self, wait_cycles: int = 0):
        """Create command context.
        
        Args:
            wait_cycles: Cycles to wait on synchronize (0 = wait forever).
        """
        self._wait_cycles = wait_cycles
        self._cmd: Optional[int] = None
    
    def __enter__(self) -> int:
        self._cmd = command_handle()
        return self._cmd
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._cmd is not None:
            _ffi_api.synchronize(self._cmd, self._wait_cycles)
        return False
    
    @property
    def handle(self) -> int:
        if self._cmd is None:
            raise RuntimeError("CommandContext not entered")
        return self._cmd
