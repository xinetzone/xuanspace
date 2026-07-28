"""Buffer class for VTA memory management."""

from typing import Optional
from . import _ffi_api


class Buffer:
    """VTA buffer wrapper with RAII semantics.

    This class provides automatic memory management for VTA device buffers.
    It supports both owning (allocates and frees memory) and non-owning
    (wraps an existing pointer) modes. Copy is disabled, context manager supported.

    Attributes:
        _size: Buffer size in bytes.
        _owns: Whether this Buffer owns the data (will free on destruction).
        _data: Raw buffer pointer as integer.
    """

    def __init__(self, size: int, data: Optional[int] = None, owns: bool = True):
        """Allocate a new buffer or wrap an existing pointer.

        Args:
            size: Buffer size in bytes.
            data: Existing buffer pointer (int), if wrapping.
            owns: Whether this Buffer owns the data (will free on destruction).
        """
        self._size = int(size)
        self._owns = owns
        if data is not None:
            self._data = int(data)
        else:
            self._data = int(_ffi_api.buffer_alloc(int(size)))

    def __del__(self):
        """Destructor - frees buffer if owns and data is valid.

        Note: We deliberately catch all exceptions here because Python
        destructors must never raise (raising in __del__ terminates the
        interpreter). In normal operation buffer_free should not fail.
        """
        if self._owns and self._data != 0:
            try:
                _ffi_api.buffer_free(self._data)
            except Exception:
                pass
            self._data = 0

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager - free resources if owns."""
        if self._owns and self._data != 0:
            try:
                _ffi_api.buffer_free(self._data)
            except Exception:
                pass
            self._data = 0
        return False

    def __repr__(self) -> str:
        """Return string representation of Buffer."""
        return (
            f"Buffer(data=0x{self._data:x}, size={self._size}, "
            f"owns={self._owns})"
        )

    @classmethod
    def from_foreign_pointer(cls, data: int, size: int) -> 'Buffer':
        """Wrap an existing foreign pointer without taking ownership.

        This is a convenience method for wrapping external pointers that
        will be managed externally. The Buffer will not free the data on
        destruction.

        Args:
            data: Existing buffer pointer as integer.
            size: Buffer size in bytes.

        Returns:
            A non-owning Buffer instance wrapping the given pointer.
        """
        return cls(size=int(size), data=int(data), owns=False)

    def reset(self) -> None:
        """Explicitly release/free the buffer.

        If the buffer owns data, calls buffer_free and resets data to 0.
        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._owns and self._data != 0:
            try:
                _ffi_api.buffer_free(self._data)
            except Exception:
                pass
            self._data = 0

    @property
    def data(self) -> int:
        """Get the raw buffer pointer as integer."""
        return self._data

    @property
    def size(self) -> int:
        """Get buffer size in bytes."""
        return self._size

    @property
    def owns_data(self) -> bool:
        """Whether this buffer owns the underlying data."""
        return self._owns

    def cpu_ptr(self, cmd: int) -> int:
        """Get CPU-accessible pointer for this buffer.

        Args:
            cmd: VTA command handle.

        Returns:
            CPU-accessible pointer as integer.
        """
        return int(_ffi_api.buffer_cpu_ptr(int(cmd), self._data))

    def write_barrier(self, cmd: int, elem_bits: int, start: int, extent: int):
        """Perform write barrier.

        Args:
            cmd: VTA command handle.
            elem_bits: Element size in bits.
            start: Start element index.
            extent: Number of elements.
        """
        _ffi_api.write_barrier(
            int(cmd), self._data, int(elem_bits), int(start), int(extent)
        )

    def read_barrier(self, cmd: int, elem_bits: int, start: int, extent: int):
        """Perform read barrier.

        Args:
            cmd: VTA command handle.
            elem_bits: Element size in bits.
            start: Start element index.
            extent: Number of elements.
        """
        _ffi_api.read_barrier(
            int(cmd), self._data, int(elem_bits), int(start), int(extent)
        )

    def __len__(self) -> int:
        """Return buffer size in bytes."""
        return self._size
