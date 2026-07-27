"""Buffer class for VTA memory management."""
from typing import Optional
from . import _ffi_api

class Buffer:
    """VTA buffer wrapper with RAII semantics."""
    
    def __init__(self, size: int, data: Optional[int] = None, owns: bool = True):
        """Allocate a new buffer or wrap an existing pointer.
        
        Args:
            size: Buffer size in bytes.
            data: Existing buffer pointer (int), if wrapping.
            owns: Whether this Buffer owns the data (will free on destruction).
        """
        self._size = size
        self._owns = owns
        if data is not None:
            self._data = int(data)
        else:
            self._data = int(_ffi_api.buffer_alloc(int(size)))
    
    def __del__(self):
        if self._owns and self._data:
            try:
                _ffi_api.buffer_free(self._data)
            except Exception:
                pass
            self._data = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()
        return False
    
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
        _ffi_api.write_barrier(int(cmd), self._data, int(elem_bits), int(start), int(extent))
    
    def read_barrier(self, cmd: int, elem_bits: int, start: int, extent: int):
        """Perform read barrier.
        
        Args:
            cmd: VTA command handle.
            elem_bits: Element size in bits.
            start: Start element index.
            extent: Number of elements.
        """
        _ffi_api.read_barrier(int(cmd), self._data, int(elem_bits), int(start), int(extent))
    
    def __len__(self) -> int:
        return self._size
