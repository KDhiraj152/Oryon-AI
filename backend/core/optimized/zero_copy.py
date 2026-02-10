"""
Zero-Copy Memory Utilities for High Performance I/O
====================================================

Provides memory-efficient data transfer mechanisms:
- Memory views for buffer reuse
- Memory-mapped file access
- Ring buffers for streaming
- Direct buffer access for serialization

Key optimizations:
1. Avoid copies via memoryview and buffer protocol
2. Memory-mapped I/O for large files
3. Pre-allocated ring buffers for streaming
4. Direct numpy array views without copies
"""

import io
import logging
import mmap
import os
import struct
import threading
from pathlib import Path
from typing import Any, BinaryIO, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Default buffer sizes (aligned to page size)
PAGE_SIZE = 4096
DEFAULT_BUFFER_SIZE = 64 * 1024  # 64KB
LARGE_BUFFER_SIZE = 1024 * 1024  # 1MB
MAX_MMAP_SIZE = 2 * 1024 * 1024 * 1024  # 2GB limit for mmap


# ============================================================================
# ZERO-COPY BUFFER
# ============================================================================


class ZeroCopyBuffer:
    """
    Buffer that supports zero-copy read/write operations.

    Uses memoryview for slicing without copying.
    Pre-allocated for reuse across operations.
    """

    __slots__ = ("_buffer", "_lock", "_position", "_size", "_view")

    def __init__(self, size: int = DEFAULT_BUFFER_SIZE):
        """
        Initialize buffer with given size.

        Args:
            size: Buffer size in bytes (rounded up to page size)
        """
        # Align to page size for efficient I/O
        self._size = ((size + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
        self._buffer = bytearray(self._size)
        self._view = memoryview(self._buffer)
        self._position = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._size

    @property
    def position(self) -> int:
        return self._position

    @property
    def remaining(self) -> int:
        return self._size - self._position

    def get_write_view(self, size: int) -> memoryview:
        """
        Get a writable view of the buffer.

        Returns memoryview for zero-copy write access.
        Raises BufferError if not enough space.
        """
        if size > self.remaining:
            raise BufferError(f"Buffer overflow: need {size}, have {self.remaining}")

        view = self._view[self._position : self._position + size]
        return view

    def commit_write(self, size: int):
        """Commit a write operation, advancing position."""
        self._position += size

    def write(self, data: bytes) -> int:
        """
        Write data to buffer.

        Returns number of bytes written.
        """
        size = len(data)
        view = self.get_write_view(size)
        view[:] = data
        self.commit_write(size)
        return size

    def write_from_numpy(self, arr: np.ndarray) -> int:
        """
        Write numpy array directly to buffer (zero-copy if possible).

        Returns number of bytes written.
        """
        data = arr.tobytes()
        return self.write(data)

    def get_read_view(self) -> memoryview:
        """Get a readable view of written data."""
        return self._view[: self._position]

    def read(self, size: int = -1) -> bytes:
        """
        Read data from buffer.

        Args:
            size: Bytes to read (-1 for all)
        """
        if size < 0:
            size = self._position
        return bytes(self._view[:size])

    def to_numpy(
        self, dtype: np.dtype, shape: tuple[int, ...] | None = None
    ) -> np.ndarray:
        """
        Convert buffer to numpy array (zero-copy view).

        Warning: The array shares memory with buffer.
        Do not use after buffer reset.
        """
        arr = np.frombuffer(self._view[: self._position], dtype=dtype)
        if shape is not None:
            arr = arr.reshape(shape)
        return arr

    def reset(self):
        """Reset buffer for reuse."""
        self._position = 0

    def clear(self):
        """Clear buffer contents and reset."""
        self._buffer[:] = b"\x00" * self._size
        self._position = 0


# ============================================================================
# MEMORY-MAPPED FILE ACCESS
# ============================================================================


class MMapFile:
    """
    Memory-mapped file for efficient large file access.

    Provides numpy-compatible interface for direct memory access
    without loading entire file into RAM.
    """

    def __init__(
        self,
        path: str | Path,
        mode: str = "r",
        size: int | None = None,
    ):
        """
        Open file for memory-mapped access.

        Args:
            path: File path
            mode: 'r' for read-only, 'w' for write, 'rw' for read-write
            size: Required size for new files (mode 'w')
        """
        self.path = Path(path)
        self.mode = mode
        self._file: BinaryIO | None = None
        self._mmap: mmap.mmap | None = None
        self._size = size

        self._open()

    def _open(self):
        """Open file and create memory map."""
        if self.mode == "r":
            self._file = open(self.path, "rb")
            self._mmap = mmap.mmap(
                self._file.fileno(),
                0,
                access=mmap.ACCESS_READ,
            )
            self._size = len(self._mmap)

        elif self.mode == "w":
            if self._size is None:
                raise ValueError("size required for write mode")

            # Create file with specified size
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "wb+")
            self._file.truncate(self._size)
            self._file.flush()

            self._mmap = mmap.mmap(
                self._file.fileno(),
                self._size,
                access=mmap.ACCESS_WRITE,
            )

        else:  # 'rw'
            self._file = open(self.path, "r+b")
            self._mmap = mmap.mmap(
                self._file.fileno(),
                0,
                access=mmap.ACCESS_WRITE,
            )
            self._size = len(self._mmap)

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, key) -> bytes:
        """Direct memory access via slicing."""
        return self._mmap[key]

    def __setitem__(self, key, value):
        """Direct memory write via slicing."""
        self._mmap[key] = value

    def read(self, offset: int, size: int) -> bytes:
        """Read bytes from offset."""
        return self._mmap[offset : offset + size]

    def write(self, offset: int, data: bytes):
        """Write bytes at offset."""
        self._mmap[offset : offset + len(data)] = data

    def as_numpy(
        self,
        dtype: np.dtype,
        offset: int = 0,
        shape: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """
        Create numpy array view of file data (zero-copy).

        Warning: Array shares memory with file.
        Changes to array will modify file if opened in write mode.

        Args:
            dtype: Numpy data type
            offset: Byte offset into file
            shape: Optional shape for reshaping

        Returns:
            Numpy array view of file data
        """
        itemsize = np.dtype(dtype).itemsize

        if shape is not None:
            count = int(np.prod(shape))
        else:
            count = (self._size - offset) // itemsize
            shape = (count,)

        arr = np.frombuffer(
            self._mmap,
            dtype=dtype,
            count=count,
            offset=offset,
        )

        return arr.reshape(shape)

    def flush(self):
        """Flush changes to disk."""
        if self._mmap:
            self._mmap.flush()

    def close(self):
        """Close memory map and file."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ============================================================================
# RING BUFFER FOR STREAMING
# ============================================================================


class RingBuffer:
    """
    Lock-free ring buffer for streaming data.

    Supports single producer, single consumer pattern
    without locks for high performance.
    """

    __slots__ = ("_buffer", "_capacity", "_head", "_size", "_tail")

    def __init__(self, capacity: int):
        """
        Initialize ring buffer.

        Args:
            capacity: Maximum items in buffer
        """
        self._capacity = capacity
        self._buffer = [None] * capacity
        self._head = 0  # Write position
        self._tail = 0  # Read position
        self._size = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_empty(self) -> bool:
        return self._size == 0

    @property
    def is_full(self) -> bool:
        return self._size == self._capacity

    def push(self, item: Any) -> bool:
        """
        Add item to buffer.

        Returns True if successful, False if full.
        """
        if self._size >= self._capacity:
            return False

        self._buffer[self._head] = item
        self._head = (self._head + 1) % self._capacity
        self._size += 1
        return True

    def pop(self) -> Any | None:
        """
        Remove and return oldest item.

        Returns None if empty.
        """
        if self._size == 0:
            return None

        item = self._buffer[self._tail]
        self._buffer[self._tail] = None  # Help GC
        self._tail = (self._tail + 1) % self._capacity
        self._size -= 1
        return item

    def peek(self) -> Any | None:
        """Return oldest item without removing."""
        if self._size == 0:
            return None
        return self._buffer[self._tail]

    def clear(self):
        """Clear all items."""
        self._buffer = [None] * self._capacity
        self._head = 0
        self._tail = 0
        self._size = 0


# ============================================================================
# NUMPY ARRAY BUFFER POOL
# ============================================================================


class NumpyBufferPool:
    """
    Pool of pre-allocated numpy arrays for reuse.

    Reduces allocation overhead by recycling arrays
    of common sizes.
    """

    def __init__(
        self,
        max_arrays: int = 100,
        max_size_bytes: int = 16 * 1024 * 1024,  # 16MB per array
    ):
        """
        Initialize buffer pool.

        Args:
            max_arrays: Maximum arrays to pool per size class
            max_size_bytes: Maximum size of pooled arrays
        """
        self._max_arrays = max_arrays
        self._max_size = max_size_bytes

        # Pool organized by (dtype, shape) tuples
        self._pools: dict[tuple, list] = {}
        self._lock = threading.Lock()

        # Statistics
        self._hits = 0
        self._misses = 0

    def acquire(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype = np.float32,
    ) -> np.ndarray:
        """
        Get array from pool or allocate new.

        Args:
            shape: Desired array shape
            dtype: Data type

        Returns:
            Numpy array (may contain garbage data)
        """
        key = (dtype, shape)

        with self._lock:
            if self._pools.get(key):
                self._hits += 1
                return self._pools[key].pop()

        self._misses += 1
        return np.empty(shape, dtype=dtype)

    def release(self, arr: np.ndarray):
        """
        Return array to pool for reuse.

        Args:
            arr: Array to release
        """
        # Check size limit
        if arr.nbytes > self._max_size:
            return  # Too large, let GC handle it

        key = (arr.dtype, arr.shape)

        with self._lock:
            if key not in self._pools:
                self._pools[key] = []

            if len(self._pools[key]) < self._max_arrays:
                self._pools[key].append(arr)

    def get_stats(self) -> dict:
        """Get pool statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
            "pool_sizes": {str(k): len(v) for k, v in self._pools.items()},
        }

    def clear(self):
        """Clear all pooled arrays."""
        with self._lock:
            self._pools.clear()


# ============================================================================
# SINGLETON INSTANCES
# ============================================================================

_buffer_pool: NumpyBufferPool | None = None
_buffer_pool_lock = threading.Lock()


def get_buffer_pool() -> NumpyBufferPool:
    """Get global buffer pool singleton."""
    global _buffer_pool

    if _buffer_pool is None:
        with _buffer_pool_lock:
            if _buffer_pool is None:
                _buffer_pool = NumpyBufferPool()

    return _buffer_pool


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def numpy_to_bytes_zerocopy(arr: np.ndarray) -> bytes:
    """
    Convert numpy array to bytes with minimal copies.

    Uses array's buffer interface directly.
    """
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr.tobytes()


def bytes_to_numpy_zerocopy(
    data: bytes,
    dtype: np.dtype,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """
    Convert bytes to numpy array without copying.

    Warning: Array shares memory with bytes object.
    Modifying array may affect original bytes.

    Args:
        data: Bytes data
        dtype: Target data type
        shape: Optional shape

    Returns:
        Numpy array view of data
    """
    arr = np.frombuffer(data, dtype=dtype)
    if shape is not None:
        arr = arr.reshape(shape)
    return arr


def streaming_numpy_save(
    arr: np.ndarray,
    path: str | Path,
) -> None:
    """
    Save large numpy array efficiently using memory mapping.

    For very large arrays, uses mmap instead of full load.
    """
    path = Path(path)

    if arr.nbytes < MAX_MMAP_SIZE:
        # Direct save for smaller arrays
        np.save(path, arr)
    else:
        # Memory-mapped save for large arrays
        header_size = 128  # NPY header size estimate
        total_size = header_size + arr.nbytes

        with MMapFile(path, mode="w", size=total_size) as mf:
            # Write header (simplified)
            magic = b"\x93NUMPY"
            version = b"\x01\x00"
            header = f"{{'descr': '{arr.dtype.str}', 'fortran_order': False, 'shape': {arr.shape}}}"
            header = header.encode("latin1")
            header_len = len(header)
            padding = 64 - ((len(magic) + len(version) + 2 + header_len) % 64)
            header = header + b" " * padding

            mf.write(0, magic + version + struct.pack("<H", len(header)) + header)

            # Write data
            mf.write(len(magic) + len(version) + 2 + len(header), arr.tobytes())


def streaming_numpy_load(
    path: str | Path,
) -> np.ndarray:
    """
    Load numpy array with memory mapping for large files.

    Returns memory-mapped array for files > 100MB.
    """
    path = Path(path)

    if path.stat().st_size < 100 * 1024 * 1024:  # < 100MB
        return np.load(path, allow_pickle=False)
    else:
        return np.load(path, mmap_mode="r", allow_pickle=False)
