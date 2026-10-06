"""Return freed native heap pages after a background job completes."""
from functools import lru_cache
import ctypes
import logging
import sys
import time

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _malloc_trim():
    if sys.platform != "linux":
        return None
    try:
        # CDLL releases the GIL while glibc trims its free allocation arenas.
        trim = ctypes.CDLL(None).malloc_trim
    except (OSError, AttributeError):
        return None  # Optional GNU libc extension, unavailable on other allocators.
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return trim


def release_free_memory() -> bool:
    """Run on the job thread after dropping buffers, never in a panel callback.

    Only already-freed whole pages can be returned. Live scene data and async
    inputs remain allocated; no process-wide allocator or GC settings change.
    Unsupported platforms simply retain their normal allocator behavior.
    """
    trim = _malloc_trim()
    if trim is None:
        return False
    started = time.perf_counter()
    released = bool(trim(0))
    if released:
        logger.info("Returned unused allocator pages in %.3fs", time.perf_counter() - started)
    return released
