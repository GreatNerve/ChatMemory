import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from app.core.memory import release_ram

_lock = asyncio.Lock()
_holder: str | None = None


def gpu_holder() -> str | None:
    return _holder


@asynccontextmanager
async def gpu_lock(owner: str) -> AsyncIterator[None]:
    """Serialize GPU-heavy work (embed batches, etc.)."""
    global _holder
    if _lock.locked():
        raise GpuBusyError(f"GPU busy: {_holder}")

    async with _lock:
        _holder = owner
        try:
            yield
        finally:
            _holder = None
            release_ram()


class GpuBusyError(Exception):
    """Raised when another job holds the GPU mutex."""
