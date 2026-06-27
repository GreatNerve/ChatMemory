"""RAM and GPU memory helpers — free VRAM before large model loads."""



from __future__ import annotations



import gc

import logging

from typing import Any



logger = logging.getLogger("chatmemory.memory")





def release_ram() -> None:

    gc.collect()





def release_cuda_memory(*, sync: bool = True) -> None:

    try:

        import torch



        if not torch.cuda.is_available():

            return

        if sync:

            torch.cuda.synchronize()

        torch.cuda.empty_cache()

        if hasattr(torch.cuda, "ipc_collect"):

            torch.cuda.ipc_collect()

    except (ImportError, OSError):

        pass

    release_ram()





def release_all_memory(*, sync_cuda: bool = True) -> None:

    release_cuda_memory(sync=sync_cuda)





def cuda_memory_stats() -> dict[str, Any] | None:

    try:

        import torch



        if not torch.cuda.is_available():

            return None

        free, total = torch.cuda.mem_get_info()

        return {

            "device": torch.cuda.get_device_name(0),

            "allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 1),

            "reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 1),

            "free_mb": round(free / (1024**2), 1),

            "total_mb": round(total / (1024**2), 1),

        }

    except (ImportError, OSError, RuntimeError):

        return None





def log_vram(label: str) -> None:

    stats = cuda_memory_stats()

    if stats:

        logger.info(

            "%s — VRAM alloc=%sMB reserved=%sMB free=%sMB / %sMB",

            label,

            stats["allocated_mb"],

            stats["reserved_mb"],

            stats["free_mb"],

            stats["total_mb"],

        )





def prepare_vram_for_large_model() -> None:

    """Unload embed models so the active embed model can fit on GPU."""

    from app.services import embed as embed_service



    embed_service.unload_embed_model()

    release_all_memory(sync_cuda=True)

    log_vram("VRAM after prepare")





def cleanup_training_artifacts(*artifacts: Any) -> None:

    """Drop references and return memory to the pool."""

    for obj in artifacts:

        try:

            del obj

        except Exception:

            pass

    release_all_memory(sync_cuda=True)

