import logging
import math
import os
from functools import lru_cache

from app.core.config import get_settings
from app.core.memory import prepare_vram_for_large_model, release_cuda_memory, release_ram

logger = logging.getLogger("chatmemory.embed")

_local_model = None


def ml_stack_available() -> tuple[bool, str | None]:
    """Check whether numpy/torch can load (Windows App Control may block DLLs)."""
    return _ml_stack_available_cached()


def cuda_available() -> tuple[bool, str | None]:
    """True when PyTorch sees an NVIDIA GPU."""
    ml_ok, ml_err = ml_stack_available()
    if not ml_ok:
        return False, ml_err
    try:
        import torch

        if torch.cuda.is_available():
            return True, None
        ver = getattr(torch, "__version__", "unknown")
        return False, f"PyTorch is CPU-only ({ver}); CUDA optional for faster ingest."
    except OSError as exc:
        return False, str(exc)


@lru_cache(maxsize=1)
def _ml_stack_available_cached() -> tuple[bool, str | None]:
    try:
        import numpy  # noqa: F401
        import torch  # noqa: F401

        return True, None
    except ImportError as exc:
        return False, str(exc)
    except OSError as exc:
        return False, str(exc)


def resolve_embed_backend() -> str:
    """Local sentence-transformers on CUDA or CPU."""
    return "local"


def active_embed_backend() -> str:
    return resolve_embed_backend()


def resolve_embed_device() -> str:
    """Resolve torch device from EMBED_DEVICE (auto | cuda | cpu)."""
    settings = get_settings()
    preference = (settings.embed_device or "auto").strip().lower()
    if preference not in ("auto", "cuda", "cpu"):
        preference = "auto"

    if preference == "cpu":
        return "cpu"

    cuda_ok, cuda_err = cuda_available()
    if preference == "cuda":
        if not cuda_ok:
            hint = (
                " Install CUDA PyTorch: "
                "uv pip install torch --index-url https://download.pytorch.org/whl/cu124"
            )
            raise RuntimeError((cuda_err or "EMBED_DEVICE=cuda but no CUDA GPU available.") + hint)
        import torch

        return f"cuda:{torch.cuda.current_device()}"

    if cuda_ok:
        import torch

        return f"cuda:{torch.cuda.current_device()}"
    return "cpu"


def embed_uses_gpu() -> bool:
    return resolve_embed_device().startswith("cuda")


def _uses_e5_prefixes(model_name: str) -> bool:
    """E5-family models require query:/passage: prefixes for best retrieval."""
    lower = model_name.lower()
    return "e5" in lower or "multilingual-e5" in lower


def _prefix_for_e5(texts: list[str], *, is_query: bool) -> list[str]:
    settings = get_settings()
    if not _uses_e5_prefixes(settings.embed_model):
        return texts
    prefix = "query: " if is_query else "passage: "
    return [f"{prefix}{t}" for t in texts]


def embed_texts(
    texts: list[str],
    batch_size: int | None = None,
    *,
    is_query: bool = False,
) -> list[list[float]]:
    if not texts:
        return []
    ok, err = ml_stack_available()
    if not ok:
        raise RuntimeError(
            err
            or "ML stack unavailable. On Windows, disable Smart App Control or run scripts/fix-windows-ml.ps1"
        )
    prepared = _prefix_for_e5(texts, is_query=is_query)
    return _embed_texts_local(prepared, batch_size)


def embed_query(text: str) -> list[float]:
    return embed_texts([text], is_query=True)[0]


def is_embed_model_loaded() -> bool:
    return _local_model is not None


def embed_ready() -> bool:
    return is_embed_model_loaded()


def _ensure_local_model() -> None:
    """Load SentenceTransformer once; reused by warmup and embed requests."""
    global _local_model

    if _local_model is not None:
        return

    import torch
    from sentence_transformers import SentenceTransformer

    settings = get_settings()
    device = resolve_embed_device()
    logger.info("Loading SentenceTransformer model from %s.", settings.embed_model)
    if device.startswith("cuda"):
        prepare_vram_for_large_model()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("Loading embed model on %s (%s)", device, gpu_name)
    else:
        logger.info("Loading embed model on cpu")
    _local_model = SentenceTransformer(settings.embed_model, device=device)


def warmup_embed_model() -> bool:
    """Preload embed model at API startup; returns False on failure (degraded mode)."""
    if os.environ.get("CHATMEMORY_SKIP_EMBED_WARMUP") == "1":
        logger.debug("Skipping embed warmup (CHATMEMORY_SKIP_EMBED_WARMUP=1)")
        return False

    if is_embed_model_loaded():
        return True

    settings = get_settings()
    logger.info("Warming up embed model...")
    try:
        ok, err = ml_stack_available()
        if not ok:
            logger.warning("Embed warmup skipped — ML stack unavailable: %s", err)
            return False

        _ensure_local_model()
        embed_query("warmup")
        device = resolve_embed_device()
        logger.info(
            "Embed model ready on %s (%s)",
            device,
            settings.embed_model,
        )
        return True
    except Exception as exc:
        logger.warning("Embed warmup failed (degraded mode): %s", exc)
        return False


def _embed_texts_local(texts: list[str], batch_size: int | None) -> list[list[float]]:
    settings = get_settings()
    _ensure_local_model()

    bs = batch_size or settings.embed_batch_size
    vectors = _local_model.encode(
        texts,
        batch_size=bs,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return [v.tolist() for v in vectors]


def unload_embed_model() -> None:
    global _local_model
    if _local_model is not None:
        try:
            _local_model.cpu()
        except Exception:
            pass
        del _local_model
        _local_model = None
    release_cuda_memory()
    release_ram()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def normalize_vector(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec))
    if n == 0:
        return vec
    return [x / n for x in vec]
