from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging_config import setup_logging
from app.core.paths import ensure_data_dirs
from app.services import embed as embed_service
from app.services import gemini as gemini_service
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

logger = logging.getLogger("chatmemory.api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging()
    ensure_data_dirs()
    settings = get_settings()
    if settings.hf_token:
        os.environ.setdefault("HF_TOKEN", settings.hf_token)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", settings.hf_token)

    ml_ok, ml_err = embed_service.ml_stack_available()
    if not ml_ok and ml_err:
        logger.warning("ML stack unavailable: %s", ml_err)

    # Warm up the embed model in a thread pool so the event loop (and HTTP server)
    # are not blocked while the model loads from disk/VRAM (~2-5 s on first boot).
    # The model will be ready shortly after startup; requests that arrive while it
    # is still loading receive a 503 from the route-level embed_ready() guard.
    asyncio.create_task(asyncio.to_thread(embed_service.warmup_embed_model))

    gemini_ok, gemini_err = gemini_service.config_status()
    if gemini_ok:
        gemini_service.warmup_gemini_client()
    elif gemini_err:
        logger.warning("%s", gemini_err)

    logger.info("ChatMemory API ready (embed model warming in background)")
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="ChatMemory API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
