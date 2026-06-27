import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging_config import setup_logging
from app.core.paths import ensure_data_dirs
from app.services import embed as embed_service
from app.services import gemini as gemini_service

logger = logging.getLogger("chatmemory.api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging()
    ensure_data_dirs()
    get_settings()

    ml_ok, ml_err = embed_service.ml_stack_available()
    if not ml_ok and ml_err:
        logger.warning("ML stack unavailable: %s", ml_err)
    embed_service.warmup_embed_model()

    gemini_ok, gemini_err = gemini_service.config_status()
    if gemini_ok:
        gemini_service.warmup_gemini_client()
    elif gemini_err:
        logger.warning("%s", gemini_err)

    logger.info("ChatMemory API ready")
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
