from app.core.config import get_settings
from app.core.gpu_lock import gpu_holder
from app.core.paths import ensure_data_dirs
from app.core.schemas import HealthResponse, SettingsResponse, SettingsUpdate, SystemStatusResponse
from app.services import embed as embed_service
from app.services import gemini as gemini_service
from app.services.vector_index import active_vector_store_mode
from fastapi import APIRouter

router = APIRouter(tags=["system"])


@router.get("/system/status", response_model=SystemStatusResponse)
def system_status() -> SystemStatusResponse:
    """Lightweight probe used by the frontend to show an embed-model loading indicator."""
    settings = get_settings()
    try:
        device = embed_service.resolve_embed_device()
    except Exception:
        device = "unknown"
    return SystemStatusResponse(
        embed_ready=embed_service.embed_ready(),
        embed_model=settings.embed_model,
        embed_device=device,
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:

    settings = get_settings()

    writable = False

    try:
        ensure_data_dirs()

        test = settings.data_root / ".write_test"

        test.write_text("ok", encoding="utf-8")

        test.unlink(missing_ok=True)

        writable = True

    except OSError:
        writable = False

    ml_ok, ml_err = embed_service.ml_stack_available()

    gemini_ok, _ = gemini_service.config_status()

    status = "ok" if writable and ml_ok and gemini_ok else "degraded"

    return HealthResponse(
        status=status,
        data_root_writable=writable,
        ml_stack_available=ml_ok,
        ml_stack_error=ml_err,
        gemini_configured=gemini_ok,
        embed_ready=embed_service.embed_ready(),
    )


@router.get("/settings", response_model=SettingsResponse)
def get_settings_route() -> SettingsResponse:

    settings = get_settings()

    holder = gpu_holder()

    cuda_ok, _ = embed_service.cuda_available()

    gemini_ok, _ = gemini_service.config_status()

    return SettingsResponse(
        data_root=str(settings.data_root),
        embed_model=settings.embed_model,
        active_embed_backend=embed_service.active_embed_backend(),
        embed_device=embed_service.resolve_embed_device(),
        vector_store=active_vector_store_mode(),
        gpu_available=cuda_ok,
        gpu_busy=holder is not None,
        active_job_id=holder,
        gemini_configured=gemini_ok,
        gemini_model=settings.gemini_model,
        thinking_show_input=settings.thinking_show_input,
    )


@router.put("/settings", response_model=SettingsResponse)
def update_settings_route(body: SettingsUpdate) -> SettingsResponse:

    # MVP: settings from env only; persist to config.json for display

    import json

    settings = get_settings()

    config_path = settings.config_path

    ensure_data_dirs()

    data = {}

    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))

    if body.data_root is not None:
        data["dataRoot"] = body.data_root

    config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    get_settings.cache_clear()

    return get_settings_route()
