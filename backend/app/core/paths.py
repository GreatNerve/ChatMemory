from pathlib import Path

from app.core.config import get_settings


def ensure_data_dirs() -> None:
    settings = get_settings()
    settings.data_root.mkdir(parents=True, exist_ok=True)
    settings.workspaces_dir.mkdir(parents=True, exist_ok=True)
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)

    config_path = settings.config_path
    if not config_path.exists():
        import json

        payload = {
            "dataRoot": str(settings.data_root),
            "embedModel": settings.embed_model,
            "personaMinMessages": settings.lora_min_messages,
            "personaThinMinMessages": settings.lora_thin_min_messages,
        }
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def workspace_path(workspace_id: str) -> Path:
    return get_settings().workspaces_dir / workspace_id


def person_path(workspace_id: str, person_id: str) -> Path:
    return workspace_path(workspace_id) / "people" / f"{person_id}.json"
