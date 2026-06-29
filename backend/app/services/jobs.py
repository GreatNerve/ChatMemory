from app.core.config import get_settings
from app.core.paths import ensure_data_dirs
from app.core.schemas import JobSnapshot, JobStatus, JobType
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import uuid


def _job_path(job_id: str) -> Path:
    ensure_data_dirs()
    return get_settings().jobs_dir / f"{job_id}.json"


def _read_raw(job_id: str) -> dict[str, Any] | None:
    path = _job_path(job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_raw(job_id: str, data: dict[str, Any]) -> None:
    path = _job_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def create_job(
    job_type: JobType,
    *,
    workspace_id: str | None = None,
    person_id: str | None = None,
    options: dict[str, Any] | None = None,
) -> JobSnapshot:
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    data = {
        "id": job_id,
        "type": job_type,
        "workspaceId": workspace_id,
        "personId": person_id,
        "status": "queued",
        "step": None,
        "percent": 0,
        "message": None,
        "error": None,
        "result": None,
        "options": options or {},
        "createdAt": now,
        "updatedAt": now,
    }
    _write_raw(job_id, data)
    return get_job(job_id)  # type: ignore[return-value]


def get_job_options(job_id: str) -> dict[str, Any]:
    raw = _read_raw(job_id)
    if raw is None:
        return {}
    opts = raw.get("options")
    return opts if isinstance(opts, dict) else {}


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def estimate_eta_seconds(raw: dict[str, Any], percent: int) -> int | None:
    """Linear ETA from elapsed time and current percent (needs a few % of progress)."""
    if percent <= 2 or percent >= 100:
        return None
    started = raw.get("startedAt") or raw.get("createdAt")
    if not started:
        return None
    elapsed = (datetime.now(timezone.utc) - _parse_ts(started)).total_seconds()
    if elapsed < 8:
        return None
    return max(0, int(elapsed * (100 - percent) / percent))


def get_job(job_id: str) -> JobSnapshot | None:
    raw = _read_raw(job_id)
    if raw is None:
        return None
    return JobSnapshot.model_validate(
        {
            "id": raw["id"],
            "type": raw["type"],
            "workspace_id": raw.get("workspaceId"),
            "person_id": raw.get("personId"),
            "status": raw["status"],
            "step": raw.get("step"),
            "percent": raw.get("percent", 0),
            "message": raw.get("message"),
            "error": raw.get("error"),
            "result": raw.get("result"),
            "eta_seconds": raw.get("etaSeconds"),
        }
    )


def get_job_updated_at(job_id: str) -> datetime | None:
    raw = _read_raw(job_id)
    if raw is None:
        return None
    ts = raw.get("updatedAt")
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def find_latest_train_job(workspace_id: str, person_id: str) -> str | None:
    jobs_dir = get_settings().jobs_dir
    if not jobs_dir.exists():
        return None
    latest_id: str | None = None
    latest_ts = ""
    for jf in jobs_dir.glob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("type") != "persona_train":
            continue
        if data.get("workspaceId") != workspace_id or data.get("personId") != person_id:
            continue
        created = data.get("createdAt", "")
        if created >= latest_ts:
            latest_ts = created
            latest_id = data.get("id")
    return latest_id


def cancel_job(job_id: str, reason: str = "Cancelled") -> None:
    raw = _read_raw(job_id)
    if raw is None:
        return
    if raw.get("status") in ("done", "error"):
        return
    update_job(job_id, status="error", error=reason, message=reason)


def update_job(
    job_id: str,
    *,
    status: JobStatus | None = None,
    step: str | None = None,
    percent: int | None = None,
    message: str | None = None,
    error: str | None = None,
    result: dict[str, Any] | None = None,
    eta_seconds: int | None = None,
) -> JobSnapshot:
    raw = _read_raw(job_id)
    if raw is None:
        raise KeyError(job_id)

    if status is not None:
        raw["status"] = status
        if status == "running" and not raw.get("startedAt"):
            raw["startedAt"] = datetime.now(timezone.utc).isoformat()
    if step is not None:
        raw["step"] = step
    if percent is not None:
        raw["percent"] = percent
    if message is not None:
        raw["message"] = message
    if error is not None:
        raw["error"] = error
    if result is not None:
        raw["result"] = result

    if status == "done":
        raw["etaSeconds"] = 0
    elif eta_seconds is not None:
        raw["etaSeconds"] = max(0, int(eta_seconds))
    elif percent is not None and status != "error":
        # Job % jumps to 40% when LoRA starts; linear ETA would show minutes not hours.
        effective_step = step if step is not None else raw.get("step")
        if effective_step != "training":
            estimated = estimate_eta_seconds(raw, percent)
            if estimated is not None:
                raw["etaSeconds"] = estimated

    raw["updatedAt"] = datetime.now(timezone.utc).isoformat()
    _write_raw(job_id, raw)
    return get_job(job_id)  # type: ignore[return-value]
