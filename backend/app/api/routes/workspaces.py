from app.core.paths import workspace_path
from app.core.schemas import WorkspaceAnalytics, WorkspaceDetail, WorkspaceSummary
from app.services import analytics as analytics_service
from app.services import jobs as job_service
from app.services import workspace as workspace_service
import asyncio
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
import logging

logger = logging.getLogger("chatmemory.workspaces")

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.get("", response_model=dict)
def list_workspaces() -> dict:
    items = workspace_service.list_workspaces()
    return {"workspaces": [w.model_dump(by_alias=True) for w in items]}


@router.get("/{workspace_id}", response_model=WorkspaceDetail)
def get_workspace(workspace_id: str) -> WorkspaceDetail:
    try:
        return workspace_service.get_workspace(workspace_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Workspace not found") from exc


@router.get("/{workspace_id}/analytics", response_model=WorkspaceAnalytics)
def get_workspace_analytics(workspace_id: str, refresh: bool = False) -> WorkspaceAnalytics:
    try:
        workspace_service.get_workspace(workspace_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Workspace not found") from exc
    try:
        raw = analytics_service.load_analytics(workspace_id, recompute=refresh)
        return WorkspaceAnalytics.model_validate(
            {
                "computed_at": raw["computedAt"],
                "group": _map_group(raw.get("group", {})),
                "people": [_map_person(p) for p in raw.get("people", [])],
                "pairs": [_map_pair(p) for p in raw.get("pairs", [])],
            }
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Analytics not available") from exc


def _map_group(g: dict) -> dict:
    sp = g.get("strongestPair")
    return {
        **g,
        "busiest_hour": g.get("busiestHour"),
        "busiest_hour_label": g.get("busiestHourLabel"),
        "busiest_day": g.get("busiestDay"),
        "avg_response_seconds": g.get("avgResponseSeconds"),
        "avg_response_label": g.get("avgResponseLabel"),
        "median_messages_per_day": g.get("medianMessagesPerDay", 0),
        "active_hours": g.get("activeHours", []),
        "active_days": g.get("activeDays", []),
        "strongest_pair": _map_pair(sp) if sp else None,
        # New time-series / heatmap fields
        "weekly_series": g.get("weeklySeries", []),
        "top_active_weeks": g.get("topActiveWeeks", []),
        "heatmap": g.get("heatmap", []),
    }


def _map_person(p: dict) -> dict:
    return {
        "person_id": p["personId"],
        "display_name": p["displayName"],
        "message_count": p["messageCount"],
        "share_percent": p.get("sharePercent", 0),
        "avg_message_length": p.get("avgMessageLength", 0),
        "avg_response_seconds": p.get("avgResponseSeconds"),
        "median_response_seconds": p.get("medianResponseSeconds"),
        "avg_response_label": p.get("avgResponseLabel"),
        "replies_given": p.get("repliesGiven", 0),
        "replies_received": p.get("repliesReceived", 0),
        "initiations": p.get("initiations", 0),
        "peak_hour": p.get("peakHour"),
        "peak_hour_label": p.get("peakHourLabel"),
        "active_hours": p.get("activeHours", []),
        "active_days": p.get("activeDays", []),
        "response_time_buckets": p.get("responseTimeBuckets", []),
    }


def _map_pair(p: dict) -> dict:
    return {
        "person_a_id": p["personAId"],
        "person_a_name": p["personAName"],
        "person_b_id": p["personBId"],
        "person_b_name": p["personBName"],
        "exchanges": p.get("exchanges", 0),
        "a_to_b_replies": p.get("aToBReplies", 0),
        "b_to_a_replies": p.get("bToAReplies", 0),
        "avg_response_seconds": p.get("avgResponseSeconds"),
        "avg_response_label": p.get("avgResponseLabel"),
        "connection_score": p.get("connectionScore", 0),
    }


@router.post("", status_code=202)
async def create_workspace(
    name: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Upload a WhatsApp .txt export")

    raw = await file.read()
    logger.info(
        "Upload received: name=%r file=%r size=%d bytes",
        name,
        file.filename,
        len(raw),
    )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    meta = workspace_service.create_workspace_record(name, file.filename)
    workspace_id = meta["id"]
    export_path = workspace_path(workspace_id) / "export.txt"
    export_path.write_text(text, encoding="utf-8")

    job = job_service.create_job("ingest", workspace_id=workspace_id)
    workspace_service.set_ingest_status(workspace_id, "running", job.id)
    job_service.update_job(
        job.id,
        status="running",
        step="queued",
        percent=1,
        message="Upload received — starting ingest",
    )

    async def _run() -> None:
        from app.graphs import ingest as ingest_graph

        await ingest_graph.run_ingest_job(job.id, workspace_id, text)

    asyncio.create_task(_run())
    logger.info(
        "Ingest queued: workspace=%s job=%s (%d chars)",
        workspace_id,
        job.id,
        len(text),
    )

    ws = WorkspaceSummary.model_validate(workspace_service.get_workspace(workspace_id))
    return {
        "workspace": ws.model_dump(by_alias=True),
        "jobId": job.id,
    }


@router.delete("/{workspace_id}", status_code=204)
def delete_workspace(workspace_id: str) -> None:
    try:
        workspace_service.delete_workspace(workspace_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Workspace not found") from exc
