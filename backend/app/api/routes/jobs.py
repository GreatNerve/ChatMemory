import asyncio
import json
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.core.schemas import JobSnapshot
from app.services import jobs as job_service

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobSnapshot)
def get_job(job_id: str) -> JobSnapshot:
    job = job_service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


async def _job_event_stream(job_id: str) -> AsyncIterator[dict]:
    last_payload = ""
    while True:
        job = job_service.get_job(job_id)
        if job is None:
            yield {"event": "error", "data": json.dumps({"message": "Job not found"})}
            break

        payload = job.model_dump_json(by_alias=True)
        if payload != last_payload:
            event = (
                "done" if job.status == "done" else "error" if job.status == "error" else "progress"
            )
            yield {"event": event, "data": payload}
            last_payload = payload

        if job.status in ("done", "error"):
            break
        await asyncio.sleep(0.5)


@router.get("/{job_id}/stream")
async def stream_job(job_id: str) -> EventSourceResponse:
    if job_service.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return EventSourceResponse(_job_event_stream(job_id))
