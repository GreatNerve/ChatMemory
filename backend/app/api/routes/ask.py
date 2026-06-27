from fastapi import APIRouter, HTTPException

from app.core.gpu_lock import GpuBusyError
from app.core.schemas import AskRequest, AskResponse
from app.services import gemini as gemini_service
from app.services import workspace as workspace_service

router = APIRouter(prefix="/workspaces/{workspace_id}", tags=["qa"])


@router.post("/ask", response_model=AskResponse)
async def ask(workspace_id: str, body: AskRequest) -> AskResponse:
    from app.graphs import qa as qa_graph

    try:
        ws = workspace_service.get_workspace(workspace_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Workspace not found") from exc

    if ws.ingest_status != "done":
        raise HTTPException(status_code=400, detail="Workspace ingest not complete")

    try:
        return await qa_graph.run_qa(
            workspace_id,
            body.question,
            speaker=body.speaker,
            date_from=body.date_from,
            date_to=body.date_to,
        )
    except gemini_service.GeminiNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GpuBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
