from fastapi import APIRouter, HTTPException

from app.core.paths import workspace_path
from app.core.schemas import PersonDetail
from app.services import workspace as workspace_service

router = APIRouter(prefix="/workspaces/{workspace_id}/people", tags=["people"])


@router.get("", response_model=dict)
def list_people(workspace_id: str) -> dict:
    if not (workspace_path(workspace_id) / "meta.json").exists():
        raise HTTPException(status_code=404, detail="Workspace not found")
    people = workspace_service.list_people(workspace_id)
    return {"people": [p.model_dump(by_alias=True) for p in people]}


@router.get("/{person_id}", response_model=PersonDetail)
def get_person(workspace_id: str, person_id: str) -> PersonDetail:
    try:
        return workspace_service.get_person(workspace_id, person_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Person not found") from exc
