import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.core.paths import person_path
from app.core.schemas import (
    PersonaChatRequest,
    PersonaChatResponse,
    PersonaSummarizeRequest,
    PersonaSummarizeResponse,
    TrainRequest,
)
from app.services import gemini as gemini_service
from app.services import jobs as job_service
from app.services import persona_chat as persona_chat_service
from app.services import workspace as workspace_service

router = APIRouter(prefix="/workspaces/{workspace_id}/people/{person_id}", tags=["persona"])


@router.post("/train", status_code=202)
async def train_persona(
    workspace_id: str,
    person_id: str,
    body: TrainRequest,
) -> dict:
    if not body.consent:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Consent required",
                    "fieldErrors": {"consent": "Required"},
                }
            },
        )

    try:
        person = workspace_service.get_person(workspace_id, person_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Person not found") from exc

    gemini_ok, gemini_err = gemini_service.config_status()
    if not gemini_ok:
        raise HTTPException(
            status_code=503,
            detail=gemini_err or "Gemini API is not configured",
        )

    settings = get_settings()
    if person.message_count < settings.lora_thin_min_messages:
        raise HTTPException(status_code=400, detail="Not enough messages to build persona")
    if (
        settings.lora_thin_min_messages <= person.message_count < settings.lora_min_messages
        and not body.force_thin
    ):
        raise HTTPException(
            status_code=400,
            detail="Thin persona — set forceThin true to proceed",
        )
    if person.persona_status == "ready_model" and not body.force_retrain:
        raise HTTPException(
            status_code=400,
            detail="Persona already active — set forceRetrain true to rebuild",
        )
    if workspace_service.training_is_active(workspace_id, person_id):
        raise HTTPException(status_code=400, detail="Persona build already in progress")

    job = job_service.create_job(
        "persona_train",
        workspace_id=workspace_id,
        person_id=person_id,
        options={"forceRetrain": body.force_retrain},
    )
    workspace_service.update_person_record(
        workspace_id,
        person_id,
        {"personaStatus": "training", "lastTrainJobId": job.id},
    )
    job_service.update_job(
        job.id,
        status="running",
        step="queued",
        percent=1,
        message="Persona activation queued",
    )

    async def _run() -> None:
        from app.graphs import persona_train as persona_train_graph

        await persona_train_graph.run_persona_train_job(job.id, workspace_id, person_id)

    asyncio.create_task(_run())
    return {"jobId": job.id, "personaStatus": "training"}


@router.post("/train/cancel", status_code=200)
def cancel_train_persona(workspace_id: str, person_id: str) -> dict:
    if not person_path(workspace_id, person_id).exists():
        raise HTTPException(status_code=404, detail="Person not found")

    workspace_service.cancel_person_training(workspace_id, person_id)
    person = workspace_service.get_person(workspace_id, person_id)
    return {
        "personaStatus": person.persona_status,
        "message": "Build cancelled — you can activate again",
    }


def _chat_history(body: PersonaChatRequest) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for turn in body.history[-10:]:
        history.append({"role": turn.role, "content": turn.content})
    return history


@router.post("/chat", response_model=PersonaChatResponse)
def persona_chat(
    workspace_id: str, person_id: str, body: PersonaChatRequest
) -> PersonaChatResponse:
    try:
        person = workspace_service.get_person(workspace_id, person_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Person not found") from exc

    if person.persona_status != "ready_model" or not person.ollama_model_name:
        raise HTTPException(status_code=400, detail="Persona not active")

    gemini_ok, gemini_err = gemini_service.config_status()
    if not gemini_ok:
        raise HTTPException(status_code=503, detail=gemini_err)

    history = _chat_history(body)
    try:
        reply, interaction_id = persona_chat_service.reply(
            workspace_id,
            person,
            history,
            body.message.strip(),
            previous_interaction_id=body.previous_interaction_id,
            conversation_summary=body.conversation_summary,
        )
    except gemini_service.GeminiNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except gemini_service.GeminiError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return PersonaChatResponse(
        reply=reply,
        model=get_settings().gemini_model,
        interaction_id=interaction_id,
    )


@router.post("/chat/stream")
def persona_chat_stream(
    workspace_id: str, person_id: str, body: PersonaChatRequest
) -> StreamingResponse:
    try:
        person = workspace_service.get_person(workspace_id, person_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Person not found") from exc

    if person.persona_status != "ready_model" or not person.ollama_model_name:
        raise HTTPException(status_code=400, detail="Persona not active")

    gemini_ok, gemini_err = gemini_service.config_status()
    if not gemini_ok:
        raise HTTPException(status_code=503, detail=gemini_err)

    history = _chat_history(body)

    return StreamingResponse(
        persona_chat_service.sse_stream(
            workspace_id,
            person,
            history,
            body.message.strip(),
            previous_interaction_id=body.previous_interaction_id,
            conversation_summary=body.conversation_summary,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/chat/summarize", response_model=PersonaSummarizeResponse)
def persona_chat_summarize(
    workspace_id: str,
    person_id: str,
    body: PersonaSummarizeRequest,
) -> PersonaSummarizeResponse:
    try:
        person = workspace_service.get_person(workspace_id, person_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Person not found") from exc

    if person.persona_status != "ready_model":
        raise HTTPException(status_code=400, detail="Persona not active")

    gemini_ok, gemini_err = gemini_service.config_status()
    if not gemini_ok:
        raise HTTPException(status_code=503, detail=gemini_err)

    history = [{"role": t.role, "content": t.content} for t in body.history]
    if not history:
        raise HTTPException(status_code=400, detail="History is empty")

    try:
        summary, count = persona_chat_service.summarize_conversation(person, history)
    except gemini_service.GeminiNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except gemini_service.GeminiError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return PersonaSummarizeResponse(summary=summary, summarized_turn_count=count)
