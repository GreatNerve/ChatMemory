import asyncio
import logging

from datetime import datetime, timezone

logger = logging.getLogger("chatmemory.persona_train")

from app.core.memory import release_all_memory
from app.services import gemini as gemini_service
from app.services import jobs as job_service
from app.services import vector_index as vector_service
from app.services import workspace as workspace_service


async def run_persona_train_job(
    job_id: str,
    workspace_id: str,
    person_id: str,
) -> None:
    job_service.update_job(job_id, status="running", step="validating", percent=5)

    try:
        if not gemini_service.is_configured():
            raise gemini_service.GeminiNotConfiguredError(
                gemini_service.config_status()[1]
                or "GEMINI_API_KEY is not set in backend/.env"
            )

        workspace_service.get_person(workspace_id, person_id)
        messages = vector_service.messages_for_person(workspace_id, person_id)
        if not messages:
            store = vector_service.resolve_vector_store(workspace_id)
            raise ValueError(
                f"No indexed messages for this person (vector store: {store}). "
                "Re-upload the workspace export to rebuild the search index."
            )

        def progress(
            percent: int,
            step: str,
            message: str,
            eta_seconds: int | None = None,
        ) -> None:
            job_service.update_job(
                job_id,
                step=step,
                percent=percent,
                message=message,
                status="running",
                eta_seconds=eta_seconds,
            )

        progress(15, "refreshing_samples", "Refreshing sample messages")
        try:
            workspace_service.refresh_person_samples(workspace_id, person_id)
        except FileNotFoundError:
            pass

        progress(40, "style_profile", "Computing style profile")
        await asyncio.to_thread(
            workspace_service.refresh_person_style_profile,
            workspace_id,
            person_id,
        )

        progress(55, "chat_analysis", "Analysing chat patterns")
        try:
            await asyncio.to_thread(
                workspace_service.refresh_person_chat_analysis,
                workspace_id,
                person_id,
            )
        except Exception as exc:
            # Non-fatal: persona still works without deep chat analysis.
            logger.warning("Chat analysis extraction failed (non-fatal): %s", exc)

        progress(65, "personality", "Extracting personality profile")
        try:
            await asyncio.to_thread(
                workspace_service.refresh_person_personality,
                workspace_id,
                person_id,
            )
        except Exception as exc:
            # Non-fatal: persona still works without personality notes.
            logger.warning("Personality extraction failed (non-fatal): %s", exc)

        progress(75, "writing_style", "Extracting writing style")
        try:
            await asyncio.to_thread(
                workspace_service.refresh_person_writing_style,
                workspace_id,
                person_id,
            )
        except Exception as exc:
            # Non-fatal: persona still works without writing style notes.
            logger.warning("Writing style extraction failed (non-fatal): %s", exc)

        progress(85, "activating", "Activating Gemini persona")
        model_name = gemini_service.GEMINI_MODEL_TAG

        workspace_service.update_person_record(
            workspace_id,
            person_id,
            {
                "personaStatus": "ready_model",
                "ollamaModelName": model_name,
                "lastTrainJobId": job_id,
                "lastTrainAt": datetime.now(timezone.utc).isoformat(),
            },
        )
        job_service.update_job(
            job_id,
            status="done",
            step="done",
            percent=100,
            message="Persona ready",
            result={"ollamaModelName": model_name, "provider": "gemini"},
        )
    except Exception as exc:
        workspace_service.update_person_record(
            workspace_id, person_id, {"personaStatus": "error"}
        )
        job_service.update_job(job_id, status="error", error=str(exc), message=str(exc))
        raise
    finally:
        await asyncio.to_thread(release_all_memory)
