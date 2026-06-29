from app.core.memory import release_all_memory
from app.services import gemini as gemini_service
from app.services import jobs as job_service
from app.services import vector_index as vector_service
from app.services import workspace as workspace_service
import asyncio
from datetime import datetime, timezone
import logging

logger = logging.getLogger("chatmemory.persona_train")


async def run_persona_train_job(
    job_id: str,
    workspace_id: str,
    person_id: str,
) -> None:
    job_service.update_job(job_id, status="running", step="validating", percent=5)

    try:
        if not gemini_service.is_configured():
            raise gemini_service.GeminiNotConfiguredError(
                gemini_service.config_status()[1] or "GEMINI_API_KEY is not set in backend/.env"
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

        progress(6, "refreshing_samples", "Refreshing sample messages")
        try:
            workspace_service.refresh_person_samples(workspace_id, person_id)
        except FileNotFoundError:
            pass

        progress(8, "style_profile", "Computing style profile")
        await asyncio.to_thread(
            workspace_service.refresh_person_style_profile,
            workspace_id,
            person_id,
        )

        # All refresh_person_* steps are awaited sequentially (Option A).
        # Each await blocks until the thread pool worker finishes before the
        # next step starts, so they never run concurrently on the same file.
        # The file-level lock in update_person_record (Option B) + atomic write
        # (Option C) defend against concurrent API reads racing the writes.

        # Old 4-call pipeline — generates chatAnalysis, personalityNotes,
        # writingStyleNotes, and activeListeningStyle alongside the v2 fields.

        progress(10, "chat_analysis", "Deep chat analysis (chunked)")
        try:
            await asyncio.to_thread(
                workspace_service.refresh_person_chat_analysis,
                workspace_id,
                person_id,
            )
        except Exception as exc:
            logger.warning("Chat analysis failed (non-fatal): %s", exc)

        progress(25, "personality", "Extracting personality notes")
        try:
            await asyncio.to_thread(
                workspace_service.refresh_person_personality,
                workspace_id,
                person_id,
            )
        except Exception as exc:
            logger.warning("Personality extraction failed (non-fatal): %s", exc)

        progress(35, "writing_style", "Extracting writing style notes")
        try:
            await asyncio.to_thread(
                workspace_service.refresh_person_writing_style,
                workspace_id,
                person_id,
            )
        except Exception as exc:
            logger.warning("Writing style extraction failed (non-fatal): %s", exc)

        progress(45, "listening_style", "Extracting listening style")
        try:
            await asyncio.to_thread(
                workspace_service.refresh_person_listening_style,
                workspace_id,
                person_id,
            )
        except Exception as exc:
            logger.warning("Listening style extraction failed (non-fatal): %s", exc)

        # v2 pipeline — generates relationshipDynamic, emotionalProfile,
        # typingFingerprint, responsePatterns, and voiceSamples.

        progress(55, "relationship_emotional", "Extracting relationship dynamic and emotional profile")
        try:
            await asyncio.to_thread(
                workspace_service.refresh_person_relationship_emotional,
                workspace_id,
                person_id,
            )
        except Exception as exc:
            logger.warning("Relationship/emotional extraction failed (non-fatal): %s", exc)

        progress(65, "typing_fingerprint", "Extracting typing fingerprint")
        try:
            await asyncio.to_thread(
                workspace_service.refresh_person_typing_fingerprint,
                workspace_id,
                person_id,
            )
        except Exception as exc:
            logger.warning("Typing fingerprint extraction failed (non-fatal): %s", exc)

        progress(75, "response_patterns", "Extracting response patterns and topic map")
        try:
            await asyncio.to_thread(
                workspace_service.refresh_person_response_patterns,
                workspace_id,
                person_id,
            )
        except Exception as exc:
            logger.warning("Response patterns extraction failed (non-fatal): %s", exc)

        progress(85, "voice_samples", "Selecting and labelling voice sample exchanges")
        try:
            await workspace_service.refresh_person_voice_samples(workspace_id, person_id)
        except Exception as exc:
            logger.warning("Voice sample selection failed (non-fatal): %s", exc)

        progress(88, "activating", "Activating Gemini persona")
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
        workspace_service.update_person_record(workspace_id, person_id, {"personaStatus": "error"})
        job_service.update_job(job_id, status="error", error=str(exc), message=str(exc))
        raise
    finally:
        await asyncio.to_thread(release_all_memory)
