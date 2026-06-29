from app.core.config import get_settings
from app.core.gpu_lock import gpu_lock
from app.core.memory import release_ram
from app.services import analytics as analytics_service
from app.services import embed as embed_service
from app.services import jobs as job_service
from app.services import vector_index as vector_service
from app.services import workspace as workspace_service
from app.services.parser.preprocess import preprocess_whatsapp_export
from app.services.parser.whatsapp import non_system_messages, parse_whatsapp_export
import asyncio
import logging
import time
from typing import TypedDict

logger = logging.getLogger("chatmemory.ingest")


class IngestState(TypedDict, total=False):
    job_id: str
    workspace_id: str
    export_text: str
    messages: list
    people_by_sender: dict[str, str]
    error: str | None


def _log_step(job_id: str, workspace_id: str, step: str, detail: str = "") -> None:
    msg = f"job={job_id[:8]} ws={workspace_id[:8]} step={step}"
    if detail:
        msg = f"{msg} — {detail}"
    logger.info(msg)


async def run_ingest_job(job_id: str, workspace_id: str, export_text: str) -> None:
    t0 = time.perf_counter()
    job_service.update_job(job_id, status="running", step="parsing", percent=5)
    workspace_service.set_ingest_status(workspace_id, "running", job_id)
    _log_step(job_id, workspace_id, "parsing", f"{len(export_text):,} chars")

    try:
        settings = get_settings()
        cleaned = await asyncio.to_thread(preprocess_whatsapp_export, export_text)
        parsed = await asyncio.to_thread(parse_whatsapp_export, cleaned)
        messages = parsed.messages
        usable = non_system_messages(messages)

        if len(usable) < settings.min_workspace_messages:
            raise ValueError(
                f"Need at least {settings.min_workspace_messages} messages; got {len(usable)}"
            )

        _log_step(job_id, workspace_id, "extracting_people", f"{len(usable):,} messages")
        job_service.update_job(job_id, step="extracting_people", percent=15)
        people_records = await asyncio.to_thread(
            workspace_service.build_people_from_messages, workspace_id, messages
        )
        sender_to_person = {r["displayName"]: pid for pid, r in people_records.items()}

        _log_step(job_id, workspace_id, "chunking", f"{len(people_records)} speakers")
        job_service.update_job(job_id, step="chunking", percent=25)
        await asyncio.to_thread(vector_service.export_bm25_corpus, workspace_id, messages)

        backend = embed_service.resolve_embed_backend()
        device = embed_service.resolve_embed_device()
        embed_label = f"Embedding via {backend} on {device}"
        needs_gpu_lock = embed_service.embed_uses_gpu()
        batch = settings.embed_batch_size
        texts = [m.text for m in usable]
        total_batches = max((len(texts) + batch - 1) // batch, 1)
        _log_step(
            job_id,
            workspace_id,
            "embedding",
            f"{embed_label} — {len(texts):,} texts in {total_batches} batches",
        )

        async def _embed_and_index() -> None:
            job_service.update_job(job_id, step="embedding", percent=30, message=embed_label)
            embeddings: list[list[float]] = []
            embed_t0 = time.perf_counter()
            for i in range(0, len(texts), batch):
                chunk = texts[i : i + batch]
                batch_num = (i // batch) + 1
                # Run blocking local embed off the event loop
                # so /jobs polling and other API calls stay responsive.
                vecs = await asyncio.to_thread(embed_service.embed_texts, chunk, batch)
                embeddings.extend(vecs)
                pct = 30 + int(65 * batch_num / total_batches)
                msg = f"{embed_label} - batch {batch_num}/{total_batches}"
                job_service.update_job(
                    job_id,
                    step="embedding",
                    percent=min(pct, 95),
                    message=msg,
                )
                if batch_num == 1 or batch_num % 20 == 0 or batch_num == total_batches:
                    elapsed = time.perf_counter() - embed_t0
                    rate = batch_num / max(elapsed, 0.1)
                    eta_s = int((total_batches - batch_num) / max(rate, 0.01))
                    _log_step(
                        job_id,
                        workspace_id,
                        "embedding",
                        f"batch {batch_num}/{total_batches}, {elapsed:.0f}s elapsed, ~{eta_s}s left",
                    )
                    if batch_num % 20 == 0:
                        release_ram()

            _log_step(job_id, workspace_id, "saving_index", f"{len(embeddings):,} vectors")
            job_service.update_job(
                job_id,
                step="saving_index",
                percent=96,
                message="Writing search index to disk...",
            )
            save_t0 = time.perf_counter()
            count = await asyncio.to_thread(
                vector_service.upsert_messages,
                workspace_id,
                usable,
                embeddings,
                sender_to_person,
            )
            store = vector_service.preferred_vector_store()
            await asyncio.to_thread(workspace_service.set_vector_store, workspace_id, store)
            save_elapsed = time.perf_counter() - save_t0
            _log_step(
                job_id,
                workspace_id,
                "saving_index",
                f"wrote {count:,} chunks in {save_elapsed:.1f}s",
            )

        if needs_gpu_lock:
            async with gpu_lock(f"ingest:{job_id}"):
                await _embed_and_index()
        else:
            await _embed_and_index()

        _log_step(job_id, workspace_id, "finalizing")
        job_service.update_job(job_id, step="finalizing", percent=98)
        await asyncio.to_thread(
            workspace_service.finalize_workspace_stats,
            workspace_id,
            messages,
            len(people_records),
        )
        await asyncio.to_thread(analytics_service.save_analytics, workspace_id, messages)
        total = time.perf_counter() - t0
        job_service.update_job(
            job_id,
            status="done",
            step="done",
            percent=100,
            result={"workspaceId": workspace_id},
            message=f"Ingest complete in {total:.0f}s",
        )
        workspace_service.set_ingest_status(workspace_id, "done", job_id)
        _log_step(job_id, workspace_id, "done", f"total {total:.0f}s")
    except Exception as exc:
        logger.exception("Ingest failed job=%s ws=%s", job_id[:8], workspace_id[:8])
        workspace_service.set_ingest_status(workspace_id, "error", job_id)
        job_service.update_job(job_id, status="error", error=str(exc), message=str(exc))
        raise
