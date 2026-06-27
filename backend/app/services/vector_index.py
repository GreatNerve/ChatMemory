"""Vector index — Chroma (LangChain) by default; JSON file store fallback when Chroma unavailable."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger("chatmemory.vector")

from app.core.config import get_settings
from app.core.paths import workspace_path
from app.services.embed import _cosine_similarity, normalize_vector
from app.services.parser.whatsapp import Message, non_system_messages


@lru_cache
def chroma_available() -> bool:
    """True when chromadb can load in this process."""
    try:
        import chromadb  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


def using_file_store() -> bool:
    """Whether new workspaces must use the JSON file vector fallback."""
    return active_vector_store_mode() == "file"


def active_vector_store_mode() -> str:
    """Configured / effective vector backend for settings and ingest."""
    settings = get_settings()
    pref = settings.vector_store.lower()
    if pref == "file":
        return "file"
    if pref == "chroma":
        return "file" if not chroma_available() else "chroma"
    # auto: prefer Chroma when importable
    return "file" if not chroma_available() else "chroma"


def preferred_vector_store() -> str:
    """Store type to persist on ingest (chroma unless forced/unavailable)."""
    return active_vector_store_mode()


def resolve_vector_store(workspace_id: str) -> str:
    """Which index holds this workspace's vectors (persisted at ingest)."""
    meta_path = workspace_path(workspace_id) / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            store = meta.get("vectorStore")
            if store in ("file", "chroma"):
                return store
        except (json.JSONDecodeError, OSError):
            pass

    chunks_path = workspace_path(workspace_id) / "vectors" / "chunks.json"
    if chunks_path.exists() and chunks_path.stat().st_size > 10:
        return "file"
    if active_vector_store_mode() == "file":
        return "file"
    return "chroma"


def _uses_file_index(workspace_id: str) -> bool:
    return resolve_vector_store(workspace_id) == "file"


def _load_file_chunks(workspace_id: str) -> list[dict[str, Any]]:
    path = workspace_path(workspace_id) / "vectors" / "chunks.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _save_file_chunks(workspace_id: str, chunks: list[dict[str, Any]]) -> None:
    path = workspace_path(workspace_id) / "vectors" / "chunks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(chunks, ensure_ascii=False, separators=(",", ":"))
    path.write_text(payload, encoding="utf-8")
    logger.info(
        "Saved %d chunks for ws=%s (%d MB)",
        len(chunks),
        workspace_id[:8],
        path.stat().st_size // (1024 * 1024),
    )


def upsert_messages(
    workspace_id: str,
    messages: list[Message],
    embeddings: list[list[float]],
    person_ids_by_sender: dict[str, str],
) -> int:
    usable = non_system_messages(messages)
    if _uses_file_index(workspace_id):
        chunks: list[dict[str, Any]] = []
        for message, emb in zip(usable, embeddings, strict=False):
            chunks.append(
                {
                    "messageId": message.id,
                    "workspaceId": workspace_id,
                    "personId": person_ids_by_sender.get(message.sender, ""),
                    "speaker": message.sender,
                    "timestamp": message.timestamp.isoformat(),
                    "text": message.text,
                    "embedding": normalize_vector(emb),
                }
            )
        _save_file_chunks(workspace_id, chunks)
        return len(usable)

    from app.services import chroma as chroma_service

    return chroma_service.upsert_messages(
        workspace_id, messages, embeddings, person_ids_by_sender
    )


def semantic_search(
    workspace_id: str,
    query_embedding: list[float],
    top_k: int,
    *,
    speaker: str | None = None,
    person_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    if _uses_file_index(workspace_id):
        query = normalize_vector(query_embedding)
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in _load_file_chunks(workspace_id):
            ts = row.get("timestamp", "")
            if person_id and row.get("personId") != person_id:
                continue
            if speaker and row.get("speaker") != speaker:
                continue
            if date_from and ts < date_from:
                continue
            if date_to and ts > date_to:
                continue
            score = _cosine_similarity(query, row.get("embedding", []))
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        items: list[dict[str, Any]] = []
        for score, row in scored[:top_k]:
            items.append(
                {
                    "message_id": row["messageId"],
                    "speaker": row.get("speaker", ""),
                    "timestamp": row.get("timestamp", ""),
                    "snippet": row.get("text", "")[:500],
                    "score": score,
                }
            )
        return items

    from app.services import chroma as chroma_service

    return chroma_service.semantic_search(
        workspace_id,
        query_embedding,
        top_k,
        speaker=speaker,
        person_id=person_id,
        date_from=date_from,
        date_to=date_to,
    )


def messages_for_person(workspace_id: str, person_id: str) -> list[dict[str, Any]]:
    if _uses_file_index(workspace_id):
        items = [
            {
                "message_id": row["messageId"],
                "text": row.get("text", ""),
                "timestamp": row.get("timestamp", ""),
            }
            for row in _load_file_chunks(workspace_id)
            if row.get("personId") == person_id
        ]
        items.sort(key=lambda row: row.get("timestamp", ""))
        return items

    from app.services import chroma as chroma_service

    return chroma_service.messages_for_person(workspace_id, person_id)


def export_bm25_corpus(workspace_id: str, messages: list[Message]) -> str:
    from app.services import chroma as chroma_service

    return chroma_service.export_bm25_corpus(workspace_id, messages)
