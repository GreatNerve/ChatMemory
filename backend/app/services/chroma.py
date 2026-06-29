"""Chroma vector store via LangChain — persisted under data/workspaces/{id}/chroma."""

from __future__ import annotations

from app.core.paths import workspace_path
from app.services.langchain_embed import get_embeddings
from app.services.parser.whatsapp import Message, non_system_messages
from functools import lru_cache
import json
from langchain_chroma import Chroma
from typing import Any


def _collection_name(workspace_id: str) -> str:
    return f"workspace_{workspace_id.replace('-', '')}"


@lru_cache(maxsize=32)
def _get_store(workspace_id: str) -> Chroma:
    path = workspace_path(workspace_id) / "chroma"
    path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=_collection_name(workspace_id),
        embedding_function=get_embeddings(),
        persist_directory=str(path),
        collection_metadata={"hnsw:space": "cosine"},
    )


def clear_store_cache(workspace_id: str | None = None) -> None:
    """Close the underlying chromadb client and drop the LangChain Chroma handle."""
    if workspace_id is not None:
        try:
            store = _get_store(workspace_id)
            # LangChain Chroma wraps a chromadb PersistentClient.
            # On Windows the SQLite + HNSW files stay locked until the client is
            # garbage-collected.  We reach into the private attribute chain and
            # close the SQLite connection explicitly before evicting the cache.
            client = getattr(store, "_client", None)
            if client is not None:
                # chromadb PersistentClient holds a SqliteDB under _db
                db = getattr(client, "_db", None)
                if db is not None:
                    conn = getattr(db, "_conn", None)
                    if conn is not None:
                        try:
                            conn.close()
                        except Exception:
                            pass
                # Also try the public delete_collection path as a best-effort flush
                try:
                    client.clear_system_cache()
                except Exception:
                    pass
        except Exception:
            pass
    _get_store.cache_clear()


def upsert_messages(
    workspace_id: str,
    messages: list[Message],
    embeddings: list[list[float]],
    person_ids_by_sender: dict[str, str],
) -> int:
    store = _get_store(workspace_id)
    usable = non_system_messages(messages)
    ids = [message.id for message in usable]
    documents = [message.text for message in usable]
    metadatas: list[dict[str, Any]] = []
    for message in usable:
        metadatas.append(
            {
                "messageId": message.id,
                "workspaceId": workspace_id,
                "personId": person_ids_by_sender.get(message.sender, ""),
                "speaker": message.sender,
                "timestamp": message.timestamp.isoformat(),
            }
        )

    batch = 100
    for i in range(0, len(ids), batch):
        store.add_texts(
            texts=documents[i : i + batch],
            metadatas=metadatas[i : i + batch],
            ids=ids[i : i + batch],
            embeddings=embeddings[i : i + batch],
        )
    return len(usable)


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
    store = _get_store(workspace_id)

    where: dict[str, Any] | None = None
    if person_id:
        where = {"personId": person_id}
    elif speaker:
        where = {"speaker": speaker}

    # LangChain Chroma: query by precomputed embedding (ingest uses same vectors).
    pairs = store.similarity_search_by_vector_with_relevance_scores(
        embedding=query_embedding,
        k=top_k,
        filter=where,
    )

    items: list[dict[str, Any]] = []
    for doc, score in pairs:
        meta = doc.metadata or {}
        ts = meta.get("timestamp", "")
        if date_from and ts < date_from:
            continue
        if date_to and ts > date_to:
            continue
        items.append(
            {
                "message_id": meta.get("messageId") or doc.id or "",
                "speaker": meta.get("speaker", ""),
                "timestamp": ts,
                "snippet": (doc.page_content or "")[:500],
                "score": float(score),
            }
        )
    return items


def messages_for_person(workspace_id: str, person_id: str) -> list[dict[str, Any]]:
    store = _get_store(workspace_id)
    result = store._collection.get(  # noqa: SLF001 — metadata filter not exposed on VectorStore
        where={"personId": person_id},
        include=["documents", "metadatas"],
    )
    items: list[dict[str, Any]] = []
    for i, msg_id in enumerate(result.get("ids") or []):
        meta = result["metadatas"][i] if result.get("metadatas") else {}
        items.append(
            {
                "message_id": msg_id,
                "text": result["documents"][i] if result.get("documents") else "",
                "timestamp": meta.get("timestamp", ""),
            }
        )
    items.sort(key=lambda row: row.get("timestamp", ""))
    return items


def export_bm25_corpus(workspace_id: str, messages: list[Message]) -> str:
    usable = non_system_messages(messages)
    corpus = [
        {
            "messageId": message.id,
            "speaker": message.sender,
            "timestamp": message.timestamp.isoformat(),
            "text": message.text,
        }
        for message in usable
    ]
    out = workspace_path(workspace_id) / "bm25" / "corpus.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")
    from app.services import bm25 as bm25_service

    bm25_service.clear_index_cache(workspace_id)
    return str(out)
