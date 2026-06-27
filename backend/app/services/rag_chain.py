"""LangChain RAG pipeline — hybrid Chroma + BM25 retrieval with LLM rerank/answer."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_settings
from app.core.schemas import AskResponse, Citation
from app.services import bm25 as bm25_service
from app.services import embed as embed_service
from app.services import gemini as gemini_service
from app.services import vector_index as vector_service
from app.services.langchain_llm import get_chat_model, uses_gemini
from app.services.rerank_utils import parse_rerank_scores

logger = logging.getLogger("chatmemory.rag")


def _llm_text(messages: list[SystemMessage | HumanMessage], *, temperature: float) -> str:
    llm = get_chat_model(temperature=temperature)
    result = llm.invoke(messages)
    content = result.content
    return content if isinstance(content, str) else str(content)


def rewrite_query(question: str) -> str:
    system = SystemMessage(
        content=(
            "Rewrite the user question for chat search. Keep Hinglish/English mix unchanged. "
            "Do not translate. Output only the rewritten question, one line."
        )
    )
    return _llm_text(
        [system, HumanMessage(content=question)],
        temperature=0.2,
    ).strip()


def rerank_chunks(question: str, chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if not chunks:
        return []
    numbered = []
    for i, c in enumerate(chunks, start=1):
        numbered.append(f"[{i}] ({c['speaker']}, {c['timestamp']}): {c['snippet'][:300]}")
    prompt = (
        "Score each snippet 0.0 to 1.0 for relevance to the question.\n"
        "Reply with ONLY a JSON array of objects. Each object must have numeric fields "
        '"id" (snippet number) and "score" (single number 0.0-1.0).\n'
        'Example: [{"id":1,"score":0.9},{"id":2,"score":0.1}]\n\n'
        f"Question: {question}\n\n" + "\n".join(numbered)
    )
    raw = _llm_text([HumanMessage(content=prompt)], temperature=0)
    score_by_id = parse_rerank_scores(raw, len(chunks))
    if not score_by_id:
        logger.warning("Rerank JSON unusable; falling back to retrieval order")
        return chunks[:top_k]

    ranked = []
    for i, c in enumerate(chunks, start=1):
        c2 = dict(c)
        c2["score"] = score_by_id.get(i, c.get("score", 0))
        ranked.append(c2)
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def grounded_answer(question: str, chunks: list[dict[str, Any]]) -> str:
    context = "\n\n".join(f"- {c['speaker']} ({c['timestamp']}): {c['snippet']}" for c in chunks)
    system = SystemMessage(
        content=(
            "Answer ONLY using the provided chat messages. "
            "Match the language style of the question (Hinglish or English). "
            "If insufficient, say you cannot find it in the chat."
        )
    )
    user = HumanMessage(content=f"Question: {question}\n\nMessages:\n{context}")
    return _llm_text([system, user], temperature=0.3)


def run_qa_pipeline(
    workspace_id: str,
    question: str,
    *,
    speaker: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> AskResponse:
    if not uses_gemini():
        _, err = gemini_service.config_status()
        raise gemini_service.GeminiNotConfiguredError(
            err or "GEMINI_API_KEY is not set in backend/.env"
        )

    settings = get_settings()

    rewritten = rewrite_query(question)
    query_vec = embed_service.embed_query(rewritten)

    semantic = vector_service.semantic_search(
        workspace_id,
        query_vec,
        settings.qa_semantic_top_k,
        speaker=speaker,
        date_from=date_from,
        date_to=date_to,
    )

    bm25 = bm25_service.load_index(workspace_id)
    keyword: list[dict[str, Any]] = []
    if bm25:
        keyword = bm25.search(rewritten, settings.qa_bm25_top_k)

    candidates = bm25_service.hybrid_merge(semantic, keyword)
    ranked = rerank_chunks(rewritten, candidates, settings.qa_rerank_top_k)

    passing = [chunk for chunk in ranked if chunk.get("score", 0) >= settings.qa_grade_threshold]
    if len(passing) < settings.qa_min_passing_chunks:
        near = [
            Citation(
                message_id=chunk["message_id"],
                speaker=chunk["speaker"],
                timestamp=chunk["timestamp"],
                snippet=chunk["snippet"],
                score=chunk.get("score"),
            )
            for chunk in ranked[:3]
        ]
        return AskResponse(
            status="not_found",
            answer=None,
            reason="No sufficiently relevant messages in this chat.",
            near_misses=near,
        )

    answer = grounded_answer(rewritten, passing)
    citations = [
        Citation(
            message_id=chunk["message_id"],
            speaker=chunk["speaker"],
            timestamp=chunk["timestamp"],
            snippet=chunk["snippet"],
            score=chunk.get("score"),
        )
        for chunk in passing
    ]
    return AskResponse(status="answered", answer=answer, citations=citations)


# Alias for import checks / docs
run_rag = run_qa_pipeline
