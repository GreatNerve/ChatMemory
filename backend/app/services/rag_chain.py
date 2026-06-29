"""LangChain RAG pipeline — hybrid Chroma + BM25 retrieval with LLM rerank/answer."""

from __future__ import annotations

from app.core.config import get_settings
from app.core.schemas import AskResponse, Citation
from app.prompts.qa import qa_grounded_answer, qa_rerank_chunks, qa_rewrite_query
from app.services import bm25 as bm25_service
from app.services import embed as embed_service
from app.services import gemini as gemini_service
from app.services import vector_index as vector_service
from app.services.langchain_llm import get_chat_model, uses_gemini
from app.services.rerank_utils import parse_rerank_scores
from app.services.retrieval import expand_hits_with_context
from langchain_core.messages import HumanMessage, SystemMessage
import logging
from typing import Any

logger = logging.getLogger("chatmemory.rag")


def _llm_text(messages: list[SystemMessage | HumanMessage], *, temperature: float) -> str:
    llm = get_chat_model(temperature=temperature)
    result = llm.invoke(messages)
    content = result.content
    return content if isinstance(content, str) else str(content)


def rewrite_query(question: str) -> list[str]:
    """Return 1-3 search queries covering different phrasings and language variants.

    The prompt asks the LLM for one query per line.  We split on newlines,
    strip empty lines, and cap at 3 to prevent runaway output.  Falls back to
    the original question when the LLM returns nothing usable.
    """
    system_text, user_text = qa_rewrite_query(question)
    raw = _llm_text(
        [SystemMessage(content=system_text), HumanMessage(content=user_text)],
        temperature=0.2,
    )
    queries = [q.strip() for q in raw.splitlines() if q.strip()]
    return queries[:3] if queries else [question]


def rerank_chunks(question: str, chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if not chunks:
        return []
    numbered = []
    for i, c in enumerate(chunks, start=1):
        numbered.append(f"[{i}] ({c['speaker']}, {c['timestamp']}): {c['snippet'][:300]}")
    prompt = qa_rerank_chunks(question, "\n".join(numbered))
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


def _format_context_blocks(blocks: list[list[dict[str, Any]]]) -> str:
    """Format structured context blocks into a readable string for the LLM.

    The directly-matched message is prefixed with '>>>' so Gemini can see which
    message triggered the retrieval, while surrounding context is indented normally.

    Example output:
        [Context around match]
        Alice (2024-03-15 21:30): oblivion kab release hua?
        >>> Dheeraj (2024-03-15 21:31): March 28 hai bhai
        Manas (2024-03-15 21:32): pre order kr le abhi
    """
    parts: list[str] = []
    for block in blocks:
        lines: list[str] = []
        for msg in block:
            prefix = ">>> " if msg.get("is_hit") else "    "
            lines.append(f"{prefix}{msg['speaker']} ({msg['timestamp']}): {msg['text']}")
        parts.append("[Context around match]\n" + "\n".join(lines))
    return "\n\n".join(parts)


def grounded_answer(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    context_blocks: list[list[dict[str, Any]]] | None = None,
) -> str:
    """Generate a grounded answer from the chat context.

    When ``context_blocks`` is provided (expanded window context from
    ``expand_hits_with_context``), they are formatted with ``>>>`` hit markers
    so the model can see the conversation around each matched message.
    Falls back to raw snippets when context expansion was unavailable.
    """
    if context_blocks:
        context = _format_context_blocks(context_blocks)
    else:
        context = "\n\n".join(
            f"- {c['speaker']} ({c['timestamp']}): {c['snippet']}" for c in chunks
        )
    system_text, user_text = qa_grounded_answer(question, context)
    return _llm_text(
        [SystemMessage(content=system_text), HumanMessage(content=user_text)],
        temperature=0.3,
    )


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

    # Generate 2-3 phrasings covering Hinglish/English variants for cross-language recall.
    queries = rewrite_query(question)
    is_multi = len(queries) > 1
    logger.info("Q&A multi-query (%d): %s (ws=%s)", len(queries), queries, workspace_id[:8])

    bm25_index = bm25_service.load_index(workspace_id)

    # BM25 keyword expansion: join all unique terms from every query so a single BM25
    # pass sees the full vocabulary across phrasings.
    all_terms = list(dict.fromkeys(term for q in queries for term in q.split()))
    combined_bm25_query = " ".join(all_terms)
    keyword: list[dict[str, Any]] = []
    if bm25_index:
        keyword = bm25_index.search(combined_bm25_query, settings.qa_bm25_top_k)

    # Run one semantic search per query variant; collect deduplicated candidates and
    # track how many queries each message was found in (used for the score boost below).
    all_candidates: dict[str, dict[str, Any]] = {}
    query_hit_count: dict[str, int] = {}

    for query in queries:
        query_vec = embed_service.embed_query(query)
        semantic = vector_service.semantic_search(
            workspace_id,
            query_vec,
            settings.qa_semantic_top_k,
            speaker=speaker,
            date_from=date_from,
            date_to=date_to,
        )
        per_query = bm25_service.hybrid_merge(semantic, keyword)
        for chunk in per_query:
            mid = chunk["message_id"]
            query_hit_count[mid] = query_hit_count.get(mid, 0) + 1
            if mid not in all_candidates or float(chunk.get("score", 0)) > float(
                all_candidates[mid].get("score", 0)
            ):
                all_candidates[mid] = chunk

    # Boost chunks retrieved by 2+ different query phrasings — appearing in multiple
    # semantic passes is a strong cross-language relevance signal.
    candidates: list[dict[str, Any]] = []
    for mid, chunk in all_candidates.items():
        c = dict(chunk)
        count = query_hit_count[mid]
        if count > 1:
            c["score"] = float(c.get("score", 0)) * (1.0 + 0.15 * (count - 1))
        candidates.append(c)
    candidates.sort(key=lambda x: float(x.get("score", 0)), reverse=True)

    ranked = rerank_chunks(queries[0], candidates, settings.qa_rerank_top_k)

    # Lower grade threshold on the multi-query path so cross-language matches that
    # individually score just below 0.6 are not discarded at grading.
    grade_threshold = (
        settings.qa_multi_query_grade_threshold if is_multi else settings.qa_grade_threshold
    )

    passing = [chunk for chunk in ranked if chunk.get("score", 0) >= grade_threshold]
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

    # Expand each passing hit with surrounding messages so Gemini sees the full
    # conversation thread — the answer often lives in the message after the match.
    context_blocks = expand_hits_with_context(
        workspace_id,
        passing,
        window_before=settings.qa_context_window_before,
        window_after=settings.qa_context_window_after,
        max_blocks=len(passing),
    )
    if context_blocks:
        logger.info(
            "Q&A context expansion: %d hits → %d blocks (ws=%s)",
            len(passing),
            len(context_blocks),
            workspace_id[:8],
        )
    else:
        logger.debug(
            "Q&A context expansion unavailable; using raw snippets (ws=%s)", workspace_id[:8]
        )

    # Use the original question (preserves Hinglish tone) for the grounded answer.
    answer = grounded_answer(question, passing, context_blocks=context_blocks or None)
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
