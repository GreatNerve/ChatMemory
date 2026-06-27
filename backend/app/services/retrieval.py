"""Fast shared retrieval — person-first with group fallback; no LLM rerank."""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import get_settings
from app.services import bm25 as bm25_service
from app.services import embed as embed_service
from app.services import vector_index as vector_service
from app.services import workspace as workspace_service

logger = logging.getLogger("chatmemory.retrieval")


def _strong_hits(hits: list[dict[str, Any]], min_score: float) -> list[dict[str, Any]]:
    return [h for h in hits if float(h.get("score") or 0) >= min_score]


def _filter_speaker(hits: list[dict[str, Any]], speaker: str) -> list[dict[str, Any]]:
    return [h for h in hits if h.get("speaker") == speaker]


def _merge_hits(*groups: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for group in groups:
        for item in group:
            mid = item.get("message_id") or ""
            if not mid:
                continue
            prev = by_id.get(mid)
            if prev is None or float(item.get("score") or 0) > float(prev.get("score") or 0):
                by_id[mid] = item
    merged = sorted(by_id.values(), key=lambda row: float(row.get("score") or 0), reverse=True)
    return merged[:limit]


def _person_pass_weak(
    person_hits: list[dict[str, Any]],
    *,
    weak_threshold: float,
    min_strong_hits: int,
    strong_score: float,
) -> bool:
    if not person_hits:
        return True
    if max(float(h.get("score") or 0) for h in person_hits) < weak_threshold:
        return True
    return len(_strong_hits(person_hits, strong_score)) < min_strong_hits


def _score_gate(hits: list[dict[str, Any]], min_score: float) -> list[dict[str, Any]]:
    """Drop hits whose similarity score is below the injection threshold.

    Weak hits are misleading: the model treats any injected context as authoritative
    and will extrapolate from loosely-related passages, fabricating plausible-sounding
    but invented facts.  Discarding them entirely is safer than injecting them.
    """
    if min_score <= 0.0:
        return hits
    return [h for h in hits if float(h.get("score") or 0) >= min_score]


def fast_retrieve(
    workspace_id: str,
    query: str,
    person_id: str,
    person_display_name: str,
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Person-scoped semantic + keyword search; widen to full group when weak.

    All returned hits are guaranteed to meet `persona_memory_inject_min_score` so
    callers never receive low-confidence hits that could trigger hallucination.
    """
    settings = get_settings()
    k = top_k or settings.persona_retrieve_top_k
    weak_threshold = settings.persona_retrieve_weak_threshold
    min_strong = settings.persona_retrieve_min_strong_hits
    strong_score = settings.persona_retrieve_strong_score
    inject_min = settings.persona_memory_inject_min_score

    # Normalize to lowercase — embedding and BM25 both work better with consistent casing;
    # ALL-CAPS questions like "VEGAS KAB GAYE THE HAM?" otherwise get poor similarity scores.
    query = query.lower()

    query_vec = embed_service.embed_query(query)

    person_semantic = vector_service.semantic_search(
        workspace_id,
        query_vec,
        k,
        person_id=person_id,
    )

    bm25 = bm25_service.load_index(workspace_id)
    keyword: list[dict[str, Any]] = []
    if bm25:
        keyword = _filter_speaker(bm25.search(query, k), person_display_name)

    if _person_pass_weak(
        person_semantic,
        weak_threshold=weak_threshold,
        min_strong_hits=min_strong,
        strong_score=strong_score,
    ):
        group_semantic = vector_service.semantic_search(workspace_id, query_vec, k)
        group_keyword: list[dict[str, Any]] = []
        if bm25:
            group_keyword = bm25.search(query, k)
        hits = _merge_hits(person_semantic, keyword, group_semantic, group_keyword, limit=k)
        hits = _score_gate(hits, inject_min)
        logger.info(
            "Retrieval widened to group (ws=%s person=%s hits=%d after_gate=%d min_score=%.2f)",
            workspace_id[:8],
            person_id[:8],
            len(hits) + len([]),  # pre-gate count not tracked separately for brevity
            len(hits),
            inject_min,
        )
        return hits

    hits = _merge_hits(person_semantic, keyword, limit=k)
    hits = _score_gate(hits, inject_min)
    logger.info(
        "Retrieval person-only (ws=%s person=%s hits=%d after_gate=%d min_score=%.2f)",
        workspace_id[:8],
        person_id[:8],
        len(hits),
        len(hits),
        inject_min,
    )
    return hits


def expand_to_turn_windows(
    workspace_id: str,
    hits: list[dict[str, Any]],
    *,
    window_before: int = 3,
    window_after: int = 2,
    max_blocks: int = 5,
    target_person: str | None = None,
    min_hit_score: float = 0.0,
) -> list[str]:
    """Expand message hits into short conversation excerpts from export.txt.

    A block is included when at least one of the following is true:
    - The hit's similarity score is >= `min_hit_score` (high-confidence hit).
    - `target_person` is set and the expanded window contains a message from or
      explicitly mentioning the person being asked about (extra guard against
      off-topic blocks from group-widened retrieval).

    When neither condition holds the block is silently dropped so weakly-related
    context never reaches the system prompt.
    """
    if not hits:
        return []

    try:
        timeline = workspace_service.load_export_timeline(workspace_id)
    except FileNotFoundError:
        # No export file — use hit snippets directly, applying person filter if possible.
        blocks: list[str] = []
        for h in hits[:max_blocks]:
            score = float(h.get("score") or 0)
            snippet = f"{h.get('speaker', '')} ({h.get('timestamp', '')}): {h.get('snippet', '')}"
            if score >= min_hit_score:
                blocks.append(snippet)
            elif target_person and target_person.lower() in snippet.lower():
                blocks.append(snippet)
        return blocks

    id_to_idx = {msg.id: idx for idx, msg in enumerate(timeline)}
    result: list[str] = []
    used_ranges: list[tuple[int, int]] = []

    for hit in hits:
        if len(result) >= max_blocks:
            break

        hit_score = float(hit.get("score") or 0)
        mid = hit.get("message_id") or ""

        if mid not in id_to_idx:
            fallback = f"{hit.get('speaker', '')} ({hit.get('timestamp', '')}): {hit.get('snippet', '')}"
            if fallback in result:
                continue
            # Apply same inclusion logic to fallback snippets.
            if hit_score >= min_hit_score or (
                target_person and target_person.lower() in fallback.lower()
            ):
                result.append(fallback)
            continue

        center = id_to_idx[mid]
        start = max(0, center - window_before)
        end = min(len(timeline), center + window_after + 1)

        overlaps = any(not (end <= s or start >= e) for s, e in used_ranges)
        if overlaps:
            continue

        window_msgs = timeline[start:end]
        lines = [f"{msg.sender}: {msg.text}" for msg in window_msgs]
        block = "\n".join(lines).strip()
        if not block:
            continue

        # Decide whether to include this block.
        include = hit_score >= min_hit_score
        if not include and target_person:
            person_lower = target_person.lower()
            include = any(
                msg.sender.lower() == person_lower or person_lower in msg.text.lower()
                for msg in window_msgs
            )

        if include:
            used_ranges.append((start, end))
            result.append(block)
        else:
            logger.debug(
                "Dropped window block (score=%.3f < %.3f, person '%s' not in window)",
                hit_score,
                min_hit_score,
                target_person,
            )

    return result
