"""Fast shared retrieval — person-first with group fallback; no LLM rerank."""

from __future__ import annotations

from app.core.config import get_settings
from app.services import bm25 as bm25_service
from app.services import embed as embed_service
from app.services import vector_index as vector_service
from app.services import workspace as workspace_service
from app.services.parser.whatsapp import is_noise_message
from datetime import datetime, timedelta, timezone
import logging
from typing import Any

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


def _apply_recency_boost(
    hits: list[dict[str, Any]],
    query_intent: str = "neutral",
) -> list[dict[str, Any]]:
    """Boost (or penalise) hits based on timestamp and the query's temporal direction.

    ``query_intent`` controls the direction of the recency adjustment:

    ``"current"`` (default boost — question is about present state):
      - Last 30 days  → +0.10  (captures the current internship, latest plans, etc.)
      - Last 31-90 days → +0.05  (recent but not the very latest context)
      - Older         → no change

    ``"historical"`` (reversed — question explicitly references the past):
      - Last 30 days  → -0.03  (penalise fresh chunks that are unlikely to be the
                                 referenced past event)
      - Older than 180 days → +0.05  (reward genuinely old chunks)
      - Between 30 and 180 days → no change

    ``"neutral"`` (half boost — ambiguous or general fact questions):
      - Last 30 days  → +0.05
      - Last 31-90 days → +0.02
      - Older         → no change

    Scores are capped at 1.0 and floored at 0.0 after adjustment.  The list is
    re-sorted by score after boosting so callers always receive a highest-first
    ordering.  Hits without a parseable timestamp are left unchanged — parse
    failures fail silently.
    """
    now = datetime.now(timezone.utc)
    cutoff_30 = now - timedelta(days=30)
    cutoff_90 = now - timedelta(days=90)
    cutoff_180 = now - timedelta(days=180)

    boosted: list[dict[str, Any]] = []
    for h in hits:
        ts_str = (h.get("timestamp") or "").strip()
        delta = 0.0
        if ts_str:
            try:
                # Handle Z suffix from some ISO serialisers.
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                if query_intent == "current":
                    if ts >= cutoff_30:
                        delta = 0.10
                    elif ts >= cutoff_90:
                        delta = 0.05

                elif query_intent == "historical":
                    if ts >= cutoff_30:
                        delta = -0.03   # penalise very recent chunks
                    elif ts < cutoff_180:
                        delta = +0.05  # reward genuinely old chunks

                else:  # "neutral"
                    if ts >= cutoff_30:
                        delta = 0.05
                    elif ts >= cutoff_90:
                        delta = 0.02

            except (ValueError, TypeError, AttributeError):
                pass

        if delta != 0.0:
            new_score = min(1.0, max(0.0, float(h.get("score") or 0) + delta))
            h = {**h, "score": new_score}
        boosted.append(h)

    return sorted(boosted, key=lambda h: float(h.get("score") or 0), reverse=True)


def _chunk_density(text: str) -> float:
    """Return an information-density bonus in range [-0.04, +0.06].

    Language-agnostic: uses only structural signals — no hardcoded words.
    Called on the chunk ``text`` (or ``snippet``) of each retrieval hit to
    distinguish fact-dense chunks (dates, amounts, proper nouns, multi-message
    context) from noise (acknowledgements, single-word reactions).

    Signals used:
      +0.03  chunk has >100 chars          — likely a substantive message
      +0.02  chunk has >200 chars          — multi-sentence / detailed content
      +0.02  contains a digit              — dates, roll numbers, amounts, etc.
      +0.02  contains a capitalised word   — proper nouns (EY, NSUT, etc.)
      -0.04  chunk has <30 chars           — very short, likely an ack/reaction
      +0.01  3+ newlines (multi-message)   — richer context block
    """
    words = text.split()
    char_count = len(text.strip())

    has_number = any(c.isdigit() for c in text)
    has_caps_word = any(
        w[0].isupper() and len(w) > 2
        for w in words
        if w.isalpha()
    )
    msg_count = text.count("\n") + 1

    score = 0.0
    if char_count > 100:
        score += 0.03
    if char_count > 200:
        score += 0.02
    if has_number:
        score += 0.02
    if has_caps_word:
        score += 0.02
    if char_count < 30:
        score -= 0.04
    if msg_count >= 3:
        score += 0.01

    return max(-0.04, min(0.06, score))


def _apply_density_score(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add an information-density bonus to each hit's score and re-sort.

    Uses ``text`` when present (structured context blocks), falling back to
    ``snippet`` (raw retrieval hits).  Scores are capped at 1.0.
    """
    result: list[dict[str, Any]] = []
    for h in hits:
        content = h.get("text") or h.get("snippet") or ""
        bonus = _chunk_density(content)
        if bonus != 0.0:
            new_score = min(1.0, max(0.0, float(h.get("score") or 0) + bonus))
            h = {**h, "score": new_score}
        result.append(h)
    return sorted(result, key=lambda h: float(h.get("score") or 0), reverse=True)


def fast_retrieve(
    workspace_id: str,
    queries: list[str],
    person_id: str,
    person_display_name: str,
    *,
    top_k: int | None = None,
    query_intent: str = "neutral",
) -> list[dict[str, Any]]:
    """Person-scoped semantic + keyword search; widen to full group when weak.

    ``queries`` is a list of 1-4 search phrases.  When multiple phrases are
    supplied (cross-language mode — e.g. Hinglish original + English
    equivalents) each phrase is embedded and searched independently; all result
    sets are merged and deduplicated by ``message_id``.  This handles the case
    where "intern lag gayi?" must match "Will be interning at EY" despite low
    direct cosine similarity.

    ``query_intent`` controls the recency boost direction:
      - ``"current"``    → full boost (+0.10 / +0.05) for recent chunks
      - ``"historical"`` → reversed: recent chunks penalised, old chunks boosted
      - ``"neutral"``    → half boost (+0.05 / +0.02) — default

    Score gating:
    - Single-query path  → ``persona_memory_inject_min_score`` (default 0.35)
    - Multi-query path   → ``persona_memory_inject_min_score_cross_lang`` (default 0.22)
      Hits that appear across 2+ query result sets get an additional ×0.65
      discount (reward for multi-signal corroboration), floored at 0.10.

    All returned hits meet the applicable threshold so callers never receive
    low-confidence hits that could trigger hallucination.
    """
    settings = get_settings()
    k = top_k or settings.persona_retrieve_top_k
    weak_threshold = settings.persona_retrieve_weak_threshold
    min_strong = settings.persona_retrieve_min_strong_hits
    strong_score = settings.persona_retrieve_strong_score
    inject_min = settings.persona_memory_inject_min_score
    cross_lang_min = settings.persona_memory_inject_min_score_cross_lang

    # Normalize to lowercase — embedding and BM25 both work better with consistent
    # casing; ALL-CAPS questions otherwise get poor similarity scores.
    normalized: list[str] = [q.lower().strip() for q in queries if q.strip()]
    if not normalized:
        return []

    is_cross_lang = len(normalized) > 1

    bm25 = bm25_service.load_index(workspace_id)

    # --- Per-query retrieval -------------------------------------------------
    # We embed each query independently so each phrasing contributes its own
    # similarity signal.  Results are tracked per-query to detect multi-signal
    # hits (same message_id returned for 2+ queries).
    per_query_person_sem: list[list[dict[str, Any]]] = []
    per_query_group_sem: list[list[dict[str, Any]]] = []

    all_keyword: list[dict[str, Any]] = []
    all_group_keyword: list[dict[str, Any]] = []

    # Track how many distinct query result sets each message_id appears in.
    mid_query_count: dict[str, int] = {}

    need_widen = False  # determined after first-pass person searches

    for q in normalized:
        query_vec = embed_service.embed_query(q)
        p_sem = vector_service.semantic_search(workspace_id, query_vec, k, person_id=person_id)
        per_query_person_sem.append(p_sem)

        for h in p_sem:
            mid = h.get("message_id") or ""
            if mid:
                mid_query_count[mid] = mid_query_count.get(mid, 0) + 1

        if bm25:
            kw = _filter_speaker(bm25.search(q, k), person_display_name)
            for h in kw:
                mid = h.get("message_id") or ""
                if mid:
                    mid_query_count[mid] = mid_query_count.get(mid, 0) + 1
            all_keyword.extend(kw)

        # Widen to group if any single-query person result is weak.
        if _person_pass_weak(
            p_sem,
            weak_threshold=weak_threshold,
            min_strong_hits=min_strong,
            strong_score=strong_score,
        ):
            need_widen = True

    # Flatten all person semantic hits
    all_person_sem: list[dict[str, Any]] = [h for group in per_query_person_sem for h in group]

    if need_widen:
        for q in normalized:
            query_vec = embed_service.embed_query(q)
            g_sem = vector_service.semantic_search(workspace_id, query_vec, k)
            per_query_group_sem.append(g_sem)
            for h in g_sem:
                mid = h.get("message_id") or ""
                if mid:
                    mid_query_count[mid] = mid_query_count.get(mid, 0) + 1
            if bm25:
                g_kw = bm25.search(q, k)
                for h in g_kw:
                    mid = h.get("message_id") or ""
                    if mid:
                        mid_query_count[mid] = mid_query_count.get(mid, 0) + 1
                all_group_keyword.extend(g_kw)

        all_group_sem = [h for group in per_query_group_sem for h in group]
        hits = _merge_hits(all_person_sem, all_keyword, all_group_sem, all_group_keyword, limit=k)
    else:
        hits = _merge_hits(all_person_sem, all_keyword, limit=k)

    # Apply query-aware recency boost before score gate: direction is controlled by
    # query_intent — current queries float recent chunks up; historical queries
    # penalise very-recent chunks and boost genuinely old ones.
    hits = _apply_recency_boost(hits, query_intent=query_intent)

    # Apply information-density bonus: fact-dense chunks (with numbers, proper
    # nouns, multi-message context) get a small score bump; short ack messages
    # get a slight penalty so they don't crowd out substantive context.
    hits = _apply_density_score(hits)

    # --- Score gate with cross-language multi-hit promotion ------------------
    if is_cross_lang:
        result: list[dict[str, Any]] = []
        for h in hits:
            mid = h.get("message_id") or ""
            score = float(h.get("score") or 0)
            count = mid_query_count.get(mid, 1)
            # Hits corroborated by 2+ query sets get a tighter floor (65% of cross_lang_min,
            # floored at 0.10) to reward multi-signal evidence while preventing noise.
            gate = max(cross_lang_min * 0.65, 0.10) if count >= 2 else cross_lang_min
            if score >= gate:
                result.append(h)
        logger.info(
            "Retrieval cross-lang (ws=%s person=%s queries=%d hits_pre=%d hits_post=%d "
            "cross_lang_min=%.2f widened=%s)",
            workspace_id[:8],
            person_id[:8],
            len(normalized),
            len(hits),
            len(result),
            cross_lang_min,
            need_widen,
        )
        return result

    # Single-query path — standard gate
    result = _score_gate(hits, inject_min)
    logger.info(
        "Retrieval single-query (ws=%s person=%s widened=%s hits=%d after_gate=%d min_score=%.2f)",
        workspace_id[:8],
        person_id[:8],
        need_widen,
        len(hits),
        len(result),
        inject_min,
    )
    return result


def _merge_window_ranges(
    ranges: list[tuple[int, int, float]],
    adjacency_gap: int = 2,
) -> list[tuple[int, int, float]]:
    """Merge sorted (start, end, max_score) tuples that overlap or are within adjacency_gap.

    Two windows are merged when the gap between them is ≤ adjacency_gap messages,
    preventing redundant context blocks from nearby hits.  The max score of all
    constituent hits is propagated to the merged range.
    """
    if not ranges:
        return []
    merged: list[tuple[int, int, float]] = [ranges[0]]
    for start, end, score in ranges[1:]:
        prev_start, prev_end, prev_score = merged[-1]
        if start <= prev_end + adjacency_gap:
            # Overlapping or close — extend the existing range
            merged[-1] = (prev_start, max(prev_end, end), max(prev_score, score))
        else:
            merged.append((start, end, score))
    return merged


def expand_to_turn_windows(
    workspace_id: str,
    hits: list[dict[str, Any]],
    *,
    window_before: int = 3,
    window_after: int = 4,
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

    Overlapping or adjacent windows (within 2 messages of each other) are merged
    into a single wider block rather than being dropped, so Q&A replies that
    follow a matched question are always captured.
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

    # Pass 1: collect per-hit window ranges (sorted by start position for merging)
    hit_windows: list[tuple[int, int, float]] = []  # (start, end, hit_score)
    fallback_snippets: list[str] = []  # hits not found in timeline

    for hit in hits:
        hit_score = float(hit.get("score") or 0)
        mid = hit.get("message_id") or ""

        if mid not in id_to_idx:
            fallback = (
                f"{hit.get('speaker', '')} ({hit.get('timestamp', '')}): {hit.get('snippet', '')}"
            )
            # Apply same inclusion logic to fallback snippets.
            if hit_score >= min_hit_score or (
                target_person and target_person.lower() in fallback.lower()
            ):
                if fallback not in fallback_snippets:
                    fallback_snippets.append(fallback)
            continue

        center = id_to_idx[mid]
        start = max(0, center - window_before)
        end = min(len(timeline), center + window_after + 1)
        hit_windows.append((start, end, hit_score))

    # Pass 2: sort by start, then merge overlapping / adjacent windows
    hit_windows.sort(key=lambda t: t[0])
    merged_ranges = _merge_window_ranges(hit_windows)

    # Pass 3: generate text blocks, apply inclusion filter
    result: list[str] = []
    for start, end, max_score in merged_ranges:
        if len(result) >= max_blocks:
            break

        window_msgs = timeline[start:end]
        lines = [f"{msg.sender}: {msg.text}" for msg in window_msgs]
        block = "\n".join(lines).strip()
        if not block:
            continue

        include = max_score >= min_hit_score
        if not include and target_person:
            person_lower = target_person.lower()
            include = any(
                msg.sender.lower() == person_lower or person_lower in msg.text.lower()
                for msg in window_msgs
            )

        if include:
            result.append(block)
        else:
            logger.debug(
                "Dropped window block (score=%.3f < %.3f, person '%s' not in window)",
                max_score,
                min_hit_score,
                target_person,
            )

    # Append fallback snippets (hits not in timeline) up to max_blocks
    for snippet in fallback_snippets:
        if len(result) >= max_blocks:
            break
        result.append(snippet)

    return result


def expand_hits_with_context(
    workspace_id: str,
    hits: list[dict[str, Any]],
    *,
    window_before: int = 3,
    window_after: int = 4,
    max_blocks: int = 5,
) -> list[list[dict[str, Any]]]:
    """Expand retrieval hits into structured conversation context blocks for Q&A.

    Each returned block is a list of message dicts with keys:
        - ``speaker``   (str)  display name of the sender
        - ``timestamp`` (str)  ISO-formatted or human-readable timestamp
        - ``text``      (str)  message body (noise messages filtered out)
        - ``is_hit``    (bool) True for the directly matched message, False for context

    Overlapping or adjacent windows from nearby hits are merged so Gemini sees
    a single coherent conversation excerpt instead of duplicate fragments.

    Falls back to empty list when export.txt is unavailable, letting the caller
    use the raw ``snippet`` fields from each hit instead.
    """
    if not hits:
        return []

    try:
        timeline = workspace_service.load_export_timeline(workspace_id)
    except FileNotFoundError:
        return []

    id_to_idx = {msg.id: idx for idx, msg in enumerate(timeline)}

    # Pass 1: collect window ranges, recording which index is the center (hit) for each
    # Each entry: (start, end, score, set_of_hit_indices)
    hit_windows: list[tuple[int, int, float, set[int]]] = []

    for hit in hits:
        mid = hit.get("message_id") or ""
        if mid not in id_to_idx:
            continue
        hit_score = float(hit.get("score") or 0)
        center = id_to_idx[mid]
        start = max(0, center - window_before)
        end = min(len(timeline), center + window_after + 1)
        hit_windows.append((start, end, hit_score, {center}))

    if not hit_windows:
        return []

    # Pass 2: sort by start position and merge overlapping / adjacent windows (gap ≤ 2)
    hit_windows.sort(key=lambda t: t[0])
    merged: list[tuple[int, int, float, set[int]]] = [hit_windows[0]]
    for start, end, score, centers in hit_windows[1:]:
        prev_start, prev_end, prev_score, prev_centers = merged[-1]
        if start <= prev_end + 2:
            # Overlapping or close — extend range and union the hit indices
            merged[-1] = (
                prev_start,
                max(prev_end, end),
                max(prev_score, score),
                prev_centers | centers,
            )
        else:
            merged.append((start, end, score, centers))

    # Pass 3: build structured blocks, filtering noise messages
    result: list[list[dict[str, Any]]] = []
    for start, end, _score, hit_centers in merged[:max_blocks]:
        block: list[dict[str, Any]] = []
        for idx in range(start, end):
            msg = timeline[idx]
            # Skip noise: media omissions, deleted messages, etc.
            if is_noise_message(msg):
                continue
            block.append(
                {
                    "speaker": msg.sender,
                    "timestamp": msg.timestamp.strftime("%Y-%m-%d %H:%M"),
                    "text": msg.text,
                    "is_hit": idx in hit_centers,
                }
            )
        if block:
            result.append(block)

    return result
