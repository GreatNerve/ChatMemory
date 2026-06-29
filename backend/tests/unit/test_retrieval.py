from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from app.services.retrieval import (
    _merge_hits,
    _merge_window_ranges,
    _person_pass_weak,
    _score_gate,
    expand_hits_with_context,
    expand_to_turn_windows,
)


def test_person_pass_weak_when_empty():
    assert _person_pass_weak([], weak_threshold=0.3, min_strong_hits=2, strong_score=0.25)


def test_person_pass_weak_when_scores_low():
    hits = [{"message_id": "a", "score": 0.1}, {"message_id": "b", "score": 0.15}]
    assert _person_pass_weak(hits, weak_threshold=0.3, min_strong_hits=2, strong_score=0.25)


def test_person_pass_strong_when_enough_hits():
    hits = [
        {"message_id": "a", "score": 0.5},
        {"message_id": "b", "score": 0.4},
    ]
    assert not _person_pass_weak(hits, weak_threshold=0.3, min_strong_hits=2, strong_score=0.25)


def test_merge_hits_dedupes_by_message_id():
    semantic = [{"message_id": "1", "score": 0.9, "snippet": "a"}]
    keyword = [{"message_id": "1", "score": 0.2, "snippet": "a"}, {"message_id": "2", "score": 0.8}]
    merged = _merge_hits(semantic, keyword, limit=5)
    assert len(merged) == 2
    assert merged[0]["message_id"] == "1"
    assert merged[0]["score"] == 0.9


# ---------------------------------------------------------------------------
# Score-gate tests — verifies that low-confidence hits are blocked from
# being injected as persona memory context.
# ---------------------------------------------------------------------------


def test_score_gate_removes_hits_below_threshold():
    hits = [
        {"message_id": "a", "score": 0.40},
        {"message_id": "b", "score": 0.20},  # below 0.35 threshold
        {"message_id": "c", "score": 0.35},  # exactly at threshold — keep
    ]
    result = _score_gate(hits, min_score=0.35)
    ids = [h["message_id"] for h in result]
    assert "b" not in ids, "Hit below threshold should be dropped"
    assert "a" in ids
    assert "c" in ids


def test_score_gate_returns_empty_when_all_below_threshold():
    hits = [
        {"message_id": "x", "score": 0.10},
        {"message_id": "y", "score": 0.15},
    ]
    result = _score_gate(hits, min_score=0.35)
    assert result == [], "All hits below threshold → empty list, no hallucination context injected"


def test_score_gate_passthrough_when_min_score_zero():
    hits = [{"message_id": "z", "score": 0.01}]
    result = _score_gate(hits, min_score=0.0)
    assert result == hits, "min_score=0 should disable the gate entirely"


def test_score_gate_handles_missing_score_field():
    hits = [{"message_id": "m"}, {"message_id": "n", "score": 0.50}]
    result = _score_gate(hits, min_score=0.35)
    ids = [h["message_id"] for h in result]
    assert "m" not in ids, "Hit with no score should be treated as 0 and dropped"
    assert "n" in ids


# ---------------------------------------------------------------------------
# _merge_window_ranges — unit tests for the overlap/adjacency merge helper
# ---------------------------------------------------------------------------


def test_merge_window_ranges_no_overlap():
    ranges = [(0, 5, 0.9), (10, 15, 0.7)]
    merged = _merge_window_ranges(ranges, adjacency_gap=2)
    assert len(merged) == 2


def test_merge_window_ranges_adjacent_within_gap():
    # end=5, next start=6 → gap of 1, within default adjacency_gap=2 → should merge
    ranges = [(0, 5, 0.9), (6, 10, 0.7)]
    merged = _merge_window_ranges(ranges, adjacency_gap=2)
    assert len(merged) == 1
    assert merged[0] == (0, 10, 0.9)


def test_merge_window_ranges_overlapping():
    ranges = [(0, 8, 0.5), (5, 12, 0.8)]
    merged = _merge_window_ranges(ranges, adjacency_gap=2)
    assert len(merged) == 1
    start, end, score = merged[0]
    assert start == 0
    assert end == 12
    assert score == 0.8  # max score propagated


def test_merge_window_ranges_three_merge_into_one():
    ranges = [(0, 4, 0.6), (3, 8, 0.7), (7, 12, 0.5)]
    merged = _merge_window_ranges(ranges, adjacency_gap=2)
    assert len(merged) == 1
    assert merged[0][0] == 0
    assert merged[0][1] == 12


# ---------------------------------------------------------------------------
# expand_hits_with_context — window expansion for Q&A
# ---------------------------------------------------------------------------


def _make_message(id_: str, sender: str, text: str, ts_offset_minutes: int = 0):
    """Build a fake Message for tests without importing the dataclass directly."""
    from app.services.parser.whatsapp import Message

    ts = datetime(2024, 3, 15, 21, 0, 0, tzinfo=timezone.utc)
    from datetime import timedelta

    return Message(
        id=id_,
        timestamp=ts + timedelta(minutes=ts_offset_minutes),
        sender=sender,
        text=text,
    )


def _make_timeline():
    """Five-message timeline: question at index 2, answer at index 3."""
    return [
        _make_message("m0", "Alice", "bhai kya scene hai", 0),
        _make_message("m1", "Bob", "kuch nahi yaar", 1),
        _make_message("m2", "Alice", "oblivion kab release hua", 2),   # hit
        _make_message("m3", "Bob", "March 28 hai", 3),                  # answer
        _make_message("m4", "Alice", "pre order kr leta hun", 4),
    ]


def test_expand_hits_with_context_basic_window():
    """Hit at index 2 with window 1 before + 2 after should include indices 1-4."""
    timeline = _make_timeline()
    hits = [{"message_id": "m2", "score": 0.9, "speaker": "Alice", "timestamp": "", "snippet": ""}]

    with patch(
        "app.services.retrieval.workspace_service.load_export_timeline",
        return_value=timeline,
    ):
        blocks = expand_hits_with_context(
            "ws1", hits, window_before=1, window_after=2, max_blocks=5
        )

    assert len(blocks) == 1
    block = blocks[0]
    texts = [m["text"] for m in block]

    # Should include the message before the hit
    assert "kuch nahi yaar" in texts
    # Matched message itself
    assert "oblivion kab release hua" in texts
    # Answer that follows the hit
    assert "March 28 hai" in texts


def test_expand_hits_with_context_is_hit_flag():
    """Only the matched message should have is_hit=True."""
    timeline = _make_timeline()
    hits = [{"message_id": "m2", "score": 0.9, "speaker": "Alice", "timestamp": "", "snippet": ""}]

    with patch(
        "app.services.retrieval.workspace_service.load_export_timeline",
        return_value=timeline,
    ):
        blocks = expand_hits_with_context("ws1", hits, window_before=1, window_after=2)

    block = blocks[0]
    hit_messages = [m for m in block if m["is_hit"]]
    non_hit_messages = [m for m in block if not m["is_hit"]]

    assert len(hit_messages) == 1
    assert hit_messages[0]["text"] == "oblivion kab release hua"
    assert all(not m["is_hit"] for m in non_hit_messages)


def test_expand_hits_with_context_merges_overlapping_windows():
    """Two hits close together (m1 and m3) should produce a single merged block."""
    timeline = _make_timeline()
    hits = [
        {"message_id": "m1", "score": 0.8, "speaker": "Bob", "timestamp": "", "snippet": ""},
        {"message_id": "m3", "score": 0.7, "speaker": "Bob", "timestamp": "", "snippet": ""},
    ]

    with patch(
        "app.services.retrieval.workspace_service.load_export_timeline",
        return_value=timeline,
    ):
        blocks = expand_hits_with_context("ws1", hits, window_before=1, window_after=1)

    # Hits at idx 1 and 3, each with window ±1 → idx 0-2 and 2-4 → merged to 0-4
    assert len(blocks) == 1
    texts = [m["text"] for m in blocks[0]]
    assert "bhai kya scene hai" in texts  # idx 0
    assert "pre order kr leta hun" in texts  # idx 4


def test_expand_hits_with_context_fallback_when_no_export():
    """Returns empty list when export.txt is not found."""
    hits = [{"message_id": "m1", "score": 0.8, "speaker": "Bob", "timestamp": "", "snippet": "x"}]

    with patch(
        "app.services.retrieval.workspace_service.load_export_timeline",
        side_effect=FileNotFoundError("export.txt"),
    ):
        blocks = expand_hits_with_context("ws1", hits)

    assert blocks == []


def test_expand_hits_with_context_unknown_message_id():
    """Hits whose message_id is not in the timeline are silently skipped."""
    timeline = _make_timeline()
    hits = [{"message_id": "UNKNOWN", "score": 0.9, "speaker": "X", "timestamp": "", "snippet": ""}]

    with patch(
        "app.services.retrieval.workspace_service.load_export_timeline",
        return_value=timeline,
    ):
        blocks = expand_hits_with_context("ws1", hits)

    assert blocks == []


# ---------------------------------------------------------------------------
# Timeline cache — same mtime returns cached result without re-parsing
# ---------------------------------------------------------------------------


def test_timeline_cache_same_mtime_returns_cached():
    """Calling load_export_timeline twice with the same mtime should only parse once."""
    import app.services.workspace as ws_module

    # Clear cache so previous test runs don't interfere
    ws_module._timeline_cache.clear()

    fake_msgs = [_make_message("c1", "Alice", "cached message")]

    first_call_count = 0

    def fake_load(workspace_id):
        """Simulate the real function but track how many times it parses."""
        nonlocal first_call_count
        export_path = ws_module.workspace_path(workspace_id) / "export.txt"
        mtime = export_path.stat().st_mtime
        cached = ws_module._timeline_cache.get(workspace_id)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        first_call_count += 1
        ws_module._timeline_cache[workspace_id] = (mtime, fake_msgs)
        return fake_msgs

    # Patch stat().st_mtime to a stable float so mtime never changes
    from unittest.mock import MagicMock

    mock_stat = MagicMock()
    mock_stat.st_mtime = 1_700_000_000.0

    with patch("app.services.workspace.workspace_path") as mock_ws_path, patch(
        "app.services.workspace.load_export_timeline", side_effect=fake_load
    ):
        mock_path = MagicMock()
        mock_path.__truediv__ = lambda self, other: mock_path
        mock_path.exists.return_value = True
        mock_path.stat.return_value = mock_stat
        mock_path.read_text.return_value = ""
        mock_ws_path.return_value = mock_path

        # Manually prime the cache (simulates first call)
        ws_module._timeline_cache["ws-cache-test"] = (1_700_000_000.0, fake_msgs)

        # Second call: should hit cache, not parse again
        result = fake_load("ws-cache-test")

    assert result == fake_msgs
    assert first_call_count == 0, "Re-parse should NOT happen when mtime is unchanged"


# ---------------------------------------------------------------------------
# fast_retrieve — multi-query / cross-language path
# ---------------------------------------------------------------------------


def _make_hit(mid: str, score: float) -> dict:
    # Use a neutral snippet (30-100 chars, no digits, no caps) so the density
    # scorer adds no bonus or penalty and existing score-gate tests stay valid.
    snippet = f"{mid} retrieved message context here"
    return {"message_id": mid, "score": score, "speaker": "Alice", "snippet": snippet}


def test_fast_retrieve_single_query_uses_standard_gate():
    """Single-query path uses persona_memory_inject_min_score (0.35 default).

    A hit at 0.30 should be dropped; a hit at 0.36 should survive.
    """
    from unittest.mock import MagicMock, patch

    from app.services.retrieval import fast_retrieve

    semantic_hits = [_make_hit("a", 0.36), _make_hit("b", 0.30)]

    with patch("app.services.retrieval.embed_service.embed_query", return_value=[0.1] * 4), \
         patch("app.services.retrieval.vector_service.semantic_search", return_value=semantic_hits), \
         patch("app.services.retrieval.bm25_service.load_index", return_value=None):
        result = fast_retrieve("ws1", ["internship"], "pid1", "Alice")

    ids = [h["message_id"] for h in result]
    assert "a" in ids, "Score 0.36 >= 0.35 should pass the gate"
    assert "b" not in ids, "Score 0.30 < 0.35 should be dropped in single-query mode"


def test_fast_retrieve_multi_query_uses_cross_lang_gate():
    """Multi-query path uses persona_memory_inject_min_score_cross_lang (0.22 default).

    A hit at 0.28 should pass the cross-lang gate even though it would fail
    the standard 0.35 gate.
    """
    from unittest.mock import patch

    from app.services.retrieval import fast_retrieve

    # Hit at 0.28 — below standard 0.35 but above cross-lang 0.22
    semantic_hits = [_make_hit("cross_lang_hit", 0.28), _make_hit("low", 0.10)]

    with patch("app.services.retrieval.embed_service.embed_query", return_value=[0.1] * 4), \
         patch("app.services.retrieval.vector_service.semantic_search", return_value=semantic_hits), \
         patch("app.services.retrieval.bm25_service.load_index", return_value=None):
        result = fast_retrieve("ws1", ["intern lag gayi", "internship", "interning"], "pid1", "Alice")

    ids = [h["message_id"] for h in result]
    assert "cross_lang_hit" in ids, "0.28 should pass cross-lang gate of 0.22"
    assert "low" not in ids, "0.10 should be dropped even in cross-lang mode"


def test_fast_retrieve_multi_query_deduplicates_hits():
    """Same message_id returned by multiple queries should appear only once in output."""
    from unittest.mock import patch

    from app.services.retrieval import fast_retrieve

    # Both queries return the same message_id with different scores
    def fake_search(workspace_id, query_vec, k, **kwargs):
        return [_make_hit("shared", 0.40), _make_hit("unique", 0.38)]

    with patch("app.services.retrieval.embed_service.embed_query", return_value=[0.1] * 4), \
         patch("app.services.retrieval.vector_service.semantic_search", side_effect=fake_search), \
         patch("app.services.retrieval.bm25_service.load_index", return_value=None):
        result = fast_retrieve("ws1", ["query one", "query two"], "pid1", "Alice")

    ids = [h["message_id"] for h in result]
    assert ids.count("shared") == 1, "Shared hit must not be duplicated in merged output"


def test_fast_retrieve_multi_hit_promotion():
    """A hit appearing in 2+ query result sets gets a lower effective gate (×0.65 of cross_lang_min).

    cross_lang_min default = 0.22 → gate for multi-hit = max(0.22 * 0.65, 0.10) ≈ 0.143.
    A hit at 0.15 that appears for 2 queries should pass; one at 0.08 should not.
    """
    from unittest.mock import patch

    from app.services.retrieval import fast_retrieve

    call_count = [0]

    def fake_search(workspace_id, query_vec, k, **kwargs):
        # Both queries return the same hit — simulating multi-signal corroboration.
        call_count[0] += 1
        return [_make_hit("multi_hit", 0.15), _make_hit("noise", 0.08)]

    with patch("app.services.retrieval.embed_service.embed_query", return_value=[0.1] * 4), \
         patch("app.services.retrieval.vector_service.semantic_search", side_effect=fake_search), \
         patch("app.services.retrieval.bm25_service.load_index", return_value=None):
        result = fast_retrieve("ws1", ["intern lag gayi", "internship"], "pid1", "Alice")

    ids = [h["message_id"] for h in result]
    assert "multi_hit" in ids, "0.15 multi-signal hit should pass the promoted gate (~0.143)"
    assert "noise" not in ids, "0.08 is below even the promoted floor of 0.10"


# ---------------------------------------------------------------------------
# _chunk_density — information density scoring
# ---------------------------------------------------------------------------


def test_chunk_density_positive_for_fact_dense_chunk():
    """A long chunk with numbers and proper nouns should get a positive bonus."""
    from app.services.retrieval import _chunk_density

    dense = (
        "Alice got her offer letter from EY yesterday — joining on July 15, 2024. "
        "Package is 9.5 LPA. She was thrilled and said NSUT placement cell was helpful."
    )
    bonus = _chunk_density(dense)
    assert bonus > 0, f"Expected positive density bonus for fact-dense chunk, got {bonus}"


def test_chunk_density_negative_for_short_ack():
    """A very short acknowledgement message should get a negative density bonus."""
    from app.services.retrieval import _chunk_density

    ack = "ok"
    bonus = _chunk_density(ack)
    assert bonus < 0, f"Expected negative density bonus for short ack, got {bonus}"


def test_apply_recency_boost_historical_penalises_recent_chunks():
    """query_intent='historical' should penalise very recent chunks (< 30 days old)."""
    from datetime import timedelta

    from app.services.retrieval import _apply_recency_boost

    now = datetime.now(timezone.utc)
    recent_ts = (now - timedelta(days=5)).isoformat()
    old_ts = (now - timedelta(days=200)).isoformat()

    hits = [
        {"message_id": "recent", "score": 0.50, "timestamp": recent_ts},
        {"message_id": "old",    "score": 0.50, "timestamp": old_ts},
        {"message_id": "no_ts",  "score": 0.50, "timestamp": ""},
    ]

    result = _apply_recency_boost(hits, query_intent="historical")
    by_id = {h["message_id"]: h for h in result}

    # Recent chunk should be penalised (-0.03)
    assert by_id["recent"]["score"] < 0.50, (
        "Recent chunk should be penalised with query_intent='historical'"
    )
    # Old chunk (>180 days) should be boosted (+0.05)
    assert by_id["old"]["score"] > 0.50, (
        "Old chunk (>180 days) should be boosted with query_intent='historical'"
    )
    # No-timestamp chunk should be unchanged
    assert by_id["no_ts"]["score"] == 0.50, "Chunk without timestamp should be unchanged"

