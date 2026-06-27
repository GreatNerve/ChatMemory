from app.services.retrieval import _merge_hits, _person_pass_weak, _score_gate


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
