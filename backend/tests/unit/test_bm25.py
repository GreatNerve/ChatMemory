"""Unit tests for the numpy inverted-index BM25 implementation.

Tests:
  - test_scorer_scores_correctly  — _Bm25Scorer ranks and filters results as expected
  - test_bm25_search_returns_correct_top_k — Bm25Index.search correctness + top-k cap
"""
import pytest

from app.services.bm25 import (
    Bm25Index,
    _Bm25Scorer,
    _tokenize,
)

CORPUS = [
    {
        "messageId": "m1",
        "speaker": "Alice",
        "timestamp": "2024-01-01T10:00:00",
        "text": "I love going to the beach on sunny days",
    },
    {
        "messageId": "m2",
        "speaker": "Bob",
        "timestamp": "2024-01-01T10:01:00",
        "text": "The beach was crowded with people",
    },
    {
        "messageId": "m3",
        "speaker": "Alice",
        "timestamp": "2024-01-01T10:02:00",
        "text": "Mountain hiking is my favourite outdoor activity",
    },
    {
        "messageId": "m4",
        "speaker": "Bob",
        "timestamp": "2024-01-01T10:03:00",
        "text": "I went hiking yesterday and it was exhausting",
    },
    {
        "messageId": "m5",
        "speaker": "Alice",
        "timestamp": "2024-01-01T10:04:00",
        "text": "The weather was perfect for outdoor sports",
    },
]


def _tokenized_corpus() -> list[list[str]]:
    return [_tokenize(row["text"]) for row in CORPUS]


def test_scorer_scores_correctly():
    """_Bm25Scorer must give positive scores to matching docs and zero to non-matching ones."""
    tokens_list = _tokenized_corpus()
    scorer = _Bm25Scorer(tokens_list)

    # "beach" appears only in docs 0 and 1
    scores = list(scorer.score(_tokenize("beach")))
    assert len(scores) == len(CORPUS)
    assert scores[0] > 0 and scores[1] > 0, "beach should score positively in m1 and m2"
    assert scores[2] == 0 and scores[3] == 0 and scores[4] == 0

    # "hiking" appears only in docs 2 and 3
    scores = list(scorer.score(_tokenize("hiking")))
    assert scores[2] > 0 and scores[3] > 0
    assert scores[0] == 0 and scores[1] == 0 and scores[4] == 0

    # Top-1 for "beach sunny" should be m1 (contains both terms)
    scores = list(scorer.score(_tokenize("beach sunny")))
    top = max(range(len(scores)), key=lambda i: scores[i])
    assert top == 0, f"Expected m1 (index 0) to rank first, got index {top}"

    # All-zero scores for a term that matches nothing
    scores = list(scorer.score(_tokenize("zzznomatch999")))
    assert all(s == 0 for s in scores)


def test_bm25_search_returns_correct_top_k():
    """search() must return only relevant docs, respect top_k, and rank correctly."""
    index = Bm25Index(CORPUS)

    # "beach" only matches m1 and m2 — they should be the top-2
    results = index.search("beach", top_k=2)
    assert len(results) == 2
    returned_ids = {r["message_id"] for r in results}
    assert returned_ids == {"m1", "m2"}

    # Scores must be positive and decreasing
    assert all(r["score"] > 0 for r in results)
    assert results[0]["score"] >= results[1]["score"]

    # "hiking" only matches m3 and m4
    results = index.search("hiking", top_k=5)
    assert len(results) == 2
    returned_ids = {r["message_id"] for r in results}
    assert returned_ids == {"m3", "m4"}

    # top_k cap is respected even when more docs match
    results = index.search("outdoor", top_k=1)
    assert len(results) == 1

    # Empty query returns empty list
    results = index.search("", top_k=5)
    assert results == []

    # Query that matches nothing returns empty list
    results = index.search("zzznomatch999", top_k=5)
    assert results == []

    # Result fields present
    results = index.search("beach sunny", top_k=5)
    for r in results:
        assert "message_id" in r
        assert "speaker" in r
        assert "timestamp" in r
        assert "snippet" in r
        assert "score" in r
