"""Tests for shared rerank score parsing."""

from app.services.rerank_utils import coerce_rerank_score, parse_rerank_scores


def test_coerce_rerank_score_percent_string():
    assert coerce_rerank_score("85%") == 0.85


def test_parse_rerank_scores_from_fenced_json():
    raw = 'Here you go:\n```json\n[{"id":1,"score":0.9},{"id":2,"score":0.1}]\n```'
    scores = parse_rerank_scores(raw, 2)
    assert scores == {1: 0.9, 2: 0.1}


def test_parse_rerank_scores_ignores_out_of_range_ids():
    raw = '[{"id":1,"score":0.8},{"id":99,"score":1.0}]'
    scores = parse_rerank_scores(raw, 2)
    assert scores == {1: 0.8}
