from app.services.rerank_utils import coerce_rerank_score, parse_rerank_scores


def test_coerce_rerank_score_handles_list():
    assert coerce_rerank_score([0.85]) == 0.85


def test_coerce_rerank_score_handles_percent_string():
    assert coerce_rerank_score("85%") == 0.85


def test_parse_rerank_scores_list_scores():
    raw = '[{"id":1,"score":[0.9,0.1]},{"id":2,"score":"0.4"}]'
    assert parse_rerank_scores(raw, 2) == {1: 0.9, 2: 0.4}


def test_parse_rerank_scores_markdown_fence():
    raw = 'Here:\n```json\n[{"id":1,"score":0.7}]\n```'
    assert parse_rerank_scores(raw, 1) == {1: 0.7}


def test_parse_rerank_scores_invalid_returns_empty():
    assert parse_rerank_scores("not json", 3) == {}
