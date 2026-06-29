from unittest.mock import patch

import pytest

from app.services.history_router import classify_history_need, fast_history_route


def test_fast_route_casual_short_replies():
    assert fast_history_route("lol") == "casual"
    assert fast_history_route("k") == "casual"
    assert fast_history_route("👍") == "casual"


def test_fast_route_obvious_memory():
    assert fast_history_route("yaad hai Goa trip kab plan hui thi?") == "memory"
    assert fast_history_route("what did we say about the meeting last time") == "memory"
    assert fast_history_route("us din kya bola tha") == "memory"


def test_fast_route_ambiguous():
    assert fast_history_route("so about that thing we discussed") == "ambiguous"
    assert fast_history_route("I was thinking about something") == "ambiguous"


# ---------------------------------------------------------------------------
# Gap 4 — "intern lag gayi?" must NOT short-circuit as casual
# ---------------------------------------------------------------------------


def test_fast_route_hinglish_intern_is_ambiguous():
    """'intern lag gayi?' — 3 Hinglish words ending with '?' — must be ambiguous.

    The question mark + 2+ words rule prevents it being dropped as a short
    casual reply.  No memory pattern matches, so it lands in 'ambiguous' and
    proceeds to the Gemini classify step.
    """
    result = fast_history_route("intern lag gayi?")
    assert result == "ambiguous", f"Expected 'ambiguous', got '{result}'"


def test_fast_route_question_with_2_words_is_ambiguous():
    """Two-word queries ending with '?' should never be 'casual'."""
    assert fast_history_route("MAC D?") == "ambiguous"
    assert fast_history_route("job kya?") == "ambiguous"
    assert fast_history_route("college kab?") == "ambiguous"


def test_fast_route_hinglish_placement_question_is_ambiguous():
    """Placement-related Hinglish queries with '?' must reach classify."""
    assert fast_history_route("placement hua?") == "ambiguous"
    assert fast_history_route("offer mila kya?") == "ambiguous"


def test_fast_route_kaha_lagi_is_ambiguous():
    """'kaha lagi?' is a short follow-up asking where an internship/job was found.

    It ends with '?' and has 2 words — must route as 'ambiguous' (not 'casual')
    so it proceeds to classify_history_need where conversation context is used to
    resolve it into a meaningful search query like 'internship company location'.
    """
    result = fast_history_route("kaha lagi?")
    assert result == "ambiguous", f"Expected 'ambiguous', got '{result}'"


# ---------------------------------------------------------------------------
# classify_history_need — returns (bool, bool, list[str], str)
# Signature: (needs_history, needs_rewrite, search_queries, query_intent)
# ---------------------------------------------------------------------------


def test_classify_history_need_returns_list_of_queries():
    """classify_history_need must return a list, not a bare string."""
    gemini_response = '{"needs_history": true, "needs_rewrite": true, "search_queries": ["intern lag gayi", "internship", "interning at company"]}'
    with patch("app.services.history_router.gemini_service.chat", return_value=gemini_response):
        needs, needs_rewrite, queries, query_intent = classify_history_need("intern lag gayi?", [])
    assert needs is True
    assert needs_rewrite is True
    assert isinstance(queries, list)
    assert len(queries) >= 1
    assert "intern lag gayi" in queries[0] or "intern" in queries[0]


def test_classify_history_need_needs_rewrite_false_for_self_contained():
    """A self-contained question should set needs_rewrite=False even if history is needed."""
    gemini_response = '{"needs_history": true, "needs_rewrite": false, "search_queries": ["EY internship location"]}'
    with patch("app.services.history_router.gemini_service.chat", return_value=gemini_response):
        needs, needs_rewrite, queries, query_intent = classify_history_need("EY mein internship kahan thi?", [])
    assert needs is True
    assert needs_rewrite is False
    assert queries == ["EY internship location"]


def test_classify_history_need_legacy_search_query_fallback():
    """Old 'search_query' string field is still accepted for backward compat."""
    gemini_response = '{"needs_history": true, "needs_rewrite": false, "search_query": "internship offer"}'
    with patch("app.services.history_router.gemini_service.chat", return_value=gemini_response):
        needs, needs_rewrite, queries, query_intent = classify_history_need("intern lag gayi?", [])
    assert needs is True
    assert isinstance(queries, list)
    assert queries == ["internship offer"]


def test_classify_history_need_false_returns_empty_list():
    """needs_history=false must return (False, False, [], 'neutral')."""
    gemini_response = '{"needs_history": false, "needs_rewrite": false, "search_queries": []}'
    with patch("app.services.history_router.gemini_service.chat", return_value=gemini_response):
        needs, needs_rewrite, queries, query_intent = classify_history_need("haha nice", [])
    assert needs is False
    assert needs_rewrite is False
    assert queries == []
    assert query_intent == "neutral"


def test_classify_history_need_false_forces_needs_rewrite_false():
    """needs_history=false must force needs_rewrite=False regardless of JSON value."""
    gemini_response = '{"needs_history": false, "needs_rewrite": true, "search_queries": []}'
    with patch("app.services.history_router.gemini_service.chat", return_value=gemini_response):
        needs, needs_rewrite, queries, query_intent = classify_history_need("haha nice", [])
    assert needs is False
    # Router contract: needs_rewrite is meaningless without retrieval.
    assert needs_rewrite is False


def test_classify_history_need_bad_json_returns_empty():
    """Malformed Gemini response must not raise — return safe defaults."""
    with patch("app.services.history_router.gemini_service.chat", return_value="NOT JSON"):
        needs, needs_rewrite, queries, query_intent = classify_history_need("test", [])
    assert needs is False
    assert needs_rewrite is False
    assert queries == []
    assert query_intent == "neutral"


def test_classify_history_need_fallback_when_queries_empty():
    """If Gemini returns needs_history=true but empty queries, fall back to user message."""
    gemini_response = '{"needs_history": true, "needs_rewrite": true, "search_queries": []}'
    with patch("app.services.history_router.gemini_service.chat", return_value=gemini_response):
        needs, needs_rewrite, queries, query_intent = classify_history_need("intern lag gayi?", [])
    assert needs is True
    assert needs_rewrite is True
    assert queries == ["intern lag gayi?"]


def test_classify_history_need_missing_needs_rewrite_defaults_false():
    """Older Gemini responses without needs_rewrite field should default to False."""
    gemini_response = '{"needs_history": true, "search_queries": ["internship location"]}'
    with patch("app.services.history_router.gemini_service.chat", return_value=gemini_response):
        needs, needs_rewrite, queries, query_intent = classify_history_need("kahan tha?", [])
    assert needs is True
    assert needs_rewrite is False  # safe default when field is absent
    assert queries == ["internship location"]
