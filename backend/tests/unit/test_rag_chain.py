from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.core.config import get_settings
from app.services import rag_chain


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _mock_llm_response(text: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=text)
    return llm


def test_rewrite_query_uses_langchain(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    get_settings.cache_clear()

    with patch(
        "app.services.rag_chain.get_chat_model",
        return_value=_mock_llm_response("Goa trip kab plan hua"),
    ) as get_model:
        text = rag_chain.rewrite_query("When was Goa trip planned?")

    assert text == "Goa trip kab plan hua"
    get_model.assert_called_once_with(temperature=0.2)


def test_grounded_answer_uses_langchain(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    get_settings.cache_clear()

    chunks = [
        {
            "speaker": "Priya",
            "timestamp": "2024-03-12T18:22:00",
            "snippet": "Goa trip final kar dete hain March end",
        }
    ]

    mock_llm = _mock_llm_response("Goa trip was in March 2024.")
    with patch("app.services.rag_chain.get_chat_model", return_value=mock_llm) as get_model:
        text = rag_chain.grounded_answer("When was the Goa trip?", chunks)

    assert text == "Goa trip was in March 2024."
    get_model.assert_called_once_with(temperature=0.3)
    prompt = mock_llm.invoke.call_args.args[0][1].content
    assert "Goa trip final" in prompt


def test_rerank_chunks_orders_by_score(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    get_settings.cache_clear()

    chunks = [
        {
            "message_id": "a",
            "speaker": "Rahul",
            "timestamp": "2024-03-12T18:22:00",
            "snippet": "low relevance",
            "score": 0.5,
        },
        {
            "message_id": "b",
            "speaker": "Priya",
            "timestamp": "2024-03-12T18:23:00",
            "snippet": "Goa trip March end",
            "score": 0.5,
        },
    ]

    with patch(
        "app.services.rag_chain.get_chat_model",
        return_value=_mock_llm_response('[{"id":1,"score":0.2},{"id":2,"score":0.9}]'),
    ):
        ranked = rag_chain.rerank_chunks("Goa trip?", chunks, top_k=2)

    assert len(ranked) == 2
    assert ranked[0]["message_id"] == "b"
    assert ranked[0]["score"] == 0.9
    assert ranked[1]["message_id"] == "a"


def test_rerank_chunks_fallback_on_bad_json(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    get_settings.cache_clear()

    chunks = [
        {
            "message_id": "a",
            "speaker": "Rahul",
            "timestamp": "2024-03-12T18:22:00",
            "snippet": "first",
            "score": 0.5,
        },
        {
            "message_id": "b",
            "speaker": "Priya",
            "timestamp": "2024-03-12T18:23:00",
            "snippet": "second",
            "score": 0.5,
        },
    ]

    with patch(
        "app.services.rag_chain.get_chat_model",
        return_value=_mock_llm_response("not json"),
    ):
        ranked = rag_chain.rerank_chunks("q", chunks, top_k=1)

    assert len(ranked) == 1
    assert ranked[0]["message_id"] == "a"
