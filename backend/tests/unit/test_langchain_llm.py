from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import get_settings
from app.services.langchain_llm import GeminiInteractionsChat, get_chat_model


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_get_chat_model_requires_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    get_settings.cache_clear()
    with pytest.raises(Exception, match="GEMINI_API_KEY"):
        get_chat_model()


def test_gemini_interactions_chat_delegates_to_gemini_service(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    get_settings.cache_clear()

    with patch("app.services.langchain_llm.gemini_service.chat", return_value="hello") as chat:
        llm = GeminiInteractionsChat(temperature=0.3)
        result = llm.invoke([SystemMessage(content="sys"), HumanMessage(content="hi")])

    assert result.content == "hello"
    chat.assert_called_once()
    assert chat.call_args.kwargs["temperature"] == 0.3
    roles = [m["role"] for m in chat.call_args.args[0]]
    assert roles == ["system", "user"]
