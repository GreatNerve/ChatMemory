from unittest.mock import MagicMock, patch

import pytest
from google.genai import errors as genai_errors

from app.core.config import get_settings
from app.services import gemini as gemini_service


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _mock_interaction(output_text: str) -> MagicMock:
    interaction = MagicMock()
    interaction.output_text = output_text
    interaction.steps = []
    return interaction


def test_warmup_gemini_client_creates_singleton(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    get_settings.cache_clear()
    gemini_service._cached_client = None

    with patch("app.services.gemini.genai.Client") as client_cls:
        client_cls.return_value = MagicMock()
        assert gemini_service.warmup_gemini_client() is True
        client_cls.assert_called_once_with(api_key="test-key")
        assert gemini_service.warmup_gemini_client() is True
        client_cls.assert_called_once()

    gemini_service._cached_client = None


def test_config_status_missing_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    get_settings.cache_clear()
    ok, msg = gemini_service.config_status()
    assert ok is False
    assert msg


def test_is_configured_accepts_google_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    get_settings.cache_clear()
    assert gemini_service.is_configured() is True


def test_is_gemini_model_name():
    assert gemini_service.is_gemini_model_name("gemini")
    assert not gemini_service.is_gemini_model_name("peft:abc")
    assert not gemini_service.is_gemini_model_name(None)


def test_chat_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    get_settings.cache_clear()

    mock_client = MagicMock()
    mock_interaction = _mock_interaction("yaar theek hai")
    mock_interaction.id = "interaction-1"
    mock_client.interactions.create.return_value = mock_interaction

    with patch("app.services.gemini._genai_client", return_value=mock_client):
        ids: list[str] = []
        text = gemini_service.chat(
            [
                {"role": "system", "content": "You are Bob"},
                {"role": "user", "content": "hey"},
            ],
            interaction_id_out=ids,
        )

    assert text == "yaar theek hai"
    assert ids == ["interaction-1"]
    call_kwargs = mock_client.interactions.create.call_args.kwargs
    # Model comes from settings; use dynamic value instead of hardcoded string.
    assert call_kwargs["model"] == get_settings().gemini_model
    assert call_kwargs["system_instruction"] == "You are Bob"
    assert call_kwargs["input"] == "hey"
    assert call_kwargs["generation_config"]["temperature"] == 0.75


def test_chat_not_configured(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    get_settings.cache_clear()
    with pytest.raises(gemini_service.GeminiNotConfiguredError):
        gemini_service.chat([{"role": "user", "content": "hi"}])


def test_chat_api_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    get_settings.cache_clear()

    mock_client = MagicMock()
    mock_client.interactions.create.side_effect = genai_errors.ClientError(
        403,
        {"error": {"message": "API key invalid"}},
    )

    with patch("app.services.gemini._genai_client", return_value=mock_client):
        with pytest.raises(gemini_service.GeminiError, match="API key invalid"):
            gemini_service.chat([{"role": "user", "content": "hi"}])


def test_chat_stream_yields_text_deltas(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    get_settings.cache_clear()

    delta1 = MagicMock()
    delta1.type = "text"
    delta1.text = "hello "
    event1 = MagicMock()
    event1.event_type = "step.delta"
    event1.delta = delta1

    delta2 = MagicMock()
    delta2.type = "text"
    delta2.text = "world"
    event2 = MagicMock()
    event2.event_type = "step.delta"
    event2.delta = delta2

    mock_client = MagicMock()
    mock_client.interactions.create.return_value = iter([event1, event2])

    with patch("app.services.gemini._genai_client", return_value=mock_client):
        chunks = list(
            gemini_service.chat_stream([{"role": "user", "content": "hi"}])
        )

    assert chunks == ["hello ", "world"]
    assert mock_client.interactions.create.call_args.kwargs["stream"] is True


def test_messages_to_interaction_input_flattens_history():
    system, user_input, prev = gemini_service._messages_to_interaction_input(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hey"},
            {"role": "assistant", "content": "hn"},
            {"role": "user", "content": "kya scene"},
        ],
        assistant_label="Kritika",
    )
    assert system == "sys"
    assert prev is None
    assert "Recent chat:" in user_input
    assert "User: hey" in user_input
    assert "Kritika: hn" in user_input
    assert user_input.endswith("User: kya scene")


def test_messages_to_interaction_input_uses_previous_id():
    system, user_input, prev = gemini_service._messages_to_interaction_input(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hey"},
            {"role": "assistant", "content": "hn"},
            {"role": "user", "content": "kya scene"},
        ],
        previous_interaction_id="prev-123",
    )
    assert system == "sys"
    assert prev == "prev-123"
    # When previous_interaction_id is provided AND there is prior history,
    # history is now flattened into the message body for belt-and-suspenders
    # context continuity (in case the Gemini chain state is stale).
    assert "Recent chat:" in user_input
    assert "User: hey" in user_input
    assert "Assistant: hn" in user_input
    assert "User: kya scene" in user_input


def test_chat_stream_raises_on_error_event(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    get_settings.cache_clear()

    err = MagicMock()
    err.code = "too_many_requests"
    err.message = "quota exceeded"
    error_event = MagicMock()
    error_event.event_type = "error"
    error_event.error = err

    mock_client = MagicMock()
    mock_client.interactions.create.return_value = iter([error_event])

    with patch("app.services.gemini._genai_client", return_value=mock_client):
        with pytest.raises(gemini_service.GeminiError, match="quota exceeded"):
            list(gemini_service.chat_stream([{"role": "user", "content": "hi"}]))
