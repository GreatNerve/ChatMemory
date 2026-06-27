from app.core.schemas import PersonDetail, SampleMessage, StyleProfile
from app.services.persona_chat import _chat_messages, _person_only_texts, build_system_prompt


def test_build_system_prompt_includes_examples():
    person = PersonDetail(
        id="p1",
        display_name="Kritika",
        message_count=100,
        style_profile=StyleProfile(
            avg_message_length=42,
            emoji_rate=0.3,
            hinglish_ratio=0.6,
        ),
        sample_messages=[
            SampleMessage(timestamp="2024-01-01T00:00:00", text="yaar kal meeting hai na?"),
            SampleMessage(timestamp="2024-01-02T00:00:00", text="haha ok done 👍"),
        ],
    )
    prompt = build_system_prompt(person, _person_only_texts(person, []), [])
    assert "Kritika" in prompt
    assert "yaar kal meeting" in prompt
    assert "not as an ai" in prompt.lower()


def test_chat_messages_includes_conversation_summary(monkeypatch):
    person = PersonDetail(
        id="p1",
        display_name="Kritika",
        message_count=100,
        ollama_model_name="gemini",
        style_profile=StyleProfile(),
    )

    def fake_build_context(*_args, **_kwargs):
        return "BASE SYSTEM", [], [], 1.0

    monkeypatch.setattr(
        "app.services.persona_chat._build_context",
        fake_build_context,
    )

    messages = _chat_messages(
        "ws1",
        person,
        [{"role": "user", "content": "hi"}],
        "hello",
        conversation_summary="They discussed weekend plans earlier.",
    )
    system = messages[0]["content"]
    assert "Earlier in this conversation (summarized):" in system
    assert "weekend plans" in system
