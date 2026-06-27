from app.core.schemas import PersonDetail, SampleMessage, StyleProfile
from app.services.persona_chat import _chat_messages, _solo_examples, build_system_prompt


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
    prompt = build_system_prompt(person, _solo_examples(person), [])
    assert "Kritika" in prompt
    assert "yaar kal meeting" in prompt
    assert "not as an ai" in prompt.lower()


def test_build_system_prompt_includes_memory_block():
    person = PersonDetail(
        id="p1",
        display_name="Kritika",
        message_count=100,
        style_profile=StyleProfile(),
    )
    memory = ["Kritika: Goa trip december mein plan tha\nFriend: hn sahi"]
    prompt = build_system_prompt(person, [], [], memory_blocks=memory)
    assert "RELEVANT PAST CHAT" in prompt
    assert "Goa trip" in prompt


def test_chat_messages_includes_conversation_summary(monkeypatch):
    person = PersonDetail(
        id="p1",
        display_name="Kritika",
        message_count=100,
        ollama_model_name="gemini",
        style_profile=StyleProfile(),
    )

    # _build_context now returns 6-tuple: (system, solo, convo, context_ms, used_memory, memory_blocks)
    def fake_build_context(*_args, **_kwargs):
        return "BASE SYSTEM", [], [], 1.0, False, []

    monkeypatch.setattr(
        "app.services.persona_chat._build_context",
        fake_build_context,
    )

    # _chat_messages now returns (turns, memory_blocks)
    turns, memory_blocks = _chat_messages(
        "ws1",
        person,
        [{"role": "user", "content": "hi"}],
        "hello",
        conversation_summary="They discussed weekend plans earlier.",
    )
    system = turns[0]["content"]
    assert "Earlier in this conversation (summarized):" in system
    assert "weekend plans" in system
    assert memory_blocks == []
