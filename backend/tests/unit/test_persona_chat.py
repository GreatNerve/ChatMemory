from app.core.schemas import PersonaChatDebugMeta, PersonDetail, SampleMessage, StyleProfile
from app.prompts.persona_chat import conversation_partner_block
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


def test_build_system_prompt_includes_active_listening_style():
    """When active_listening_style is set, build_system_prompt includes it in the output."""
    person = PersonDetail(
        id="p1",
        display_name="Rohan",
        message_count=150,
        style_profile=StyleProfile(),
        active_listening_style=(
            "Rohan typically responds to vents with a short 'bc yaar' and then pivots to "
            "asking a sharp follow-up question. He rarely validates feelings outright — "
            "he redirects quickly to solutions or dark humour."
        ),
    )
    prompt = build_system_prompt(person, [], [])
    assert "bc yaar" in prompt
    assert "mirror this in how you react" in prompt


# ---------------------------------------------------------------------------
# conversation_partner_block tests
# ---------------------------------------------------------------------------


def test_conversation_partner_block_empty_returns_empty():
    """No partners → empty string (1-person workspace silently skips the block)."""
    assert conversation_partner_block([]) == ""


def test_conversation_partner_block_includes_name_and_stats():
    """Partner name, personality notes, and computed stat labels appear in the block.

    writingStyleNotes and activeListeningStyle are intentionally excluded from the
    partner block to keep it compact (personality only — 1 paragraph max).
    """
    partner = {
        "displayName": "Dheeraj Sharma",
        "personalityNotes": "Direct and pragmatic communicator.",
        "writingStyleNotes": "Lowercase, sparse punctuation.",
        "activeListeningStyle": "Pivots quickly to action items after brief acknowledgement.",
        "styleProfile": {
            "avgMessageLength": 27.0,
            "emojiRate": 0.002,       # < 0.02 → "low"
            "hinglishRatio": 0.10,    # 0.05–0.15 → "medium"
        },
    }
    block = conversation_partner_block([partner])
    assert "CONVERSATION PARTNER" in block
    assert "Dheeraj Sharma" in block
    assert "Direct and pragmatic" in block
    # writingStyleNotes and activeListeningStyle are excluded — partner block is 1 paragraph max
    assert "Lowercase, sparse" not in block
    assert "Pivots quickly" not in block
    # Stat labels
    assert "avg msg length: 27 chars" in block
    assert "emoji rate: low" in block
    assert "hinglish: medium" in block


def test_conversation_partner_block_skips_missing_fields():
    """Fields that are None/empty are omitted without errors."""
    partner = {
        "displayName": "Alex",
        "personalityNotes": None,
        "writingStyleNotes": "",
        "activeListeningStyle": None,
        "styleProfile": {},
    }
    block = conversation_partner_block([partner])
    assert "CONVERSATION PARTNER" in block
    assert "Alex" in block
    # None/empty personality must not produce the "About them:" label
    assert "About them:" not in block


def test_conversation_partner_block_truncates_long_personality_notes():
    """Personality notes longer than 300 chars are truncated with '...'."""
    long_notes = "Y" * 400
    partner = {
        "displayName": "Bob",
        "personalityNotes": long_notes,
        "styleProfile": {},
    }
    block = conversation_partner_block([partner])
    # Should contain the first 300 chars followed by ellipsis
    assert "Y" * 300 + "..." in block
    # Must NOT include char 301+
    assert "Y" * 301 not in block


def test_conversation_partner_block_emoji_rate_labels():
    """Emoji rate bucket boundaries map to correct labels."""
    def _make(emoji_rate: float) -> str:
        return conversation_partner_block([{
            "displayName": "T",
            "styleProfile": {"avgMessageLength": 10, "emojiRate": emoji_rate, "hinglishRatio": 0},
        }])

    assert "emoji rate: low" in _make(0.0)
    assert "emoji rate: low" in _make(0.019)
    assert "emoji rate: medium" in _make(0.02)
    assert "emoji rate: medium" in _make(0.079)
    assert "emoji rate: high" in _make(0.08)


def test_conversation_partner_block_hinglish_labels():
    """Hinglish ratio bucket boundaries map to correct labels."""
    def _make(hinglish: float) -> str:
        return conversation_partner_block([{
            "displayName": "T",
            "styleProfile": {"avgMessageLength": 10, "emojiRate": 0, "hinglishRatio": hinglish},
        }])

    assert "hinglish: low" in _make(0.0)
    assert "hinglish: low" in _make(0.049)
    assert "hinglish: medium" in _make(0.05)
    assert "hinglish: medium" in _make(0.149)
    assert "hinglish: high" in _make(0.15)


def test_build_system_prompt_with_partner_includes_block():
    """build_system_prompt with a partner injects CONVERSATION PARTNER into the prompt."""
    person = PersonDetail(
        id="kratika-id",
        display_name="Kratika",
        message_count=100,
        style_profile=StyleProfile(),
    )
    partners = [{
        "displayName": "Dheeraj Sharma",
        "personalityNotes": "Pragmatic and caring teammate.",
        "styleProfile": {
            "avgMessageLength": 27.0,
            "emojiRate": 0.002,
            "hinglishRatio": 0.10,
        },
    }]
    prompt = build_system_prompt(person, [], [], partners=partners)
    assert "CONVERSATION PARTNER" in prompt
    assert "Dheeraj Sharma" in prompt
    assert "Pragmatic and caring" in prompt
    # Partner block must appear BEFORE REPLY RULES
    partner_pos = prompt.index("CONVERSATION PARTNER")
    rules_pos = prompt.index("REPLY RULES")
    assert partner_pos < rules_pos


def test_build_system_prompt_no_partners_omits_block():
    """When partners=None, no CONVERSATION PARTNER section appears."""
    person = PersonDetail(
        id="p1",
        display_name="Solo",
        message_count=50,
        style_profile=StyleProfile(),
    )
    prompt = build_system_prompt(person, [], [])
    assert "CONVERSATION PARTNER" not in prompt


def test_chat_messages_includes_conversation_summary(monkeypatch):
    person = PersonDetail(
        id="p1",
        display_name="Kritika",
        message_count=100,
        ollama_model_name="gemini",
        style_profile=StyleProfile(),
    )

    # _build_context now returns 8-tuple:
    # (system, solo, convo, context_ms, used_memory, memory_blocks, rewritten_query, debug_meta)
    def fake_build_context(*_args, **_kwargs):
        return "BASE SYSTEM", [], [], 1.0, False, [], "", PersonaChatDebugMeta()

    monkeypatch.setattr(
        "app.services.persona_chat._build_context",
        fake_build_context,
    )

    # _chat_messages now returns (turns, memory_blocks, debug_meta)
    turns, memory_blocks, _ = _chat_messages(
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
