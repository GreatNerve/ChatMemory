"""Unit tests for paired-exchange training data pipeline.

Tests cover:
- _build_paired_exchanges: the core helper that pairs target messages with preceding
  other-sender messages for relational-context training.
- refresh_person_personality / refresh_person_listening_style: verify that the paired
  exchange format flows through to the Gemini prompt.
- update_person_record: race-condition protection (per-file lock, atomic write, .bak fallback).
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import json
import threading
import pytest

from app.services.parser.whatsapp import Message
from app.services.workspace import _build_paired_exchanges, _select_voice_samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(id_: str, sender: str, text: str, minute: int = 0) -> Message:
    """Convenience constructor for test Message objects."""
    return Message(
        id=id_,
        timestamp=datetime(2024, 6, 1, 12, minute, tzinfo=timezone.utc),
        sender=sender,
        text=text,
        is_system=False,
    )


# ---------------------------------------------------------------------------
# _build_paired_exchanges
# ---------------------------------------------------------------------------

class TestBuildPairedExchanges:
    """Core helper: pairs target messages with preceding other-sender context."""

    def test_pairs_immediately_preceding_other_sender(self):
        """When the message directly before the target is from another sender, they are paired."""
        timeline = [
            _msg("1", "Alice", "How are you?", 0),
            _msg("2", "Kratika", "I'm doing well!", 1),
        ]
        result = _build_paired_exchanges(timeline, "Kratika", [timeline[1]])
        assert len(result) == 1
        assert "[Alice]: How are you?" in result[0]
        assert "[Kratika]: I'm doing well!" in result[0]

    def test_standalone_when_target_is_first_message(self):
        """Target message with no preceding message at all is rendered standalone."""
        timeline = [_msg("1", "Kratika", "Hello!", 0)]
        result = _build_paired_exchanges(timeline, "Kratika", [timeline[0]])
        assert result == ["[Kratika]: Hello!"]

    def test_standalone_when_no_other_sender_within_8_positions(self):
        """If 8+ consecutive target messages precede the target, it stays standalone."""
        # Build 9 consecutive target messages — more than the 8-position lookback limit.
        msgs = [_msg(str(i), "Kratika", f"msg{i}", i) for i in range(9)]
        timeline = msgs
        # Last message in the block — all 8 positions back are also "Kratika" messages.
        result = _build_paired_exchanges(timeline, "Kratika", [msgs[-1]])
        assert result == ["[Kratika]: msg8"]

    def test_looks_back_past_consecutive_target_messages(self):
        """Skips consecutive target messages to find the nearest other-sender message."""
        timeline = [
            _msg("1", "Alice", "Context from Alice", 0),
            _msg("2", "Kratika", "msg a", 1),
            _msg("3", "Kratika", "msg b", 2),
            _msg("4", "Kratika", "final response", 3),
        ]
        result = _build_paired_exchanges(timeline, "Kratika", [timeline[3]])
        # Alice's message is 3 positions back — within the 8-position window.
        assert "[Alice]: Context from Alice" in result[0]
        assert "[Kratika]: final response" in result[0]

    def test_mixed_standalone_and_paired(self):
        """Mixed list of standalone and paired messages is handled correctly."""
        timeline = [
            _msg("1", "Kratika", "standalone", 0),
            _msg("2", "Alice", "question?", 1),
            _msg("3", "Kratika", "answer", 2),
        ]
        target_msgs = [timeline[0], timeline[2]]
        result = _build_paired_exchanges(timeline, "Kratika", target_msgs)
        assert result[0] == "[Kratika]: standalone"
        assert "[Alice]: question?" in result[1]
        assert "[Kratika]: answer" in result[1]

    def test_single_speaker_workspace_all_standalone(self):
        """All messages from same sender → all standalone (backward-compat fallback)."""
        timeline = [
            _msg("1", "Kratika", "first", 0),
            _msg("2", "Kratika", "second", 1),
            _msg("3", "Kratika", "third", 2),
        ]
        result = _build_paired_exchanges(timeline, "Kratika", timeline)
        assert all(r.startswith("[Kratika]:") for r in result)
        assert not any("[" in r.split("]", 1)[1] for r in result)  # no second [...]

    def test_skips_noise_messages_for_context(self):
        """Noise messages (media omitted, deleted) are skipped when seeking context."""
        timeline = [
            _msg("1", "Alice", "<Media omitted>", 0),
            _msg("2", "Kratika", "Hello!", 1),
        ]
        result = _build_paired_exchanges(timeline, "Kratika", [timeline[1]])
        # "<Media omitted>" is noise — no valid preceding message → standalone.
        assert result == ["[Kratika]: Hello!"]

    def test_skips_blank_other_sender_messages(self):
        """Blank messages from other senders are also skipped when seeking context."""
        timeline = [
            _msg("1", "Alice", "   ", 0),
            _msg("2", "Kratika", "hello", 1),
        ]
        result = _build_paired_exchanges(timeline, "Kratika", [timeline[1]])
        assert result == ["[Kratika]: hello"]

    def test_returns_empty_for_empty_target_list(self):
        """Empty target list returns empty exchange list."""
        timeline = [_msg("1", "Alice", "hi")]
        result = _build_paired_exchanges(timeline, "Kratika", [])
        assert result == []

    def test_target_messages_order_independent(self):
        """The order of target_msgs does not affect correctness (reversed order test)."""
        timeline = [
            _msg("1", "Alice", "first context", 0),
            _msg("2", "Kratika", "first response", 1),
            _msg("3", "Alice", "second context", 2),
            _msg("4", "Kratika", "second response", 3),
        ]
        # Pass in reverse chronological order — should still produce correct pairs.
        result = _build_paired_exchanges(timeline, "Kratika", [timeline[3], timeline[1]])
        # Both should be paired.
        for r in result:
            assert "[Alice]:" in r
            assert "[Kratika]:" in r

    def test_chooses_most_recent_other_sender_within_window(self):
        """When multiple other senders precede the target, takes the nearest one."""
        timeline = [
            _msg("1", "Bob", "old context", 0),
            _msg("2", "Alice", "closer context", 1),
            _msg("3", "Kratika", "response", 2),
        ]
        result = _build_paired_exchanges(timeline, "Kratika", [timeline[2]])
        # "Alice" is the most recent other sender — should be chosen.
        assert "[Alice]: closer context" in result[0]
        assert "[Bob]" not in result[0]

    def test_message_not_in_timeline_treated_as_standalone(self):
        """Message whose ID is absent from the timeline index is rendered standalone."""
        timeline = [_msg("1", "Alice", "hi", 0)]
        orphan = _msg("orphan", "Kratika", "I'm not in the timeline")
        result = _build_paired_exchanges(timeline, "Kratika", [orphan])
        assert result == ["[Kratika]: I'm not in the timeline"]


# ---------------------------------------------------------------------------
# _select_voice_samples — algorithmic voice sample selection (no Gemini)
# ---------------------------------------------------------------------------

class TestSelectVoiceSamples:
    """Algorithmic voice sample selection: cluster → pick → cap → Gemini label → filter."""

    @pytest.fixture(autouse=True)
    def _mock_label_voice_samples(self, monkeypatch):
        """Patch _label_voice_samples so tests never make real Gemini calls.

        Returns a simple list of generic labels in the same order as the input
        exchanges so the rest of the pipeline (context assignment, dedup) still works.
        """
        def _fake_label(exchanges: list, person_name: str) -> list[str]:
            short = person_name.split(",")[0].strip()
            return [f"{short} speaking"] * len(exchanges)

        monkeypatch.setattr("app.services.workspace._label_voice_samples", _fake_label)

    def _run(self, coro):
        """Drive an async coroutine synchronously inside a sync test."""
        return asyncio.run(coro)

    def _msg(self, id_: str, sender: str, text: str, minute: int = 0) -> Message:
        return Message(
            id=id_,
            timestamp=datetime(2024, 6, 1, 12, minute, tzinfo=timezone.utc),
            sender=sender,
            text=text,
            is_system=False,
        )

    def _multi_word(self, text: str) -> bool:
        """True if text has more than 1 word — samples must pass this check."""
        return len(text.strip().split()) > 1

    def test_empty_timeline_returns_empty(self):
        """Empty timeline → no candidates → empty result."""
        result = self._run(_select_voice_samples([], "Kratika"))
        assert result == []

    def test_single_speaker_returns_empty(self):
        """Timeline with only the target sender → no paired exchanges → empty."""
        timeline = [
            self._msg("1", "Kratika", "Hello there", 0),
            self._msg("2", "Kratika", "How are you", 1),
            self._msg("3", "Kratika", "Let me know", 2),
        ]
        result = self._run(_select_voice_samples(timeline, "Kratika"))
        assert result == []

    def test_filters_one_word_messages(self):
        """Exchanges containing any 1-word message are excluded."""
        # All messages from the other sender are single-word — should be filtered.
        timeline = [
            self._msg("1", "Dheeraj", "Ok", 0),          # 1-word — filtered
            self._msg("2", "Kratika", "Haan theek hai", 1),
            self._msg("3", "Dheeraj", "Fine", 2),         # 1-word — filtered
            self._msg("4", "Kratika", "Kal milte hai", 3),
        ]
        result = self._run(_select_voice_samples(timeline, "Kratika"))
        # Every candidate exchange includes a 1-word Dheeraj message → all filtered.
        assert result == []

    def test_basic_exchange_included(self):
        """A valid 4-turn exchange with both speakers and multi-word messages is selected."""
        timeline = [
            self._msg("1", "Dheeraj", "website deploy kiya kya", 0),
            self._msg("2", "Kratika", "Haan kr diya check kr le", 1),
            self._msg("3", "Dheeraj", "thik acha thanks bhai", 2),
            self._msg("4", "Kratika", "fst fst presentation bhi hai kal", 3),
        ]
        result = self._run(_select_voice_samples(timeline, "Kratika"))
        assert len(result) >= 1
        # Every sample must have an exchange list and a context field.
        for sample in result:
            assert "context" in sample
            assert "exchange" in sample
            assert isinstance(sample["exchange"], list)
            assert len(sample["exchange"]) >= 1

    def test_exchange_capped_at_4_turns(self):
        """No exchange in the result has more than 4 turns."""
        timeline = [
            self._msg(str(i), "Dheeraj" if i % 2 == 0 else "Kratika", f"message number {i} here", i)
            for i in range(20)
        ]
        result = self._run(_select_voice_samples(timeline, "Kratika"))
        for sample in result:
            assert len(sample["exchange"]) <= 4, (
                f"Exchange has {len(sample['exchange'])} turns — should be capped at 4"
            )

    def test_exchange_contains_both_senders(self):
        """Every selected exchange must include the target person and at least one other sender."""
        timeline = [
            self._msg("1", "Dheeraj", "kya chal raha hai project", 0),
            self._msg("2", "Kratika", "sab badhiya chal raha hai", 1),
            self._msg("3", "Dheeraj", "deadline kab hai batao", 2),
            self._msg("4", "Kratika", "kal tak submit krna hai", 3),
            self._msg("5", "Dheeraj", "design ready hai kya abhi", 4),
            self._msg("6", "Kratika", "haan design final ho gaya", 5),
        ]
        result = self._run(_select_voice_samples(timeline, "Kratika"))
        for sample in result:
            senders = {m["sender"] for m in sample["exchange"]}
            assert "Kratika" in senders, "Target person missing from exchange"
            assert len(senders) >= 2, "Exchange must involve at least 2 senders"

    def test_topic_buckets_produce_diverse_contexts(self):
        """Multiple distinct topic regions produce multiple context-labelled samples."""
        # Mix of topic keyword regions so the classifier assigns different buckets.
        timeline = [
            # scheduling region
            self._msg("1", "Dheeraj", "kal meeting kab hai bhai", 0),
            self._msg("2", "Kratika", "kal 10 bje se hai meeting", 1),
            self._msg("3", "Dheeraj", "thik hai confirm kr dena", 2),
            self._msg("4", "Kratika", "haan confirm ho gaya schedule", 3),
            # project_work region
            self._msg("5", "Dheeraj", "website deploy kar diya kya", 10),
            self._msg("6", "Kratika", "haan deploy ho gaya server pe", 11),
            self._msg("7", "Dheeraj", "github push bhi kar dena please", 12),
            self._msg("8", "Kratika", "push kar diya dekh lo github", 13),
            # casual/social region
            self._msg("9", "Dheeraj", "yaar kya maza aa gaya aaj", 20),
            self._msg("10", "Kratika", "haan bilkul noice tha ekdum bhai", 21),
            self._msg("11", "Dheeraj", "chill bhai aaj bohot maza aya", 22),
            self._msg("12", "Kratika", "haan fun tha ekdum chill raha", 23),
        ]
        result = self._run(_select_voice_samples(timeline, "Kratika"))
        assert len(result) >= 2, "Expected at least 2 samples from different topic regions"
        # The mock labeler returns the same generic label, so we check count not diversity.
        contexts = [s["context"] for s in result]
        assert all(isinstance(c, str) and len(c) > 0 for c in contexts), (
            f"All contexts must be non-empty strings, got: {contexts}"
        )

    def test_result_within_sample_count_bounds(self):
        """Result length is between 0 and max_samples (8) — never exceeds the cap."""
        timeline = [
            self._msg(str(i), "Dheeraj" if i % 2 == 0 else "Kratika", f"some text message {i} here", i)
            for i in range(40)
        ]
        result = self._run(_select_voice_samples(timeline, "Kratika"))
        assert len(result) <= 8

    def test_noise_messages_excluded_from_exchange_windows(self):
        """System noise messages (media omitted) are not included in exchange candidates."""
        timeline = [
            self._msg("1", "Dheeraj", "deployment kab hoga project ka", 0),
            self._msg("2", "Kratika", "<Media omitted>", 1),  # noise — skipped
            self._msg("3", "Kratika", "kal tak ho jaega deployment sure", 2),
            self._msg("4", "Dheeraj", "thik hai acha bata dena please", 3),
        ]
        result = self._run(_select_voice_samples(timeline, "Kratika"))
        # Any included exchange must not contain "<Media omitted>" text.
        for sample in result:
            for msg in sample["exchange"]:
                assert "<Media omitted>" not in msg["text"]

    def test_exchange_sender_text_keys_present(self):
        """Every message in every exchange has 'sender' and 'text' keys."""
        timeline = [
            self._msg("1", "Dheeraj", "bhai kya chal raha hai yaar", 0),
            self._msg("2", "Kratika", "sab theek hai bas kaam chal raha", 1),
            self._msg("3", "Dheeraj", "kab free hoge aaj bata dena", 2),
            self._msg("4", "Kratika", "shaam ko free hoon bata dunga", 3),
        ]
        result = self._run(_select_voice_samples(timeline, "Kratika"))
        for sample in result:
            for msg in sample["exchange"]:
                assert "sender" in msg
                assert "text" in msg
                assert isinstance(msg["sender"], str)
                assert isinstance(msg["text"], str)


# ---------------------------------------------------------------------------
# refresh_person_personality integration (mocked)
# ---------------------------------------------------------------------------

class TestRefreshPersonPersonalityUsesPairedFormat:
    """Verify that the paired exchange format reaches the Gemini prompt."""

    def test_prompt_contains_exchange_format(self, tmp_path, monkeypatch):
        """Personality extraction prompt must include '[Sender]:' style exchanges."""
        # Write a minimal workspace + person record.
        ws_id = "test-ws"
        person_id = "test-person"

        ws_dir = tmp_path / "workspaces" / ws_id
        people_dir = ws_dir / "people"
        people_dir.mkdir(parents=True)

        record = {
            "id": person_id,
            "workspaceId": ws_id,
            "displayName": "Kratika",
            "aliases": [],
            "messageCount": 100,
            "firstSeen": "2024-01-01T00:00:00+00:00",
            "lastSeen": "2024-12-01T00:00:00+00:00",
            "personaStatus": "ready",
            "ollamaModelName": None,
            "lastTrainJobId": None,
            "lastTrainAt": None,
            "styleProfile": {"avgMessageLength": 30, "emojiRate": 0.0, "hinglishRatio": 0.0},
            "sampleMessages": [],
        }
        (people_dir / f"{person_id}.json").write_text(json.dumps(record))

        # Write meta.json so load_export_timeline and workspace lookups work.
        meta = {
            "id": ws_id,
            "name": "Test WS",
            "createdAt": "2024-01-01T00:00:00+00:00",
            "ingestStatus": "done",
            "messageCount": 3,
            "speakerCount": 2,
        }
        (ws_dir / "meta.json").write_text(json.dumps(meta))

        # Patch get_settings so paths resolve to tmp_path.
        settings_mock = MagicMock()
        settings_mock.workspaces_dir = tmp_path / "workspaces"
        settings_mock.lora_thin_min_messages = 30
        settings_mock.lora_min_messages = 100
        monkeypatch.setattr("app.services.workspace.get_settings", lambda: settings_mock)
        monkeypatch.setattr("app.core.paths.get_settings", lambda: settings_mock)

        # Fake timeline with paired messages.
        fake_timeline = [
            _msg("m1", "Dheeraj", "Kya chal raha hai?", 0),
            _msg("m2", "Kratika", "Sab badhiya!", 1),
            _msg("m3", "Dheeraj", "Hackathon?", 2),
            _msg("m4", "Kratika", "Haan bilkul!", 3),
        ]
        monkeypatch.setattr(
            "app.services.workspace.load_export_timeline",
            lambda wid: fake_timeline,
        )

        # Capture the prompt sent to Gemini.
        captured_prompts: list[str] = []

        def fake_chat(messages, temperature=0.4):
            captured_prompts.append(messages[0]["content"])
            return "Kratika is enthusiastic and direct."

        monkeypatch.setattr("app.services.workspace.gemini_service.chat", fake_chat)
        monkeypatch.setattr(
            "app.services.workspace._rate_limiter.acquire", lambda tokens: None
        )
        monkeypatch.setattr(
            "app.services.workspace._rate_limiter.record", lambda tokens: None
        )

        from app.services.workspace import refresh_person_personality
        refresh_person_personality(ws_id, person_id)

        assert captured_prompts, "Gemini was never called"
        prompt = captured_prompts[0]
        # Paired exchange format uses [Sender]: pattern.
        assert "[Dheeraj]:" in prompt
        assert "[Kratika]:" in prompt
        # Should NOT use the old "- message" bullet format as the primary structure.
        assert "Exchanges:" in prompt


# ---------------------------------------------------------------------------
# refresh_person_listening_style integration (mocked)
# ---------------------------------------------------------------------------

class TestRefreshPersonListeningStyleUsesPairedFormat:
    """Verify that listening style prompt receives other-sender context."""

    def test_prompt_contains_other_sender_messages(self, tmp_path, monkeypatch):
        """Listening style extraction must receive the stimulus messages from the other person."""
        ws_id = "test-ws-ls"
        person_id = "test-person-ls"

        ws_dir = tmp_path / "workspaces" / ws_id
        people_dir = ws_dir / "people"
        people_dir.mkdir(parents=True)

        record = {
            "id": person_id,
            "workspaceId": ws_id,
            "displayName": "Kratika",
            "aliases": [],
            "messageCount": 100,
            "firstSeen": "2024-01-01T00:00:00+00:00",
            "lastSeen": "2024-12-01T00:00:00+00:00",
            "personaStatus": "ready",
            "ollamaModelName": None,
            "lastTrainJobId": None,
            "lastTrainAt": None,
            "styleProfile": {"avgMessageLength": 30, "emojiRate": 0.0, "hinglishRatio": 0.0},
            "sampleMessages": [],
        }
        (people_dir / f"{person_id}.json").write_text(json.dumps(record))
        (ws_dir / "meta.json").write_text(json.dumps({
            "id": ws_id,
            "name": "Test LS WS",
            "createdAt": "2024-01-01T00:00:00+00:00",
            "ingestStatus": "done",
            "messageCount": 4,
            "speakerCount": 2,
        }))

        settings_mock = MagicMock()
        settings_mock.workspaces_dir = tmp_path / "workspaces"
        settings_mock.lora_thin_min_messages = 30
        settings_mock.lora_min_messages = 100
        monkeypatch.setattr("app.services.workspace.get_settings", lambda: settings_mock)
        monkeypatch.setattr("app.core.paths.get_settings", lambda: settings_mock)

        fake_timeline = [
            _msg("m1", "Dheeraj", "Yaar bura lag raha hai aaj", 0),
            _msg("m2", "Kratika", "Kya hua? Bata na", 1),
            _msg("m3", "Dheeraj", "Sab theek hai", 2),
            _msg("m4", "Kratika", "Ok good", 3),
        ]
        monkeypatch.setattr(
            "app.services.workspace.load_export_timeline",
            lambda wid: fake_timeline,
        )

        captured_prompts: list[str] = []

        def fake_chat(messages, temperature=0.3):
            captured_prompts.append(messages[0]["content"])
            return "Kratika asks clarifying questions."

        monkeypatch.setattr("app.services.workspace.gemini_service.chat", fake_chat)
        monkeypatch.setattr("app.services.workspace._rate_limiter.acquire", lambda tokens: None)
        monkeypatch.setattr("app.services.workspace._rate_limiter.record", lambda tokens: None)

        from app.services.workspace import refresh_person_listening_style
        refresh_person_listening_style(ws_id, person_id)

        assert captured_prompts, "Gemini was never called"
        prompt = captured_prompts[0]
        # The other person's message must appear in the prompt.
        assert "Yaar bura lag raha hai aaj" in prompt or "[Dheeraj]:" in prompt
        assert "Exchanges:" in prompt


# ---------------------------------------------------------------------------
# update_person_record — race-condition protection (B + C)
# ---------------------------------------------------------------------------

class TestUpdatePersonRecord:
    """Verify per-file locking, atomic write, and empty-file .bak fallback."""

    def _write_record(self, people_dir, person_id: str, extra: dict | None = None) -> dict:
        """Write a minimal person JSON record and return the dict."""
        record = {
            "id": person_id,
            "workspaceId": "test-ws-race",
            "displayName": "TestUser",
            "aliases": [],
            "messageCount": 10,
            "firstSeen": "2024-01-01T00:00:00+00:00",
            "lastSeen": "2024-12-01T00:00:00+00:00",
            "personaStatus": "ready",
            "ollamaModelName": None,
            "lastTrainJobId": None,
            "lastTrainAt": None,
            "styleProfile": {"avgMessageLength": 20, "emojiRate": 0.0, "hinglishRatio": 0.0},
            "sampleMessages": [],
        }
        if extra:
            record.update(extra)
        (people_dir / f"{person_id}.json").write_text(json.dumps(record), encoding="utf-8")
        return record

    def _patch_settings(self, monkeypatch, tmp_path):
        settings_mock = MagicMock()
        settings_mock.workspaces_dir = tmp_path / "workspaces"
        settings_mock.lora_thin_min_messages = 5
        settings_mock.lora_min_messages = 20
        monkeypatch.setattr("app.services.workspace.get_settings", lambda: settings_mock)
        monkeypatch.setattr("app.core.paths.get_settings", lambda: settings_mock)
        return settings_mock

    def test_normal_update_merges_fields(self, tmp_path, monkeypatch):
        """update_person_record merges new fields into the existing record."""
        ws_id = "test-ws-race"
        person_id = "pid-normal"
        ws_dir = tmp_path / "workspaces" / ws_id
        people_dir = ws_dir / "people"
        people_dir.mkdir(parents=True)
        self._write_record(people_dir, person_id)
        self._patch_settings(monkeypatch, tmp_path)

        from app.services.workspace import update_person_record
        update_person_record(ws_id, person_id, {"personalityNotes": "Direct and warm"})

        result = json.loads((people_dir / f"{person_id}.json").read_text(encoding="utf-8"))
        assert result["personalityNotes"] == "Direct and warm"
        assert result["displayName"] == "TestUser"  # existing fields preserved

    def test_atomic_write_creates_bak(self, tmp_path, monkeypatch):
        """After update_person_record, a .bak snapshot of the previous state exists."""
        ws_id = "test-ws-race"
        person_id = "pid-bak"
        ws_dir = tmp_path / "workspaces" / ws_id
        people_dir = ws_dir / "people"
        people_dir.mkdir(parents=True)
        original = self._write_record(people_dir, person_id)
        self._patch_settings(monkeypatch, tmp_path)

        from app.services.workspace import update_person_record
        update_person_record(ws_id, person_id, {"writingStyleNotes": "Terse"})

        bak_path = people_dir / f"{person_id}.bak"
        assert bak_path.exists(), ".bak file must be created by atomic write"
        bak_data = json.loads(bak_path.read_text(encoding="utf-8"))
        # .bak should reflect state BEFORE the update (no writingStyleNotes).
        assert "writingStyleNotes" not in bak_data

    def test_no_tmp_file_left_behind(self, tmp_path, monkeypatch):
        """The .tmp scratch file must not exist after a successful update."""
        ws_id = "test-ws-race"
        person_id = "pid-tmp"
        ws_dir = tmp_path / "workspaces" / ws_id
        people_dir = ws_dir / "people"
        people_dir.mkdir(parents=True)
        self._write_record(people_dir, person_id)
        self._patch_settings(monkeypatch, tmp_path)

        from app.services.workspace import update_person_record
        update_person_record(ws_id, person_id, {"typingFingerprint": {"speed": "fast"}})

        tmp_path_file = people_dir / f"{person_id}.tmp"
        assert not tmp_path_file.exists(), ".tmp must be renamed away by os.replace()"

    def test_empty_file_recovers_from_bak(self, tmp_path, monkeypatch):
        """If the .json is empty (truncated mid-write), update_person_record recovers from .bak."""
        ws_id = "test-ws-race"
        person_id = "pid-empty"
        ws_dir = tmp_path / "workspaces" / ws_id
        people_dir = ws_dir / "people"
        people_dir.mkdir(parents=True)
        original = self._write_record(people_dir, person_id)
        self._patch_settings(monkeypatch, tmp_path)

        # Write good data to .bak, then truncate the live .json to simulate a crash.
        bak_path = people_dir / f"{person_id}.bak"
        bak_path.write_text(json.dumps(original), encoding="utf-8")
        (people_dir / f"{person_id}.json").write_text("", encoding="utf-8")

        from app.services.workspace import update_person_record
        # Should not raise — must recover from .bak.
        update_person_record(ws_id, person_id, {"chatAnalysis": "Friendly"})

        result = json.loads((people_dir / f"{person_id}.json").read_text(encoding="utf-8"))
        assert result["chatAnalysis"] == "Friendly"
        assert result["displayName"] == "TestUser"

    def test_empty_file_no_bak_raises_clear_error(self, tmp_path, monkeypatch):
        """Empty .json with no .bak raises a descriptive ValueError (not JSONDecodeError)."""
        ws_id = "test-ws-race"
        person_id = "pid-corrupt"
        ws_dir = tmp_path / "workspaces" / ws_id
        people_dir = ws_dir / "people"
        people_dir.mkdir(parents=True)
        (people_dir / f"{person_id}.json").write_text("", encoding="utf-8")
        self._patch_settings(monkeypatch, tmp_path)

        from app.services.workspace import update_person_record
        with pytest.raises(ValueError, match="empty"):
            update_person_record(ws_id, person_id, {"x": 1})

    def test_concurrent_updates_do_not_corrupt(self, tmp_path, monkeypatch):
        """Multiple threads calling update_person_record concurrently must not corrupt the file.

        Each thread appends a unique key; after all threads complete, every key must
        be present in the final record — proving that no update was lost to a race.
        """
        ws_id = "test-ws-race"
        person_id = "pid-concurrent"
        ws_dir = tmp_path / "workspaces" / ws_id
        people_dir = ws_dir / "people"
        people_dir.mkdir(parents=True)
        self._write_record(people_dir, person_id)
        self._patch_settings(monkeypatch, tmp_path)

        from app.services.workspace import update_person_record

        errors: list[Exception] = []
        n_threads = 10

        def worker(i: int) -> None:
            try:
                update_person_record(ws_id, person_id, {f"field_{i}": f"value_{i}"})
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent updates raised: {errors}"
        result = json.loads((people_dir / f"{person_id}.json").read_text(encoding="utf-8"))
        for i in range(n_threads):
            assert f"field_{i}" in result, f"field_{i} was lost to a concurrent write"
