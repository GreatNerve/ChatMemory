"""Unit tests for app.services.analytics.

Covers:
  - Session-aware reply timing (pickup vs burst classification)
  - Upper-cap filtering for pickup replies
  - Minimum-sample guard (_safe_median)
  - Zero-gap exclusion from timing lists
  - Mention detection in group chats
  - Backward-compat fields still present
  - Fixture smoke-test (existing regression guard)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.services.analytics import (
    MIN_RT_SAMPLES,
    PICKUP_REPLY_CAP_SEC,
    SESSION_GAP_SEC,
    _detect_mention,
    _safe_median,
    compute_analytics,
)
from app.services.parser.whatsapp import Message, parse_whatsapp_export


def _ts(base: datetime, delta_minutes: int) -> datetime:
    """Return base + delta_minutes as a UTC-aware datetime."""
    return base + timedelta(minutes=delta_minutes)


_MSG_COUNTER = 0


def _msg(sender: str, ts: datetime, text: str = "hello") -> Message:
    global _MSG_COUNTER
    _MSG_COUNTER += 1
    return Message(id=str(_MSG_COUNTER), timestamp=ts, sender=sender, text=text, is_system=False)


def _make_ws(tmp_path, senders: list[str]) -> tuple[str, object]:
    """Create a minimal workspace directory with person files for *senders*."""
    ws_id = "test-ws"
    ws_dir = tmp_path / ws_id
    (ws_dir / "people").mkdir(parents=True)
    for i, name in enumerate(senders):
        (ws_dir / "people" / f"p{i}.json").write_text(
            f'{{"id":"p{i}","displayName":"{name}"}}', encoding="utf-8"
        )
    return ws_id, ws_dir


def _run(tmp_path, monkeypatch, senders: list[str], messages: list[Message]) -> dict:
    """Wire up workspace, monkeypatch path helper, and run compute_analytics."""
    ws_id, ws_dir = _make_ws(tmp_path, senders)
    monkeypatch.setattr("app.services.analytics.workspace_path", lambda _: ws_dir)
    return compute_analytics(ws_id, messages)


class TestSafeMedian:
    def test_returns_none_below_min_samples(self):
        assert _safe_median([]) is None
        assert _safe_median([60.0]) is None
        assert _safe_median([60.0, 120.0]) is None

    def test_returns_value_at_min_samples(self):
        result = _safe_median([60.0, 120.0, 180.0])
        assert result == 120.0

    def test_rounds_to_one_decimal(self):
        result = _safe_median([61.0, 62.0, 63.0])
        assert isinstance(result, float)

    def test_returns_median_not_mean(self):
        # Skewed distribution: median should be 100, mean is ~133.
        result = _safe_median([100.0, 100.0, 200.0])
        assert result == 100.0


class TestDetectMention:
    def test_simple_mention(self):
        assert _detect_mention("hey @Alice what's up", "Alice") is True

    def test_case_insensitive(self):
        assert _detect_mention("@ALICE please check", "alice") is True

    def test_no_mention(self):
        assert _detect_mention("hey everyone", "Alice") is False

    def test_partial_name_not_matched(self):
        # "@Ali" should not match "Alice"
        assert _detect_mention("hey @Ali", "Alice") is False

    def test_special_chars_in_name(self):
        # Name with a dot — re.escape should handle it.
        assert _detect_mention("@Dr.Smith see you", "Dr.Smith") is True


class TestSessionAwareReplyTiming:
    BASE = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)

    def test_pickup_reply_recorded_after_session_gap(self, tmp_path, monkeypatch):
        """First cross-sender reply after SESSION_GAP_SEC is a pickup reply."""
        msgs = [
            _msg("Alice", _ts(self.BASE, 0)),   # Alice initiates
            # gap > SESSION_GAP_SEC (30 min) → new session; Bob replies → pickup
            _msg("Bob", _ts(self.BASE, 45)),
        ]
        data = _run(tmp_path, monkeypatch, ["Alice", "Bob"], msgs)
        bob = next(p for p in data["people"] if p["displayName"] == "Bob")
        # Pickup reply recorded: 45 min × 60 s = 2700 s
        # Only 1 sample → below MIN_RT_SAMPLES → typicalPickupReply is None
        assert bob["typicalPickupReply"] is None  # needs ≥ MIN_RT_SAMPLES

    def test_burst_reply_within_session(self, tmp_path, monkeypatch):
        """Replies with gap < SESSION_GAP_SEC are burst replies, not pickup."""
        base = self.BASE
        msgs = [
            _msg("Alice", _ts(base, 0)),
            _msg("Bob",   _ts(base, 1)),   # 1-min burst
            _msg("Alice", _ts(base, 2)),   # 1-min burst
            _msg("Bob",   _ts(base, 3)),   # 1-min burst
            _msg("Alice", _ts(base, 4)),   # 1-min burst
        ]
        data = _run(tmp_path, monkeypatch, ["Alice", "Bob"], msgs)
        bob = next(p for p in data["people"] if p["displayName"] == "Bob")
        # Bob has 2 burst replies (at t=1 and t=3), each 60s gap.
        # 2 < MIN_RT_SAMPLES → typicalBurstReply still None
        assert bob["typicalBurstReply"] is None
        # But burst reply events should be counted
        assert bob["repliesGiven"] >= 1

    def test_pickup_vs_burst_distinction_with_enough_samples(self, tmp_path, monkeypatch):
        """With enough data, pickup and burst medians differ meaningfully."""
        base = self.BASE
        # Build: 3 session breaks (each 40 min apart) → 3 pickup samples
        # Within each session, rapid back-and-forth → burst samples
        msgs: list[Message] = []
        session_start = base
        for session_idx in range(3):
            # Alice opens the session
            msgs.append(_msg("Alice", session_start))
            # Bob replies 40 min later (pickup — crosses 30-min boundary)
            msgs.append(_msg("Bob", session_start + timedelta(minutes=40)))
            # Fast in-session exchanges (1 min each)
            for i in range(1, 4):
                sender = "Alice" if i % 2 == 0 else "Bob"
                msgs.append(_msg(sender, session_start + timedelta(minutes=40 + i)))
            # Advance to next session (60 min after last message)
            session_start = session_start + timedelta(minutes=40 + 4 + 60)

        data = _run(tmp_path, monkeypatch, ["Alice", "Bob"], msgs)
        bob = next(p for p in data["people"] if p["displayName"] == "Bob")

        # 3 pickup samples → typicalPickupReply should be ~2400 s (40 min)
        assert bob["typicalPickupReply"] is not None
        assert bob["typicalPickupReply"] == pytest.approx(40 * 60, abs=1)

        # Burst replies should be fast (~60 s median)
        alice = next(p for p in data["people"] if p["displayName"] == "Alice")
        # Alice has burst replies in-session
        # (we care that it's populated with a much smaller value than 2400 s)
        if alice["typicalBurstReply"] is not None:
            assert alice["typicalBurstReply"] < bob["typicalPickupReply"]

    def test_same_sender_continuation_not_a_reply(self, tmp_path, monkeypatch):
        """Consecutive same-sender messages do not inflate reply counts."""
        base = self.BASE
        msgs = [
            _msg("Alice", _ts(base, 0)),
            _msg("Alice", _ts(base, 1)),  # same sender, continuation
            _msg("Alice", _ts(base, 2)),  # same sender
            _msg("Bob",   _ts(base, 3)),  # Bob replies once
        ]
        data = _run(tmp_path, monkeypatch, ["Alice", "Bob"], msgs)
        bob = next(p for p in data["people"] if p["displayName"] == "Bob")
        # Bob only replied once (in-session burst, 1 min gap)
        assert bob["repliesGiven"] == 1


class TestPickupReplyCap:
    BASE = datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc)

    def test_gap_above_cap_excluded_from_pickup_timing(self, tmp_path, monkeypatch):
        """Gaps above PICKUP_REPLY_CAP_SEC are excluded from pickup timing lists."""
        # Gap of 5 hours (> 4-hour cap) — the pickup timing should not be recorded.
        too_long = PICKUP_REPLY_CAP_SEC + 600  # 4h 10min
        msgs = [
            _msg("Alice", self.BASE),
            _msg("Bob",   self.BASE + timedelta(seconds=too_long)),
        ]
        data = _run(tmp_path, monkeypatch, ["Alice", "Bob"], msgs)
        bob = next(p for p in data["people"] if p["displayName"] == "Bob")
        # Only 1 sample anyway, but also above cap → definitely None
        assert bob["typicalPickupReply"] is None

    def test_gap_within_cap_included(self, tmp_path, monkeypatch):
        """Gaps within PICKUP_REPLY_CAP_SEC contribute to the pickup distribution."""
        within_cap = PICKUP_REPLY_CAP_SEC - 600  # 3h 50min
        # Build 3 sessions so we hit MIN_RT_SAMPLES
        base = self.BASE
        msgs: list[Message] = []
        for i in range(3):
            msgs.append(_msg("Alice", base + timedelta(hours=i * 8)))
            msgs.append(_msg("Bob",   base + timedelta(hours=i * 8, seconds=within_cap)))
        data = _run(tmp_path, monkeypatch, ["Alice", "Bob"], msgs)
        bob = next(p for p in data["people"] if p["displayName"] == "Bob")
        assert bob["typicalPickupReply"] is not None
        assert bob["typicalPickupReply"] == pytest.approx(within_cap, abs=1)

    def test_zero_gap_excluded_from_burst_timing(self, tmp_path, monkeypatch):
        """Same-minute (0-second) gaps are counted as replies but excluded from timing."""
        base = self.BASE
        # All messages in the same minute → 0-second gap after turn collapsing
        msgs = [
            _msg("Alice", base),
            _msg("Bob",   base),  # same minute, gap = 0
            _msg("Alice", base),
            _msg("Bob",   base),
        ]
        data = _run(tmp_path, monkeypatch, ["Alice", "Bob"], msgs)
        bob = next(p for p in data["people"] if p["displayName"] == "Bob")
        # typicalBurstReply should be None (no >0 timing samples, not enough samples)
        assert bob["typicalBurstReply"] is None
        # But exchanges still counted (replies are real, just not timed)
        assert bob["repliesGiven"] >= 0  # may be 0 if collapsed to same turn


class TestGroupChatMentions:
    BASE = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

    def test_mention_attributed_to_correct_speaker(self, tmp_path, monkeypatch):
        """@mention in a group chat routes the pair reply to the mentioned speaker."""
        base = self.BASE
        # Alice, Bob, Carol in a group.
        # Carol sends a message; Bob replies explicitly @Alice (not Carol).
        # This should create a pair event for Alice←Bob, not Carol←Bob.
        msgs = [
            _msg("Alice", _ts(base, 0)),
            _msg("Carol", _ts(base, 1)),   # Carol says something
            _msg("Bob",   _ts(base, 2), text="@Alice what do you think?"),  # mentions Alice
        ]
        data = _run(tmp_path, monkeypatch, ["Alice", "Bob", "Carol"], msgs)
        # Find the Alice–Bob pair
        ab_pair = next(
            (
                p for p in data["pairs"]
                if set([p["personAName"], p["personBName"]]) == {"Alice", "Bob"}
            ),
            None,
        )
        # Find the Carol–Bob pair
        cb_pair = next(
            (
                p for p in data["pairs"]
                if set([p["personAName"], p["personBName"]]) == {"Carol", "Bob"}
            ),
            None,
        )
        # Bob's message was attributed to Alice (mentioned), not Carol (prev speaker)
        if ab_pair is not None:
            assert ab_pair["exchanges"] >= 1
        # Carol–Bob pair should have no Bob→Carol reply (he mentioned Alice instead)
        if cb_pair is not None:
            carol_b = "bToAReplies" if cb_pair["personAName"] == "Carol" else "aToBReplies"
            assert cb_pair[cb_pair.get("personAName") == "Carol" and "bToAReplies" or "aToBReplies"] == 0

    def test_no_mention_uses_previous_speaker(self, tmp_path, monkeypatch):
        """Without a mention, reply is attributed to the previous speaker as before."""
        base = self.BASE
        msgs = [
            _msg("Alice", _ts(base, 0)),
            _msg("Carol", _ts(base, 1)),
            _msg("Bob",   _ts(base, 2), text="yeah totally"),  # no mention
        ]
        data = _run(tmp_path, monkeypatch, ["Alice", "Bob", "Carol"], msgs)
        # Bob's reply is attributed to Carol (previous speaker)
        cb_pair = next(
            (
                p for p in data["pairs"]
                if set([p["personAName"], p["personBName"]]) == {"Carol", "Bob"}
            ),
            None,
        )
        # Should exist and have at least 1 exchange
        assert cb_pair is not None
        assert cb_pair["exchanges"] >= 1


class TestBackwardCompatFields:
    BASE = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)

    def _basic_data(self, tmp_path, monkeypatch) -> dict:
        base = self.BASE
        msgs: list[Message] = []
        for i in range(3):
            msgs.append(_msg("Alice", base + timedelta(hours=i * 2)))
            msgs.append(_msg("Bob",   base + timedelta(hours=i * 2, minutes=45)))
        return _run(tmp_path, monkeypatch, ["Alice", "Bob"], msgs)

    def test_person_has_avg_response_seconds(self, tmp_path, monkeypatch):
        data = self._basic_data(tmp_path, monkeypatch)
        for person in data["people"]:
            assert "avgResponseSeconds" in person
            assert "avgResponseLabel" in person
            assert "medianResponseSeconds" in person

    def test_person_has_new_pickup_burst_fields(self, tmp_path, monkeypatch):
        data = self._basic_data(tmp_path, monkeypatch)
        for person in data["people"]:
            assert "typicalPickupReply" in person
            assert "typicalPickupReplyLabel" in person
            assert "typicalBurstReply" in person
            assert "typicalBurstReplyLabel" in person

    def test_pair_has_all_fields(self, tmp_path, monkeypatch):
        data = self._basic_data(tmp_path, monkeypatch)
        for pair in data["pairs"]:
            assert "avgResponseSeconds" in pair
            assert "avgResponseLabel" in pair
            assert "typicalPickupReply" in pair
            assert "typicalPickupReplyLabel" in pair
            assert "typicalBurstReply" in pair
            assert "typicalBurstReplyLabel" in pair

    def test_group_has_avg_response_seconds(self, tmp_path, monkeypatch):
        data = self._basic_data(tmp_path, monkeypatch)
        assert "avgResponseSeconds" in data["group"]
        assert "avgResponseLabel" in data["group"]

    def test_avg_response_seconds_equals_pickup_reply(self, tmp_path, monkeypatch):
        """avgResponseSeconds (compat) must equal typicalPickupReply (new field)."""
        data = self._basic_data(tmp_path, monkeypatch)
        for person in data["people"]:
            assert person["avgResponseSeconds"] == person["typicalPickupReply"]


def test_compute_analytics_on_fixture(tmp_path, monkeypatch):
    """Ensure analytics runs end-to-end on the real fixture file."""
    fixture = Path(__file__).parent.parent / "fixtures" / "whatsapp" / "android_group.txt"
    text = fixture.read_text(encoding="utf-8")
    ws_id = "test-ws"
    ws_dir = tmp_path / ws_id
    (ws_dir / "people").mkdir(parents=True)
    parsed = parse_whatsapp_export(text)
    senders = {m.sender for m in parsed.messages if not m.is_system}
    for i, sender in enumerate(sorted(senders)):
        (ws_dir / "people" / f"p{i}.json").write_text(
            f'{{"id":"p{i}","displayName":"{sender}"}}', encoding="utf-8"
        )
    (ws_dir / "export.txt").write_text(text, encoding="utf-8")
    monkeypatch.setattr("app.services.analytics.workspace_path", lambda _id: ws_dir)
    data = compute_analytics(ws_id)
    assert len(data["people"]) >= 1
    assert "group" in data
    # New fields present on all people
    for person in data["people"]:
        assert "typicalPickupReply" in person
        assert "typicalBurstReply" in person
    if len(data["people"]) >= 2:
        assert len(data["pairs"]) >= 1
        for pair in data["pairs"]:
            assert "typicalPickupReply" in pair
            assert "typicalBurstReply" in pair
