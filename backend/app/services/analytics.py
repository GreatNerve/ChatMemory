"""Workspace chat analytics — response times, activity windows, pair connectivity."""

from __future__ import annotations

from app.core.paths import workspace_path
from app.services.parser.preprocess import preprocess_whatsapp_export
from app.services.parser.whatsapp import Message, non_system_messages, parse_whatsapp_export
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import re
import statistics
from typing import Any

# A new conversation *session* begins when consecutive turns are separated by
# more than this many seconds.  Within a session, replies are "burst" replies.
SESSION_GAP_SEC = 30 * 60  # 30 min

# Pickup replies (first reply after a session break) are only included when
# the gap is ≤ this cap.  Larger gaps mean the person was sleeping or offline
# for the day — including those would skew the distribution badly.
PICKUP_REPLY_CAP_SEC = 4 * 60 * 60  # 4 h

# Minimum number of samples required to produce a meaningful median.
# Fewer samples produce None so the UI shows "—" rather than a spurious value.
MIN_RT_SAMPLES = 3

# Messages shorter than this are excluded from avgMessageLength calculations.
_MIN_LEN_FOR_AVG = 2

_DAY_LABELS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# Response-time histogram bins: (lo_sec, hi_sec, display_label).
# Lower bound inclusive, upper exclusive.
_RT_BINS: list[tuple[float, float, str]] = [
    (0, 60, "<1m"),
    (60, 300, "1–5m"),
    (300, 1800, "5–30m"),
    (1800, float("inf"), "30m+"),
]


def _hour_label(hour: int) -> str:
    if hour == 0:
        return "12 AM"
    if hour < 12:
        return f"{hour} AM"
    if hour == 12:
        return "12 PM"
    return f"{hour - 12} PM"


def _format_seconds(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s" if secs else f"{mins}m"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours}h {mins}m" if mins else f"{hours}h"


def _safe_median(values: list[float]) -> float | None:
    """Return the median only when at least MIN_RT_SAMPLES exist, else None.

    Prevents single-sample or two-sample medians from appearing as reliable
    statistics in the UI.
    """
    if len(values) >= MIN_RT_SAMPLES:
        return round(statistics.median(values), 1)
    return None


def _detect_mention(text: str, name: str) -> bool:
    """Return True if *text* contains an @mention of *name* (case-insensitive).

    Used in group chats to attribute a reply to the mentioned person even when
    other speakers have sent messages in between.
    """
    return bool(re.search(r"@" + re.escape(name), text, re.I))


def _load_sender_to_person(workspace_id: str) -> dict[str, dict[str, str]]:
    people_dir = workspace_path(workspace_id) / "people"
    mapping: dict[str, dict[str, str]] = {}
    if not people_dir.exists():
        return mapping
    for pf in people_dir.glob("*.json"):
        pdata = json.loads(pf.read_text(encoding="utf-8"))
        mapping[pdata["displayName"]] = {
            "id": pdata["id"],
            "displayName": pdata["displayName"],
        }
    return mapping


def _rt_buckets(rts: list[float]) -> list[dict[str, Any]]:
    """Bin response times into fixed ranges for histogram display."""
    return [
        {"label": label, "count": sum(1 for rt in rts if lo <= rt < hi)}
        for lo, hi, label in _RT_BINS
    ]


def _week_label(week_key: str) -> str:
    """Return a short display label for an ISO-week key like '2024-W03'."""
    yr_str, wk_str = week_key.split("-W")
    dt = datetime.fromisocalendar(int(yr_str), int(wk_str), 1)
    return dt.strftime("%b %d")


def _top_buckets(
    counts: dict[int, int], labels: dict[int, str], n: int = 3
) -> list[dict[str, Any]]:
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]
    return [{"key": k, "label": labels.get(k, str(k)), "count": c} for k, c in ranked]


@dataclass
class _Turn:
    """One uninterrupted block of messages from the same sender."""

    sender: str
    first_ts: datetime
    msg_count: int = field(default=1)
    # Concatenation of all messages in this turn — used for mention detection.
    text: str = field(default="")


def _group_into_turns(messages: list[Message]) -> list[_Turn]:
    """Merge consecutive same-sender messages into a single turn.

    Why: WhatsApp users often send 2–3 short messages in a burst.
    Without this, each burst message looks like a separate reply event,
    making response-time stats falsely fast (e.g. 3-second "replies").
    The *text* field accumulates all messages so that mention detection
    in group chats can scan the full turn content.
    """
    turns: list[_Turn] = []
    for msg in messages:
        if turns and turns[-1].sender == msg.sender:
            turns[-1].msg_count += 1
            turns[-1].text = turns[-1].text + " " + msg.text
        else:
            turns.append(_Turn(sender=msg.sender, first_ts=msg.timestamp, text=msg.text))
    return turns


def compute_analytics(
    workspace_id: str,
    messages: list[Message] | None = None,
) -> dict[str, Any]:
    if messages is None:
        export_path = workspace_path(workspace_id) / "export.txt"
        if not export_path.exists():
            raise FileNotFoundError(workspace_id)
        raw = export_path.read_text(encoding="utf-8")
        parsed = parse_whatsapp_export(preprocess_whatsapp_export(raw))
        messages = non_system_messages(parsed.messages)
    else:
        messages = non_system_messages(messages)

    sender_map = _load_sender_to_person(workspace_id)
    usable = [m for m in messages if m.sender in sender_map]

    if not usable:
        return {
            "computedAt": datetime.now(timezone.utc).isoformat(),
            "group": {},
            "people": [],
            "pairs": [],
        }

    total = len(usable)
    hour_counts: dict[int, int] = defaultdict(int)
    day_counts: dict[int, int] = defaultdict(int)
    msg_lengths: dict[str, list[int]] = defaultdict(list)
    msg_counts: dict[str, int] = defaultdict(int)
    replies_given: dict[str, int] = defaultdict(int)
    replies_received: dict[str, int] = defaultdict(int)
    initiations: dict[str, int] = defaultdict(int)

    # pickup_times[sender] = first-reply-per-session gaps (session-crossing replies)
    pickup_times: dict[str, list[float]] = defaultdict(list)
    # burst_times[sender] = within-session cross-sender reply gaps (> 0 s)
    burst_times: dict[str, list[float]] = defaultdict(list)

    # pair_pickup[(a, b)] = pickup reply gaps when b replied to a's session-end message
    pair_pickup: dict[tuple[str, str], list[float]] = defaultdict(list)
    # pair_burst[(a, b)] = burst reply gaps when b replied to a within a session
    pair_burst: dict[tuple[str, str], list[float]] = defaultdict(list)
    # pair_a_to_b counts all directional cross-sender events (pickup + burst)
    pair_a_to_b: dict[tuple[str, str], int] = defaultdict(int)

    # Group-level pickup times — used for group.avgResponseSeconds
    all_pickup_times: list[float] = []

    week_counts: dict[str, int] = defaultdict(int)
    hour_day_grid: dict[tuple[int, int], int] = defaultdict(int)

    # Group chats (>2 unique senders) support @mention attribution.
    is_group = len(sender_map) > 2

    for msg in usable:
        sender = msg.sender
        msg_counts[sender] += 1
        if len(msg.text) >= _MIN_LEN_FOR_AVG:
            msg_lengths[sender].append(len(msg.text))
        hour_counts[msg.timestamp.hour] += 1
        day_counts[msg.timestamp.weekday()] += 1
        iso_cal = msg.timestamp.isocalendar()
        week_key = f"{iso_cal.year}-W{iso_cal.week:02d}"
        week_counts[week_key] += 1
        hour_day_grid[(msg.timestamp.hour, msg.timestamp.weekday())] += 1

    # Each turn is either:
    #   • initiation    — first turn ever, or first turn of a new session
    #   • pickup reply  — first cross-sender reply after a session break
    #                     (SESSION_GAP_SEC < gap ≤ PICKUP_REPLY_CAP_SEC)
    #   • burst reply   — cross-sender reply within an active session (0 < gap ≤ SESSION_GAP_SEC)
    #   • gap == 0      — same-minute WhatsApp precision; reply is counted but
    #                     gap is excluded from timing lists (not a real duration)
    #   • same sender   — continuation, ignored for reply stats
    #
    # For group chats, @mention in the replying turn overrides the default
    # attribution (last speaker) so pair stats reflect directed replies.
    turns = _group_into_turns(usable)
    prev_turn: _Turn | None = None

    for turn in turns:
        sender = turn.sender

        if prev_turn is None:
            initiations[sender] += 1
            prev_turn = turn
            continue

        gap = (turn.first_ts - prev_turn.first_ts).total_seconds()

        if gap < 0:
            # Timestamp artefact (D/M vs M/D ambiguity) — skip without recording.
            prev_turn = turn
            continue

        if gap > SESSION_GAP_SEC:
            initiations[sender] += 1

            # Pickup reply: different sender comes back within the 4-hour cap.
            # The gap here is the true "how long until they picked up the phone"
            # stat — much more meaningful than rapid in-session exchanges.
            if prev_turn.sender != sender and gap <= PICKUP_REPLY_CAP_SEC:
                replies_given[sender] += 1
                replies_received[prev_turn.sender] += 1
                pickup_times[sender].append(gap)
                all_pickup_times.append(gap)

                # In group chats, honour @mentions for directed attribution.
                attr_target = prev_turn.sender
                if is_group:
                    for name in sender_map:
                        if name != sender and _detect_mention(turn.text, name):
                            attr_target = name
                            break

                pair_pickup[(attr_target, sender)].append(gap)
                pair_a_to_b[(attr_target, sender)] += 1

        else:
            if prev_turn.sender != sender:
                # Cross-sender reply (burst).  Always count the exchange; only
                # record timing when gap > 0 to exclude same-minute artefacts.
                replies_given[sender] += 1
                replies_received[prev_turn.sender] += 1

                if gap > 0:
                    burst_times[sender].append(gap)

                attr_target = prev_turn.sender
                if is_group:
                    for name in sender_map:
                        if name != sender and _detect_mention(turn.text, name):
                            attr_target = name
                            break

                pair_burst[(attr_target, sender)].append(gap)
                pair_a_to_b[(attr_target, sender)] += 1

        prev_turn = turn

    weekly_series: list[dict[str, Any]] = sorted(
        [{"week": k, "label": _week_label(k), "count": v} for k, v in week_counts.items()],
        key=lambda x: x["week"],
    )
    top_active_weeks = sorted(weekly_series, key=lambda x: x["count"], reverse=True)[:5]

    heatmap: list[dict[str, Any]] = [
        {"hour": h, "day": d, "count": hour_day_grid[(h, d)]}
        for d in range(7)
        for h in range(24)
        if hour_day_grid[(h, d)] > 0
    ]

    hour_labels = {h: _hour_label(h) for h in range(24)}
    day_labels = {i: _DAY_LABELS[i] for i in range(7)}

    busiest_hour = max(hour_counts, key=hour_counts.get) if hour_counts else None
    busiest_day_idx = max(day_counts, key=day_counts.get) if day_counts else None

    span_days = max(
        1,
        (usable[-1].timestamp.date() - usable[0].timestamp.date()).days + 1,
    )

    people_out: list[dict[str, Any]] = []
    for sender, count in sorted(msg_counts.items(), key=lambda x: x[1], reverse=True):
        person = sender_map[sender]

        person_hours: dict[int, int] = defaultdict(int)
        person_days: dict[int, int] = defaultdict(int)
        for m in usable:
            if m.sender != sender:
                continue
            person_hours[m.timestamp.hour] += 1
            person_days[m.timestamp.weekday()] += 1

        peak_h = max(person_hours, key=person_hours.get) if person_hours else None
        lengths = msg_lengths[sender]
        median_len = round(statistics.median(lengths), 1) if lengths else 0

        # typicalPickupReply: median first-reply-per-session — the meaningful
        # "how long until they respond" stat shown as "Typical reply" in the UI.
        pickup_rt = _safe_median(pickup_times.get(sender, []))

        # typicalBurstReply: median within-session reply speed — how fast they
        # type back once the conversation is already active.
        burst_rt = _safe_median([g for g in burst_times.get(sender, []) if g > 0])

        # avgResponseSeconds / avgResponseLabel kept for backward compatibility.
        # Now reflects the more meaningful pickup-reply median.
        compat_rt = pickup_rt

        people_out.append(
            {
                "personId": person["id"],
                "displayName": person["displayName"],
                "messageCount": count,
                "sharePercent": round(100.0 * count / total, 1),
                "avgMessageLength": median_len,
                "avgResponseSeconds": compat_rt,
                "medianResponseSeconds": compat_rt,
                "avgResponseLabel": _format_seconds(compat_rt),
                # typicalPickupReply: median gap until they reply after a break
                "typicalPickupReply": pickup_rt,
                "typicalPickupReplyLabel": _format_seconds(pickup_rt),
                # typicalBurstReply: median gap during active back-and-forth
                "typicalBurstReply": burst_rt,
                "typicalBurstReplyLabel": _format_seconds(burst_rt),
                "repliesGiven": replies_given.get(sender, 0),
                "repliesReceived": replies_received.get(sender, 0),
                "initiations": initiations.get(sender, 0),
                "peakHour": peak_h,
                "peakHourLabel": _hour_label(peak_h) if peak_h is not None else None,
                "activeHours": _top_buckets(dict(person_hours), hour_labels),
                "activeDays": _top_buckets(dict(person_days), day_labels),
                # Histogram uses burst times — shows in-session reply speed distribution.
                "responseTimeBuckets": _rt_buckets(
                    [g for g in burst_times.get(sender, []) if g > 0]
                ),
            }
        )

    # Build the union of all speaker pairs that had at least one reply event.
    all_pair_keys: set[tuple[str, str]] = set(pair_pickup.keys()) | set(pair_burst.keys())
    pairs_out: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for a, b in all_pair_keys:
        pair_key = tuple(sorted((a, b)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        a_id, b_id = pair_key

        ab = pair_a_to_b.get((a_id, b_id), 0)
        ba = pair_a_to_b.get((b_id, a_id), 0)
        exchanges = ab + ba

        # Pickup reply times for this pair (both directions combined).
        pair_pickup_combined = pair_pickup.get((a_id, b_id), []) + pair_pickup.get((b_id, a_id), [])
        pickup_pair_rt = _safe_median(pair_pickup_combined)

        # Burst reply times (for secondary stat / future use).
        pair_burst_combined = [g for g in pair_burst.get((a_id, b_id), []) if g > 0] + [
            g for g in pair_burst.get((b_id, a_id), []) if g > 0
        ]
        burst_pair_rt = _safe_median(pair_burst_combined)

        # Connection score: fraction of turns that were mutual exchanges,
        # based on turn count not raw messages to avoid burst-sender inflation.
        na = sum(1 for t in turns if t.sender == a_id)
        nb = sum(1 for t in turns if t.sender == b_id)
        connection = min(100.0, round(100.0 * exchanges / max(1, (na + nb) / 2), 1))

        pairs_out.append(
            {
                "personAId": sender_map[a_id]["id"],
                "personAName": a_id,
                "personBId": sender_map[b_id]["id"],
                "personBName": b_id,
                "exchanges": exchanges,
                "aToBReplies": ab,
                "bToAReplies": ba,
                # Backward-compat fields now use pickup reply timing.
                "avgResponseSeconds": pickup_pair_rt,
                "avgResponseLabel": _format_seconds(pickup_pair_rt),
                # New fields.
                "typicalPickupReply": pickup_pair_rt,
                "typicalPickupReplyLabel": _format_seconds(pickup_pair_rt),
                "typicalBurstReply": burst_pair_rt,
                "typicalBurstReplyLabel": _format_seconds(burst_pair_rt),
                "connectionScore": connection,
            }
        )
    pairs_out.sort(key=lambda p: p["connectionScore"], reverse=True)

    # Group-level pickup median — shown as the workspace "Typical reply" stat.
    group_pickup_rt = _safe_median(all_pickup_times)

    return {
        "computedAt": datetime.now(timezone.utc).isoformat(),
        "group": {
            "busiestHour": busiest_hour,
            "busiestHourLabel": _hour_label(busiest_hour) if busiest_hour is not None else None,
            "busiestDay": _DAY_LABELS[busiest_day_idx] if busiest_day_idx is not None else None,
            # Backward-compat group response time, now session-aware.
            "avgResponseSeconds": group_pickup_rt,
            "avgResponseLabel": _format_seconds(group_pickup_rt),
            "medianMessagesPerDay": round(total / span_days, 1),
            "activeHours": _top_buckets(dict(hour_counts), hour_labels, 5),
            "activeDays": _top_buckets(dict(day_counts), day_labels, 7),
            "strongestPair": pairs_out[0] if pairs_out else None,
            "weeklySeries": weekly_series,
            "topActiveWeeks": top_active_weeks,
            "heatmap": heatmap,
        },
        "people": people_out,
        "pairs": pairs_out,
    }


def analytics_file(workspace_id: str):
    return workspace_path(workspace_id) / "analytics.json"


def save_analytics(workspace_id: str, messages: list[Message] | None = None) -> dict[str, Any]:
    data = compute_analytics(workspace_id, messages)
    path = analytics_file(workspace_id)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def load_analytics(workspace_id: str, *, recompute: bool = False) -> dict[str, Any]:
    path = analytics_file(workspace_id)
    if path.exists() and not recompute:
        return json.loads(path.read_text(encoding="utf-8"))
    return save_analytics(workspace_id)
