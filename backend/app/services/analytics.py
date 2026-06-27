"""Workspace chat analytics — response times, activity windows, pair connectivity."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.core.paths import workspace_path
from app.services.parser.whatsapp import Message, non_system_messages, parse_whatsapp_export
from app.services.parser.preprocess import preprocess_whatsapp_export

# Reply if same thread turn within 2 hours
RESPONSE_WINDOW_SEC = 2 * 60 * 60
# New "conversation" if gap exceeds 30 minutes
SESSION_GAP_SEC = 30 * 60
# Messages shorter than this are excluded from avgMessageLength
# (omit truly empty or near-empty entries that would pull the median down)
_MIN_LEN_FOR_AVG = 2

_DAY_LABELS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# Response-time histogram bins: (lo_sec, hi_sec, display_label)
# Lower bound is inclusive, upper bound is exclusive (lo <= rt < hi).
# WhatsApp exports are minute-precision, so the smallest measurable gap is
# 0 s (same-minute reply). Those land in the "<1m" bucket, which is correct.
# The next smallest possible gap is exactly 60 s (next-minute reply), which
# intentionally falls in "1–5m" — 60 s == 1 m, not less than 1 m.
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
        # Omit trailing "0s" for cleaner display (e.g. "1m" not "1m 0s")
        return f"{mins}m {secs}s" if secs else f"{mins}m"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours}h {mins}m" if mins else f"{hours}h"


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


def _top_buckets(counts: dict[int, int], labels: dict[int, str], n: int = 3) -> list[dict[str, Any]]:
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]
    return [{"key": k, "label": labels.get(k, str(k)), "count": c} for k, c in ranked]


@dataclass
class _Turn:
    """One uninterrupted block of messages from the same sender."""
    sender: str
    first_ts: datetime
    msg_count: int = field(default=1)


def _group_into_turns(messages: list[Message]) -> list[_Turn]:
    """Merge consecutive same-sender messages into a single turn.

    Why: WhatsApp users often send 2–3 short messages in a burst.
    Without this, each burst message looks like a separate reply event,
    making response-time stats falsely fast (e.g. 3-second "replies").
    """
    turns: list[_Turn] = []
    for msg in messages:
        if turns and turns[-1].sender == msg.sender:
            turns[-1].msg_count += 1
        else:
            turns.append(_Turn(sender=msg.sender, first_ts=msg.timestamp))
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
    response_times: dict[str, list[float]] = defaultdict(list)
    replies_given: dict[str, int] = defaultdict(int)
    replies_received: dict[str, int] = defaultdict(int)
    initiations: dict[str, int] = defaultdict(int)

    pair_replies: dict[tuple[str, str], list[float]] = defaultdict(list)
    pair_a_to_b: dict[tuple[str, str], int] = defaultdict(int)
    all_response_times: list[float] = []

    # New: weekly message counts and hour×day heatmap
    week_counts: dict[str, int] = defaultdict(int)
    hour_day_grid: dict[tuple[int, int], int] = defaultdict(int)

    # ── Pass 1: per-message stats (counts, lengths, activity patterns) ──────────
    for msg in usable:
        sender = msg.sender
        msg_counts[sender] += 1
        # Only include messages with real content for length stats.
        if len(msg.text) >= _MIN_LEN_FOR_AVG:
            msg_lengths[sender].append(len(msg.text))
        hour_counts[msg.timestamp.hour] += 1
        day_counts[msg.timestamp.weekday()] += 1
        iso_cal = msg.timestamp.isocalendar()
        week_key = f"{iso_cal.year}-W{iso_cal.week:02d}"
        week_counts[week_key] += 1
        hour_day_grid[(msg.timestamp.hour, msg.timestamp.weekday())] += 1

    # ── Pass 2: response-time stats computed on turns, not raw messages ──────
    # Grouping consecutive same-sender messages into turns prevents burst
    # messages (multiple short messages sent in a row) from inflating reply
    # counts and making response times falsely fast.
    turns = _group_into_turns(usable)
    prev_turn: _Turn | None = None
    for turn in turns:
        sender = turn.sender
        if prev_turn is None:
            initiations[sender] += 1
        else:
            gap = (turn.first_ts - prev_turn.first_ts).total_seconds()
            if gap > SESSION_GAP_SEC:
                # Large gap → new conversation session; this turn is an initiation.
                initiations[sender] += 1
            elif 0 <= gap <= RESPONSE_WINDOW_SEC:
                # Valid reply: includes same-minute replies (gap == 0) which are
                # genuine quick responses — WhatsApp timestamps are minute-precision
                # so 0 s means "both parties sent within the same minute".
                # Negative gaps (timestamp parsing artefacts from D/M vs M/D
                # ambiguity) are excluded by the >= 0 guard.
                all_response_times.append(gap)
                response_times[sender].append(gap)
                replies_given[sender] += 1
                replies_received[prev_turn.sender] += 1
                key = (prev_turn.sender, sender)
                pair_replies[key].append(gap)
                pair_a_to_b[key] += 1
        prev_turn = turn

    # Build weekly time series (sorted ascending by ISO week key)
    weekly_series: list[dict[str, Any]] = sorted(
        [
            {"week": k, "label": _week_label(k), "count": v}
            for k, v in week_counts.items()
        ],
        key=lambda x: x["week"],
    )
    top_active_weeks = sorted(weekly_series, key=lambda x: x["count"], reverse=True)[:5]

    # Build hour×day heatmap — only emit non-zero cells to keep payload small
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
        rts = response_times.get(sender, [])
        person_hours = defaultdict(int)
        person_days = defaultdict(int)
        for m in usable:
            if m.sender != sender:
                continue
            person_hours[m.timestamp.hour] += 1
            person_days[m.timestamp.weekday()] += 1
        peak_h = max(person_hours, key=person_hours.get) if person_hours else None
        lengths = msg_lengths[sender]
        median_len = round(statistics.median(lengths), 1) if lengths else 0
        median_rt = round(statistics.median(rts), 1) if rts else None
        people_out.append(
            {
                "personId": person["id"],
                "displayName": person["displayName"],
                "messageCount": count,
                "sharePercent": round(100.0 * count / total, 1),
                # Median is robust to one-off long messages that would pull mean up.
                "avgMessageLength": median_len,
                # Keep raw seconds for callers that want to do their own math.
                "avgResponseSeconds": median_rt,
                "medianResponseSeconds": median_rt,
                # Label uses median so a single burst-ack doesn't look like sub-second reply.
                "avgResponseLabel": _format_seconds(median_rt),
                "repliesGiven": replies_given.get(sender, 0),
                "repliesReceived": replies_received.get(sender, 0),
                "initiations": initiations.get(sender, 0),
                "peakHour": peak_h,
                "peakHourLabel": _hour_label(peak_h) if peak_h is not None else None,
                "activeHours": _top_buckets(dict(person_hours), hour_labels),
                "activeDays": _top_buckets(dict(person_days), day_labels),
                "responseTimeBuckets": _rt_buckets(rts),
            }
        )

    pairs_out: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for (a, b), deltas in pair_replies.items():
        pair_key = tuple(sorted((a, b)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        a_id, b_id = pair_key
        ab = pair_a_to_b.get((a_id, b_id), 0)
        ba = pair_a_to_b.get((b_id, a_id), 0)
        exchanges = ab + ba
        combined = pair_replies.get((a_id, b_id), []) + pair_replies.get((b_id, a_id), [])
        # Connection score based on turn count, not raw message count, to avoid
        # burst-heavy senders inflating the score.
        na = sum(1 for t in turns if t.sender == a_id)
        nb = sum(1 for t in turns if t.sender == b_id)
        connection = min(100.0, round(100.0 * exchanges / max(1, (na + nb) / 2), 1))
        median_pair_rt = round(statistics.median(combined), 1) if combined else None
        pairs_out.append(
            {
                "personAId": sender_map[a_id]["id"],
                "personAName": a_id,
                "personBId": sender_map[b_id]["id"],
                "personBName": b_id,
                "exchanges": exchanges,
                "aToBReplies": ab,
                "bToAReplies": ba,
                "avgResponseSeconds": median_pair_rt,
                "avgResponseLabel": _format_seconds(median_pair_rt),
                "connectionScore": connection,
            }
        )
    pairs_out.sort(key=lambda p: p["connectionScore"], reverse=True)

    return {
        "computedAt": datetime.now(timezone.utc).isoformat(),
        "group": {
            "busiestHour": busiest_hour,
            "busiestHourLabel": _hour_label(busiest_hour) if busiest_hour is not None else None,
            "busiestDay": _DAY_LABELS[busiest_day_idx] if busiest_day_idx is not None else None,
            # Group-level response time: median across all turns, robust to burst outliers.
            "avgResponseSeconds": round(statistics.median(all_response_times), 1)
            if all_response_times
            else None,
            "avgResponseLabel": _format_seconds(
                statistics.median(all_response_times) if all_response_times else None
            ),
            "medianMessagesPerDay": round(total / span_days, 1),
            "activeHours": _top_buckets(dict(hour_counts), hour_labels, 5),
            "activeDays": _top_buckets(dict(day_counts), day_labels, 7),
            "strongestPair": pairs_out[0] if pairs_out else None,
            # New: time-series and heatmap data
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
