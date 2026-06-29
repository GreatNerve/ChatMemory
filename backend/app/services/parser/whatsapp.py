from dataclasses import dataclass, field
from datetime import datetime
import re
import uuid


@dataclass
class Message:
    id: str
    timestamp: datetime
    sender: str
    text: str
    is_system: bool = False


@dataclass
class ParseResult:
    messages: list[Message] = field(default_factory=list)
    format: str = "unknown"


# WhatsApp uses narrow no-break space (U+202F) before AM/PM on some exports.
_TIME_SUFFIX = r"(?:[\s\u202f\u00a0]*[APMapm]{2})?"

# Android: 9/21/23, 8:23 PM - Sender: text  OR  - system line (no ": ")
_ANDROID_LINE_RE = re.compile(
    rf"^(\d{{1,2}}/\d{{1,2}}/\d{{2,4}}),\s+"
    rf"(\d{{1,2}}:\d{{2}}(?::\d{{2}})?{_TIME_SUFFIX})"
    r"\s+-\s+(.+)$"
)

# iOS: [9/21/23, 8:23:15 PM] Sender: text
_IOS_LINE_RE = re.compile(
    rf"^\[(\d{{1,2}}/\d{{1,2}}/\d{{2,4}}),\s+"
    rf"(\d{{1,2}}:\d{{2}}(?::\d{{2}})?{_TIME_SUFFIX})\]"
    r"\s+(.+)$"
)


def _normalize_time_part(time_part: str) -> str:
    return time_part.replace("\u202f", " ").replace("\u00a0", " ").strip()


def _parse_datetime(date_part: str, time_part: str) -> datetime:
    time_part = _normalize_time_part(time_part)
    combo = f"{date_part} {time_part}"
    formats = [
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%Y %I:%M %p",
        "%m/%d/%Y %I:%M %p",
        "%d/%m/%Y %I:%M:%S %p",
        "%m/%d/%Y %I:%M:%S %p",
        "%d/%m/%y %H:%M",
        "%m/%d/%y %H:%M",
        "%d/%m/%y %I:%M %p",
        "%m/%d/%y %I:%M %p",
        "%d/%m/%y %I:%M:%S %p",
        "%m/%d/%y %I:%M:%S %p",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(combo, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unparseable datetime: {date_part} {time_part}")


def _split_sender_body(rest: str) -> tuple[str, str, bool]:
    """User lines use 'Name: message'; system/contact lines have no ': '."""
    if ": " in rest:
        sender, body = rest.split(": ", 1)
        return sender.strip(), body, False
    return "System", rest.strip(), True


def _detect_format(lines: list[str]) -> str:
    for line in lines[:40]:
        line = line.strip("\ufeff").strip()
        if not line:
            continue
        if _IOS_LINE_RE.match(line):
            return "ios"
        if _ANDROID_LINE_RE.match(line):
            return "android"
    return "android"


def _append_message(
    messages: list[Message],
    *,
    date_part: str,
    time_part: str,
    rest: str,
) -> Message:
    ts = _parse_datetime(date_part, time_part)
    sender, body, is_system = _split_sender_body(rest)
    msg = Message(
        id=str(uuid.uuid4()),
        timestamp=ts,
        sender=sender,
        text=body,
        is_system=is_system,
    )
    messages.append(msg)
    return msg


def parse_whatsapp_export(text: str) -> ParseResult:
    lines = text.splitlines()
    if not lines:
        return ParseResult()

    fmt = _detect_format(lines)
    messages: list[Message] = []
    current: Message | None = None

    line_re = _IOS_LINE_RE if fmt == "ios" else _ANDROID_LINE_RE

    for raw_line in lines:
        line = raw_line.strip("\ufeff").rstrip()
        if not line:
            continue

        m = line_re.match(line)
        if m:
            date_part, time_part, rest = m.groups()
            current = _append_message(
                messages,
                date_part=date_part,
                time_part=time_part,
                rest=rest,
            )
            continue

        if current is not None:
            current.text = f"{current.text}\n{line}" if current.text else line

    return ParseResult(messages=messages, format=fmt)


# Noise bodies that carry no retrievable content: media placeholders, deleted-message
# markers, and WhatsApp's attachment-omitted lines (after zero-width-char stripping).
# Note: preprocessor converts "This message was deleted" → "[message deleted]" and
# strips the leading U+200E from "‎image omitted" → "image omitted" etc.
_NOISE_RE = re.compile(
    r"^\s*(?:"
    r"<Media omitted>"
    r"|\[message deleted\]"
    r"|This message was deleted\.?"
    r"|message was deleted"
    r"|(?:image|video|audio|document|sticker|GIF)\s+omitted"
    r")\s*$",
    re.IGNORECASE,
)


def is_noise_message(msg: Message) -> bool:
    """Return True for messages that carry no retrievable content.

    Covers media placeholders, deleted-message markers, and WhatsApp's
    attachment-omitted lines (image/video/audio/document/sticker/GIF omitted).
    Blank/whitespace-only messages are also considered noise.
    """
    text = msg.text.strip()
    if not text:
        return True
    return bool(_NOISE_RE.match(text))


def non_system_messages(messages: list[Message]) -> list[Message]:
    """Return non-system messages that contain usable text content.

    Excludes system events, blank messages, and noise placeholders such as
    '<Media omitted>', '[message deleted]', and '*omitted' attachment lines.
    """
    return [m for m in messages if not m.is_system and not is_noise_message(m)]
