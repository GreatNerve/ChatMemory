"""Normalize raw WhatsApp .txt exports before parsing.

Runs before ``parse_whatsapp_export``. See ``docs/ingest/whatsapp-export.md``.
"""

from __future__ import annotations

import re

# Strip BOM and common invisible / bidi marks that break header regex matching.
_ZERO_WIDTH_CHARS = (
    "\ufeff",  # BOM
    "\u200b",  # zero width space
    "\u200c",  # ZWNJ
    "\u200d",  # ZWJ
    "\u200e",  # LRM
    "\u200f",  # RLM
    "\u202a",  # LRE
    "\u202b",  # RLE
    "\u202c",  # PDF
    "\u202d",  # LRO
    "\u202e",  # RLO
    "\u2060",  # word joiner
)
_ZERO_WIDTH_RE = re.compile(f"[{re.escape(''.join(_ZERO_WIDTH_CHARS))}]")

_SPACE_NORMALIZE = str.maketrans({"\u202f": " ", "\u00a0": " "})

_EDITED_SUFFIX_RE = re.compile(
    r"\s*<This message was edited>\s*$",
    re.IGNORECASE,
)

_DELETED_BODY_RE = re.compile(
    r"^This message was deleted\.?$",
    re.IGNORECASE,
)

# Android / iOS header prefix — used to detect continuation vs new message.
_ANDROID_HEADER_RE = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}",
)
_IOS_HEADER_RE = re.compile(
    r"^\[\d{1,2}/\d{1,2}/\d{2,4},\s+\d{1,2}:\d{2}",
)

# Placeholder kept for message-count stats; embed layer may skip empty bodies.
DELETED_PLACEHOLDER = "[message deleted]"
MEDIA_PLACEHOLDER = "<Media omitted>"


def _strip_edited_suffix(line: str) -> str:
    return _EDITED_SUFFIX_RE.sub("", line)


def _normalize_deleted_body_in_header(line: str) -> str:
    """Tag standalone deleted-message bodies on header lines."""
    if ": " not in line:
        return line
    prefix, body = line.rsplit(": ", 1)
    if _DELETED_BODY_RE.match(body.strip()):
        return f"{prefix}: {DELETED_PLACEHOLDER}"
    return line


def _is_message_header(line: str) -> bool:
    return bool(_ANDROID_HEADER_RE.match(line) or _IOS_HEADER_RE.match(line))


def preprocess_whatsapp_export(text: str) -> str:
    """Return cleaned export text ready for ``parse_whatsapp_export``.

    Steps:
    1. Normalize CRLF / lone CR to LF.
    2. Remove BOM and zero-width / bidi control characters.
    3. Replace narrow no-break space (U+202F) and NBSP with ASCII space.
    4. Strip ``<This message was edited>`` suffixes from message bodies.
    5. Tag ``This message was deleted`` bodies with a stable placeholder.
    6. Collapse blank lines inside multi-line messages (parser continuation).
    """
    if not text:
        return text

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.lstrip("\ufeff")
    text = _ZERO_WIDTH_RE.sub("", text)
    text = text.translate(_SPACE_NORMALIZE)

    lines = text.split("\n")
    out: list[str] = []
    in_message = False

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line:
            # Blank line inside a message breaks nothing, but adds noise; drop it.
            if in_message:
                continue
            out.append(line)
            continue

        if _is_message_header(line):
            line = _strip_edited_suffix(line)
            line = _normalize_deleted_body_in_header(line)
            out.append(line)
            in_message = True
            continue

        # Continuation line (multi-line paste, Hinglish line breaks, etc.)
        line = _strip_edited_suffix(line)
        if _DELETED_BODY_RE.match(line.strip()):
            line = DELETED_PLACEHOLDER
        out.append(line)
        in_message = True

    return "\n".join(out)
