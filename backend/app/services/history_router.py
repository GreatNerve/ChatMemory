"""Two-stage router: fast heuristics, then Gemini for ambiguous memory-intent detection."""

from __future__ import annotations

import json
import re
from typing import Literal

from app.services import gemini as gemini_service

FastRoute = Literal["casual", "memory", "ambiguous"]

_CASUAL_EXACT = frozenset(
    {
        "k",
        "ok",
        "okay",
        "hn",
        "haan",
        "ha",
        "nhi",
        "nah",
        "no",
        "yes",
        "ya",
        "yep",
        "yup",
        "lol",
        "lmao",
        "haha",
        "hehe",
        "hehee",
        "hmm",
        "hm",
        "nice",
        "cool",
        "wow",
        "thanks",
        "ty",
        "thx",
        "sure",
        "done",
        "see",
        "bye",
        "hi",
        "hello",
        "hey",
        "sup",
        "good",
        "great",
        "same",
        "true",
        "right",
        "fine",
        "isee",
        "i see",
        "got it",
        "achha",
        "accha",
        "theek",
        "thik",
        "sahi",
        "haye",
        "arey",
        "arre",
        "oye",
        "uff",
        "ugh",
    }
)

_MEMORY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\byaad\b", re.I),
    re.compile(r"\bremember\b", re.I),
    re.compile(r"\brecall\b", re.I),
    re.compile(r"\bkab\s+(tha|thi|the|hui|hua|hoga|hogi|plan|fix|decide)\b", re.I),
    re.compile(r"\bkya\s+(bola|boli|bolta|bolti|kaha|kahi|likha|likhi)\b", re.I),
    re.compile(r"\bwoh\s+wala\b", re.I),
    re.compile(r"\bus\s+din\b", re.I),
    re.compile(r"\bus\s+time\b", re.I),
    re.compile(r"\bpehle\b", re.I),
    re.compile(r"\bpurane?\b", re.I),
    re.compile(r"\blast\s+time\b", re.I),
    re.compile(r"\bwhen\s+did\b", re.I),
    re.compile(r"\bwhat\s+did\s+(we|you|i|they)\b", re.I),
    re.compile(r"\bkitne\s+din\b", re.I),
    re.compile(r"\bkis\s+din\b", re.I),
    re.compile(r"\bkons[aei]\s+din\b", re.I),
    re.compile(r"\b(remind|bata\s+na|batao\s+na)\b", re.I),
    re.compile(r"\b(trip|plan|meeting|event)\s+(kab|when|kya)\b", re.I),
    re.compile(r"\bhistory\b", re.I),
    re.compile(r"\bpast\s+(message|chat|conversation)\b", re.I),
)

_EMOJI_ONLY = re.compile(
    r"^[\s\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0000FE00-\U0000FE0F"
    r"\U0000200D\U0001F1E0-\U0001F1FF!?.,]+$"
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def fast_history_route(message: str) -> FastRoute:
    """Instant route for obvious casual or obvious memory questions."""
    raw = message.strip()
    if not raw:
        return "casual"

    norm = _normalize(raw)
    if len(norm) <= 2:
        return "casual"
    if _EMOJI_ONLY.match(raw):
        return "casual"
    if norm in _CASUAL_EXACT:
        return "casual"

    words = norm.split()

    # Any message that ends with "?" and has 2+ words is at minimum ambiguous — never
    # short-circuit as casual, even if it looks short (e.g. "MAC D?", "OBLIVION KAB HA?").
    if "?" in norm and len(words) >= 2:
        # Still check memory patterns first so "yaad kab tha?" can fast-path to memory.
        if any(p.search(norm) for p in _MEMORY_PATTERNS):
            return "memory"
        return "ambiguous"

    # Short reactions without question marks — likely casual.
    if len(norm) <= 12 and "?" not in norm and not any(p.search(norm) for p in _MEMORY_PATTERNS):
        if len(words) <= 2:
            return "casual"

    if any(p.search(norm) for p in _MEMORY_PATTERNS):
        return "memory"

    return "ambiguous"


def classify_history_need(
    user_message: str,
    history: list[dict[str, str]],
) -> tuple[bool, str]:
    """Gemini structured classify for ambiguous turns. Returns (needs_history, search_query)."""
    recent = history[-4:]
    context_lines: list[str] = []
    for turn in recent:
        label = "User" if turn.get("role") == "user" else "Persona"
        context_lines.append(f"{label}: {turn.get('content', '')}")
    context_block = "\n".join(context_lines) if context_lines else "(no prior turns)"

    # Lowercase the message before classification — ALL-CAPS queries confuse the model
    # and may generate a poor search_query; lowercasing normalises without losing meaning.
    normalized_message = user_message.strip().lower()

    prompt = (
        "Decide if the latest user message needs factual recall from old WhatsApp chat history.\n"
        'Return ONLY JSON: {"needs_history": boolean, "search_query": string}\n'
        "- needs_history=true when they ask about past events, dates, plans, what was said, "
        "or references something from before this live chat.\n"
        "- needs_history=false for casual chat, reactions, opinions, or continuing the current topic.\n"
        "- search_query: short search phrase in the same language mix as the user (Hinglish/English). "
        "Empty string when needs_history=false.\n\n"
        f"Recent chat:\n{context_block}\n\n"
        f"Latest user message: {normalized_message}"
    )

    raw = gemini_service.chat(
        [{"role": "user", "content": prompt}],
        temperature=0,
    ).strip()

    text = raw
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.I)
        if match:
            text = match.group(1)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return False, ""

    try:
        parsed = json.loads(text[start:end])
    except json.JSONDecodeError:
        return False, ""

    needs = bool(parsed.get("needs_history"))
    query = str(parsed.get("search_query") or "").strip()
    if needs and not query:
        query = normalized_message
    return needs, query
