"""Two-stage router: fast heuristics, then Gemini for ambiguous memory-intent detection."""

from __future__ import annotations

from app.prompts.routing import persona_classify_history_need
from app.services import gemini as gemini_service
import json
import re
from typing import Literal

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
) -> tuple[bool, bool, list[str], str]:
    """Gemini structured classify for ambiguous turns.

    Returns ``(needs_history, needs_rewrite, search_queries, query_intent)`` where:
    - ``needs_history`` — whether old chat memory retrieval is required.
    - ``needs_rewrite`` — whether the raw message is too pronoun-heavy or brief
      to be a useful standalone retrieval query (set by the router, not by a
      word-count heuristic in the graph node).
    - ``search_queries`` — list of 1-4 fully context-resolved phrases covering
      different language phrasings of the topic to maximise cross-language
      Hinglish↔English retrieval via parallel embedding searches.
    - ``query_intent`` — temporal direction: "current" | "historical" | "neutral".
      Controls recency boost direction in the retrieval scoring pipeline.

    Falls back to ``(False, False, [], "neutral")`` on any parse error.
    """
    # Use last 6 turns (3 exchanges) so the classifier has enough context to
    # resolve pronouns and topic references in short follow-up messages.
    recent = history[-6:]
    context_lines: list[str] = []
    for turn in recent:
        label = "User" if turn.get("role") == "user" else "Persona"
        context_lines.append(f"{label}: {turn.get('content', '')}")
    context_block = "\n".join(context_lines) if context_lines else "(no prior turns)"

    # Lowercase the message before classification — ALL-CAPS queries confuse the model
    # and may generate poor search queries; lowercasing normalises without losing meaning.
    normalized_message = user_message.strip().lower()

    prompt = persona_classify_history_need(context_block, normalized_message)

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
        return False, False, [], "neutral"

    try:
        parsed = json.loads(text[start:end])
    except json.JSONDecodeError:
        return False, False, [], "neutral"

    needs = bool(parsed.get("needs_history"))

    if not needs:
        # needs_rewrite and query_intent are only meaningful when retrieval will happen.
        return False, False, [], "neutral"

    needs_rewrite = bool(parsed.get("needs_rewrite", False))

    # Parse query_intent — default to "neutral" for unknown/missing values.
    raw_intent = str(parsed.get("query_intent", "neutral")).strip().lower()
    query_intent = raw_intent if raw_intent in ("current", "historical", "neutral") else "neutral"

    # Accept either "search_queries" (new list form) or legacy "search_query" (str).
    raw_queries = parsed.get("search_queries")
    if isinstance(raw_queries, list):
        queries = [str(q).strip() for q in raw_queries if str(q).strip()]
    else:
        # Legacy fallback: single string field
        legacy = str(parsed.get("search_query") or "").strip()
        queries = [legacy] if legacy else []

    # Always ensure at least the normalised message is present as a fallback.
    if not queries:
        queries = [normalized_message]

    return True, needs_rewrite, queries, query_intent
