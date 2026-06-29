"""Hallucination validation and safe-regeneration prompts for persona chat.

Used by ``graphs/persona_chat._gen_validate_factual_claims`` and
``_gen_regenerate_safe``.
"""

from __future__ import annotations


def persona_validate_factual_claims(
    persona_background: str,
    memory_blocks_text: str,
    history_text: str,
    reply: str,
) -> str:
    """Return the full validation prompt string for hallucination detection.

    The model must return ONLY JSON: ``{"has_hallucination": bool, "reason": str}``.
    Fails safe — if uncertain, the model should return ``has_hallucination=false``.

    Parameters
    ----------
    persona_background:
        Personality notes + chat analysis + listening style concatenated — facts the
        persona legitimately knows from their profile (prevents false-positive flags).
    memory_blocks_text:
        Retrieved memory excerpts joined by "\\n\\n---\\n", or "(none)".
    history_text:
        Current conversation turns (skipping system message) formatted as
        "User: …\\n{name}: …" lines, or "(none)".
    reply:
        The generated persona reply being evaluated.
    """
    return (
        "You are checking if a persona reply invents SPECIFIC VERIFIABLE FACTS that don't exist "
        "anywhere in the provided sources.\n\n"
        "ONLY flag has_hallucination=true when the reply contains ALL of:\n"
        "1. A specific factual claim (exact date, specific event name, named action someone took, "
        "technical detail, location, code or project name)\n"
        "2. That specific claim does NOT appear in: persona background, memory excerpts, OR conversation history\n"
        "3. The claim is presented as fact, not as uncertainty "
        '("yaad nahi", "shayad", "I think", "kuch aisa", "lagta hai")\n\n'
        "Do NOT flag:\n"
        "- Opinions, reactions, or emotional responses\n"
        "- Vague references (\u201cwoh incident\u201d, \u201cus cheez ke baare mein\u201d, \u201cwoh wala\u201d)\n"
        "- Casual name-drops of people known from the group (they appear in persona background)\n"
        "- Slang, humor, or personality-consistent responses\n"
        "- Saying they don't remember something (\u201cyaad nahi\u201d, \u201cpata nahi\u201d, \u201cbhool gaya\u201d)\n"
        "- Short acknowledgements or reactions\n\n"
        "SOURCES AVAILABLE TO THIS PERSONA:\n\n"
        f"PERSONA BACKGROUND (facts legitimately known by this persona):\n{persona_background or '(none)'}\n\n"
        f"RETRIEVED MEMORY EXCERPTS (specific context for this query):\n{memory_blocks_text}\n\n"
        f"CONVERSATION (current exchange):\n{history_text}\n\n"
        f"GENERATED REPLY:\n{reply}\n\n"
        "Return JSON only, no markdown: "
        '{"has_hallucination": bool, "reason": "brief explanation"}\n'
        "If uncertain, return has_hallucination=false. Only flag clear, specific invented facts."
    )


def persona_regenerate_safe(topic: str) -> str:
    """Return the regeneration-note string prepended to the system prompt on hallucination.

    ``topic`` is the hallucination reason string from the validator (used verbatim),
    or a generic fallback phrase when the reason is empty.

    The note is intentionally short so it doesn\u2019t strangle the persona voice.
    Only attempted once per reply generation cycle.
    """
    return (
        f"Note: Your previous reply may have invented a specific detail about '{topic}' that isn't in the "
        "chat history. Please avoid stating that specific detail as fact. "
        'If you\'re unsure, be vague in character: "yaad nahi", "kuch aisa hi tha shayad".\n\n'
    )
