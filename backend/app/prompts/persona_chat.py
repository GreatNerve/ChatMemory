"""Persona chat runtime prompts — system prompt assembly and conversation summarization.

``persona_system_prompt`` assembles the full system prompt from pre-built section strings
and style metrics.  The caller (``services/persona_chat.py``) is responsible for building
the section strings from the ``PersonDetail`` object before calling this function.

``conversation_partner_block`` builds the optional CONVERSATION PARTNER section that tells
the persona who they are talking to (the other person's profile from the same workspace).

``persona_summarize_conversation`` returns a single-turn summarization prompt string
for rolling context compression (called via ``gemini_service.chat``).
"""

from __future__ import annotations


def conversation_partner_block(partners: list[dict]) -> str:
    """Build the CONVERSATION PARTNER context block for the persona system prompt.

    For each partner, extract display name, personality/writing/listening notes and
    style stats. Fields that are None or empty are silently skipped.
    Returns an empty string when ``partners`` is empty (1-person workspace).

    Parameters
    ----------
    partners:
        List of raw person JSON dicts (camelCase keys as stored on disk) for every
        person in the workspace other than the current persona.
    """
    if not partners:
        return ""

    blocks: list[str] = []
    for partner in partners:
        name = (partner.get("displayName") or "").strip()
        if not name:
            continue

        lines: list[str] = [
            "== CONVERSATION PARTNER ==",
            f"You are talking to: {name}",
        ]

        # Only personality notes — writingStyleNotes and activeListeningStyle are excluded
        # to keep the partner block compact (1 paragraph max).
        personality = (partner.get("personalityNotes") or "").strip()
        if personality:
            # Truncate long personality notes to stay concise.
            truncated = personality[:300] + "..." if len(personality) > 300 else personality
            lines.append(f"About them: {truncated}")

        sp = partner.get("styleProfile") or {}
        avg_len_p: float = sp.get("avgMessageLength") or sp.get("avg_message_length") or 0.0
        emoji_r: float = sp.get("emojiRate") or sp.get("emoji_rate") or 0.0
        hinglish_r: float = sp.get("hinglishRatio") or sp.get("hinglish_ratio") or 0.0

        emoji_label = "low" if emoji_r < 0.02 else ("medium" if emoji_r < 0.08 else "high")
        hinglish_label = "low" if hinglish_r < 0.05 else ("medium" if hinglish_r < 0.15 else "high")
        lines.append(
            f"Stats: avg msg length: {avg_len_p:.0f} chars, "
            f"emoji rate: {emoji_label}, hinglish: {hinglish_label}"
        )

        blocks.append("\n".join(lines))

    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n\n"


def persona_system_prompt(
    name: str,
    personality_section: str,
    chat_analysis_section: str,
    writing_style_section: str,
    listening_style_section: str,
    partner_block: str,
    memory_section: str,
    avg_len: int,
    hinglish_ratio: float,
    emoji_rate: float,
    terse_note: str,
    solo_block: str,
    convo_block: str,
    burst_sep: str,
    *,
    voice_style_section: str = "",
    relationship_section: str = "",
    behavioral_patterns_section: str = "",
    emotional_section: str = "",
) -> str:
    """Assemble the full persona system prompt string.

    Parameters
    ----------
    name:
        The persona's display name (used in several places in the prompt).
    personality_section:
        Pre-built block including header "Who they are…" and the personality notes,
        or an empty string if personality notes are absent.  Used only when the v2
        schema fields are absent (backward-compat fallback).
    chat_analysis_section:
        Pre-built block including header "Chat patterns…" and analysis text,
        or an empty string if chat analysis is absent.  Backward-compat fallback.
    writing_style_section:
        Pre-built block including header "Writing style…" and style notes,
        or an empty string if writing style notes are absent.  Backward-compat fallback.
    listening_style_section:
        Pre-built block including header "How they listen…" and listening notes,
        or an empty string if listening style is absent.  Backward-compat fallback.
    partner_block:
        Pre-built "== CONVERSATION PARTNER ==" block for the other person(s) in
        the workspace, or an empty string for single-person workspaces.
    memory_section:
        Pre-built "=== RELEVANT PAST CHAT ===" block with injected excerpts,
        or an empty string if no memory blocks are being injected.
    avg_len:
        Average message length in characters (int) — used in REPLY RULES.
    hinglish_ratio:
        Hinglish ratio (0.0–1.0) — used in old-schema fingerprint block.
    emoji_rate:
        Emoji rate (emojis per message) — used in old-schema fingerprint block.
    terse_note:
        Extra note appended to avg-length bullet when avg_len ≤ 15, else "".
    solo_block:
        Bullet-list string of solo example messages ("• text\\n• text…").
    convo_block:
        Conversation-snippet string separated by "\\n\\n---\\n" markers.
    burst_sep:
        The burst-message separator token (e.g. "||").
    voice_style_section:
        (v2) Pre-built "=== VOICE & STYLE ===" block with typing fingerprint and
        curated voice sample exchanges, or "" if v2 fields are absent.
    relationship_section:
        (v2) Pre-built "=== HOW YOU RELATE TO THEM ===" block, or "".
    behavioral_patterns_section:
        (v2) Pre-built "=== YOUR BEHAVIORAL PATTERNS ===" block with bulleted
        stimulus→response patterns, or "".
    emotional_section:
        (v2) Pre-built "=== YOUR EMOTIONAL STYLE ===" block, or "".
    """
    # When v2 schema fields are present, use the structured v2 sections as the primary
    # profile block. Any legacy fields (from the old 4-call pipeline) are appended as
    # supplementary context under a separate header — both are now generated together.
    # When only legacy fields exist (pre-v2 persona), use the original WHO THEY ARE block.
    has_v2 = bool(voice_style_section or relationship_section)

    if has_v2:
        primary_block = (
            voice_style_section
            + relationship_section
            + behavioral_patterns_section
            + emotional_section
        )
        # Build supplementary legacy block if any old fields are present.
        legacy_parts = (
            personality_section
            + chat_analysis_section
            + writing_style_section
            + listening_style_section
        )
        if legacy_parts.strip():
            supplementary_block = (
                f"=== ADDITIONAL CONTEXT (from deep analysis) ===\n"
                f"{legacy_parts}"
            )
        else:
            supplementary_block = ""
        profile_block = primary_block + supplementary_block
    else:
        profile_block = (
            f"=== WHO THEY ARE ===\n"
            f"{personality_section}"
            f"{chat_analysis_section}"
            f"Messaging fingerprint:\n"
            f"- Avg message length: ~{avg_len} chars (but length varies wildly — see examples){terse_note}\n"
            f"- Hinglish ratio: ~{hinglish_ratio:.0%}\n"
            f"- Emoji use: ~{emoji_rate:.1f} per message\n\n"
            f"{writing_style_section}"
            f"{listening_style_section}"
        )

    return (
        f"You are {name}. Reply as them, not as an AI.\n\n"
        f"{profile_block}"
        f"{partner_block}"
        f"=== THEIR REAL MESSAGES ===\n"
        f"Study these for vocabulary, rhythm, casing, punctuation, and energy. Reproduce the style exactly:\n"
        f"{solo_block}\n\n"
        f"=== REAL CONVERSATIONS (they are {name}) ===\n"
        f"{convo_block}\n\n"
        f"{memory_section}"
        f"=== REPLY RULES (follow every one) ===\n\n"
        f"LENGTH — vary dramatically:\n"
        f'- A quick confirmation or reaction \u2192 1\u20134 chars ("Hn", "k", "Nope", "lol")\n'
        f"- A simple answer \u2192 5\u201315 chars\n"
        f"- An involved reply \u2192 20\u201350 chars max\n"
        f"- Match the energy of what\u2019s being said. Short question \u2192 short reply. Don\u2019t pad.\n"
        f"- Never default to ~{avg_len} chars every time; that\u2019s the average, not the rule.\n\n"
        f"CASING & PUNCTUATION:\n"
        f"- Look at the real messages above \u2014 if they rarely capitalise, you rarely capitalise.\n"
        f"- If they skip punctuation (no full stops, no question marks), you skip it too.\n"
        f"- Reproduce the casing pattern exactly as seen in the examples. Don\u2019t regularise it.\n\n"
        f"VOCABULARY:\n"
        f"- Use their abbreviations: yr=yaar, hn=haan, nhi=nahi, sb=sab, kl=kal, bta=bata, etc.\n"
        f"- Only use words and phrases that appear in their real messages or close variants.\n"
        f"- No greetings, no \u2018sure\u2019, no \u2018of course\u2019, no \u2018absolutely\u2019, no sign-offs.\n\n"
        f"SENTENCE STYLE:\n"
        f"- Prefer fragments over complete sentences. If they say \u2018Nhi ho rha\u2019 not \u2018No, it\u2019s not happening\u2019, do the same.\n"
        f"- Never produce a grammatically complete formal sentence if the person never does in their real messages.\n"
        f"- No AI hedges, no politeness filler, no explanations unless they naturally explain things.\n\n"
        f"BURST MESSAGES (use {burst_sep!r} as separator — only when genuinely needed):\n"
        f"- DEFAULT IS ONE MESSAGE. Only use {burst_sep!r} for genuine bursts where the person hit send, then typed a second unrelated thought.\n"
        f"- A burst is NOT two parts of the same sentence or idea split for style.\n"
        f"- A burst IS: first reaction → then separate follow-up thought (e.g. 'Hn yr{burst_sep}kl kab hai').\n"
        f"- 'thoda relief mila yr' after 'hn abhi offer letter aaya' is ONE message — do NOT burst that.\n"
        f"- If the thought flows naturally from the previous, it is ONE message.\n"
        f"- When in doubt: single message. No burst.\n"
        f"- Wrong: Nhi ho rha{burst_sep}mujhse nhi hoga — one continuous thought, not a burst.\n\n"
        f"CONVERSATION FLOW:\n"
        f"- Never give the same type of reply more than twice in a row \u2014 vary your reaction even if your position stays the same\n"
        f"- If the other person is clearly emotional or escalating, shift tone \u2014 not necessarily gentler, but different (curious, blunt, tired, direct)\n"
        f"- Conversations move forward \u2014 after 3\u20134 exchanges on the same point, change angle or close the topic\n"
        f"- React to what\u2019s actually being said in this specific message, not just the general situation\n\n"
        f"HARD RULES:\n"
        f"- FACTS: Never invent specific events, actions, dates, project names, technical details, or things\n"
        f"  people said/did. Only reference facts that appear in RELEVANT PAST CHAT above or the current conversation.\n"
        f"- NO MEMORY = VAGUE: If you don\u2019t have evidence for something specific, reply vaguely in your voice\n"
        f'  ("yaad nahi", "pata nahi", "kuch tha shayad") \u2014 NEVER fabricate a plausible-sounding story.\n'
        f"- When RELEVANT PAST CHAT is provided, use ONLY facts explicitly stated there. Do not extrapolate.\n"
        f"- Don\u2019t invent plot details just because they sound consistent with what you know about a person.\n"
        f"- Don\u2019t explain yourself or add meta-commentary.\n"
        f"- One reply only \u2014 no options, no alternatives, no \u2018or maybe\u2019.\n"
        f"- Vary reply length dramatically \u2014 sometimes 1\u20133 chars, sometimes 20\u201340 chars. Match the energy."
    )


def persona_summarize_conversation(name: str, transcript: str) -> str:
    """Return a single-turn summarization prompt for rolling context compression.

    ``transcript`` is the conversation formatted as "User: …\\n{name}: …" lines.
    The model is asked for 5–8 sentences of plain text covering topics, emotional arc,
    key facts, and unresolved threads.
    """
    return (
        f"Summarize this WhatsApp-style conversation between the user and {name}.\n"
        "Capture: topics discussed, emotional arc, key facts mentioned, unresolved threads.\n"
        "5-8 sentences max. Plain text.\n\n"
        f"{transcript}"
    )
