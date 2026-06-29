"""Persona chat — Gemini style mimic with local RAG context."""

from __future__ import annotations

from app.core.schemas import PersonaChatDebugMeta, PersonDetail
from app.graphs.persona_chat import run_persona_context, run_persona_generation
from app.prompts.persona_chat import (
    conversation_partner_block,
    persona_summarize_conversation,
    persona_system_prompt,
)
from app.services import gemini as gemini_service
from app.services import workspace as workspace_service
from app.services.parser.whatsapp import Message
from app.services.rate_limit import _rate_limiter, estimate_tokens
from collections.abc import Iterator
from functools import lru_cache
import json
import logging
import queue
import re
import time

logger = logging.getLogger("chatmemory.persona_chat")

# Lighter retrieval is handled by graphs/persona_chat (two-stage router + fast retrieve).
_SOLO_EXAMPLE_LIMIT = 8
_CONVO_SNIPPET_LIMIT = 4
_CONVO_WINDOW = 6

# Burst-message protocol: Gemini separates multiple messages with this token.
# Chosen to be visually obvious in prompts and unlikely in natural chat.
_BURST_SEP = "||"

# History window: more turns = better recall of what's already been said, but costs more
# tokens. We prefer 30 turns and fall back to 20 if the raw char volume is too high.
_MAX_HISTORY_TURNS = 30
_FALLBACK_HISTORY_TURNS = 20
_MAX_HISTORY_CHARS = 8000

# How long (seconds) to pause between burst messages — simulates switching tabs to type.
_BURST_PAUSE = 0.7

# Delay between words — simulates realistic typing speed.
_WORD_DELAY = 0.04


def _typing_fingerprint_to_text(fp: dict) -> str:
    """Convert a structured typingFingerprint dict to compact readable text for the prompt.

    Example output:
    "caps: mostly_lowercase | abbreviations: nhi=nahi, yr=yaar, kr=kar | emojis: 🥹😭 |
     punctuation: skips periods; occasional ? and ! | emphasis: elongation (fstttt, noiceee)"
    """
    parts: list[str] = []
    if fp.get("capsStyle"):
        parts.append(f"caps: {fp['capsStyle']}")
    abbrevs: list[dict] = fp.get("abbreviations") or []
    if abbrevs:
        abbrev_str = ", ".join(
            f"{a['from']}={a['to']}" for a in abbrevs[:8] if a.get("from") and a.get("to")
        )
        if abbrev_str:
            parts.append(f"abbreviations: {abbrev_str}")
    emojis: list[str] = fp.get("emojis") or []
    if emojis:
        parts.append(f"emojis: {''.join(str(e) for e in emojis[:8])}")
    if fp.get("punctuation"):
        parts.append(f"punctuation: {fp['punctuation']}")
    emphasis = (fp.get("emphasisStyle") or "").strip()
    if emphasis and emphasis.lower() != "none":
        parts.append(f"emphasis: {emphasis}")
    if fp.get("avgMessageLength"):
        parts.append(f"avg length: {fp['avgMessageLength']} chars")
    return " | ".join(parts) if parts else ""


def _voice_samples_to_dialogue(samples: list[dict]) -> str:
    """Format voiceSamples list as readable dialogue blocks for the system prompt.

    Each sample becomes:
      [context label]
      Sender: text
      Sender: text
      …
    Blocks are separated by a blank line.
    """
    blocks: list[str] = []
    for sample in samples:
        context = (sample.get("context") or "").strip()
        exchange: list[dict] = sample.get("exchange") or []
        if not exchange:
            continue
        lines: list[str] = []
        if context:
            lines.append(f"[{context}]")
        for msg in exchange:
            sender = msg.get("sender", "")
            text = msg.get("text", "")
            if sender and text:
                lines.append(f"{sender}: {text}")
        if len(lines) > 1:  # at least context label + 1 message
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _persona_background(person: PersonDetail) -> str:
    """Build a compact background string for the hallucination validator.

    Includes both v2 and legacy fields when both are present — after the new
    training pipeline both sets are generated together. The validator uses this
    to know which names, people, and topics the persona legitimately knows.
    """
    parts: list[str] = []

    # v2: relationship dynamic + emotional profile + response patterns
    if person.relationship_dynamic:
        parts.append(f"Relationship dynamic:\n{person.relationship_dynamic}")
    if person.emotional_profile:
        parts.append(f"Emotional profile:\n{person.emotional_profile}")
    if person.response_patterns:
        parts.append(f"Response patterns:\n{person.response_patterns}")

    # Legacy: personality, chat analysis, listening style — supplement v2 when present
    if person.personality_notes:
        parts.append(f"Personality notes:\n{person.personality_notes}")
    if person.chat_analysis:
        parts.append(f"Chat analysis:\n{person.chat_analysis}")
    if person.active_listening_style:
        parts.append(f"Active listening style:\n{person.active_listening_style}")

    return "\n\n".join(parts)


def _require_gemini_persona(person: PersonDetail) -> None:
    if not gemini_service.is_gemini_model_name(person.ollama_model_name):
        raise gemini_service.GeminiError(
            "Legacy persona model is no longer supported. Re-activate the persona from the UI."
        )


def _solo_examples(person: PersonDetail, limit: int = _SOLO_EXAMPLE_LIMIT) -> list[str]:
    """Style samples from activation — not query-time retrieval."""
    seen: set[str] = set()
    out: list[str] = []
    for sample in person.sample_messages:
        text = (sample.text or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
        if len(out) >= limit:
            break
    return out[:limit]


def _conversation_snippets(
    timeline: list[Message],
    person_name: str,
    *,
    limit: int = _CONVO_SNIPPET_LIMIT,
    window: int = _CONVO_WINDOW,
) -> list[str]:
    """Back-and-forth excerpts ending with this person's message, spread across the chat."""
    candidates: list[str] = []
    for i, msg in enumerate(timeline):
        if msg.sender != person_name or not msg.text.strip():
            continue
        start = max(0, i - window)
        block = timeline[start : i + 1]
        if len(block) < 2:
            continue
        lines = [f"{m.sender}: {m.text}" for m in block]
        snippet = "\n".join(lines)
        if snippet not in candidates:
            candidates.append(snippet)

    if len(candidates) <= limit:
        return candidates

    step = max((len(candidates) - 1) // (limit - 1), 1)
    picked = [candidates[j] for j in range(0, len(candidates), step)]
    if picked[-1] != candidates[-1]:
        picked.append(candidates[-1])
    return sorted(set(picked), key=candidates.index)[:limit]


@lru_cache(maxsize=32)
def _cached_convo_snippets(workspace_id: str, person_name: str) -> tuple[str, ...]:
    try:
        timeline = workspace_service.load_export_timeline(workspace_id)
        return tuple(_conversation_snippets(timeline, person_name))
    except FileNotFoundError:
        return ()


def load_workspace_partners(workspace_id: str, exclude_person_id: str) -> list[dict]:
    """Load all other people in the workspace (excluding the current persona).

    Reads every .json file from ``data/workspaces/{workspace_id}/people/``,
    skips the file whose ``id`` matches ``exclude_person_id``, and returns the
    remaining raw dicts.  Returns an empty list if the directory is missing or
    all files fail to parse.  I/O is fast for typical 1-3 person workspaces.
    """
    from app.core.paths import workspace_path

    people_dir = workspace_path(workspace_id) / "people"
    if not people_dir.exists():
        return []

    partners: list[dict] = []
    for pf in sorted(people_dir.glob("*.json")):
        try:
            pdata = json.loads(pf.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to read partner file %s — skipping", pf)
            continue
        if pdata.get("id") == exclude_person_id:
            continue
        partners.append(pdata)
    return partners


def build_system_prompt(
    person: PersonDetail,
    solo_examples: list[str],
    convo_snippets: list[str],
    memory_blocks: list[str] | None = None,
    partners: list[dict] | None = None,
) -> str:
    """Assemble the persona system prompt from PersonDetail fields and example samples.

    Delegates the actual prompt text to ``app.prompts.persona_chat.persona_system_prompt``
    after extracting primitives and building the intermediate section strings.

    Parameters
    ----------
    partners:
        Raw person JSON dicts for every other person in the workspace.  When provided,
        a CONVERSATION PARTNER block is injected after the persona's profile sections
        so the model knows who it is talking to.  Pass ``None`` or ``[]`` for
        single-person workspaces (the block is silently omitted).
    """
    sp = person.style_profile
    avg_len = int(sp.avg_message_length)

    solo_block = (
        "\n".join(f"\u2022 {t}" for t in solo_examples) if solo_examples else "\u2022 (none)"
    )
    convo_block = (
        "\n\n---\n".join(convo_snippets) if convo_snippets else "(no conversation samples)"
    )

    # Derived casing hint: if avg length is short, they're terse; mention it explicitly.
    terse_note = " Most of their messages are under 15 chars." if avg_len <= 15 else ""

    # Build conversation partner context (empty string for single-person workspaces).
    partner_block = conversation_partner_block(partners) if partners else ""

    memory_section = ""
    if memory_blocks:
        joined = "\n\n---\n".join(memory_blocks)
        memory_section = (
            f"=== RELEVANT PAST CHAT (real WhatsApp history) ===\n"
            f"Use these excerpts for factual recall. Stay in character \u2014 weave facts naturally, "
            f"do not sound like you are reading a log.\n"
            f"{joined}\n\n"
        )

    # Determine which schema generation to use for the profile block.
    # Always build both v2 and legacy sections — the prompt assembler and cap logic
    # decide what to include. New fields are preferred when the cap fires.
    has_v2 = any([
        person.relationship_dynamic,
        person.typing_fingerprint,
        person.response_patterns,
        person.emotional_profile,
        person.voice_samples,
    ])

    # v2 profile sections (empty string when not trained with v2 pipeline yet).
    voice_style_parts: list[str] = []
    if person.typing_fingerprint:
        fp_text = _typing_fingerprint_to_text(person.typing_fingerprint)
        if fp_text:
            voice_style_parts.append(f"Typing rules: {fp_text}")
    if person.voice_samples:
        samples_text = _voice_samples_to_dialogue(person.voice_samples)
        if samples_text:
            voice_style_parts.append(f"Sample exchanges:\n{samples_text}")
    voice_style_section = (
        "=== VOICE & STYLE ===\n" + "\n\n".join(voice_style_parts) + "\n\n"
        if voice_style_parts else ""
    )

    relationship_section = (
        f"=== HOW YOU RELATE TO THEM ===\n{person.relationship_dynamic}\n\n"
        if person.relationship_dynamic else ""
    )

    behavioral_patterns_section = (
        f"=== YOUR BEHAVIORAL PATTERNS ===\n{person.response_patterns}\n\n"
        if person.response_patterns else ""
    )

    emotional_section = (
        f"=== YOUR EMOTIONAL STYLE ===\n{person.emotional_profile}\n\n"
        if person.emotional_profile else ""
    )

    # Legacy profile sections — always built when data exists (supplement v2 or sole source).
    personality_section = ""
    if person.personality_notes:
        personality_section = (
            f"Who they are (from analysing their real messages \u2014 let this shape every reply):\n"
            f"{person.personality_notes}\n"
            f"Do not describe these traits; embody them. Let them dictate word choice, energy, and pacing.\n\n"
        )

    writing_style_section = ""
    if person.writing_style_notes:
        writing_style_section = (
            f"Writing style (how they actually type \u2014 mirror this precisely in every reply):\n"
            f"{person.writing_style_notes}\n\n"
        )

    listening_style_section = ""
    if person.active_listening_style:
        listening_style_section = (
            f"How they listen and respond when others share problems or news "
            f"(mirror this in how you react \u2014 this is their actual pattern):\n"
            f"{person.active_listening_style}\n\n"
        )

    chat_analysis_section = ""
    if person.chat_analysis:
        chat_analysis_section = (
            f"Chat patterns (deep analysis of full message history \u2014 use this for richer context):\n"
            f"{person.chat_analysis}\n\n"
        )

    # --- Prompt size cap (excludes memory_section which varies per request) ---
    # Measure the non-memory content so we can warn early and truncate before
    # the sections are concatenated into an immutable string.
    _PROMPT_SIZE_WARN = 2500
    non_memory_size = (
        len(solo_block) + len(convo_block)
        + len(personality_section) + len(chat_analysis_section)
        + len(writing_style_section) + len(listening_style_section)
        + len(partner_block)
        + len(voice_style_section) + len(relationship_section)
        + len(behavioral_patterns_section) + len(emotional_section)
    )
    if non_memory_size > _PROMPT_SIZE_WARN:
        logger.warning(
            "Persona prompt non-memory content is large: %d chars (threshold=%d). "
            "Breakdown — personality=%d chat_analysis=%d writing=%d listening=%d "
            "partner=%d voice=%d relationship=%d behavioral=%d emotional=%d examples=%d",
            non_memory_size, _PROMPT_SIZE_WARN,
            len(personality_section), len(chat_analysis_section),
            len(writing_style_section), len(listening_style_section),
            len(partner_block),
            len(voice_style_section), len(relationship_section),
            len(behavioral_patterns_section), len(emotional_section),
            len(solo_block) + len(convo_block),
        )
        # Truncate in priority order — drop legacy (old) fields first since v2 fields
        # are preferred; trim v2 voice samples last as a last resort.
        if listening_style_section and non_memory_size > _PROMPT_SIZE_WARN:
            non_memory_size -= len(listening_style_section)
            listening_style_section = ""
            logger.warning("Prompt cap: dropped listening_style_section (saves ~%d chars)", len(listening_style_section))
        if chat_analysis_section and non_memory_size > _PROMPT_SIZE_WARN:
            non_memory_size -= len(chat_analysis_section)
            chat_analysis_section = ""
            logger.warning("Prompt cap: dropped chat_analysis_section (saves ~%d chars)", len(chat_analysis_section))
        if writing_style_section and non_memory_size > _PROMPT_SIZE_WARN:
            non_memory_size -= len(writing_style_section)
            writing_style_section = ""
            logger.warning("Prompt cap: dropped writing_style_section (saves ~%d chars)", len(writing_style_section))
        if personality_section and non_memory_size > _PROMPT_SIZE_WARN:
            non_memory_size -= len(personality_section)
            personality_section = ""
            logger.warning("Prompt cap: dropped personality_section (saves ~%d chars)", len(personality_section))
        # For v2 voice samples: trim to first 3 exchanges if still over cap.
        if voice_style_section and non_memory_size > _PROMPT_SIZE_WARN and person.voice_samples:
            capped_samples = _voice_samples_to_dialogue(person.voice_samples[:3])
            fp_text = _typing_fingerprint_to_text(person.typing_fingerprint) if person.typing_fingerprint else ""
            parts_v: list[str] = []
            if fp_text:
                parts_v.append(f"Typing rules: {fp_text}")
            if capped_samples:
                parts_v.append(f"Sample exchanges:\n{capped_samples}")
            if parts_v:
                voice_style_section = "=== VOICE & STYLE ===\n" + "\n\n".join(parts_v) + "\n\n"
            logger.warning("Prompt cap: trimmed voice_samples to 3 exchanges")

    return persona_system_prompt(
        name=person.display_name,
        personality_section=personality_section,
        chat_analysis_section=chat_analysis_section,
        writing_style_section=writing_style_section,
        listening_style_section=listening_style_section,
        partner_block=partner_block,
        memory_section=memory_section,
        avg_len=avg_len,
        hinglish_ratio=sp.hinglish_ratio,
        emoji_rate=sp.emoji_rate,
        terse_note=terse_note,
        solo_block=solo_block,
        convo_block=convo_block,
        burst_sep=_BURST_SEP,
        voice_style_section=voice_style_section,
        relationship_section=relationship_section,
        behavioral_patterns_section=behavioral_patterns_section,
        emotional_section=emotional_section,
    )


def _build_context(
    workspace_id: str,
    person: PersonDetail,
    user_message: str,
    history: list[dict[str, str]],
    *,
    stage_queue: queue.Queue | None = None,
) -> tuple[str, list[str], list[str], float, bool, list[str], str, PersonaChatDebugMeta]:
    """Build persona system prompt.

    Returns (system, solo, convo, context_ms, used_memory, memory_blocks,
             rewritten_query, debug_meta).
    memory_blocks is returned raw so callers can pass it to the validation graph.
    rewritten_query is the context-resolved standalone query used for retrieval (may equal
    user_message when no rewrite was performed), surfaced so callers can hint the persona
    about which memory angle was searched.
    debug_meta contains all routing decisions for the frontend thinking accordion.
    stage_queue: when provided, graph nodes emit stage events into this queue for SSE delivery.
    """
    t0 = time.perf_counter()
    memory_blocks: list[str] = []
    rewritten_query = ""
    used_memory = False
    debug_meta = PersonaChatDebugMeta()
    try:
        ctx = run_persona_context(
            workspace_id,
            person.id,
            person.display_name,
            user_message,
            history,
            stage_queue=stage_queue,
        )
        memory_blocks = list(ctx.get("memory_blocks") or [])
        used_memory = bool(memory_blocks)
        rewritten_query = ctx.get("rewritten_query", "") or ""
        debug_meta = PersonaChatDebugMeta(
            route=ctx.get("fast_route"),
            needs_history=ctx.get("needs_history"),
            needs_rewrite=ctx.get("needs_rewrite"),
            rewritten_query=rewritten_query or None,
            search_queries=list(ctx.get("search_queries") or []) or None,
            blocks_retrieved=len(memory_blocks),
        )
    except Exception as exc:
        logger.warning("Persona context graph failed: %s", exc)

    solo = _solo_examples(person)
    convo = list(_cached_convo_snippets(workspace_id, person.display_name))
    # Load partners once per request — fast I/O for 1-2 JSON files.
    partners = load_workspace_partners(workspace_id, person.id)
    system = build_system_prompt(person, solo, convo, memory_blocks=memory_blocks or None, partners=partners)
    context_ms = (time.perf_counter() - t0) * 1000
    return system, solo, convo, context_ms, used_memory, memory_blocks, rewritten_query, debug_meta


def summarize_conversation(
    person: PersonDetail,
    history: list[dict[str, str]],
) -> tuple[str, int]:
    """Summarize older chat turns via Gemini for rolling context compression."""
    if not history:
        raise gemini_service.GeminiError("No history to summarize")

    lines: list[str] = []
    for turn in history:
        label = "User" if turn["role"] == "user" else person.display_name
        lines.append(f"{label}: {turn['content']}")
    transcript = "\n".join(lines)

    prompt = persona_summarize_conversation(person.display_name, transcript)

    est = estimate_tokens(prompt)
    _rate_limiter.acquire(est)
    try:
        summary = gemini_service.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.4,
        ).strip()
    finally:
        _rate_limiter.record(est)

    if not summary:
        raise gemini_service.GeminiError("Gemini returned an empty summary")
    return summary, len(history)


def _chat_messages(
    workspace_id: str,
    person: PersonDetail,
    history: list[dict[str, str]],
    user_message: str,
    *,
    previous_interaction_id: str | None = None,
    conversation_summary: str | None = None,
    stage_queue: queue.Queue | None = None,
) -> tuple[list[dict[str, str]], list[str], PersonaChatDebugMeta]:
    """Build Gemini message list for a persona turn.

    Returns (turns, memory_blocks, debug_meta).  memory_blocks is passed separately to
    the generation + validation graph so it can check replies against injected context.
    debug_meta carries routing decisions for the frontend thinking accordion.
    stage_queue: forwarded to _build_context so graph nodes can emit stage SSE events.
    """
    # Every turn runs the two-stage router; follow-ups can still ask about past chat.
    system, _, _, context_ms, used_memory, memory_blocks, rewritten_query, debug_meta = (
        _build_context(workspace_id, person, user_message, history, stage_queue=stage_queue)
    )
    logger.info(
        "Persona context built in %.0fms (memory=%s, workspace=%s person=%s)",
        context_ms,
        used_memory,
        workspace_id,
        person.id,
    )

    # Smarter history window: up to 30 turns for richer recall, trimmed to 20 if the
    # combined char count is very large (rough guard against blowing the token budget).
    recent = history[-_MAX_HISTORY_TURNS:]
    if sum(len(t.get("content", "")) for t in recent) > _MAX_HISTORY_CHARS:
        recent = history[-_FALLBACK_HISTORY_TURNS:]

    # Surface explicit conversation-depth awareness so the model knows it is deep into
    # an exchange and should not recycle the same response patterns it has already used.
    n = len(history)
    if n > 6:
        system = (
            system + f"\n\nYou are {n} exchanges into this conversation. "
            "Do not repeat response patterns you've already used."
        )

    if conversation_summary:
        system = system + f"\n\nEarlier in this conversation (summarized):\n{conversation_summary}"

    turns: list[dict[str, str]] = [{"role": "system", "content": system}]
    for turn in recent:
        turns.append({"role": turn["role"], "content": turn["content"]})

    # When retrieval used a rewritten/context-resolved query that differs from the raw
    # user message, surface it as a brief inline hint so the persona knows which memory
    # angle was searched.  The actual question the persona answers is still user_message.
    # This only fires when rewrite happened (rewritten_query != user_message and non-empty).
    user_content = user_message
    if rewritten_query and rewritten_query.strip() != user_message.strip():
        user_content = f"[Memory searched for: {rewritten_query}]\n{user_message}"

    turns.append({"role": "user", "content": user_content})
    return turns, memory_blocks, debug_meta


def reply(
    workspace_id: str,
    person: PersonDetail,
    history: list[dict[str, str]],
    user_message: str,
    *,
    temperature: float = 0.85,
    previous_interaction_id: str | None = None,
    conversation_summary: str | None = None,
) -> tuple[str, str | None, PersonaChatDebugMeta]:
    _require_gemini_persona(person)
    messages, memory_blocks, debug_meta = _chat_messages(
        workspace_id,
        person,
        history,
        user_message,
        previous_interaction_id=previous_interaction_id,
        conversation_summary=conversation_summary,
    )
    interaction_ids: list[str] = []
    full_text = run_persona_generation(
        person.display_name,
        messages,
        memory_blocks,
        persona_background=_persona_background(person),
        temperature=temperature,
        previous_interaction_id=previous_interaction_id,
        interaction_id_out=interaction_ids,
    )
    if not full_text:
        raise gemini_service.GeminiError("Gemini returned an empty response")
    # Collapse burst parts to a single space for the non-streaming endpoint.
    # Normalize \n\n within each part so paragraph breaks don't survive into the output.
    text = " ".join(
        re.sub(r"\n{2,}", " ", p).strip()
        for p in full_text.split(_BURST_SEP)
        if p.strip()
    )
    return text, interaction_ids[0] if interaction_ids else None, debug_meta


def sse_stream(
    workspace_id: str,
    person: PersonDetail,
    history: list[dict[str, str]],
    user_message: str,
    *,
    previous_interaction_id: str | None = None,
    conversation_summary: str | None = None,
) -> Iterator[str]:
    """
    SSE generator that yields persona reply tokens with burst-message support.

    Protocol:
      {"status": "thinking"}          — first event, before API call
      {"type":"stage", "stage":"...", "status":"running"|"done", ...}
                                      — real-time stage progress events
      {"token": "<text>"}             — word-level tokens for current bubble
      {"msg_break": true}             — commit current bubble, start new one
      {"done": true, "interactionId": "..."}   — all bubbles complete
      {"error": "<message>"}          — on failure
    """
    t0 = time.perf_counter()
    interaction_ids: list[str] = []

    try:
        _require_gemini_persona(person)
        yield f"data: {json.dumps({'status': 'thinking'})}\n\n"

        # Thread-safe queue for stage events emitted by graph nodes during context execution.
        stage_q: queue.Queue[dict] = queue.Queue()

        messages, memory_blocks, debug_meta = _chat_messages(
            workspace_id,
            person,
            history,
            user_message,
            previous_interaction_id=previous_interaction_id,
            conversation_summary=conversation_summary,
            stage_queue=stage_q,
        )

        # Drain stage events accumulated during context graph execution and forward to SSE.
        # All context stages (route/classify/rewrite/retrieve) are emitted here, before tokens.
        while not stage_q.empty():
            try:
                ev = stage_q.get_nowait()
                yield f"data: {json.dumps(ev)}\n\n"
            except queue.Empty:
                break

        # Emit generate stage start before the generation graph runs.
        generate_input: dict = {
            "blocks_injected": len(memory_blocks),
            "rewrite_used": bool(
                debug_meta.rewritten_query
                and debug_meta.rewritten_query.strip() != user_message.strip()
            ),
        }
        yield f"data: {json.dumps({'type': 'stage', 'stage': 'generate', 'status': 'running', 'input': generate_input, 'output': None})}\n\n"

        # Use non-streaming generation so we can split burst messages before emitting,
        # and so the validation graph can inspect the full reply before delivery.
        full_text = run_persona_generation(
            person.display_name,
            messages,
            memory_blocks,
            persona_background=_persona_background(person),
            temperature=0.85,
            previous_interaction_id=previous_interaction_id,
            interaction_id_out=interaction_ids,
        )

        # Emit generate stage done (response_length only — full text comes via token events).
        yield f"data: {json.dumps({'type': 'stage', 'stage': 'generate', 'status': 'done', 'input': generate_input, 'output': {'response_length': len(full_text)}})}\n\n"

        first_response_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Persona response in %.0fms (workspace=%s person=%s)",
            first_response_ms,
            workspace_id,
            person.id,
        )

        # Split strictly on the explicit burst separator — never on double-newlines.
        # Normalize any stray multi-newlines within each part to a single space so
        # the model's paragraph breaks don't create visual message-breaks on the frontend.
        parts = [
            re.sub(r"\n{2,}", " ", p).strip()
            for p in full_text.split(_BURST_SEP)
            if p.strip()
        ]
        if not parts:
            raise gemini_service.GeminiError("Gemini returned an empty response")

        logger.info(
            "Persona burst: %d message(s) (workspace=%s person=%s)",
            len(parts),
            workspace_id,
            person.id,
        )

        for i, part in enumerate(parts):
            if i > 0:
                # Signal frontend to commit the current bubble and show typing indicator.
                yield f"data: {json.dumps({'msg_break': True})}\n\n"
                time.sleep(_BURST_PAUSE)

            # Stream word by word for realistic typing feel.
            words = part.split(" ")
            for j, word in enumerate(words):
                token = word if j == 0 else f" {word}"
                yield f"data: {json.dumps({'token': token})}\n\n"
                if j < len(words) - 1:
                    time.sleep(_WORD_DELAY)

        done_payload: dict[str, object] = {"done": True}
        if interaction_ids:
            done_payload["interactionId"] = interaction_ids[0]
        # Include routing debug metadata for the frontend thinking accordion.
        done_payload["debugMeta"] = json.loads(debug_meta.model_dump_json(by_alias=True))
        yield f"data: {json.dumps(done_payload)}\n\n"

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Persona stream done in %.0fms (workspace=%s)",
            total_ms,
            workspace_id,
        )

    except gemini_service.GeminiError as exc:
        logger.warning("Persona chat error: %s", exc)
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
    except Exception as exc:
        logger.exception("Persona chat failed")
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
