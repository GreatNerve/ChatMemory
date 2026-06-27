"""Persona chat — Gemini style mimic with local RAG context."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from functools import lru_cache

from app.core.schemas import PersonDetail
from app.services import embed as embed_service
from app.services import gemini as gemini_service
from app.services import vector_index as vector_service
from app.services import workspace as workspace_service
from app.services.parser.whatsapp import Message
from app.services.rate_limit import _rate_limiter, estimate_tokens

logger = logging.getLogger("chatmemory.persona_chat")

# Lighter retrieval + fewer snippets to keep prompts small and fast.
_PERSONA_RAG_TOP_K = 3
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


def _require_gemini_persona(person: PersonDetail) -> None:
    if not gemini_service.is_gemini_model_name(person.ollama_model_name):
        raise gemini_service.GeminiError(
            "Legacy persona model is no longer supported. Re-activate the persona from the UI."
        )


def _person_only_texts(person: PersonDetail, rag: list[dict], limit: int = _SOLO_EXAMPLE_LIMIT) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for sample in person.sample_messages:
        text = (sample.text or "").strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    for row in rag:
        text = (row.get("snippet") or "").strip()
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


def build_system_prompt(
    person: PersonDetail,
    solo_examples: list[str],
    convo_snippets: list[str],
) -> str:
    sp = person.style_profile
    avg_len = int(sp.avg_message_length)

    solo_block = "\n".join(f"• {t}" for t in solo_examples) if solo_examples else "• (none)"
    convo_block = (
        "\n\n---\n".join(convo_snippets) if convo_snippets else "(no conversation samples)"
    )

    # Personality notes integrated directly into style guidance when present.
    personality_section = ""
    if person.personality_notes:
        personality_section = (
            f"Who they are (from analysing their real messages — let this shape every reply):\n"
            f"{person.personality_notes}\n"
            f"Do not describe these traits; embody them. Let them dictate word choice, energy, and pacing.\n\n"
        )

    # Writing style notes: HOW they type — casing, punctuation, abbreviations, emoji patterns.
    # Extracted at build time from real messages; injected verbatim so the model mirrors exact habits.
    writing_style_section = ""
    if person.writing_style_notes:
        writing_style_section = (
            f"Writing style (how they actually type — mirror this precisely in every reply):\n"
            f"{person.writing_style_notes}\n\n"
        )

    # Deep chat-pattern analysis: vocabulary habits, recurring topics, emotional patterns,
    # relationship dynamics. Extracted from the full corpus via chunked analysis at build time.
    chat_analysis_section = ""
    if person.chat_analysis:
        chat_analysis_section = (
            f"Chat patterns (deep analysis of full message history — use this for richer context):\n"
            f"{person.chat_analysis}\n\n"
        )

    # Derived casing hint: if avg length is short, they're terse; mention it explicitly.
    terse_note = " Most of their messages are under 15 chars." if avg_len <= 15 else ""

    return (
        f"You are {person.display_name}. Reply as them, not as an AI.\n\n"
        f"=== WHO THEY ARE ===\n"
        f"{personality_section}"
        f"{chat_analysis_section}"
        f"Messaging fingerprint:\n"
        f"- Avg message length: ~{avg_len} chars (but length varies wildly — see examples){terse_note}\n"
        f"- Hinglish ratio: ~{sp.hinglish_ratio:.0%}\n"
        f"- Emoji use: ~{sp.emoji_rate:.1f} per message\n\n"
        f"{writing_style_section}"
        f"=== THEIR REAL MESSAGES ===\n"
        f"Study these for vocabulary, rhythm, casing, punctuation, and energy. Reproduce the style exactly:\n"
        f"{solo_block}\n\n"
        f"=== REAL CONVERSATIONS (they are {person.display_name}) ===\n"
        f"{convo_block}\n\n"
        f"=== REPLY RULES (follow every one) ===\n\n"
        f"LENGTH — vary dramatically:\n"
        f"- A quick confirmation or reaction → 1–4 chars (\"Hn\", \"k\", \"Nope\", \"lol\")\n"
        f"- A simple answer → 5–15 chars\n"
        f"- An involved reply → 20–50 chars max\n"
        f"- Match the energy of what's being said. Short question → short reply. Don't pad.\n"
        f"- Never default to ~{avg_len} chars every time; that's the average, not the rule.\n\n"
        f"CASING & PUNCTUATION:\n"
        f"- Look at the real messages above — if they rarely capitalise, you rarely capitalise.\n"
        f"- If they skip punctuation (no full stops, no question marks), you skip it too.\n"
        f"- Reproduce the casing pattern exactly as seen in the examples. Don't regularise it.\n\n"
        f"VOCABULARY:\n"
        f"- Use their abbreviations: yr=yaar, hn=haan, nhi=nahi, sb=sab, kl=kal, bta=bata, etc.\n"
        f"- Only use words and phrases that appear in their real messages or close variants.\n"
        f"- No greetings, no 'sure', no 'of course', no 'absolutely', no sign-offs.\n\n"
        f"SENTENCE STYLE:\n"
        f"- Prefer fragments over complete sentences. If they say 'Nhi ho rha' not 'No, it's not happening', do the same.\n"
        f"- Never produce a grammatically complete formal sentence if the person never does in their real messages.\n"
        f"- No AI hedges, no politeness filler, no explanations unless they naturally explain things.\n\n"
        f"BURST MESSAGES (use {_BURST_SEP!r} as separator):\n"
        f"- Only burst when it feels like they hit send and then had a second thought or reaction.\n"
        f"- A burst is NOT two halves of one planned sentence split across messages.\n"
        f"- A burst IS: a reaction then a follow-up, or a statement then an afterthought.\n"
        f"- When in doubt, send one message. Do not force bursts.\n"
        f"- Example of authentic burst: Hn yr||kl kab hai\n"
        f"- Example of forced/wrong burst: Nhi ho rha||mujhse nhi hoga (this is just one thought)\n\n"
        f"CONVERSATION FLOW:\n"
        f"- Never give the same type of reply more than twice in a row — vary your reaction even if your position stays the same\n"
        f"- If the other person is clearly emotional or escalating, shift tone — not necessarily gentler, but different (curious, blunt, tired, direct)\n"
        f"- Conversations move forward — after 3–4 exchanges on the same point, change angle or close the topic\n"
        f"- React to what's actually being said in this specific message, not just the general situation\n\n"
        f"HARD RULES:\n"
        f"- Don't invent facts, names, events, or plans not visible in the conversation.\n"
        f"- Don't explain yourself or add meta-commentary.\n"
        f"- One reply only — no options, no alternatives, no 'or maybe'.\n"
        f"- Vary reply length dramatically — sometimes 1–3 chars, sometimes 20–40 chars. Match the energy."
    )


def _build_context(
    workspace_id: str,
    person: PersonDetail,
    user_message: str,
    *,
    skip_rag: bool = False,
) -> tuple[str, list[str], list[str], float]:
    """Build persona system prompt parts. Returns (system, solo, convo, context_ms)."""
    t0 = time.perf_counter()
    rag: list[dict] = []
    if not skip_rag:
        try:
            query_vec = embed_service.embed_query(user_message)
            rag = vector_service.semantic_search(
                workspace_id,
                query_vec,
                top_k=_PERSONA_RAG_TOP_K,
                person_id=person.id,
            )
        except Exception as exc:
            logger.warning("Persona RAG lookup failed: %s", exc)

    solo = _person_only_texts(person, rag)
    convo = list(_cached_convo_snippets(workspace_id, person.display_name))
    system = build_system_prompt(person, solo, convo)
    context_ms = (time.perf_counter() - t0) * 1000
    return system, solo, convo, context_ms


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

    prompt = (
        f"Summarize this WhatsApp-style conversation between the user and {person.display_name}.\n"
        "Capture: topics discussed, emotional arc, key facts mentioned, unresolved threads.\n"
        "5-8 sentences max. Plain text.\n\n"
        f"{transcript}"
    )

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
) -> list[dict[str, str]]:
    # Follow-up turns already carry chat state server-side — skip heavy RAG + rebuild.
    skip_rag = bool(previous_interaction_id)
    system, _, _, context_ms = _build_context(
        workspace_id, person, user_message, skip_rag=skip_rag
    )
    logger.info(
        "Persona context built in %.0fms (rag=%s, workspace=%s person=%s)",
        context_ms,
        not skip_rag,
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
            system
            + f"\n\nYou are {n} exchanges into this conversation. "
            "Do not repeat response patterns you've already used."
        )

    if conversation_summary:
        system = (
            system
            + f"\n\nEarlier in this conversation (summarized):\n{conversation_summary}"
        )

    turns: list[dict[str, str]] = [{"role": "system", "content": system}]
    for turn in recent:
        turns.append({"role": turn["role"], "content": turn["content"]})
    turns.append({"role": "user", "content": user_message})
    return turns


def reply(
    workspace_id: str,
    person: PersonDetail,
    history: list[dict[str, str]],
    user_message: str,
    *,
    temperature: float = 0.85,
    previous_interaction_id: str | None = None,
    conversation_summary: str | None = None,
) -> tuple[str, str | None]:
    _require_gemini_persona(person)
    messages = _chat_messages(
        workspace_id,
        person,
        history,
        user_message,
        previous_interaction_id=previous_interaction_id,
        conversation_summary=conversation_summary,
    )
    interaction_ids: list[str] = []
    full_text = gemini_service.chat(
        messages,
        temperature=temperature,
        previous_interaction_id=previous_interaction_id,
        assistant_label=person.display_name,
        interaction_id_out=interaction_ids,
    ).strip()
    if not full_text:
        raise gemini_service.GeminiError("Gemini returned an empty response")
    # Collapse burst separator to a space for the non-streaming single-reply endpoint.
    text = " ".join(p.strip() for p in full_text.split(_BURST_SEP) if p.strip())
    return text, interaction_ids[0] if interaction_ids else None


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

        messages = _chat_messages(
            workspace_id,
            person,
            history,
            user_message,
            previous_interaction_id=previous_interaction_id,
            conversation_summary=conversation_summary,
        )

        # Use non-streaming call so we can split burst messages before emitting.
        full_text = gemini_service.chat(
            messages,
            temperature=0.85,
            previous_interaction_id=previous_interaction_id,
            assistant_label=person.display_name,
            interaction_id_out=interaction_ids,
        ).strip()

        first_response_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Persona response in %.0fms (workspace=%s person=%s)",
            first_response_ms,
            workspace_id,
            person.id,
        )

        # Split on burst separator; filter empty parts.
        parts = [p.strip() for p in full_text.split(_BURST_SEP) if p.strip()]
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
