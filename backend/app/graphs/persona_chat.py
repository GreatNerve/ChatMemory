"""LangGraph — persona chat context (history router + fast retrieval) and generation + validation."""

from __future__ import annotations

from app.core.config import get_settings
from app.prompts.routing import rewrite_query_prompt
from app.prompts.validation import persona_regenerate_safe, persona_validate_factual_claims
from app.services import gemini as gemini_service
from app.services import history_router
from app.services import retrieval as retrieval_service
import json
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
import logging
import queue as stdlib_queue
import re
from typing import Any, Literal, TypedDict

logger = logging.getLogger("chatmemory.persona_chat_graph")


def _emit_stage(
    config: RunnableConfig,
    stage: str,
    status: str,
    input_data: dict[str, Any] | None = None,
    output_data: dict[str, Any] | None = None,
) -> None:
    """Put a stage event on the SSE queue if one was provided via config.

    Silently no-ops when called from non-streaming paths (stage_queue absent).
    Each node emits "running" at entry and "done" at exit so the frontend can
    light up each stage in real time as the context graph executes.
    """
    stage_queue: stdlib_queue.Queue | None = (config.get("configurable") or {}).get("stage_queue")
    if stage_queue is not None:
        try:
            stage_queue.put_nowait(
                {
                    "type": "stage",
                    "stage": stage,
                    "status": status,
                    "input": input_data,
                    "output": output_data,
                }
            )
        except Exception:
            pass  # never crash the graph if the queue is full or closed


# Graph 1: context routing + retrieval


class PersonaContextState(TypedDict, total=False):
    workspace_id: str
    person_id: str
    person_display_name: str
    user_message: str
    history: list[dict[str, str]]
    fast_route: Literal["casual", "memory", "ambiguous"]
    needs_history: bool
    # Set by the router (classify_history_need or fast-path heuristic), not by
    # a word-count guard in the graph node.  True when the raw user message
    # contains unresolved pronouns or implicit context references that make it
    # a poor standalone retrieval query.
    needs_rewrite: bool
    # Temporal direction of the query, set by classify_history_need.
    # Controls recency boost direction in the retrieval scoring pipeline:
    #   "current"    → full boost (recent chunks rank higher)
    #   "historical" → reversed boost (older chunks rank higher)
    #   "neutral"    → half boost (default when ambiguous)
    query_intent: str
    # Multi-query list replaces single search_query — enables cross-language
    # Hinglish↔English retrieval by running parallel embedding searches.
    search_queries: list[str]
    memory_blocks: list[str]
    # Rewritten standalone query for short follow-ups like "kaha?" or "kab?".
    # Only retrieval uses this; the original user_message is still sent to generation
    # so the persona replies naturally without seeing the expanded query.
    rewritten_query: str


def _node_fast_route(state: PersonaContextState, config: RunnableConfig) -> PersonaContextState:
    user_msg = state["user_message"]
    _emit_stage(config, "route", "running", input_data={"message": user_msg})
    route = history_router.fast_history_route(user_msg)
    _emit_stage(config, "route", "done", input_data={"message": user_msg}, output_data={"route": route})
    return {"fast_route": route}


def _node_classify(state: PersonaContextState, config: RunnableConfig) -> PersonaContextState:
    # Unpack the extended 4-tuple: the router now returns query_intent so the graph
    # can pass the temporal direction straight through to the retrieval layer.
    user_msg = state["user_message"]
    history = state.get("history") or []
    _emit_stage(config, "classify", "running", input_data={
        "message": user_msg,
        "context_turns": len(history),
    })
    needs, needs_rewrite, queries, query_intent = history_router.classify_history_need(user_msg, history)
    _emit_stage(config, "classify", "done",
                input_data={"message": user_msg, "context_turns": len(history)},
                output_data={"needs_history": needs, "needs_rewrite": needs_rewrite,
                             "query_intent": query_intent, "search_queries": queries})
    return {"needs_history": needs, "needs_rewrite": needs_rewrite,
            "query_intent": query_intent, "search_queries": queries}


def _node_retrieve(state: PersonaContextState, config: RunnableConfig) -> PersonaContextState:
    settings = get_settings()
    # Use Gemini-generated multi-queries when available; fall back to the raw user message.
    queries: list[str] = state.get("search_queries") or [state["user_message"].strip()]
    workspace_id = state["workspace_id"]
    person_id = state["person_id"]
    person_name = state["person_display_name"]
    # Temporal direction set by classify — controls recency boost in retrieval scoring.
    query_intent: str = state.get("query_intent") or "neutral"

    _emit_stage(config, "retrieve", "running", input_data={
        "queries": queries,
        "query_count": len(queries),
        "query_intent": query_intent,
    })

    # fast_retrieve and expand_to_turn_windows are fully synchronous; no asyncio.run
    # or gpu_lock wrapper needed here — the GPU lock is only acquired by the async
    # FastAPI route layer (ingest/train jobs) before calling into this sync graph.
    hits = retrieval_service.fast_retrieve(
        workspace_id,
        queries,
        person_id,
        person_name,
        query_intent=query_intent,
    )

    # Use the lower cross-language score gate for expand_to_turn_windows when
    # multi-query mode was used, so borderline matches aren't dropped a second time.
    min_score = (
        settings.persona_memory_inject_min_score_cross_lang
        if len(queries) > 1
        else settings.persona_memory_inject_min_score
    )

    blocks = retrieval_service.expand_to_turn_windows(
        workspace_id,
        hits,
        window_before=settings.persona_memory_window_before,
        window_after=settings.persona_memory_window_after,
        max_blocks=settings.persona_memory_max_blocks,
        # Pass the person being discussed so blocks that don't mention them are dropped,
        # providing a second line of defence on top of the score gate in fast_retrieve.
        target_person=person_name,
        min_hit_score=min_score,
    )

    # Truncate top_snippet to 80 chars to keep SSE payloads small.
    top_snippet = (blocks[0][:80] + "…") if blocks and len(blocks[0]) > 80 else (blocks[0] if blocks else None)
    _emit_stage(config, "retrieve", "done",
                input_data={"queries": queries, "query_count": len(queries)},
                output_data={"blocks_retrieved": len(blocks), "top_snippet": top_snippet})
    return {"memory_blocks": blocks}


def _after_fast_route(state: PersonaContextState) -> str:
    route = state.get("fast_route", "ambiguous")
    if route == "casual":
        return "skip_retrieve"
    if route == "memory":
        return "retrieve"
    return "classify"


def _after_classify(state: PersonaContextState) -> str:
    if state.get("needs_history"):
        return "maybe_rewrite_query"
    return "skip_retrieve"


def _node_skip_retrieve(state: PersonaContextState) -> PersonaContextState:
    return {"memory_blocks": []}


def _node_prepare_memory_route(state: PersonaContextState, config: RunnableConfig) -> PersonaContextState:
    # Fast "memory" path — user message matched a memory pattern heuristically.
    # Messages on this path are by definition follow-ups about past facts, so they
    # always benefit from query rewriting (needs_rewrite=True unconditionally).
    # For very short follow-ups (≤4 words) we also expand the query by appending
    # the last assistant turn's content as an additional retrieval hint, so the
    # retrieval layer gets a context-enriched query rather than an isolated
    # pronoun-heavy fragment like "kaha lagi?".
    # No Gemini call; this is a cheap heuristic that runs in-process.
    msg = state["user_message"].strip()
    history = state.get("history") or []

    # Emit as "classify" stage since this node is the fast-path substitute for
    # _node_classify on the "memory" route — same user-visible purpose.
    _emit_stage(config, "classify", "running", input_data={
        "message": msg,
        "context_turns": len(history),
    })

    queries: list[str] = [msg]
    if history and len(msg.split()) <= 4:
        # Grab the last two turns for context (last assistant reply + previous user turn).
        last_turns = history[-2:]
        context_snippet = " ".join(t.get("content", "") for t in last_turns)
        # Cap at 200 chars to avoid bloating the embedding query.
        combined = f"{msg} {context_snippet}".strip()[:200]
        if combined != msg:
            queries.append(combined)

    _emit_stage(config, "classify", "done",
                input_data={"message": msg, "context_turns": len(history)},
                output_data={"needs_history": True, "needs_rewrite": True, "search_queries": queries})
    return {
        "needs_history": True,
        # Memory-path messages are follow-ups about past facts by definition —
        # always rewrite so retrieval gets a context-resolved query.
        "needs_rewrite": True,
        # Memory fast-path questions are typically "what happened / what is the status"
        # style — default to "current" so recent chunks rank higher for follow-ups
        # like "lag gayi?" that are asking about the current outcome of a past topic.
        "query_intent": "current",
        "search_queries": queries,
    }


def _node_maybe_rewrite_query(state: PersonaContextState, config: RunnableConfig) -> PersonaContextState:
    """Rewrite pronoun-heavy or context-dependent queries into 2-3 standalone search variations.

    Activated only when the router has set both needs_history=True AND
    needs_rewrite=True.  The rewrite decision is made upstream (by
    classify_history_need or the fast memory-path heuristic) rather than by a
    crude word-count guard here — so longer messages like "aur wahan ka experience
    kaisa tha?" are also rewritten when they contain unresolved references.

    A single low-temperature Gemini call returns a JSON array of 2-3 query
    variations for e.g. "kaha?" after an internship discussion:
        ["internship location city", "intern company location", "where internship"]
    so BM25/Chroma can run parallel searches across all phrasings, catching
    cross-language Hinglish↔English matches that a single query would miss.

    CRITICAL: the prompt instructs the model NOT to fill in facts (company names,
    dates, locations) — only to describe what to search for.  This prevents the
    rewrite from hallucinating facts like filling in the wrong organisation name
    inferred from the persona's display name rather than the actual chat content.

    The original user_message is preserved untouched for the generation step so the
    persona replies naturally without seeing the rewritten form.  All variations and
    the original are kept in search_queries so neither path loses recall.
    """
    user_msg = state["user_message"].strip()
    history = state.get("history") or []

    # Skip rewrite when the router did not request it — avoids an unnecessary
    # Gemini call for self-contained questions that retrieval can handle as-is.
    if not state.get("needs_history") or not state.get("needs_rewrite"):
        return {"rewritten_query": user_msg}

    # Build a concise context block from the last 4 live conversation turns.
    # When history is empty (first turn or client omitted it), fall back to the
    # classify-generated search_queries as topic hints — they already encode
    # the context the classify step resolved, and are far better than "(no prior
    # turns)" which causes the model to hallucinate unrelated topics.
    recent = history[-4:] if len(history) >= 4 else history
    context_lines: list[str] = []
    for turn in recent:
        role = "User" if turn.get("role") == "user" else state.get("person_display_name", "Persona")
        context_lines.append(f"{role}: {turn.get('content', '')}")

    if context_lines:
        context_block = "\n".join(context_lines)
    else:
        # No live history available — use the classify-generated queries as a
        # lightweight context proxy so the rewrite model has some topic signal.
        existing_queries = state.get("search_queries") or []
        if existing_queries:
            context_block = "Topic context (from query classifier): " + "; ".join(existing_queries[:3])
        else:
            context_block = "(no prior turns)"

    # First 100 chars of context block for the stage event input (avoids huge payloads).
    context_snippet = context_block[:100] + ("…" if len(context_block) > 100 else "")
    _emit_stage(config, "rewrite", "running", input_data={
        "original": user_msg,
        "context_snippet": context_snippet,
    })

    prompt = rewrite_query_prompt(user_msg, context_block)

    # Parse the JSON array of query variations returned by the model.
    # Fall back to just the original message if parsing fails for any reason.
    variations: list[str] = []
    primary = user_msg  # safe fallback for rewritten_query
    try:
        raw = gemini_service.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        ).strip()

        # Strip markdown code fences the model may add despite the instructions.
        clean = raw
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(
                lines[1:-1] if lines and lines[-1].strip() in ("```", "") else lines[1:]
            )

        parsed = json.loads(clean.strip())
        if isinstance(parsed, list):
            variations = [str(v).strip() for v in parsed if str(v).strip()]
            if variations:
                primary = variations[0]
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Query rewrite JSON parse failed for message=%r, falling back to original: %s",
            user_msg,
            exc,
        )
    except Exception as exc:
        logger.warning("Query rewrite failed for message=%r, using original: %s", user_msg, exc)

    # Sanity-check: if the user message is very short (≤3 words) AND none of the
    # content words from the most-recent 2 live turns appear in any rewritten query,
    # the rewrite almost certainly hallucinated an unrelated topic (e.g. "ey?" →
    # "how to build trust in a conversation").  Discard it so the better
    # classify-generated search_queries carry retrieval instead.
    if variations and len(user_msg.split()) <= 3 and history:
        last_2_text = " ".join(t.get("content", "") for t in history[-2:]).lower()
        context_words = {w for w in re.split(r"\W+", last_2_text) if len(w) > 2}
        all_var_text = " ".join(variations).lower()
        if context_words and not any(w in all_var_text for w in context_words):
            logger.warning(
                "Query rewrite sanity-check failed for %r — none of the recent-turn "
                "words appear in rewritten queries %r; discarding rewrite, keeping "
                "original + classify queries.",
                user_msg,
                variations,
            )
            variations = []
            primary = user_msg

    # Merge: interleave rewrite variations with classify-generated queries so BM25
    # gets the Hinglish classify queries tried early rather than buried at the end.
    # Pattern: [rewrite[0], classify[0], rewrite[1], classify[1], …, original]
    # Cap at 8 total (deduped) to keep retrieval latency bounded.
    MAX_QUERIES = 8
    existing = list(state.get("search_queries") or [])
    seen: set[str] = set()
    merged: list[str] = []

    interleave_len = max(len(variations), len(existing))
    for i in range(interleave_len):
        for source in (variations, existing):
            if i < len(source):
                q = source[i]
                if q and q not in seen:
                    seen.add(q)
                    merged.append(q)
    # Always include the original as a fallback anchor if not already present.
    if user_msg and user_msg not in seen:
        merged.append(user_msg)
    # Hard cap to avoid retrieval slowdown.
    merged = merged[:MAX_QUERIES]

    logger.info(
        "Query rewrite: %r → %r (total search_queries=%d ws=%s)",
        user_msg,
        primary,
        len(merged),
        state.get("workspace_id", "")[:8],
    )
    _emit_stage(config, "rewrite", "done",
                input_data={"original": user_msg, "context_snippet": context_snippet},
                output_data={"rewritten_query": primary, "all_queries": merged})
    return {"rewritten_query": primary, "search_queries": merged}


def _build_context_graph():
    graph = StateGraph(PersonaContextState)
    graph.add_node("fast_route", _node_fast_route)
    graph.add_node("prepare_memory_route", _node_prepare_memory_route)
    graph.add_node("classify", _node_classify)
    graph.add_node("maybe_rewrite_query", _node_maybe_rewrite_query)
    graph.add_node("retrieve", _node_retrieve)
    graph.add_node("skip_retrieve", _node_skip_retrieve)

    graph.add_edge(START, "fast_route")
    graph.add_conditional_edges(
        "fast_route",
        _after_fast_route,
        {
            "skip_retrieve": "skip_retrieve",
            "retrieve": "prepare_memory_route",
            "classify": "classify",
        },
    )
    # memory fast-path → rewrite → retrieve
    graph.add_edge("prepare_memory_route", "maybe_rewrite_query")
    graph.add_conditional_edges(
        "classify",
        _after_classify,
        {
            "maybe_rewrite_query": "maybe_rewrite_query",
            "skip_retrieve": "skip_retrieve",
        },
    )
    graph.add_edge("maybe_rewrite_query", "retrieve")
    graph.add_edge("retrieve", END)
    graph.add_edge("skip_retrieve", END)
    return graph.compile()


_compiled_context_graph = None


def run_persona_context(
    workspace_id: str,
    person_id: str,
    person_display_name: str,
    user_message: str,
    history: list[dict[str, str]],
    *,
    stage_queue: stdlib_queue.Queue | None = None,
) -> PersonaContextState:
    """Run two-stage router and optional fast retrieval for persona chat.

    stage_queue: when provided (SSE streaming path), each graph node emits stage
    events into this queue so the SSE stream can forward them to the frontend in
    real time.  Pass None (default) for non-streaming callers — nodes silently skip
    the emit calls.
    """
    global _compiled_context_graph
    if _compiled_context_graph is None:
        _compiled_context_graph = _build_context_graph()

    initial: PersonaContextState = {
        "workspace_id": workspace_id,
        "person_id": person_id,
        "person_display_name": person_display_name,
        "user_message": user_message,
        "history": history,
        "memory_blocks": [],
        "needs_history": False,
        "needs_rewrite": False,
        "query_intent": "neutral",
        "search_queries": [],
        "rewritten_query": "",
    }
    graph_config: dict = {}
    if stage_queue is not None:
        graph_config = {"configurable": {"stage_queue": stage_queue}}
    result = _compiled_context_graph.invoke(initial, graph_config)
    logger.info(
        "Persona context route=%s needs_history=%s blocks=%d ws=%s",
        result.get("fast_route"),
        result.get("needs_history"),
        len(result.get("memory_blocks") or []),
        workspace_id[:8],
    )
    return result


# Graph 2: generation + factual validation


class PersonaGenerationState(TypedDict, total=False):
    person_name: str
    messages: list[dict[str, str]]  # full turn list including system prompt
    memory_blocks: list[str]
    # Persona background knowledge (personality_notes + chat_analysis) passed to the
    # validator so it doesn't flag names/facts that the persona legitimately knows.
    persona_background: str
    temperature: float
    previous_interaction_id: str | None
    reply: str
    interaction_id: str | None  # captured from the initial generate call
    has_hallucination: bool
    hallucination_reason: str
    regeneration_attempt: int  # 0 = not yet regenerated


def _gen_generate_reply(state: PersonaGenerationState) -> PersonaGenerationState:
    """Call Gemini for the initial persona reply."""
    interaction_ids: list[str] = []
    text = gemini_service.chat(
        state["messages"],
        temperature=state.get("temperature", 0.85),
        previous_interaction_id=state.get("previous_interaction_id"),
        assistant_label=state.get("person_name", "Assistant"),
        interaction_id_out=interaction_ids,
    ).strip()
    return {
        "reply": text,
        "interaction_id": interaction_ids[0] if interaction_ids else None,
        "regeneration_attempt": 0,
    }


def _gen_validate_factual_claims(state: PersonaGenerationState) -> PersonaGenerationState:
    """Second Gemini call that checks whether the reply invents facts absent from context.

    Structured output: {"has_hallucination": bool, "reason": str}
    Fails safe (no hallucination flagged) on any error so the flow never crashes.

    Skipped entirely for casual/short replies — there are no facts to hallucinate
    in "hn", "lol", "yaad nahi", or similar one-liners.
    """
    reply = state.get("reply", "")

    # Skip validation for casual or very short replies — they carry no verifiable facts.
    # fast_history_route is a cheap regex check, no Gemini call required.
    if len(reply.strip()) < 20 or history_router.fast_history_route(reply) == "casual":
        logger.debug(
            "Skipping hallucination validation for short/casual reply (person=%s)",
            state.get("person_name"),
        )
        return {"has_hallucination": False, "hallucination_reason": ""}

    memory_blocks = state.get("memory_blocks") or []
    messages = state.get("messages") or []
    persona_background = (state.get("persona_background") or "").strip()

    # Build a compact conversation history string (skip the system turn).
    history_lines: list[str] = []
    for msg in messages:
        if msg.get("role") == "system":
            continue
        label = "User" if msg.get("role") == "user" else state.get("person_name", "Person")
        history_lines.append(f"{label}: {msg.get('content', '')}")
    history_text = "\n".join(history_lines) if history_lines else "(none)"

    memory_blocks_text = "\n\n---\n".join(memory_blocks) if memory_blocks else "(none)"

    validation_prompt = persona_validate_factual_claims(
        persona_background, memory_blocks_text, history_text, reply
    )

    has_hallucination = False
    reason = ""
    validation_response = ""
    try:
        validation_response = gemini_service.chat(
            [{"role": "user", "content": validation_prompt}],
            temperature=0.1,  # low temperature for deterministic fact-checking
        ).strip()
        # Strip markdown code fences that the model may still add despite instructions.
        clean = validation_response
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(
                lines[1:-1] if lines and lines[-1].strip() in ("```", "") else lines[1:]
            )
        data = json.loads(clean.strip())
        has_hallucination = bool(data.get("has_hallucination", False))
        reason = str(data.get("reason", ""))
        if has_hallucination:
            logger.warning(
                "Hallucination detected in persona reply (person=%s): %s",
                state.get("person_name"),
                reason,
            )
    except json.JSONDecodeError as exc:
        logger.warning(
            "Validation response not valid JSON (skipping): %s | raw=%.200s",
            exc,
            validation_response,
        )
    except gemini_service.GeminiError as exc:
        logger.warning("Validation Gemini call failed (skipping): %s", exc)
    except Exception as exc:
        logger.warning("Unexpected validation error (skipping): %s", exc)

    return {"has_hallucination": has_hallucination, "hallucination_reason": reason}


def _after_validate(state: PersonaGenerationState) -> str:
    """Route to regeneration only on the first hallucination; pass through after that."""
    if state.get("has_hallucination") and state.get("regeneration_attempt", 0) == 0:
        return "regenerate"
    return "end"


def _gen_regenerate_safe(state: PersonaGenerationState) -> PersonaGenerationState:
    """Regenerate with a gentle, targeted note prepended to the system prompt.

    Only attempted once.  If this second attempt is also flagged the reply passes
    through with a warning so the user always gets a response.

    Uses the specific hallucination reason so the note is minimal and targeted —
    the persona voice is preserved rather than being strangled by a blanket ban.
    """
    messages = list(state.get("messages") or [])

    # Extract the topic from the hallucination reason for a targeted note.
    # Fall back to a generic phrase so the note always makes sense.
    raw_reason = (state.get("hallucination_reason") or "").strip()
    topic = raw_reason if raw_reason else "unverified specific events"
    regen_note = persona_regenerate_safe(topic)

    # Prepend the targeted note to the existing system prompt.
    augmented: list[dict[str, str]] = []
    for msg in messages:
        if msg.get("role") == "system":
            augmented.append(
                {
                    "role": "system",
                    "content": regen_note + msg.get("content", ""),
                }
            )
        else:
            augmented.append(msg)

    interaction_ids: list[str] = []
    text = gemini_service.chat(
        augmented,
        temperature=max(0.0, (state.get("temperature") or 0.85) - 0.2),
        previous_interaction_id=state.get("previous_interaction_id"),
        assistant_label=state.get("person_name", "Assistant"),
        interaction_id_out=interaction_ids,
    ).strip()

    logger.info(
        "Regenerated safe reply for person=%s (hallucination reason: %s)",
        state.get("person_name"),
        state.get("hallucination_reason"),
    )
    return {
        "reply": text,
        "interaction_id": interaction_ids[0] if interaction_ids else state.get("interaction_id"),
        "regeneration_attempt": 1,
    }


def _build_generation_graph():
    graph = StateGraph(PersonaGenerationState)
    graph.add_node("generate_reply", _gen_generate_reply)
    graph.add_node("validate_factual_claims", _gen_validate_factual_claims)
    graph.add_node("regenerate_safe", _gen_regenerate_safe)

    graph.add_edge(START, "generate_reply")
    graph.add_edge("generate_reply", "validate_factual_claims")
    graph.add_conditional_edges(
        "validate_factual_claims",
        _after_validate,
        {
            "regenerate": "regenerate_safe",
            "end": END,
        },
    )
    graph.add_edge("regenerate_safe", END)
    return graph.compile()


_compiled_generation_graph = None


def run_persona_generation(
    person_name: str,
    messages: list[dict[str, str]],
    memory_blocks: list[str],
    *,
    persona_background: str = "",
    temperature: float = 0.85,
    previous_interaction_id: str | None = None,
    interaction_id_out: list[str] | None = None,
) -> str:
    """Generate a validated persona reply via LangGraph.

    Flow: generate_reply → validate_factual_claims → (regenerate_safe if hallucinated) → END

    persona_background should contain the persona's personality_notes and chat_analysis so
    the validator doesn't flag names and facts already known from the persona's profile.

    Returns the validated reply string.  If Gemini is not configured or validation
    itself fails the original reply is returned unchanged so callers always get text.
    """
    global _compiled_generation_graph
    if _compiled_generation_graph is None:
        _compiled_generation_graph = _build_generation_graph()

    initial: PersonaGenerationState = {
        "person_name": person_name,
        "messages": messages,
        "memory_blocks": memory_blocks,
        "persona_background": persona_background,
        "temperature": temperature,
        "previous_interaction_id": previous_interaction_id,
        "reply": "",
        "interaction_id": None,
        "has_hallucination": False,
        "hallucination_reason": "",
        "regeneration_attempt": 0,
    }
    result = _compiled_generation_graph.invoke(initial)

    reply_text = result.get("reply", "").strip()
    interaction_id = result.get("interaction_id")
    if interaction_id and interaction_id_out is not None:
        interaction_id_out.append(interaction_id)

    logger.info(
        "Persona generation done (person=%s hallucination=%s regenerated=%s)",
        person_name,
        result.get("has_hallucination"),
        bool(result.get("regeneration_attempt", 0)),
    )
    return reply_text
