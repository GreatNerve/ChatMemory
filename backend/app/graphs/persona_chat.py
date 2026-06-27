"""LangGraph — persona chat context (history router + fast retrieval) and generation + validation."""

from __future__ import annotations

import json
import logging
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.core.config import get_settings
from app.services import gemini as gemini_service
from app.services import history_router
from app.services import retrieval as retrieval_service

logger = logging.getLogger("chatmemory.persona_chat_graph")


# ---------------------------------------------------------------------------
# Graph 1: context routing + retrieval
# ---------------------------------------------------------------------------


class PersonaContextState(TypedDict, total=False):
    workspace_id: str
    person_id: str
    person_display_name: str
    user_message: str
    history: list[dict[str, str]]
    fast_route: Literal["casual", "memory", "ambiguous"]
    needs_history: bool
    search_query: str
    memory_blocks: list[str]


def _node_fast_route(state: PersonaContextState) -> PersonaContextState:
    route = history_router.fast_history_route(state["user_message"])
    return {"fast_route": route}


def _node_classify(state: PersonaContextState) -> PersonaContextState:
    needs, query = history_router.classify_history_need(
        state["user_message"],
        state.get("history") or [],
    )
    return {"needs_history": needs, "search_query": query}


def _node_retrieve(state: PersonaContextState) -> PersonaContextState:
    settings = get_settings()
    query = (state.get("search_query") or state["user_message"]).strip()
    workspace_id = state["workspace_id"]
    person_id = state["person_id"]
    person_name = state["person_display_name"]

    # fast_retrieve and expand_to_turn_windows are fully synchronous; no asyncio.run
    # or gpu_lock wrapper needed here — the GPU lock is only acquired by the async
    # FastAPI route layer (ingest/train jobs) before calling into this sync graph.
    hits = retrieval_service.fast_retrieve(
        workspace_id,
        query,
        person_id,
        person_name,
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
        min_hit_score=settings.persona_memory_inject_min_score,
    )
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
        return "retrieve"
    return "skip_retrieve"


def _node_skip_retrieve(state: PersonaContextState) -> PersonaContextState:
    return {"memory_blocks": []}


def _node_prepare_memory_route(state: PersonaContextState) -> PersonaContextState:
    return {
        "needs_history": True,
        "search_query": state["user_message"].strip(),
    }


def _build_context_graph():
    graph = StateGraph(PersonaContextState)
    graph.add_node("fast_route", _node_fast_route)
    graph.add_node("prepare_memory_route", _node_prepare_memory_route)
    graph.add_node("classify", _node_classify)
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
    graph.add_edge("prepare_memory_route", "retrieve")
    graph.add_conditional_edges(
        "classify",
        _after_classify,
        {
            "retrieve": "retrieve",
            "skip_retrieve": "skip_retrieve",
        },
    )
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
) -> PersonaContextState:
    """Run two-stage router and optional fast retrieval for persona chat."""
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
        "search_query": "",
    }
    result = _compiled_context_graph.invoke(initial)
    logger.info(
        "Persona context route=%s needs_history=%s blocks=%d ws=%s",
        result.get("fast_route"),
        result.get("needs_history"),
        len(result.get("memory_blocks") or []),
        workspace_id[:8],
    )
    return result


# ---------------------------------------------------------------------------
# Graph 2: generation + factual validation
# ---------------------------------------------------------------------------

_STRICT_RECALL_PREFIX = (
    "STRICT RECALL RULE: Your previous response contained invented facts not found in the chat history. "
    "Do NOT mention specific events, actions, technical details, or things people did unless they appear "
    'verbatim in the RELEVANT PAST CHAT section above or the current conversation. '
    'If you don\'t know something specific, reply vaguely in character: "pata nahi", "yaad nahi", '
    '"kuch aisa hi tha shayad" — NOT a fabricated story.\n\n'
)


class PersonaGenerationState(TypedDict, total=False):
    person_name: str
    messages: list[dict[str, str]]          # full turn list including system prompt
    memory_blocks: list[str]
    # Persona background knowledge (personality_notes + chat_analysis) passed to the
    # validator so it doesn't flag names/facts that the persona legitimately knows.
    persona_background: str
    temperature: float
    previous_interaction_id: str | None
    reply: str
    interaction_id: str | None              # captured from the initial generate call
    has_hallucination: bool
    hallucination_reason: str
    regeneration_attempt: int               # 0 = not yet regenerated


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
    """
    reply = state.get("reply", "")
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
    conversation_text = "\n".join(history_lines) if history_lines else "(none)"

    memory_text = "\n\n---\n".join(memory_blocks) if memory_blocks else "(none)"

    # Build the background section only when background knowledge is available so the
    # validator understands which names / facts the persona legitimately knows from their
    # own chat analysis without those being flagged as hallucinations.
    background_section = ""
    if persona_background:
        background_section = (
            f"PERSONA BACKGROUND KNOWLEDGE (from their real chat history analysis — "
            f"names, people, places, and topics mentioned here are legitimate):\n"
            f"{persona_background}\n\n"
        )

    validation_prompt = (
        "You are a strict fact-checker for AI-generated chat messages.\n\n"
        f"{background_section}"
        f"RETRIEVED MEMORY EXCERPTS (specific snippets retrieved for this query):\n{memory_text}\n\n"
        f"CURRENT CONVERSATION:\n{conversation_text}\n\n"
        f"GENERATED REPLY:\n{reply}\n\n"
        "Does the reply contain specific factual claims — specific events, names, actions, dates, "
        "technical details, project names, things someone did or said — that are NOT present in "
        "any of the above sources (background knowledge, memory excerpts, or current conversation)?\n\n"
        'Respond with JSON only, no markdown: {"has_hallucination": true/false, "reason": "brief explanation"}'
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
    """Regenerate with the STRICT RECALL prefix prepended to the system prompt.

    Only attempted once.  If this second attempt is also flagged the reply passes
    through with a warning so the user always gets a response.
    """
    messages = list(state.get("messages") or [])

    # Prepend the strict recall instruction to the existing system prompt.
    augmented: list[dict[str, str]] = []
    for msg in messages:
        if msg.get("role") == "system":
            augmented.append({
                "role": "system",
                "content": _STRICT_RECALL_PREFIX + msg.get("content", ""),
            })
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
