"""Google Gemini chat for persona replies and RAG answer generation."""

from __future__ import annotations

from app.core.config import get_settings
from collections.abc import Iterator
from google import genai
from google.genai import errors as genai_errors
import logging
import os
from typing import Any

logger = logging.getLogger("chatmemory.gemini")

GEMINI_MODEL_TAG = "gemini"

_cached_client: genai.Client | None = None


class GeminiError(Exception):
    pass


class GeminiNotConfiguredError(GeminiError):
    pass


def is_gemini_model_name(model_name: str | None) -> bool:
    return model_name == GEMINI_MODEL_TAG


def _resolve_api_key() -> str:
    settings = get_settings()
    key = settings.gemini_api_key.strip()
    if key:
        return key
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if key:
        return key
    return ""


def is_configured() -> bool:
    return bool(_resolve_api_key())


def config_status() -> tuple[bool, str | None]:
    if is_configured():
        return True, None
    return (
        False,
        "Set GEMINI_API_KEY (or GOOGLE_API_KEY) in backend/.env to enable persona chat and RAG answers.",
    )


def warmup_gemini_client() -> bool:
    """Create genai.Client once at startup when API key is configured."""
    global _cached_client
    if not is_configured():
        return False
    if _cached_client is not None:
        return True
    try:
        _cached_client = genai.Client(api_key=_resolve_api_key())
        logger.info("Gemini client ready")
        return True
    except Exception as exc:
        logger.warning("Gemini warmup failed: %s", exc)
        return False


def _require_configured() -> str:
    key = _resolve_api_key()
    if not key:
        raise GeminiNotConfiguredError(
            "GEMINI_API_KEY is not set. Add it to backend/.env (or set GOOGLE_API_KEY) and restart the API."
        )
    return key


def _genai_client() -> genai.Client:
    global _cached_client
    if _cached_client is not None:
        return _cached_client
    _cached_client = genai.Client(api_key=_require_configured())
    return _cached_client


def _flatten_history(
    history: list[dict[str, str]],
    *,
    assistant_label: str = "Assistant",
) -> str:
    lines: list[str] = []
    for turn in history:
        label = "User" if turn["role"] == "user" else assistant_label
        lines.append(f"{label}: {turn['content']}")
    return "\n".join(lines)


def _messages_to_interaction_input(
    messages: list[dict[str, str]],
    *,
    previous_interaction_id: str | None = None,
    assistant_label: str = "Assistant",
) -> tuple[str | None, str, str | None]:
    """Map chat messages to Interactions API input (single string + optional previous id).

    The Interactions API does not accept legacy multi-turn Turn lists — use
    previous_interaction_id for follow-ups, or flatten prior turns into one prompt.
    """
    system_text: str | None = None
    turns: list[dict[str, str]] = []
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        else:
            turns.append({"role": msg["role"], "content": msg["content"]})

    if not turns or turns[-1]["role"] != "user":
        raise GeminiError("No user message to send")

    user_message = turns[-1]["content"]
    history = turns[:-1]

    if previous_interaction_id:
        # Gemini holds server-side state for this chain; send only the new user
        # message plus the interaction ID.  We also inline the flattened history
        # as a "Recent chat:" prefix so Gemini has full context even if its
        # server-side state has drifted or the model needs a reminder — this is
        # belt-and-suspenders and does not hurt well-functioning chains.
        if history:
            prior = _flatten_history(history, assistant_label=assistant_label)
            combined = f"Recent chat:\n{prior}\n\nUser: {user_message}"
            return system_text, combined, previous_interaction_id
        return system_text, user_message, previous_interaction_id

    if history:
        prior = _flatten_history(history, assistant_label=assistant_label)
        combined = f"Recent chat:\n{prior}\n\nUser: {user_message}"
        return system_text, combined, None

    return system_text, user_message, None


def _extract_output_text(interaction: Any) -> str:
    text = getattr(interaction, "output_text", None)
    if text:
        return text

    parts: list[str] = []
    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", None) != "model_output":
            continue
        for content in getattr(step, "content", None) or []:
            if getattr(content, "type", None) == "text":
                chunk = getattr(content, "text", "")
                if chunk:
                    parts.append(chunk)

    if parts:
        return "".join(parts)

    status = getattr(interaction, "status", None)
    if status == "failed":
        err = getattr(interaction, "error", None)
        detail = getattr(err, "message", None) if err else None
        raise GeminiError(detail or "Gemini interaction failed")
    raise GeminiError("Gemini returned no output text")


def _raise_from_genai_error(exc: Exception) -> None:
    message = getattr(exc, "message", None) or str(exc)
    raise GeminiError(message) from exc


def _interaction_kwargs(
    *,
    input_value: str,
    system_instruction: str | None,
    temperature: float,
    previous_interaction_id: str | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    kwargs: dict[str, Any] = {
        "model": settings.gemini_model,
        "input": input_value,
        "generation_config": {
            "temperature": temperature,
            "thinking_level": "minimal",
        },
    }
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    if previous_interaction_id:
        kwargs["previous_interaction_id"] = previous_interaction_id
    return kwargs


def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.75,
    previous_interaction_id: str | None = None,
    assistant_label: str = "Assistant",
    interaction_id_out: list[str] | None = None,
) -> str:
    system_instruction, input_value, prev_id = _messages_to_interaction_input(
        messages,
        previous_interaction_id=previous_interaction_id,
        assistant_label=assistant_label,
    )
    client = _genai_client()
    try:
        interaction = client.interactions.create(
            **_interaction_kwargs(
                input_value=input_value,
                system_instruction=system_instruction,
                temperature=temperature,
                previous_interaction_id=prev_id,
            )
        )
    except (genai_errors.ClientError, genai_errors.ServerError, genai_errors.APIError) as exc:
        _raise_from_genai_error(exc)
    interaction_id = getattr(interaction, "id", None)
    if interaction_id and interaction_id_out is not None:
        interaction_id_out.append(interaction_id)
    return _extract_output_text(interaction)


def chat_stream(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.75,
    previous_interaction_id: str | None = None,
    assistant_label: str = "Assistant",
    interaction_id_out: list[str] | None = None,
) -> Iterator[str]:
    system_instruction, input_value, prev_id = _messages_to_interaction_input(
        messages,
        previous_interaction_id=previous_interaction_id,
        assistant_label=assistant_label,
    )
    client = _genai_client()
    try:
        stream = client.interactions.create(
            stream=True,
            **_interaction_kwargs(
                input_value=input_value,
                system_instruction=system_instruction,
                temperature=temperature,
                previous_interaction_id=prev_id,
            ),
        )
    except (genai_errors.ClientError, genai_errors.ServerError, genai_errors.APIError) as exc:
        _raise_from_genai_error(exc)

    yielded = False
    interaction_id: str | None = None
    completed_interaction: Any | None = None

    for event in stream:
        event_type = getattr(event, "event_type", None)
        if event_type == "step.delta":
            delta = getattr(event, "delta", None)
            if delta is None or getattr(delta, "type", None) != "text":
                continue
            text = getattr(delta, "text", None)
            if text:
                yielded = True
                yield text
        elif event_type == "interaction.completed":
            interaction = getattr(event, "interaction", None)
            if interaction is not None:
                completed_interaction = interaction
                interaction_id = getattr(interaction, "id", None)
        elif event_type == "error":
            err = getattr(event, "error", None)
            code = getattr(err, "code", None) if err else None
            message = getattr(err, "message", None) if err else None
            detail = message or code or "Gemini stream error"
            raise GeminiError(str(detail))

    if not yielded and completed_interaction is not None:
        fallback = _extract_output_text(completed_interaction)
        if fallback:
            logger.warning(
                "Gemini stream had no text deltas; used interaction.completed output (%d chars)",
                len(fallback),
            )
            yield fallback
            yielded = True

    if not yielded:
        logger.error(
            "Gemini stream completed with no text (interaction_id=%s, had_prev=%s)",
            interaction_id,
            bool(prev_id),
        )
        raise GeminiError("Gemini returned an empty response")

    if interaction_id and interaction_id_out is not None:
        interaction_id_out.append(interaction_id)
