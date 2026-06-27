"""LangChain chat models for RAG — wraps Gemini Interactions API via gemini.py."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.services import gemini as gemini_service


def _to_gemini_messages(messages: list[BaseMessage]) -> list[dict[str, str]]:
    converted: list[dict[str, str]] = []
    for message in messages:
        role = message.type
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"
        elif role == "system":
            pass
        else:
            role = "user"
        content = message.content
        if isinstance(content, str):
            text = content
        else:
            text = str(content)
        converted.append({"role": role, "content": text})
    return converted


class GeminiInteractionsChat(BaseChatModel):
    """Thin LangChain adapter over the official google-genai Interactions API."""

    temperature: float = 0.3

    @property
    def _llm_type(self) -> str:
        return "gemini-interactions"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        text = gemini_service.chat(
            _to_gemini_messages(messages),
            temperature=self.temperature,
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def get_chat_model(*, temperature: float = 0.3) -> BaseChatModel:
    if not gemini_service.is_configured():
        raise gemini_service.GeminiNotConfiguredError(
            gemini_service.config_status()[1]
            or "GEMINI_API_KEY is not set in backend/.env"
        )
    return GeminiInteractionsChat(temperature=temperature)


def uses_gemini() -> bool:
    return gemini_service.is_configured()
