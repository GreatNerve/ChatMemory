"""ChatMemory prompt library — centralised LLM prompt strings.

Each sub-module owns one domain. Import from here for convenience,
or import directly from the sub-module for clarity.

Sub-modules
-----------
qa              Q&A pipeline: rewrite, rerank, grounded answer
persona_build   Build-time persona activation: personality, writing style, chat analysis, listening style
persona_chat    Runtime persona chat: system prompt assembly, conversation summarization
routing         History-router Gemini classify prompt
validation      Hallucination validate + safe-regeneration prompts
"""

from app.prompts.persona_build import (
    persona_extract_chat_analysis,
    persona_extract_chat_analysis_consolidate,
    persona_extract_listening_style,
    persona_extract_personality,
    persona_extract_writing_style,
)
from app.prompts.persona_chat import (
    persona_summarize_conversation,
    persona_system_prompt,
)
from app.prompts.qa import (
    qa_grounded_answer,
    qa_rerank_chunks,
    qa_rewrite_query,
)
from app.prompts.routing import persona_classify_history_need
from app.prompts.validation import persona_regenerate_safe, persona_validate_factual_claims

__all__ = [
    # qa
    "qa_rewrite_query",
    "qa_rerank_chunks",
    "qa_grounded_answer",
    # persona_build
    "persona_extract_personality",
    "persona_extract_writing_style",
    "persona_extract_chat_analysis",
    "persona_extract_chat_analysis_consolidate",
    "persona_extract_listening_style",
    # persona_chat
    "persona_system_prompt",
    "persona_summarize_conversation",
    # routing
    "persona_classify_history_need",
    # validation
    "persona_validate_factual_claims",
    "persona_regenerate_safe",
]
