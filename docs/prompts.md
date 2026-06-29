# ChatMemory — Prompt Library

All LLM prompt strings live under `backend/app/prompts/`. Each domain gets its own module; `__init__.py` re-exports everything. No prompt string should appear inline in a service or graph file.

## Package layout

```
backend/app/prompts/
├── __init__.py          # re-exports all prompt functions
├── qa.py               # Q&A / Ask pipeline prompts
├── persona_build.py    # Persona activation/build prompts (workspace.py)
├── persona_chat.py     # Persona chat runtime prompts (system prompt, summarize)
├── routing.py          # history_router classify prompt
└── validation.py       # hallucination validate + regenerate prompts
```

## Prompt function reference

### `qa.py` — Q&A pipeline

| Function | Returns | Pipeline step | Caller |
|----------|---------|---------------|--------|
| `qa_rewrite_query(question)` | `tuple[str, str]` (system, user) | Query rewrite — Gemini expands the question for vector search | `services/rag_chain.rewrite_query` |
| `qa_rerank_chunks(question, numbered_snippets)` | `str` | LLM rerank — scores retrieved snippets 0–1 for relevance | `services/rag_chain.rerank_chunks` |
| `qa_grounded_answer(question, context)` | `tuple[str, str]` (system, user) | Final answer — grounded strictly in retrieved chat messages | `services/rag_chain.grounded_answer` |

### `persona_build.py` — Build-time persona activation

Called during `graphs/persona_train` → `workspace.py` refresh functions. All use `gemini_service.chat([{"role": "user", "content": prompt}], ...)` with the shared rate limiter.

| Function | Returns | Purpose | Caller |
|----------|---------|---------|--------|
| `persona_extract_personality(name, samples)` | `str` | 3–6 sentence third-person personality keynote | `workspace.refresh_person_personality` |
| `persona_extract_writing_style(name, samples)` | `str` | 3–5 sentence surface typing-pattern description (HOW they type) | `workspace.refresh_person_writing_style` |
| `persona_extract_chat_analysis(name, chunk, chunk_num, total_chunks)` | `str` | 3–5 bullet observations per message chunk (vocabulary, topics, tone, dynamics) | `workspace.refresh_person_chat_analysis` (per chunk) |
| `persona_extract_chat_analysis_consolidate(name, analyses, num_chunks)` | `str` | Synthesises per-chunk observations into 5–10 sentence chat-pattern analysis | `workspace.refresh_person_chat_analysis` (consolidation) |
| `persona_extract_listening_style(name, samples)` | `str` | 3–5 sentence reactive listening behaviour (NOT generic empathy) | `workspace.refresh_person_listening_style` |

### `persona_chat.py` — Runtime persona chat

| Function | Returns | Purpose | Caller |
|----------|---------|---------|--------|
| `persona_system_prompt(name, personality_section, chat_analysis_section, writing_style_section, listening_style_section, memory_section, avg_len, hinglish_ratio, emoji_rate, terse_note, solo_block, convo_block, burst_sep)` | `str` | Full system prompt assembly from pre-built section strings + style metrics | `services/persona_chat.build_system_prompt` |
| `persona_summarize_conversation(name, transcript)` | `str` | Rolling context compression — 5–8 sentence WhatsApp conversation summary | `services/persona_chat.summarize_conversation` |

`build_system_prompt` in `services/persona_chat.py` is the thin assembly layer: it extracts primitives from `PersonDetail` and calls `persona_system_prompt`.

### `routing.py` — History router

| Function | Returns | Purpose | Caller |
|----------|---------|---------|--------|
| `persona_classify_history_need(context_block, user_message)` | `str` | Gemini structured classify — `{"needs_history": bool, "search_query": str}` for ambiguous turns | `services/history_router.classify_history_need` |

### `validation.py` — Hallucination validation

| Function | Returns | Purpose | Caller |
|----------|---------|---------|--------|
| `persona_validate_factual_claims(persona_background, memory_blocks_text, history_text, reply)` | `str` | Validates generated reply for invented facts absent from all sources; returns JSON `{"has_hallucination": bool, "reason": str}` | `graphs/persona_chat._gen_validate_factual_claims` |
| `persona_regenerate_safe(topic)` | `str` | Targeted note prepended to system prompt on first hallucination detection; signals the model to be vague about the specific topic | `graphs/persona_chat._gen_regenerate_safe` |

## Conventions

- Prompt functions are **pure** — no imports from `services/` or `graphs/`. They may import from `app.core.*` only if needed (currently none do).
- All functions return `str` or `tuple[str, str]` (system, user). No LangChain message objects.
- Callers own the message-list construction (`SystemMessage`, `HumanMessage`) and the LLM call.
- Temperature choices live in the caller, not the prompt function.
- Unicode special characters (→, –, •, etc.) are embedded as `\uXXXX` escapes to avoid encoding issues on Windows.
