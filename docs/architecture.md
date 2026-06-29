# ChatMemory — Architecture

Technical design: local monorepo, Next.js UI, FastAPI + LangGraph backend, Gemini LLM, Chroma + local embeddings.

## Overview

```
┌──────────────────┐     REST + SSE      ┌─────────────────────────────┐
│  Next.js :3000   │ ──────────────────► │  FastAPI :8000              │
│  (browser UI)    │                     │  LangGraph · services       │
└──────────────────┘                     └───────────┬─────────────────┘
                                                   │
                     ┌─────────────────────────────┼─────────────────────────────┐
                     │                             │                             │
                     ▼                             ▼                             ▼
            ┌────────────────┐           ┌────────────────┐           ┌────────────────┐
            │  ./data/       │           │  Chroma        │           │  Google Gemini │
            │  workspaces    │           │  per workspace │           │  (API)         │
            └────────────────┘           └────────────────┘           └────────────────┘
                                                   │
                                                   ▼
                                          ┌────────────────┐
                                          │  CUDA / CPU    │
                                          │  e5-large embed│
                                          └────────────────┘
```

**Operator model:** single human on one machine. No accounts. Chat exports stay on disk; only Gemini API calls leave the machine (Q&A, persona build analysis, persona chat, summarization).

## Process layout

| Process | Port | Role |
|---------|------|------|
| Next.js dev server | 3000 | UI only |
| FastAPI (uvicorn) | 8000 | API, LangGraph, embed, job orchestration |

Two terminals in development. Browser talks only to Next.js and (via fetch) FastAPI.

## Monorepo structure

```
ChatMemory/
├── AGENTS.md
├── CONTEXT.md
├── COMMANDS.md
├── .pre-commit-config.yaml    # ruff on backend/; install from repo root
├── docs/
├── backend/
│   ├── pyproject.toml
│   └── app/
│       ├── main.py
│       ├── api/routes/
│       ├── graphs/            # ingest, qa, persona_train, persona_chat
│       ├── prompts/           # centralised LLM prompt strings (qa, persona_build, persona_chat, routing, validation)
│       ├── services/          # parser, chroma, embed, gemini, rag_chain, jobs, analytics
│       └── core/              # config, gpu_lock, schemas
├── frontend/
└── data/                      # gitignored runtime root
```

**Git:** single monorepo at repo root (not `backend/.git` only). `.agents/` and `no-push/` are gitignored.

## Layer responsibilities

### Frontend (`frontend/`)

- Neo-brutalism dark UI per [ui-design.md](./ui-design.md)
- TanStack Query for FastAPI; SSE for ingest/persona jobs and persona chat stream
- **Does not** call Gemini, Chroma, or read `data/` directly

### LangGraph (`backend/app/graphs/`)

| Graph | Trigger | Sync/async |
|-------|---------|------------|
| `ingest` | Workspace create + file upload | Async job + SSE |
| `qa` | POST ask | Sync (seconds) |
| `persona_train` | POST train (Gemini activation) | Async job + SSE |
| `persona_chat` | POST chat / chat/stream | Sync (seconds) |

Persona **summarize** remains a direct route on the `persona_chat` service (no graph).

### Prompts (`backend/app/prompts/`)

Centralised LLM prompt strings — no inline prompt text in services or graphs. See [prompts.md](./prompts.md) for the full function reference.

| Module | Domain |
|--------|--------|
| `qa.py` | Q&A rewrite, rerank, grounded answer |
| `persona_build.py` | Personality, writing style, chat analysis, listening style extraction |
| `persona_chat.py` | System prompt assembly, conversation summarization |
| `routing.py` | History-router Gemini classify |
| `validation.py` | Hallucination validate + safe-regeneration note |

### Services (`backend/app/services/`)

| Service | Responsibility |
|---------|----------------|
| `parser` + `preprocess` | WhatsApp `.txt` → cleaned messages + speakers |
| `embed` | `multilingual-e5-large` via sentence-transformers (CUDA or CPU) |
| `chroma` | LangChain Chroma collection per workspace |
| `bm25` | Keyword index for hybrid retrieval (cached per workspace; cleared on ingest) |
| `retrieval` | Persona fast retrieve + turn-window expansion |
| `history_router` | Two-stage persona memory router (heuristics + Gemini classify) |
| `langchain_llm` | `GeminiInteractionsChat` → `gemini.py` Interactions API |
| `rag_chain` | LangChain Gemini: rewrite, rerank, grounded answer |
| `gemini` | Low-level Interactions API; persona chat |
| `persona_chat` | System prompt, LangGraph context + generation, burst `||`, history window, summarization |
| `workspace` | CRUD meta, paths, build-time LLM extraction (personality, style, chat analysis) |
| `analytics` | Turn-based response times, weekly/monthly growth series, heatmap |
| `rate_limit` | Gemini RPM/TPM guard for build-time analysis calls |
| `jobs` | Job registry, SSE progress |

## Data flow

### Ingest

```
upload .txt → preprocess → parse → upsert people → chunk messages
  → multilingual-e5-large embed → Chroma upsert (+ BM25 corpus)
  → compute analytics → save analytics.json
  → update workspace stats
```

### Q&A (strict RAG)

```
question → qa_graph → rag_chain
  → Gemini query rewrite
  → Chroma semantic (top-20) + BM25 (top-20) → merge
  → Gemini LLM rerank → top-8
  → grade (≥2 chunks score ≥ 0.6)
  → if fail: NOT_FOUND + nearMisses
  → else Gemini grounded answer + citations
```

### Persona activation

```
train + consent → persona_train_graph
  → refresh samples (recency-biased monthly spread)
  → style profile (deterministic metrics)
  → chat analysis (~55%, Gemini, rate-limited, non-fatal on fail)
  → personality notes (~65%, recency-weighted sample)
  → writing style notes (~75%, recency-weighted sample)
  → activate → personaStatus ready_model
```

Build-time Gemini calls share a **14 RPM / 100k TPM** sliding-window limiter (`rate_limit.py`).

### Persona chat

Two LangGraph pipelines in `persona_chat.py`; orchestrated by `persona_chat` service. Full node detail: [langgraph/persona-chat.md](./langgraph/persona-chat.md).

**Context graph** (`run_persona_context`):

```
message + history
  → fast_route (heuristics: casual | memory | ambiguous)
  → casual → skip_retrieve
  → memory → retrieve (person-first Chroma + BM25, group fallback if weak)
  → ambiguous → classify (Gemini JSON) → retrieve OR skip_retrieve
  → expand_to_turn_windows (3 before + 2 after from export.txt)
  → memory_blocks → === RELEVANT PAST CHAT === in system prompt
```

**Generation graph** (`run_persona_generation`):

```
system + history + user message
  → generate_reply (Gemini)
  → validate_factual_claims (Gemini JSON hallucination check)
  → if flagged: regenerate_safe once (STRICT RECALL prefix)
  → reply; optional burst via || in SSE stream
```

Every turn runs the context router (no memory skip on follow-ups). Style comes from activation fields + samples; memory blocks are factual excerpts only. HARD RULES: no invented facts; vague ("yaad nahi") when evidence is missing. Low-score hits dropped at `persona_memory_inject_min_score` (default 0.35). Parser noise filter (`is_noise_message`) applies at index/read — not on raw export.

`previousInteractionId` chains Gemini Interactions API only; it does not skip memory routing.

When UI history exceeds **24 turns**, client calls `/chat/summarize`, keeps last **10** verbatim, passes `conversationSummary` on subsequent chats.

### Workspace analytics

Computed at end of ingest and cached in `analytics.json`. `GET .../analytics?refresh=true` recomputes from export.

- **Turn-based** reply stats: consecutive same-sender messages merged into one turn before measuring gaps.
- **Median** (not mean) for typical reply time — robust to burst outliers.
- Same-minute replies (0s gap, WhatsApp minute precision) count in `<1m` bucket.
- `weeklySeries` for conversation growth; UI toggles weekly (last 52 weeks, capped at today) vs monthly (all time, aggregated from weeks).
- `isGroup` on workspace: `speakerCount > 2` — UI shows "Group rhythm" vs "Conversation rhythm".

## GPU strategy

| Workload | Device | Notes |
|----------|--------|-------|
| Embedding at ingest | CUDA if available | Mutex via `gpu_lock` |
| Q&A embed query | CUDA if available | Same mutex |
| LLM (Q&A, persona build, persona chat) | Gemini API | No local GPU |

## Models

| Purpose | Model |
|---------|-------|
| Embeddings | `intfloat/multilingual-e5-large` (local, Hinglish-friendly) |
| Q&A + persona LLM | `gemini-3.5-flash` (configurable via `GEMINI_MODEL`) |
| Vector store | Chroma per workspace |

## Configuration

| Variable | Default | Where |
|----------|---------|-------|
| `DATA_ROOT` | `../data` | backend `.env` |
| `GEMINI_API_KEY` | — | backend `.env` (required) |
| `GEMINI_MODEL` | `gemini-3.5-flash` | backend `.env` |
| `EMBED_MODEL` | `intfloat/multilingual-e5-large` | backend `.env` |
| `VECTOR_STORE` | `chroma` | backend `.env` |
| `persona_memory_inject_min_score` | `0.35` | backend config |
| `persona_retrieve_top_k` | `8` | backend config |
| `persona_memory_window_before` / `after` | `3` / `2` | backend config |
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8000/api/v1` | frontend `.env.local` |

## Pre-commit

From repo root (via backend venv):

```bash
cd backend && uv sync
uv run pre-commit install -c ../.pre-commit-config.yaml
```

Hooks: trailing whitespace, EOF, YAML, large files; **ruff** lint + format on `backend/` only. Frontend ESLint is manual (`pnpm lint`).

## Related docs

- [api.md](./api.md) — HTTP contract
- [data-layout.md](./data-layout.md) — disk schema
- [langgraph/](./langgraph/) — node-level flows
- [decisions.md](./decisions.md) — ADR log
