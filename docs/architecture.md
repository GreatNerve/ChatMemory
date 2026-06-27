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

**Operator model:** single human on one machine. No accounts. Chat exports stay on disk; only Gemini API calls leave the machine (Q&A + persona chat).

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
├── docs/
├── backend/
│   ├── pyproject.toml
│   └── app/
│       ├── main.py
│       ├── api/routes/
│       ├── graphs/            # ingest, qa, persona_train
│       ├── services/          # parser, chroma, embed, gemini, rag_chain, jobs
│       └── core/              # config, gpu_lock, schemas
├── frontend/
└── data/                      # gitignored runtime root
```

## Layer responsibilities

### Frontend (`frontend/`)

- Neo-brutalism dark UI per [ui-design.md](./ui-design.md)
- TanStack Query for FastAPI; SSE for ingest/persona jobs
- **Does not** call Gemini, Chroma, or read `data/` directly

### LangGraph (`backend/app/graphs/`)

| Graph | Trigger | Sync/async |
|-------|---------|------------|
| `ingest` | Workspace create + file upload | Async job + SSE |
| `qa` | POST ask | Sync (seconds) |
| `persona_train` | POST train (Gemini activation) | Async job + SSE |

Persona **chat** is a direct route (`persona_chat` service), not a graph.

### Services (`backend/app/services/`)

| Service | Responsibility |
|---------|----------------|
| `parser` + `preprocess` | WhatsApp `.txt` → cleaned messages + speakers |
| `embed` | `multilingual-e5-large` via sentence-transformers (CUDA or CPU) |
| `chroma` | LangChain Chroma collection per workspace |
| `bm25` | Keyword index for hybrid retrieval |
| `langchain_llm` | `GeminiInteractionsChat` → `gemini.py` Interactions API |
| `rag_chain` | LangChain Gemini: rewrite, rerank, grounded answer |
| `gemini` | Low-level Interactions API; persona chat stream |
| `persona_chat` | System prompt + Gemini stream |
| `jobs` | Job registry, SSE progress |
| `workspace` | CRUD meta, paths on disk |

## Data flow

### Ingest

```
upload .txt → preprocess → parse → upsert people → chunk messages
  → multilingual-e5-large embed → Chroma upsert (+ BM25 corpus)
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
  → refresh samples + style profile
  → set personaStatus ready_model (provider gemini)
```

### Persona chat

```
message → style profile + samples + optional RAG context
  → Gemini chat → reply (style only, not factual lookup)
```

## GPU strategy

| Workload | Device | Notes |
|----------|--------|-------|
| Embedding at ingest | CUDA if available | Mutex via `gpu_lock` |
| Q&A embed query | CUDA if available | Same mutex |
| LLM (Q&A, persona) | Gemini API | No local GPU |

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
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8000` | frontend `.env.local` |

## Related docs

- [api.md](./api.md) — HTTP contract
- [data-layout.md](./data-layout.md) — disk schema
- [langgraph/](./langgraph/) — node-level flows
- [decisions.md](./decisions.md) — ADR log
