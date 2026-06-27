# ChatMemory

Local-first WhatsApp chat RAG and per-speaker persona mimic — search your exports, ask grounded questions, and chat in someone's voice. Hinglish and English.

## Features

- **Import WhatsApp export** — group or 1-on-1 `.txt` exports with preprocess + parse pipeline
- **RAG Q&A** — hybrid semantic + keyword retrieval, Gemini rerank, strict grounding with citations
- **Persona chat** — Gemini style mimic per speaker; burst messages, history summarization, PDF export
- **Workspace analytics** — turn-based reply metrics, weekly/monthly conversation growth, activity heatmap
- **Build-time persona analysis** — personality notes, writing style, and chat-pattern extraction at activation

## Stack

| Layer | Choice |
|-------|--------|
| Frontend | Next.js (App Router), pnpm, TanStack Query, Zod |
| Backend | FastAPI, uv, LangGraph |
| Reads / Q&A | Chroma + `multilingual-e5-large` + hybrid BM25 + Gemini rerank/generate |
| Persona | Gemini activation (style profile + samples) |
| Embeddings | `intfloat/multilingual-e5-large` via sentence-transformers (CUDA or CPU) |
| Data | `./data/` at repo root (gitignored) |
| UI | Neo-brutalism, dark mode only |

## Quick start

**Prerequisites:** [uv](https://docs.astral.sh/uv/) (Python 3.12+), [Node.js](https://nodejs.org/) 20+, [pnpm](https://pnpm.io/), and a [Gemini API key](https://aistudio.google.com/apikey).

**Terminal 1 — backend**

```bash
cd backend
cp .env.example .env
# Set GEMINI_API_KEY in .env
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

**Terminal 2 — frontend**

```bash
cd frontend
cp .env.local.example .env.local
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000). API base: `http://127.0.0.1:8000/api/v1`.

Set `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000/api/v1` in `frontend/.env.local`.

Full setup (CUDA, Windows ML policy, pre-commit, troubleshooting): [COMMANDS.md](./COMMANDS.md).

## Project layout

| Path | Role |
|------|------|
| `backend/app/` | FastAPI routes, LangGraph graphs, services, core |
| `frontend/src/` | Next.js App Router, neo-brutalism UI, TanStack Query |
| `docs/` | Architecture, API, LangGraph flows, design system |
| `data/` | Runtime workspaces, Chroma, exports — **gitignored** |

Package-level notes: [backend/README.md](./backend/README.md), [frontend/README.md](./frontend/README.md).

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/README.md](./docs/README.md) | Documentation index and build order |
| [CONTEXT.md](./CONTEXT.md) | Domain terms, locked decisions, stack |
| [AGENTS.md](./AGENTS.md) | Agent workflow, layout, dev commands |
| [docs/architecture.md](./docs/architecture.md) | System overview, data flows, GPU strategy |
| [docs/api.md](./docs/api.md) | REST + SSE contract |
| [docs/ui-design.md](./docs/ui-design.md) | Neo-brutalism design system |
| [docs/data-layout.md](./docs/data-layout.md) | On-disk schema under `./data/` |
| [docs/langgraph/](./docs/langgraph/) | Ingest, Q&A, and persona activation graphs |

## License & disclaimer

ChatMemory is for **personal, local use** on your own machine. Chat exports stay on disk; only Gemini API calls leave the machine (Q&A, persona build, persona chat).

No warranty. You are responsible for consent when building personas from real conversations. Examples in this repo use fictional workspace and speaker names only.
