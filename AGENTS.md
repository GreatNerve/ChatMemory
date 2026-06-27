# AGENTS — ChatMemory



Workflow. Domain: [CONTEXT.md](./CONTEXT.md). Detail: `docs/`.



## Before code



1. Read `CONTEXT.md`

2. Read relevant `docs/` — start with `docs/architecture.md` and `docs/api.md`

3. For UI work: `docs/ui-design.md`

4. For LangGraph: `docs/langgraph/` — [persona-chat.md](./docs/langgraph/persona-chat.md) for memory recall flow



## Layout



| Path | Role |

|------|------|

| `backend/app/` | FastAPI routes, LangGraph graphs, services, core |

| `frontend/src/` | Next.js App Router, brutal UI, TanStack Query, Zod |

| `docs/` | All technical + design markdown (except root CONTEXT/AGENTS) |

| `data/` | Runtime data — **gitignored, never commit** |



## Rules



**Backend**



- UUID for workspace and person IDs

- API JSON camelCase; Python snake_case

- All file I/O under `data/` via config — never hardcode paths outside `app/core/config.py`

- Stock business logic in `services/` — routes and graphs stay thin

- GPU mutex: only one heavy GPU job (ingest embed batch, Q&A embed) at a time

- `GEMINI_API_KEY` required for Q&A and persona chat



**Frontend**



- pnpm only

- TanStack Query for all FastAPI reads/writes

- SSE via `useJobStream` for ingest/train jobs

- No direct Gemini or Chroma calls from browser — FastAPI only

- Neo-brutalism dark per `docs/ui-design.md` — no shadcn, no light mode

- Custom components in `components/` — never patch third-party UI primitives by hand



**Generated / never hand-edit**



- `data/**` (runtime)

- Future: `backend/app/grpc/gen/` if added



**Git**



- Commit docs when asked; never commit `data/` or secrets



## Commands



```bash

# Terminal 1 — backend

cd backend && uv sync && uv run uvicorn app.main:app --reload --port 8000



# Terminal 2 — frontend

cd frontend && pnpm install && pnpm dev

```



Env (backend): copy `backend/.env.example` → `backend/.env` and set `GEMINI_API_KEY`.  

Env (frontend): `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000`



## Ambiguous product



One question → recommend → record in `CONTEXT.md` grill table + `docs/decisions.md`



## LangGraph graphs

| Graph | File | Doc |
|-------|------|-----|
| `ingest` | `graphs/ingest.py` | `docs/langgraph/ingest.md` |
| `qa` | `graphs/qa.py` | `docs/langgraph/qa.md` |
| `persona_train` | `graphs/persona_train.py` | `docs/langgraph/persona-train.md` |
| `persona_chat` | `graphs/persona_chat.py` | `docs/langgraph/persona-chat.md` |

