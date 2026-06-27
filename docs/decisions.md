# ChatMemory — Decisions (ADR log)

Architecture Decision Records from product grill. New decisions append at bottom.

---

## ADR-001: Local single-operator app

**Status:** Accepted

**Context:** WhatsApp exports are sensitive. User has RTX 3050 laptop.

**Decision:** No accounts, no cloud. One operator per install. Data on local disk.

**Consequences:** No auth in API. No multi-tenant isolation needed.

---

## ADR-002: Person = chat speaker

**Status:** Accepted

**Decision:** "People" in UI are speakers parsed from exports, not app user profiles.

---

## ADR-003: One workspace per WhatsApp group (MVP)

**Status:** Accepted (future: global library)

**Decision:** Each upload creates an isolated workspace with own Chroma collection and people set.

---

## ADR-004: Persona via LoRA + Ollama

**Status:** Superseded by ADR-021

**Decision:** Train QLoRA in Python; serve via `ollama create` per speaker. Not RAG-only style mimic.

**Consequences:** GPU mutex with Ollama; train requires stopping Ollama models.

---

## ADR-005: Next.js + FastAPI two-terminal dev

**Status:** Accepted (rejected Electron)

**Decision:** Frontend Next.js; backend FastAPI. No Electron shell.

---

## ADR-006: Ollama for inference, Python for training

**Status:** Superseded by ADR-021

**Decision:** Q&A and persona chat use Ollama GPU. Embed and LoRA use Python/CUDA.

---

## ADR-007: Neo-brutalism dark-only UI

**Status:** Accepted

**Decision:** Custom components, no shadcn, no light mode in MVP.

---

## ADR-008: WhatsApp `.txt` only (MVP)

**Status:** Accepted

**Decision:** Single parser supporting Android and iOS export variants.

---

## ADR-009: Local embeddings + Chroma

**Status:** Accepted (updated — model changed)

**Decision:** Local sentence-transformers embeddings on CUDA; Chroma persistent per workspace. Current model: `intfloat/multilingual-e5-large` (supersedes early `BAAI/bge-m3` choice for Hinglish coverage).

---

## ADR-010: Three LangGraph graphs

**Status:** Accepted

**Decision:** Separate `ingest`, `qa`, `persona_train`, and `persona_chat` graphs. Persona summarize remains a direct service route.

**Note:** Superseded in part by ADR-026 (`persona_chat` graph added).

---

## ADR-011: Data root `./data/` (dev)

**Status:** Accepted

**Decision:** Repo-adjacent data folder, gitignored. Settings override deferred post-MVP.

---

## ADR-012: Persona activation gates

**Status:** Accepted

| Rule | Value |
|------|-------|
| Block | &lt; 50 messages |
| Warn + force | 50–199 |
| Normal | ≥ 200 |
| Sample cap | 60 messages for personality/style extraction; full corpus for chat analysis |
| Consent | Required checkbox |

---

## ADR-013: Strict RAG for Q&A

**Status:** Accepted

**Decision:** Refuse with `not_found` when &lt; 2 chunks pass relevance threshold. Citations required on answers.

**Consequences:** Persona chat uses optional memory recall (ADR-026), not strict Q&A refusal rules.

---

## ADR-014: Hybrid retrieve + LLM rerank

**Status:** Accepted (updated — reranker changed)

**Decision:** Semantic (Chroma) + BM25 merge, then Gemini scores chunks. Rejected CPU cross-encoder for MVP. (Originally Ollama; superseded in ADR-021.)

---

## ADR-015: Hinglish + English only

**Status:** Accepted

**Decision:** No translation. Answer in same language mix as question (prompt rule).

---

## ADR-016: REST + SSE API

**Status:** Accepted

**Decision:** REST for CRUD and actions; SSE for ingest/train job progress.

---

## ADR-017: TanStack Query frontend

**Status:** Accepted

**Decision:** pnpm, TanStack Query, Zod. No Apollo.

---

## ADR-018: GPU aggressive with mutex

**Status:** Accepted (updated — Ollama removed)

**Context:** User wants maximum GPU use on RTX 3050 6GB.

**Decision:** Embed on CUDA; serialize heavy embed jobs via `gpu_lock`. LLM work is Gemini API (no local GPU).

---

## ADR-019: Docs at repo root + `docs/`

**Status:** Accepted

**Decision:** `AGENTS.md` and `CONTEXT.md` at root; all other markdown under `docs/`.

---

## ADR-020: Backend scaffold — procedural graph runners

**Status:** Accepted

**Decision:** MVP implements `graphs/*.py` as async procedural pipelines matching LangGraph node sequences. Upgrade to `langgraph.StateGraph` when adding branching/retry without changing HTTP contract.

---

## ADR-021: Gemini + Chroma — supersede Ollama/LoRA

**Status:** Accepted (supersedes ADR-004, ADR-006, ADR-014 in part)

**Context:** Local LoRA + Ollama added ops burden (model pulls, VRAM juggling, Windows path issues). Gemini API gives strong Hinglish/English style mimic without training.

**Decision:**

- Q&A rewrite, rerank, and answer via **Google Gemini** (`GEMINI_API_KEY` required)
- Persona activation builds style profile + samples; chat uses Gemini
- Embeddings stay local (`multilingual-e5-large` + sentence-transformers)
- Vector store: **Chroma** per workspace via LangChain
- WhatsApp ingest uses **preprocess** pipeline before parse

**Consequences:** No Ollama or `uv sync --extra train` in default path. Legacy `data/.../lora/` folders may remain on disk but are unused.

---

## ADR-022: Build-time Gemini persona analysis

**Status:** Accepted

**Context:** Style profile + samples alone miss personality depth and typing habits. Unbounded chat history cannot fit in one prompt.

**Decision:** At persona activation, run rate-limited Gemini extraction steps: `chatAnalysis` (chunked full corpus), `personalityNotes`, and `writingStyleNotes`. Store on person JSON. Non-fatal failures — activation completes with partial notes.

**Consequences:** Build job takes longer and consumes Gemini quota. Recency-weighted sampling (60% from recent third) for personality/style calls.

---

## ADR-023: Turn-based analytics with median reply times

**Status:** Accepted

**Context:** Raw per-message gaps inflate reply counts when users send burst messages. Mean reply time is skewed by outliers.

**Decision:** Merge consecutive same-sender messages into turns before measuring response gaps. Report **median** seconds as typical reply. Include 0s (same-minute) gaps in `<1m` bucket. Cache in `analytics.json` at ingest.

**Consequences:** Overview UI shows "Conversation rhythm" (1-on-1) vs "Group rhythm" (3+ speakers) using `isGroup` (`speakerCount > 2`).

---

## ADR-024: Rolling persona chat summarization

**Status:** Accepted

**Context:** Long persona chats exceed practical history windows without blowing token budgets.

**Decision:** When client history exceeds 24 turns, call `POST .../chat/summarize`, keep last 10 verbatim, pass `conversationSummary` on subsequent chat requests. Service uses up to 30 turns internally (20 if char budget exceeded).

**Consequences:** Extra Gemini call on threshold crossing; `previousInteractionId` chain resets after summarize.

---

## ADR-025: Monorepo git at repo root

**Status:** Accepted

**Decision:** Single git repository at repo root (`backend/` + `frontend/` + `docs/`). Pre-commit at root (ruff on `backend/`). `.agents/` and `no-push/` gitignored.

**Consequences:** Install hooks with `uv run pre-commit install -c ../.pre-commit-config.yaml` from `backend/`.

---

## ADR-026: Persona chat memory recall (router B + validation)

**Status:** Accepted

**Context:** Persona chat was style-only (ADR-013 consequence). Users ask personas about past events ("yaad hai?", "kab plan kiya tha?"). Q&A strict RAG is too heavy and refuses too often for conversational mimic. Follow-up turns can still need history lookup.

**Decision:**

- Fourth LangGraph: `persona_chat` with two compiled subgraphs — **context** (`run_persona_context`) and **generation** (`run_persona_generation`).
- **Router B:** `fast_history_route()` heuristics first; Gemini JSON `classify_history_need()` only on `ambiguous`. Casual → no retrieval; obvious memory → retrieve; ambiguous → classify then retrieve or skip.
- **Retrieval scope C:** Person-first Chroma + BM25; widen to full group when person hits are weak. No LLM rerank. Score gate at `persona_memory_inject_min_score`. Turn windows from `export.txt` (3 before, 2 after) with `target_person` filter.
- Memory injected in `=== RELEVANT PAST CHAT ===` — separate from style samples. Every chat turn runs the router (no skip on `previousInteractionId`).
- **Validation node:** After generation, Gemini checks for invented facts vs memory + conversation; one `regenerate_safe` retry with STRICT RECALL prefix.
- WhatsApp **noise filter** at index/read (`is_noise_message`); BM25 cache per workspace, invalidated on ingest.

**Consequences:** Extra Gemini calls on ambiguous routes and on hallucination retry. Persona chat is not strict Q&A — vague in-character replies when memory is empty. API contract unchanged (memory is server-side). Supersedes ADR-010/ADR-013 persona-chat consequences in part.

---

## Template for new ADRs

```
## ADR-NNN: Title

**Status:** Proposed | Accepted | Superseded

**Context:** ...

**Decision:** ...

**Consequences:** ...
```
