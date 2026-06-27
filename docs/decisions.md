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

**Status:** Accepted

**Decision:** Train QLoRA in Python; serve via `ollama create` per speaker. Not RAG-only style mimic.

**Consequences:** GPU mutex with Ollama; train requires stopping Ollama models.

---

## ADR-005: Next.js + FastAPI two-terminal dev

**Status:** Accepted (rejected Electron)

**Decision:** Frontend Next.js; backend FastAPI. No Electron shell.

---

## ADR-006: Ollama for inference, Python for training

**Status:** Accepted

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

## ADR-009: bge-m3 + Chroma

**Status:** Accepted

**Decision:** `BAAI/bge-m3` embeddings on CUDA; Chroma persistent per workspace.

---

## ADR-010: Three LangGraph graphs

**Status:** Accepted

**Decision:** Separate `ingest`, `qa`, `persona_train` graphs. Persona chat is a direct route.

---

## ADR-011: Data root `./data/` (dev)

**Status:** Accepted

**Decision:** Repo-adjacent data folder, gitignored. Settings override deferred post-MVP.

---

## ADR-012: LoRA training gates

**Status:** Accepted

| Rule | Value |
|------|-------|
| Block | &lt; 50 messages |
| Warn + force | 50–199 |
| Normal | ≥ 200 |
| Train cap | 5000 messages sampled |
| Consent | Required checkbox |

---

## ADR-013: Strict RAG for Q&A

**Status:** Accepted

**Decision:** Refuse with `not_found` when &lt; 2 chunks pass relevance threshold. Citations required on answers.

**Consequences:** Persona chat does not use RAG in MVP.

---

## ADR-014: Hybrid retrieve + Ollama LLM rerank

**Status:** Accepted

**Decision:** Semantic (Chroma) + BM25 merge, then Ollama scores chunks. Rejected CPU cross-encoder for MVP.

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

**Status:** Accepted

**Context:** User wants maximum GPU use on RTX 3050 6GB.

**Decision:** Embed on CUDA; Ollama on GPU; serialize heavy jobs via `gpu_lock`.

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
- Embeddings stay local (`bge-m3` + sentence-transformers)
- Vector store: **Chroma** per workspace via LangChain
- WhatsApp ingest uses **preprocess** pipeline before parse

**Consequences:** No Ollama or `uv sync --extra train` in default path. Legacy `data/.../lora/` folders may remain on disk but are unused.

---

## Template for new ADRs

```
## ADR-NNN: Title

**Status:** Proposed | Accepted | Superseded

**Context:** ...

**Decision:** ...

**Consequences:** ...
```
