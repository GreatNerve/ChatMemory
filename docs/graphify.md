# Graphify — codebase knowledge graph

[Graphify](https://pypi.org/project/graphifyy/) builds a queryable knowledge graph from this monorepo (`backend/`, `frontend/`, `docs/`). Use it to trace LangGraph flows, service dependencies, and module connections without repo-wide grep.

Output lives in `graphify-out/` at repo root — **gitignored**, never commit.

## Corpus exclusions (`.graphifyignore`)

Graphify reads **`.graphifyignore`** at the repo root (gitignore syntax, including `!` negation). When present, it replaces `.gitignore` for that directory — so this file carries both graph-specific rules and the exclusions we need from git.

**What we exclude and why:**

| Pattern | Why |
|---------|-----|
| `data/` | Runtime workspace data (exports, Chroma, jobs) — not source architecture |
| `node_modules/`, `.next/`, `dist/`, `build/` | Dependencies and build artifacts |
| `__pycache__/`, `.venv/`, `.uv/` | Python env and bytecode |
| `graphify-out/` | Graphify’s own output — never re-index |
| `.git/`, `terminals/` | VCS metadata and local session logs |
| `uv.lock`, `pnpm-lock.yaml` | Large lockfiles, low architecture signal |
| `*.pyc`, `*.min.js` | Generated / minified blobs |
| `.agents/`, `no-push/`, `models/` | Local-only or cache paths |

**What stays in the corpus:** `backend/app/**`, `frontend/src/**`, `docs/**`, root `README.md`, `CONTEXT.md`, `AGENTS.md`, plus tests and config at repo edges.

Verify after edits:

```bash
graphify detect .
# or: uv tool run --from graphifyy python -c "from graphify.detect import detect; from pathlib import Path; print(detect(Path('.'))['total_files'])"
```

## One-time setup

Pick one:

```bash
# Recommended (isolated tool)
uv tool install graphifyy

# Or pip
pip install graphifyy
```

Verify:

```bash
graphify detect .
```

## Build

From repo root:

```bash
/graphify .
```

Or manually:

```bash
graphify detect .
graphify build . --no-viz    # JSON + report, skip HTML (faster)
```

Incremental update after code changes:

```bash
graphify build . --update --no-viz
```

Requires no API keys for a code-only corpus. Full extraction with `--mode deep` may call an LLM if configured.

## Query (after build)

```bash
graphify query "How does persona chat memory retrieval work?"
graphify path "persona_chat" "retrieval"
graphify explain "fast_retrieve"
```

Agent rule: [.cursor/rules/graphify.mdc](../.cursor/rules/graphify.mdc).

## What gets generated

| Path | Purpose |
|------|---------|
| `graphify-out/graph.json` | GraphRAG-ready node/edge JSON |
| `graphify-out/GRAPH_REPORT.md` | Plain-language audit report |
| `graphify-out/*.html` | Interactive viz (unless `--no-viz`) |
| `graphify-out/.graphify_python` | Local interpreter hint (gitignored via `*.graphify_*`) |

## Suggested queries for ChatMemory

| Topic | Example query |
|-------|---------------|
| Persona chat flow | `"persona chat fast_route classify retrieve generate"` |
| Retrieval scoring | `"fast_retrieve merge recency density score gate"` |
| Persona training | `"persona_train chat_analysis voice_samples steps"` |
| Ingest pipeline | `"ingest graph embed chroma"` |

## Related

- [architecture.md](./architecture.md) — canonical system overview
- [langgraph/persona-chat.md](./langgraph/persona-chat.md) — persona chat node detail
- [AGENTS.md](../AGENTS.md) — agent workflow + knowledge graph section
