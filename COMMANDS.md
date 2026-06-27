# ChatMemory — setup & run commands

Windows paths assume repo at `D:\@2026\RAG_TEST`. Use **Git Bash** unless noted.

---

## 1. One-time prerequisites

### Install


| Tool                                                           | Purpose                       |
| -------------------------------------------------------------- | ----------------------------- |
| [uv](https://docs.astral.sh/uv/)                               | Python 3.12+ and backend deps |
| [Node.js](https://nodejs.org/) 20+                             | Next.js frontend              |
| [pnpm](https://pnpm.io/)                                       | Frontend package manager      |
| [Google AI Studio](https://aistudio.google.com/apikey) API key | Gemini Q&A + persona chat     |


**Not required:** Ollama, local LLM servers, or LoRA training stacks.

### Backend env (`backend/.env`)

```bash
cd D:/@2026/RAG_TEST/backend
cp .env.example .env
```

Edit `backend/.env` — important variables:

```bash
# Data (repo-relative from backend cwd)
DATA_ROOT=../data

# Local embeddings — sentence-transformers (CUDA or CPU)
EMBED_MODEL=intfloat/multilingual-e5-large
EMBED_DEVICE=auto

# Vector index: chroma (default) | file | auto
VECTOR_STORE=chroma

# API
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Required for Q&A and persona chat (either name works)
GEMINI_API_KEY=your-key-from-aistudio.google.com
# GOOGLE_API_KEY=your-key-from-aistudio.google.com
GEMINI_MODEL=gemini-3.5-flash

# Persona activation message gates (legacy env names)
LORA_MIN_MESSAGES=200
LORA_THIN_MIN_MESSAGES=50
```

**Stack summary**

- **LLM:** Gemini via official `google-genai` SDK (Interactions API), default `gemini-3.5-flash`. LangChain wraps it for RAG (`rag_chain.py`).
- **Embeddings:** local `intfloat/multilingual-e5-large` (E5 `query:` / `passage:` prefixes). Always `activeEmbedBackend: "local"` — no remote embed API.
- **Vector store:** Chroma per workspace at `data/workspaces/<id>/chroma/`.
- **Retrieval:** hybrid Chroma semantic + BM25 keyword merge, Gemini rerank + answer.
- **Persona:** fast Gemini activation (style profile + samples) — no LoRA training.

**After changing** `EMBED_MODEL`**:** first startup downloads the new model (~1–2 GB). **Re-ingest every workspace** so Chroma vectors match the new embedder.

**Startup preload:** On API boot the embed model loads once (CUDA when available) plus a tiny warmup encode. First-ever run still downloads weights; later starts preload before the first request. Check `/api/v1/health` → `embedReady: true` when loaded.

### Frontend env (`frontend/.env.local`)

```bash
cd D:/@2026/RAG_TEST/frontend
cp .env.local.example .env.local
```

```bash
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000/api/v1
```



### Python deps

```bash
cd D:/@2026/RAG_TEST/backend
uv sync
```

LangChain + Chroma + `google-genai` are default deps. After first ingest, each workspace has `data/workspaces/<id>/chroma/`.

### Run backend — two workflows

`uv sync` creates `.venv` automatically; both options below use the same environment.

**Option A — `uv run` (recommended, no activate)**

```bash
cd D:/@2026/RAG_TEST/backend
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

**Option B — activate venv manually (Git Bash on Windows)**

```bash
cd D:/@2026/RAG_TEST/backend
uv sync
source .venv/Scripts/activate
uvicorn app.main:app --reload --port 8000
```

PowerShell:

```powershell
cd D:\@2026\RAG_TEST\backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```

### Frontend deps

```bash
cd D:/@2026/RAG_TEST/frontend
pnpm install
```

### Pre-commit (format + lint)

```bash
cd D:/@2026/RAG_TEST/backend && uv sync
uv run pre-commit install -c ../.pre-commit-config.yaml
uv run pre-commit run --all-files -c ../.pre-commit-config.yaml   # one-off check
```

Frontend ESLint: `cd frontend && pnpm lint` (not in pre-commit; run manually or in CI).



### Windows: PyTorch / NumPy blocked

If `import torch` fails with **Application Control** (common on Windows):

1. Windows Security → App & browser control → **Smart App Control** → **Off**
2. Restart PC
3. PowerShell **as Administrator**:

```powershell
cd D:\@2026\RAG_TEST\backend\scripts
.\fix-windows-ml.ps1
```

**CUDA PyTorch (NVIDIA GPUs)** for faster ingest embed. `backend/pyproject.toml` pins `torch` to the `pytorch-cu124` index — use `uv sync` (not PyPI CPU torch):

```bash
cd D:/@2026/RAG_TEST/backend
uv sync
uv run python -c "import torch; print(torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

If you still see `2.x+cpu` or `cuda False`, reinstall the pinned wheel:

```bash
cd D:/@2026/RAG_TEST/backend
uv sync --reinstall-package torch
```

`fix-windows-ml.ps1` also runs the cu124 install after SAC/Defender checks.

**Verify GPU embed** (RTX 3050 etc. — needs CUDA PyTorch above; CPU-only torch shows `cuda False`):

```bash
cd D:/@2026/RAG_TEST/backend
uv run python -c "from app.services.embed import cuda_available, resolve_embed_device; print('cuda_available', cuda_available()); print('embed_device', resolve_embed_device())"
curl -s http://127.0.0.1:8000/api/v1/settings | python -m json.tool
# embedDevice should be "cuda:0" (not "cpu"); Settings UI shows Embed device row
# On API startup, backend logs embed warmup (not on first ingest/Q&A):
#   Warming up embed model...
#   Loading SentenceTransformer model from intfloat/multilingual-e5-large.
#   Loading embed model on cuda:0 (NVIDIA GeForce RTX 3050 ...)
#   Embed model ready on cuda:0 (intfloat/multilingual-e5-large)
# curl -s http://127.0.0.1:8000/api/v1/health → embedReady: true
```

`EMBED_DEVICE`: `auto` (default) uses GPU when available; `cpu` forces CPU; `cuda` errors if no GPU.

---



## 2. Every session — backend (Terminal 1)

**Option A — `uv run` (recommended):**

```bash
cd D:/@2026/RAG_TEST/backend
uv run uvicorn app.main:app --reload --port 8000
```

**Option B — manual venv (Git Bash):**

```bash
cd D:/@2026/RAG_TEST/backend
source .venv/Scripts/activate
uvicorn app.main:app --reload --port 8000
```

See §1 for PowerShell activate (`.\.venv\Scripts\Activate.ps1`).

**Note:** `--reload` spawns a new worker on code changes, which re-runs embed warmup. Production (no `--reload`) warms once per process.

API base: [http://127.0.0.1:8000/api/v1](http://127.0.0.1:8000/api/v1)

OpenAPI docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---



## 3. Every session — frontend (Terminal 2)

```bash
cd D:/@2026/RAG_TEST/frontend
pnpm dev
```

UI: [http://localhost:3000](http://localhost:3000)

---



## 4. PowerShell equivalents

```powershell
cd D:\@2026\RAG_TEST\backend
uv run uvicorn app.main:app --reload --port 8000
```

```powershell
cd D:\@2026\RAG_TEST\frontend
pnpm dev
```

---



## 5. First-time / health checks

With the backend running:

```bash
curl -s http://127.0.0.1:8000/api/v1/health | python -m json.tool
```

Expect `status: "ok"` when `dataRootWritable`, `mlStackAvailable`, and `geminiConfigured` are all true. `degraded` usually means missing `GEMINI_API_KEY` or a broken ML stack.

```bash
curl -s http://127.0.0.1:8000/api/v1/settings | python -m json.tool
```

Sanity checks:


| Field                | Expected                                            |
| -------------------- | --------------------------------------------------- |
| `embedModel`         | `intfloat/multilingual-e5-large` (or your override) |
| `activeEmbedBackend` | `"local"`                                           |
| `vectorStore`        | `"chroma"`                                          |
| `geminiConfigured`   | `true`                                              |
| `geminiModel`        | `gemini-3.5-flash`                                  |


Or open **Settings** in the UI after `pnpm dev`.

---



## 6. Ingest workflow (WhatsApp export)

**UI:** Home → create workspace → upload WhatsApp `.txt` export → watch SSE progress.

**Pipeline:** preprocess export → parse messages → index speakers → BM25 corpus → embed (local E5) → write Chroma under `data/workspaces/<id>/chroma/`.

**curl example:**

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/workspaces \
  -F "name=College Gang" \
  -F "file=@/path/to/WhatsApp Chat with Group.txt" | python -m json.tool
```

Save `jobId` from the response. Stream progress:

```bash
curl -N http://127.0.0.1:8000/api/v1/jobs/<jobId>/stream
```

Poll workspace until `ingestStatus` is `done`:

```bash
curl -s http://127.0.0.1:8000/api/v1/workspaces/<workspaceId> | python -m json.tool
```

Minimum **50** non-system messages required (`min_workspace_messages`).

---



## 7. Persona activate + chat

Persona = Gemini style mimic for one speaker (not factual memory lookup).

**UI:** Workspace → speaker → **Activate persona** (consent required) → wait for job → **Chat**.

**API — activate** (endpoint name is still `/train`; no LoRA involved):

```bash
curl -s -X POST \
  "http://127.0.0.1:8000/api/v1/workspaces/<workspaceId>/people/<personId>/train" \
  -H "Content-Type: application/json" \
  -d '{"consent": true, "forceThin": false}' | python -m json.tool
```


| Messages | Behavior                               |
| -------- | -------------------------------------- |
| < 50     | 400 — not enough data                  |
| 50–199   | thin persona — set `"forceThin": true` |
| ≥ 200    | normal activation                      |


Stream job via `GET /jobs/<jobId>/stream`. When `personaStatus` is `ready_model`, chat:

```bash
curl -s -X POST \
  "http://127.0.0.1:8000/api/v1/workspaces/<workspaceId>/people/<personId>/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "kya chal raha hai?", "history": []}' | python -m json.tool
```

**Cancel stuck activation:** UI **Cancel activation**, or:

```bash
curl -s -X POST \
  "http://127.0.0.1:8000/api/v1/workspaces/<workspaceId>/people/<personId>/train/cancel"
```

---



## 8. Q&A usage (grounded RAG)

**UI:** Workspace → **Ask** → question (+ optional speaker / date filters).

**API:**

```bash
curl -s -X POST \
  "http://127.0.0.1:8000/api/v1/workspaces/<workspaceId>/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "When did we plan the Goa trip?",
    "speaker": null,
    "dateFrom": null,
    "dateTo": null
  }' | python -m json.tool
```

Requires `ingestStatus: "done"`, `GEMINI_API_KEY`, and working local embed + Chroma. Response `status` is `answered` (with citations) or `not_found`.

---



## 9. Useful commands


| Task                      | Command                                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------- |
| All unit tests            | `cd backend && uv run pytest tests/unit -q`                                                                   |
| Full test suite           | `cd backend && uv run pytest`                                                                                 |
| Chroma / RAG tests        | `cd backend && uv run pytest tests/unit/test_chroma.py tests/unit/test_qa.py tests/unit/test_rag_chain.py -v` |
| WhatsApp preprocess tests | `cd backend && uv run pytest tests/unit/test_whatsapp_preprocess.py -v`                                       |
| Health integration test   | `cd backend && uv run pytest tests/integration/test_api_health.py -v`                                         |
| Fix Windows ML DLLs       | `backend/scripts/fix-windows-ml.ps1` (Admin PowerShell)                                                       |


---



## 10. What needs what


| Feature          | GEMINI_API_KEY | ML stack (torch)       | Chroma             |
| ---------------- | -------------- | ---------------------- | ------------------ |
| Ingest workspace | —              | yes (local embed)      | created on ingest  |
| Ask (Q&A)        | yes            | yes (embed query)      | yes                |
| Activate persona | yes            | — (style profile only) | samples from index |
| Persona chat     | yes            | optional (RAG context) | yes                |


Only one heavy GPU job (ingest embed batch or Q&A embed) runs at a time.

---



## 11. Troubleshooting

- **Persona chat / Ask returns 503** → set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in `backend/.env` and restart API
- `geminiConfigured: false` **in** `/settings` → same as above; key must be non-empty
- **Ingest fails on embed** → run `fix-windows-ml.ps1` or use CPU torch; check `mlStackAvailable` in `/health`
- **Embed on CPU /** `cuda False` **with NVIDIA GPU** → `cd backend && uv sync --reinstall-package torch`; verify §1 CUDA commands; restart API
- **Changed** `EMBED_MODEL` → delete workspace and re-upload, or re-ingest; old Chroma vectors are incompatible
- **Chroma path** → `data/workspaces/<workspaceId>/chroma/` (under `DATA_ROOT`, default `../data` from backend)
- **Persona stuck activating** → **Cancel activation** in UI or `POST .../train/cancel`; restart backend if needed
- **Ask 409 GPU busy** → wait for ingest embed to finish (or cancel that job)
- **Legacy** `data/.../lora/` **folders** → safe to ignore; old LoRA adapters are not used
- **Upload rejected** → file must be WhatsApp `.txt` export (UTF-8 or Latin-1)

