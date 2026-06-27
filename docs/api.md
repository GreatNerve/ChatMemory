# ChatMemory — API

FastAPI REST contract. All paths prefixed with `/api/v1` (recommended). JSON bodies use **camelCase** in HTTP; Python uses snake_case internally.

Base URL (dev): `http://127.0.0.1:8000/api/v1`

No authentication headers in MVP (local single-operator).

---

## Conventions

### Errors

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Human-readable summary",
    "fieldErrors": { "consent": "Required" }
  }
}
```

| HTTP | When |
|------|------|
| 400 | Validation, bad file, job preconditions failed |
| 404 | Workspace, person, or job not found |
| 409 | GPU busy — another heavy job running |
| 500 | Unexpected server error |
| 503 | Gemini not configured |

### Job statuses (SSE + poll)

`queued` → `running` → `done` | `error`

Persona train substeps (in `message` or `step` field):

`validating` → `refreshing_samples` → `style_profile` → `chat_analysis` → `personality` → `writing_style` → `activating` → `done`

### IDs

UUID strings for `workspaceId`, `personId`, `jobId`.

---

## Workspaces

### List workspaces

```
GET /workspaces
```

**Response 200**

```json
{
  "workspaces": [
    {
      "id": "uuid",
      "name": "Test Group",
      "createdAt": "2026-06-24T10:00:00Z",
      "messageCount": 12450,
      "speakerCount": 8,
      "dateFrom": "2023-01-01",
      "dateTo": "2026-06-01",
      "ingestStatus": "done",
      "isGroup": true
    }
  ]
}
```

`ingestStatus`: `pending` | `running` | `done` | `error`

`isGroup`: computed — `true` when `speakerCount > 2` (group chat); `false` for 1-on-1.

---

### Create workspace (upload + ingest)

```
POST /workspaces
Content-Type: multipart/form-data
```

| Field | Type | Required |
|-------|------|----------|
| `name` | string | yes |
| `file` | file (.txt) | yes |

**Response 202** (ingest started)

```json
{
  "workspace": { "id": "uuid", "name": "Test Group", "ingestStatus": "running" },
  "jobId": "uuid"
}
```

Client opens `GET /jobs/{jobId}/stream` for progress.

---

### Get workspace

```
GET /workspaces/{workspaceId}
```

**Response 200**

```json
{
  "id": "uuid",
  "name": "Test Group",
  "createdAt": "2026-06-24T10:00:00Z",
  "messageCount": 12450,
  "speakerCount": 8,
  "dateFrom": "2023-01-01",
  "dateTo": "2026-06-01",
  "ingestStatus": "done",
  "isGroup": true,
  "topSpeakers": [
    { "personId": "uuid", "displayName": "Alice", "messageCount": 3200 }
  ]
}
```

---

### Get workspace analytics

```
GET /workspaces/{workspaceId}/analytics?refresh=false
```

| Query | Default | Meaning |
|-------|---------|---------|
| `refresh` | `false` | When `true`, recompute from `export.txt` and overwrite cache |

**Response 200** — see [architecture.md](./architecture.md#workspace-analytics) for field semantics.

Key points:

- Per-person **typical reply time** uses **median** response seconds (turn-based; burst messages grouped).
- Response-time histogram buckets: `<1m` (includes same-minute / 0s replies), `1–5m`, `5–30m`, `30m+`.
- `group.weeklySeries` — all weeks, sorted ascending; UI shows last 52 weeks capped at today.
- `group.heatmap` — hour×day message frequency (non-zero cells only).

---

### Delete workspace

```
DELETE /workspaces/{workspaceId}
```

Removes workspace folder under `data/workspaces/{id}/`.

**Response 204**

---

## People (speakers)

### List people in workspace

```
GET /workspaces/{workspaceId}/people
```

**Response 200**

```json
{
  "people": [
    {
      "id": "uuid",
      "displayName": "Alice",
      "messageCount": 3200,
      "firstSeen": "2023-01-01",
      "lastSeen": "2026-06-01",
      "personaStatus": "ready_model"
    }
  ]
}
```

`personaStatus`:

| Value | Meaning |
|-------|---------|
| `not_enough` | &lt; 50 messages |
| `thin` | 50–199 messages (train with warning) |
| `ready` | ≥ 200 messages, no model yet |
| `training` | Job in progress |
| `ready_model` | Gemini persona active |
| `error` | Last train failed |

---

### Get person detail

```
GET /workspaces/{workspaceId}/people/{personId}
```

**Response 200**

```json
{
  "id": "uuid",
  "displayName": "Alice",
  "messageCount": 3200,
  "personaStatus": "ready_model",
  "ollamaModelName": "gemini-3.5-flash",
  "styleProfile": {
    "avgMessageLength": 42,
    "emojiRate": 0.12,
    "hinglishRatio": 0.65
  },
  "sampleMessages": [
    { "timestamp": "2024-03-12T18:22:00", "text": "yaar kal meeting hai" }
  ],
  "personalityNotes": "Alice tends to reply in short bursts...",
  "writingStyleNotes": "Mostly lowercase, skips punctuation...",
  "chatAnalysis": "Recurring topics include weekend plans...",
  "trainEligible": true,
  "trainWarning": null,
  "lastTrainJobId": "uuid"
}
```

`personalityNotes`, `writingStyleNotes`, and `chatAnalysis` are populated at persona **build** time via Gemini. `null` for personas activated before this feature or when a non-fatal extraction step fails.

---

## Q&A (RAG)

### Ask question

```
POST /workspaces/{workspaceId}/ask
```

**Request**

```json
{
  "question": "When did we plan the Goa trip?",
  "speaker": "Alice",
  "dateFrom": "2024-01-01",
  "dateTo": null
}
```

`speaker`, `dateFrom`, `dateTo` optional filters.

**Response 200 — success**

```json
{
  "status": "answered",
  "answer": "The Goa trip was discussed on 12 March 2024.",
  "citations": [
    {
      "messageId": "uuid",
      "speaker": "Bob",
      "timestamp": "2024-03-12T18:22:00",
      "snippet": "Goa trip final kar dete hain March end"
    }
  ]
}
```

**Response 200 — not found (strict RAG)**

```json
{
  "status": "not_found",
  "answer": null,
  "reason": "No sufficiently relevant messages in this chat.",
  "nearMisses": [
    {
      "messageId": "uuid",
      "speaker": "Bob",
      "timestamp": "2024-02-01T10:00:00",
      "snippet": "trip pe chalenge kahi",
      "score": 0.45
    }
  ]
}
```

**Response 409** if GPU mutex blocks embed.

Q&A uses Google Gemini for rewrite, rerank, and answer (`GEMINI_API_KEY` required). Retrieval is local: `multilingual-e5-large` embed, Chroma semantic search, BM25 keyword merge. Synchronous (no SSE).

---

## Persona

### Start persona training (activation)

```
POST /workspaces/{workspaceId}/people/{personId}/train
```

**Request**

```json
{
  "consent": true,
  "forceThin": false,
  "forceRetrain": false
}
```

| Field | Rule |
|-------|------|
| `consent` | Must be `true` |
| `forceThin` | Allow train when 50–199 messages (show warning in UI) |
| `forceRetrain` | Rebuild when `personaStatus` is already `ready_model` |

**Response 202**

```json
{
  "jobId": "uuid",
  "personaStatus": "training"
}
```

**Response 400** if &lt; 50 messages, consent false, already training, or already active without `forceRetrain`.

### Cancel persona training

```
POST /workspaces/{workspaceId}/people/{personId}/train/cancel
```

**Response 200**

```json
{
  "personaStatus": "ready",
  "message": "Build cancelled — you can activate again"
}
```

---

### Persona chat (JSON)

```
POST /workspaces/{workspaceId}/people/{personId}/chat
```

**Request**

```json
{
  "message": "kya chal raha hai?",
  "history": [
    { "role": "user", "content": "hello" },
    { "role": "assistant", "content": "bol bhai" }
  ],
  "previousInteractionId": "interactions/abc123",
  "conversationSummary": "Earlier they discussed weekend plans..."
}
```

| Field | Rule |
|-------|------|
| `history` | Optional multi-turn; route passes last 10 turns to the service (service may use up to 30 internally) |
| `previousInteractionId` | Gemini Interactions API chain ID from prior reply; does **not** skip memory routing |
| `conversationSummary` | Rolling summary of older turns (from `/chat/summarize`); injected into system prompt |

**Response 200**

```json
{
  "reply": "kuch nahi yaar, tu bata",
  "model": "gemini-3.5-flash",
  "interactionId": "interactions/abc123"
}
```

Burst separators (`||`) in the model output are collapsed to spaces in this non-streaming endpoint.

**Response 400** if `personaStatus` is not `ready_model`.

---

### Persona chat (SSE stream)

```
POST /workspaces/{workspaceId}/people/{personId}/chat/stream
Content-Type: application/json
Accept: text/event-stream
```

Same request body as `/chat`.

**Events** (`data:` JSON per line)

| Payload | Meaning |
|---------|---------|
| `{"status":"thinking"}` | Before Gemini call |
| `{"token":"<text>"}` | Word-level token for current bubble |
| `{"msgBreak":true}` | Commit bubble; start next (burst reply) |
| `{"done":true,"interactionId":"..."}` | All bubbles complete |
| `{"error":"<message>"}` | Failure |

Optional **burst replies**: model may separate multiple WhatsApp-style messages with `||`. Not forced every reply.

---

### Summarize persona chat history

```
POST /workspaces/{workspaceId}/people/{personId}/chat/summarize
```

**Request**

```json
{
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ],
  "keepRecent": 10
}
```

`keepRecent` is accepted for API symmetry; the client trims history before calling. Used when local history exceeds **24 turns** — older turns are summarized, last 10 kept verbatim.

**Response 200**

```json
{
  "summary": "They discussed weekend plans and Alice seemed tired...",
  "summarizedTurnCount": 14
}
```

---

## Jobs (SSE)

### Stream job progress

```
GET /jobs/{jobId}/stream
Accept: text/event-stream
```

**Events**

```
event: progress
data: {"status":"running","step":"embedding","percent":45,"message":"Embedding batch 9/20"}

event: done
data: {"status":"done","result":{"workspaceId":"uuid"}}

event: error
data: {"status":"error","message":"CUDA OOM during embedding"}
```

### Poll job (fallback)

```
GET /jobs/{jobId}
```

Same fields as latest SSE payload.

---

## Settings

### Get settings

```
GET /settings
```

**Response 200**

```json
{
  "dataRoot": "./data",
  "embedModel": "intfloat/multilingual-e5-large",
  "activeEmbedBackend": "local",
  "embedDevice": "cuda:0",
  "vectorStore": "chroma",
  "gpuAvailable": true,
  "gpuBusy": false,
  "activeJobId": null,
  "geminiConfigured": true,
  "geminiModel": "gemini-3.5-flash"
}
```

### Update settings

```
PUT /settings
```

**Request** (partial)

```json
{
  "dataRoot": "./data"
}
```

**Response 200** — full settings object.

---

## Health

```
GET /health
```

**Response 200**

```json
{
  "status": "ok",
  "dataRootWritable": true,
  "mlStackAvailable": true,
  "geminiConfigured": true,
  "embedReady": true
}
```

---

## CORS

FastAPI allows `http://localhost:3000` and `http://127.0.0.1:3000` in development.

---

## Related

- [architecture.md](./architecture.md)
- [langgraph/qa.md](./langgraph/qa.md)
- [langgraph/persona-train.md](./langgraph/persona-train.md)
