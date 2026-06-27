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

### Job statuses (SSE + poll)

`queued` → `running` → `done` | `error`

Persona train substeps (in `message` or `step` field):

`validating` → `refreshing_samples` → `style_profile` → `activating` → `done`

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
      "name": "College Gang",
      "createdAt": "2026-06-24T10:00:00Z",
      "messageCount": 12450,
      "speakerCount": 8,
      "dateFrom": "2023-01-01",
      "dateTo": "2026-06-01",
      "ingestStatus": "done"
    }
  ]
}
```

`ingestStatus`: `pending` | `running` | `done` | `error`

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
  "workspace": { "id": "uuid", "name": "College Gang", "ingestStatus": "running" },
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
  "name": "College Gang",
  "createdAt": "2026-06-24T10:00:00Z",
  "messageCount": 12450,
  "speakerCount": 8,
  "dateFrom": "2023-01-01",
  "dateTo": "2026-06-01",
  "ingestStatus": "done",
  "topSpeakers": [
    { "personId": "uuid", "displayName": "Rahul", "messageCount": 3200 }
  ]
}
```

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
      "displayName": "Rahul",
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
| `ready_model` | Gemini persona active (`ollamaModelName`: `"gemini"`) |
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
  "displayName": "Rahul",
  "messageCount": 3200,
  "personaStatus": "ready",
  "ollamaModelName": null,
  "styleProfile": {
    "avgMessageLength": 42,
    "emojiRate": 0.12,
    "hinglishRatio": 0.65
  },
  "sampleMessages": [
    { "timestamp": "2024-03-12T18:22:00", "text": "yaar kal meeting hai" }
  ],
  "trainEligible": true,
  "trainWarning": null
}
```

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
  "speaker": "Rahul",
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
      "speaker": "Priya",
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
      "speaker": "Amit",
      "timestamp": "2024-02-01T10:00:00",
      "snippet": "trip pe chalenge kahi",
      "score": 0.45
    }
  ]
}
```

**Response 409** if GPU mutex blocks (rare for Q&A — document if train holds lock).

Q&A uses Google Gemini for rewrite, rerank, and answer (`GEMINI_API_KEY` required). Retrieval is local: bge-m3 embed, Chroma semantic search, BM25 keyword merge. Synchronous (no SSE).

---

## Persona

### Start persona training

```
POST /workspaces/{workspaceId}/people/{personId}/train
```

**Request**

```json
{
  "consent": true,
  "forceThin": false
}
```

| Field | Rule |
|-------|------|
| `consent` | Must be `true` |
| `forceThin` | Allow train when 50–199 messages (show warning in UI) |

**Response 202**

```json
{
  "jobId": "uuid",
  "personaStatus": "training"
}
```

**Response 400** if &lt; 50 messages, consent false, or already training.

---

### Persona chat

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
  ]
}
```

`history` optional for multi-turn (last N turns).

**Response 200**

```json
{
  "reply": "kuch nahi yaar, tu bata",
  "model": "chatmemory-rahul-{workspaceId-short}"
}
```

**Response 400** if `personaStatus` is not `ready_model`.

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
data: {"status":"error","message":"CUDA OOM during training"}
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
  "embedModel": "BAAI/bge-m3",
  "activeEmbedBackend": "local",
  "vectorStore": "chroma",
  "gpuAvailable": true,
  "gpuBusy": false,
  "activeJobId": null,
  "geminiConfigured": true,
  "geminiModel": "gemini-2.0-flash"
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
  "geminiConfigured": true
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
