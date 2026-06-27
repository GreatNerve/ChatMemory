# ChatMemory — Data layout

All runtime data lives under `DATA_ROOT` (default `./data/` at repo root). **Gitignored.**

FastAPI is the only writer. Next.js never reads these paths directly.

---

## Tree

```
data/
├── config.json                 # operator settings (data root, embed model)
├── jobs/
│   └── {jobId}.json            # ephemeral job state for SSE/poll
└── workspaces/
    └── {workspaceId}/
        ├── meta.json           # name, dates, counts, ingest status
        ├── export.txt          # raw WhatsApp upload (immutable source)
        ├── chroma/               # Chroma persistent client path
        ├── bm25/                 # keyword index artifacts
        └── people/
            └── {personId}.json   # speaker profile + persona status
```

Legacy installs may still have `lora/{personId}/` from older LoRA training — **unused**, safe to delete manually.

---

## `config.json`

```json
{
  "dataRoot": "./data",
  "embedModel": "BAAI/bge-m3",
  "personaMinMessages": 200,
  "personaThinMinMessages": 50
}
```

---

## `workspaces/{id}/meta.json`

```json
{
  "id": "uuid",
  "name": "College Gang",
  "createdAt": "2026-06-24T10:00:00Z",
  "ingestStatus": "done",
  "ingestJobId": "uuid",
  "messageCount": 12450,
  "speakerCount": 8,
  "dateFrom": "2023-01-01T00:00:00",
  "dateTo": "2026-06-01T23:59:00",
  "exportFilename": "WhatsApp Chat with College Gang.txt"
}
```

---

## `people/{personId}.json`

```json
{
  "id": "uuid",
  "workspaceId": "uuid",
  "displayName": "Rahul",
  "aliases": ["Rahul Bhai"],
  "messageCount": 3200,
  "firstSeen": "2023-01-01T12:00:00",
  "lastSeen": "2026-06-01T18:00:00",
  "personaStatus": "ready_model",
  "ollamaModelName": "gemini",
  "lastTrainJobId": "uuid",
  "lastTrainAt": "2026-06-24T11:00:00Z",
  "styleProfile": {
    "avgMessageLength": 42,
    "emojiRate": 0.12,
    "hinglishRatio": 0.65
  },
  "sampleMessages": [
    {
      "messageId": "uuid",
      "timestamp": "2024-03-12T18:22:00",
      "text": "yaar kal meeting hai"
    }
  ]
}
```

`sampleMessages`: auto-picked diverse short messages for UI preview and persona prompts.

---

## Chroma metadata (per chunk)

| Field | Type | Purpose |
|-------|------|---------|
| `messageId` | string | UUID |
| `workspaceId` | string | UUID |
| `personId` | string | Speaker UUID |
| `speaker` | string | Display name (denormalized for debug) |
| `timestamp` | string | ISO 8601 |
| `text` | string | Full message body |

Collection name: `workspace_{workspaceId}` (no hyphens if Chroma restricts).

---

## `jobs/{jobId}.json`

```json
{
  "id": "uuid",
  "type": "ingest",
  "workspaceId": "uuid",
  "personId": null,
  "status": "running",
  "step": "embedding",
  "percent": 45,
  "message": "Embedding batch 9/20",
  "createdAt": "2026-06-24T10:00:00Z",
  "updatedAt": "2026-06-24T10:05:00Z",
  "error": null
}
```

`type`: `ingest` | `persona_train`

Jobs may be deleted after `done` + TTL (e.g. 24h) — implementation choice.

---

## Migration to global library (future)

When cross-group search ships:

- Move Chroma to shared collection with `workspaceId` filter (already on chunks)
- Add `people_global` merge table — **not in MVP**
- Optional migrate `dataRoot` to `%APPDATA%\ChatMemory` via settings

---

## Related

- [architecture.md](./architecture.md)
- [ingest/whatsapp-export.md](./ingest/whatsapp-export.md)
