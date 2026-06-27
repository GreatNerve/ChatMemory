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
        ├── analytics.json      # cached workspace analytics (recomputable)
        ├── chroma/             # Chroma persistent client path
        ├── bm25/               # keyword index artifacts
        └── people/
            └── {personId}.json # speaker profile + persona status + LLM notes
```

Legacy installs may still have `lora/{personId}/` from older LoRA training — **unused**, safe to delete manually.

---

## `config.json`

```json
{
  "dataRoot": "./data",
  "embedModel": "intfloat/multilingual-e5-large",
  "personaMinMessages": 200,
  "personaThinMinMessages": 50
}
```

---

## `workspaces/{id}/meta.json`

```json
{
  "id": "uuid",
  "name": "Test Group",
  "createdAt": "2026-06-24T10:00:00Z",
  "ingestStatus": "done",
  "ingestJobId": "uuid",
  "messageCount": 12450,
  "speakerCount": 8,
  "dateFrom": "2023-01-01T00:00:00",
  "dateTo": "2026-06-01T23:59:00",
  "exportFilename": "WhatsApp Chat with Test Group.txt"
}
```

`isGroup` is **not** stored on disk — computed at API layer as `speakerCount > 2`.

---

## `people/{personId}.json`

```json
{
  "id": "uuid",
  "workspaceId": "uuid",
  "displayName": "Alice",
  "aliases": [],
  "messageCount": 3200,
  "firstSeen": "2023-01-01T12:00:00",
  "lastSeen": "2026-06-01T18:00:00",
  "personaStatus": "ready_model",
  "ollamaModelName": "gemini-3.5-flash",
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
  ],
  "personalityNotes": "Alice tends to reply in short bursts...",
  "writingStyleNotes": "Mostly lowercase, skips punctuation...",
  "chatAnalysis": "Recurring topics include weekend plans...",
  "activeListeningStyle": "When others vent, Alice typically responds with a short reaction..."
}
```

| Field | Set when |
|-------|----------|
| `sampleMessages` | Ingest + refreshed at persona build (recency-biased monthly spread) |
| `personalityNotes` | Persona build — Gemini, recency-weighted sample (~60% from recent third) |
| `writingStyleNotes` | Persona build — Gemini, same sampling |
| `chatAnalysis` | Persona build — chunked Gemini analysis over full corpus |
| `activeListeningStyle` | Persona build — Gemini, recency-weighted sample (~80%, same as writing style) |

All four LLM fields may be absent (`null`) if build predates the feature or a non-fatal extraction step fails.

---

## `analytics.json`

Cached output of `analytics.compute_analytics()`. Written at end of ingest; refreshed via `GET .../analytics?refresh=true`.

Contains `computedAt`, `group` (rhythm stats, `weeklySeries`, `heatmap`, `strongestPair`), per-person stats, and pair connectivity. See [api.md](./api.md#get-workspace-analytics).

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
