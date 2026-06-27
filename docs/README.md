# ChatMemory documentation

Start here after [CONTEXT.md](../CONTEXT.md) and [AGENTS.md](../AGENTS.md).

## Index

| Doc | Description |
|-----|-------------|
| [architecture.md](./architecture.md) | System overview, monorepo layout, data flows |
| [api.md](./api.md) | FastAPI REST + SSE contract |
| [ui-design.md](./ui-design.md) | Neo-brutalism dark design system |
| [data-layout.md](./data-layout.md) | `./data/` on-disk schema |
| [decisions.md](./decisions.md) | ADR log from product grill |

### LangGraph

| Doc | Graph |
|-----|-------|
| [langgraph/ingest.md](./langgraph/ingest.md) | Upload → parse → embed → Chroma |
| [langgraph/qa.md](./langgraph/qa.md) | Strict RAG Q&A |
| [langgraph/persona-train.md](./langgraph/persona-train.md) | Gemini persona activation |

### Ingest

| Doc | Topic |
|-----|-------|
| [ingest/whatsapp-export.md](./ingest/whatsapp-export.md) | WhatsApp `.txt` parser spec |

## Build order (recommended)

1. Backend skeleton + `data/` paths + health route
2. WhatsApp parser + ingest graph
3. Q&A graph + ask UI
4. Persona train graph + chat UI (build-time analysis, burst SSE, summarization)
5. Workspace analytics + overview panel
6. Settings + job SSE polish
