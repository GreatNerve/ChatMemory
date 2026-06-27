# LangGraph flows

Node-level documentation for each compiled graph in `backend/app/graphs/`.

| Graph | Doc | Trigger |
|-------|-----|---------|
| Ingest | [ingest.md](./ingest.md) | Workspace upload |
| Q&A | [qa.md](./qa.md) | `POST .../ask` |
| Persona train | [persona-train.md](./persona-train.md) | `POST .../train` |
| Persona chat | [persona-chat.md](./persona-chat.md) | `POST .../chat` |

See [../architecture.md](../architecture.md) for how graphs fit the monorepo.
