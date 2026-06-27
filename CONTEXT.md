# CONTEXT — ChatMemory



Domain and locked decisions. Workflow: [AGENTS.md](./AGENTS.md).



## What it is



**ChatMemory** is a local-first desktop web app: upload a WhatsApp group export, search past conversations (RAG), and build per-speaker persona chatbots (Gemini style mimic) that reply in that person's voice. Hinglish and English only.



## Terms



| Term | Meaning |

|------|---------|

| Operator | The human using the app on their machine (single user, no login) |

| Workspace | One imported WhatsApp group chat |

| Speaker / Person | Someone who sent messages in the export (not an app account) |

| Message chunk | One chat line stored and embedded for retrieval (usually 1 WhatsApp message) |

| Persona | Gemini-powered style mimic for a speaker (style profile + samples) |

| Q&A | Grounded RAG over chat history with citations |

| Persona chat | Style mimic only — not factual memory lookup |

| Ingest | Preprocess export → parse → index speakers → embed → Chroma |

| Job | Background work (ingest or persona activation) with SSE progress |



## Stack (locked)



| Layer | Choice |

|-------|--------|

| Frontend | Next.js (App Router), pnpm, TanStack Query, Zod |

| Backend | FastAPI, uv, LangGraph |

| Reads / Q&A | Chroma + multilingual-e5-large + hybrid BM25 + Gemini rerank/generate |

| Persona | Gemini activation (style profile + samples) |

| Embeddings | `intfloat/multilingual-e5-large` via sentence-transformers (CUDA or CPU) |

| Data | `./data/` at repo root (gitignored) |

| UI | Neo-brutalism, dark mode only |



## Grill (locked)



| # | Decision |

|---|----------|

| 1 | Local single-operator app — no auth |

| 2 | Person = chat speaker |

| 3 | MVP: one workspace per group; future: global library |

| 4 | Persona via Gemini style mimic (not LoRA) |

| 5 | Target hardware: RTX 3050 6GB, Windows — GPU for embed when available |

| 6 | Next.js frontend + FastAPI backend — two terminals |

| 7 | Gemini for LLM; local Python for embed + Chroma |

| 8 | Neo-brutalism dark UI — no shadcn |

| 9 | MVP routes: workspaces, overview, ask, people, person detail, settings |

| 10 | Input: WhatsApp `.txt` export only (with preprocess pipeline) |

| 11 | Embeddings: `intfloat/multilingual-e5-large` at ingest |

| 12 | Vector DB: Chroma per workspace (LangChain adapter) |

| 13 | LangGraph: 3 graphs (ingest, qa, persona_train) |

| 14 | Data root: `./data/` (dev); gitignored |

| 15 | Persona: min 50 msgs (thin), recommend 200+, consent checkbox |

| 16 | Languages: Hinglish + English; answer in question's language mix |

| 17 | API: REST + SSE for jobs |

| 18 | Q&A: strict grounding — refuse when retrieval weak |

| 19 | RAG: hybrid retrieve + Gemini LLM rerank |

| 20 | Frontend: TanStack Query + pnpm + Zod |

| 21 | Monorepo: `backend/` + `frontend/` + `docs/` |



## Out of scope (MVP)



- Multi-user accounts / cloud sync

- Telegram, Discord, or other chat formats

- Cross-group speaker merge (future)

- Analytics dashboards (topic charts, word clouds)

- Persona chat with RAG facts (future hybrid)

- Light mode / theme toggle

- Translation between Hindi script and Roman Hinglish

- Local LoRA / Ollama inference



## Docs



| Path | Contents |

|------|----------|

| [docs/architecture.md](./docs/architecture.md) | System overview |

| [docs/api.md](./docs/api.md) | REST + SSE contract |

| [docs/ui-design.md](./docs/ui-design.md) | Neo-brutalism design system |

| [docs/data-layout.md](./docs/data-layout.md) | On-disk layout |

| [docs/decisions.md](./docs/decisions.md) | ADR log |

| [docs/langgraph/](./docs/langgraph/) | Graph flows |

| [docs/ingest/whatsapp-export.md](./docs/ingest/whatsapp-export.md) | Parser spec |

