# ChatMemory — UI design

Neo-brutalism, **dark mode only** (MVP). Custom components — no shadcn. Next.js App Router.

Design goal: feel like a forensic tool for your chats — raw, loud, precise. Not a soft SaaS dashboard.

---

## Design principles

1. **Hard edges** — zero border radius on containers and buttons
2. **High contrast** — near-black surfaces, off-white text, one acid accent
3. **Visible structure** — thick borders instead of subtle shadows (except primary CTA offset shadow)
4. **Monospace for system voice** — labels, nav, metadata, timestamps
5. **Density where data is heavy** — tables, citations, chat logs are compact
6. **One accent, used sparingly** — lime/yellow for primary actions and active nav only
7. **No purple gradients, no glassmorphism, no rounded cards**

---

## Color tokens

Define in `frontend/src/app/globals.css` as CSS variables:

```css
:root {
  --cm-bg: #0a0a0a;
  --cm-surface: #141414;
  --cm-surface-raised: #1a1a1a;
  --cm-border: #f5f5f5;
  --cm-border-muted: #444444;
  --cm-text: #f5f5f5;
  --cm-text-muted: #a3a3a3;
  --cm-accent: #e8ff00;
  --cm-accent-fg: #0a0a0a;
  --cm-error: #ff3333;
  --cm-warning: #ffaa00;
  --cm-success: #00ff88;
}
```

| Token | Use |
|-------|-----|
| `--cm-bg` | Page background |
| `--cm-surface` | Panels, sidebar, cards |
| `--cm-border` | 2–4px structural borders |
| `--cm-accent` | Primary button, active nav, focus ring |
| `--cm-error` | NOT_FOUND banners, validation |
| `--cm-warning` | Thin persona warning, force train |
| `--cm-success` | Job done, model ready |

---

## Typography

| Role | Font | Weight | Use |
|------|------|--------|-----|
| Display / UI chrome | [Space Mono](https://fonts.google.com/specimen/Space+Mono) | 400, 700 | Nav, headings, labels, stats, buttons |
| Body | [IBM Plex Sans](https://fonts.google.com/specimen/IBM+Plex+Sans) | 400, 500 | Paragraphs, chat bubbles, answers |

Load via `next/font/google`.

| Element | Size |
|---------|------|
| Page title | `text-2xl` / `text-3xl` mono uppercase tracking-tight |
| Section label | `text-xs` mono uppercase tracking-widest text-muted |
| Body | `text-sm` or `text-base` |
| Metadata | `text-xs` mono text-muted |

---

## Spacing and borders

| Rule | Value |
|------|-------|
| Border width | `2px` default; `4px` for page frame / modal |
| Border radius | `0` everywhere |
| Panel padding | `p-4` compact; `p-6` for hero upload zone only |
| Gap | `gap-2` / `gap-4` — avoid `space-y-*` (use flex/grid gap) |
| Max content width | `max-w-6xl` for main column |

---

## Shadows (neo-brutalist)

Primary button only:

```css
.cm-shadow-brutal {
  box-shadow: 4px 4px 0 var(--cm-accent);
}
.cm-shadow-brutal:active {
  box-shadow: 2px 2px 0 var(--cm-accent);
  transform: translate(2px, 2px);
}
```

No shadow on cards — border carries structure.

---

## Layout shell

```
┌────────────────────────────────────────────────────────────┐
│  CHATMEMORY          [workspace name]              SETTINGS │
├──────────────┬─────────────────────────────────────────────┤
│  NAV         │  MAIN                                        │
│              │                                              │
│  WORKSPACES  │  Page content                                │
│  > active    │                                              │
│    Overview  │                                              │
│    Ask       │                                              │
│    People    │                                              │
│              │                                              │
└──────────────┴─────────────────────────────────────────────┘
```

- **Sidebar** fixed left `w-56`, `border-r-4 border-[var(--cm-border)]`, `bg-surface`
- **Header** `border-b-2`, mono app name
- **Mobile** (`< md`): sidebar → full-width drawer with same brutal borders (no soft sheet animation — slide or instant)

---

## Components (custom)

Build under `frontend/src/components/`:

| Component | Purpose |
|-----------|---------|
| `BrutalButton` | primary / ghost / danger variants |
| `BrutalPanel` | bordered surface container |
| `BrutalInput`, `BrutalTextarea` | thick border, no radius |
| `BrutalBadge` | status chips (persona status, ingest) |
| `BrutalTable` | dense rows, `border-b` dividers |
| `CitationBlock` | Q&A source snippet |
| `JobProgress` | SSE step + percent bar (square) |
| `UploadZone` | dashed `4px` border, mono "DROP .TXT" |
| `ChatThread` | persona chat messages (burst bubbles) |
| `PersonaChatPanel` | chat input, PDF export, fullscreen |
| `WorkspaceAnalyticsPanel` | overview rhythm stats + growth chart |
| `AppShell` | sidebar + header layout |

### Button variants

| Variant | Style |
|---------|-------|
| Primary | `bg-accent text-accent-fg border-2 border-border brutal shadow` |
| Ghost | `bg-transparent border-2 border-border hover:bg-surface-raised` |
| Danger | `border-error text-error` |

### Persona status badges

| Status | Color |
|--------|-------|
| `not_enough` | muted border |
| `thin` | warning |
| `ready` | accent outline |
| `training` | accent fill pulse |
| `ready_model` | success |
| `error` | error |

---

## Page specs

### `/` — Workspaces

- Large mono heading: `WORKSPACES`
- Grid of workspace cards: name, msg count, date range, ingest badge
- Prominent `+ NEW WORKSPACE` opens upload panel
- Upload: name input + file picker + `INGEST` primary button
- On submit → redirect to workspace or show job progress inline

### `/workspace/[id]` — Overview

- Stat row: messages, speakers, date range (mono numbers)
- Top speakers list with links to people
- Quick links: `ASK` / `PEOPLE` as brutal buttons
- **Workspace analytics** panel (`WorkspaceAnalyticsPanel`):
  - **Conversation rhythm** (1-on-1, ≤2 speakers) or **Group rhythm** (3+ speakers) — busiest hour/day, typical reply (median), messages per day
  - **Conversation growth** chart with **W** / **M** toggle: weekly (last 52 weeks, capped at today) vs monthly (all time, aggregated)
  - Hour×day heatmap, per-person reply histograms (`<1m` includes same-minute replies), pair connectivity table (group only)

### `/workspace/[id]/ask`

- Textarea for question
- Optional filters: speaker select, date from/to
- Submit → loading state (mono `SEARCHING...`)
- Answer panel with body text
- **Citations** below: each in `CitationBlock` — speaker, timestamp, snippet, thick left border accent
- **NOT_FOUND**: red banner, near-misses in muted panels

### `/workspace/[id]/people`

- Grid of speaker cards: name, msg count, persona badge
- Click → person detail

### `/workspace/[id]/people/[pid]`

- Header: display name + status badge
- Style stats row (avg length, emoji rate, hinglish ratio)
- Build-time notes when present: personality, writing style, chat analysis (compact panels)
- Sample messages list (scrollable, compact)
- **Build persona** section:
  - Consent checkbox (required): "I have permission to mimic this person for personal use"
  - Train button disabled until consent + eligible
  - Warning banner if thin (50–199 msgs)
  - `JobProgress` during train (steps: samples → style → chat analysis → personality → writing style → activate)
- **Chat** section below (only if `ready_model`):
  - Thread with burst bubbles (`msgBreak` SSE events)
  - Indicator when earlier turns are summarized
  - **Download PDF** — exports session history
  - **Fullscreen** overlay (Esc to exit)
  - Rolling summarization when history &gt; 24 turns (keeps last 10 verbatim)

### `/settings`

- Data root path (read-only display for MVP)
- Embed model + device, vector store, Gemini model status
- `Open data folder` button (if OS integration added later)
- GPU status: available / busy / active job
- Health indicators from `GET /health` (`embedReady`, `geminiConfigured`)

---

## Motion

Restrained — brutalism avoids bouncy UI.

| Interaction | Motion |
|-------------|--------|
| Button press | 2px translate + shadow shrink (CSS) |
| Page enter | optional 150ms fade — no slide cascades |
| Job progress | width transition on bar only |
| No | parallax, blur, spring physics |

---

## Accessibility

- Minimum touch target `44px` on mobile buttons
- Focus: `outline-2 outline-offset-2 outline-accent`
- Error text not color-only — include label
- Citation snippets: readable contrast on `--cm-surface-raised`

---

## Tailwind notes

- Use `cn()` utility for class merge
- Prefer semantic CSS variables over raw hex in components
- Do **not** install shadcn/ui
- Dark only — no `prefers-color-scheme` light branch

---

## Reference mood

- Raw grid, newspaper harshness
- Acid yellow on black (rave flyer energy)
- Terminal metadata + human chat body font pairing

---

## Related

- [architecture.md](./architecture.md) — routes
- [api.md](./api.md) — data for UI states
