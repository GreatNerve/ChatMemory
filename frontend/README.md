# ChatMemory Frontend

Next.js neo-brutalism UI — see repo `docs/ui-design.md`.

## Setup

```bash
cd frontend
cp .env.local.example .env.local
pnpm install
```

If `pnpm install` warns about ignored build scripts, `pnpm-workspace.yaml` already allows `sharp` and `unrs-resolver` (required by Next.js / ESLint). Or run `pnpm approve-builds` interactively.

## Run

```bash
pnpm dev
```

Open `http://localhost:3000`. Backend must run on `http://127.0.0.1:8000`.

## Stack

- Next.js App Router
- TanStack Query + Zod
- Tailwind v4
- Custom brutal components (no shadcn)
