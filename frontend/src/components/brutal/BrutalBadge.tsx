import { cn } from "@/lib/cn";
import { PersonaStatus, IngestStatus } from "@/lib/api/types";

const personaLabels: Record<PersonaStatus, string> = {
  not_enough: "NOT ENOUGH",
  thin: "THIN",
  ready: "READY",
  training: "TRAINING",
  ready_model: "MODEL READY",
  error: "ERROR",
};

const personaColors: Record<PersonaStatus, string> = {
  not_enough: "border-[var(--cm-border-muted)] text-[var(--cm-text-muted)]",
  thin: "border-[var(--cm-warning)] text-[var(--cm-warning)]",
  ready: "border-[var(--cm-accent)] text-[var(--cm-accent)]",
  training: "bg-[var(--cm-accent)] text-[var(--cm-accent-fg)] border-[var(--cm-accent)] animate-pulse",
  ready_model: "border-[var(--cm-success)] text-[var(--cm-success)]",
  error: "border-[var(--cm-error)] text-[var(--cm-error)]",
};

const ingestLabels: Record<IngestStatus, string> = {
  pending: "PENDING",
  running: "INGESTING",
  done: "READY",
  error: "ERROR",
};

export function PersonaBadge({ status }: { status: PersonaStatus }) {
  return (
    <span
      className={cn(
        "inline-block border-2 px-2 py-0.5 font-mono text-[10px] uppercase tracking-widest",
        personaColors[status],
      )}
    >
      {personaLabels[status]}
    </span>
  );
}

export function IngestBadge({ status }: { status: IngestStatus }) {
  const color =
    status === "done"
      ? "border-[var(--cm-success)] text-[var(--cm-success)]"
      : status === "error"
        ? "border-[var(--cm-error)] text-[var(--cm-error)]"
        : status === "running"
          ? "border-[var(--cm-accent)] text-[var(--cm-accent)] animate-pulse"
          : "border-[var(--cm-border-muted)] text-[var(--cm-text-muted)]";

  return (
    <span
      className={cn(
        "inline-block border-2 px-2 py-0.5 font-mono text-[10px] uppercase tracking-widest",
        color,
      )}
    >
      {ingestLabels[status]}
    </span>
  );
}
