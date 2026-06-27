import { Citation } from "@/lib/api/types";

export function CitationBlock({ citation }: { citation: Citation }) {
  return (
    <div className="border-l-4 border-[var(--cm-accent)] bg-[var(--cm-surface-raised)] p-3">
      <p className="font-mono text-[10px] uppercase tracking-widest text-[var(--cm-text-muted)]">
        {citation.speaker} · {new Date(citation.timestamp).toLocaleString()}
        {citation.score != null ? ` · ${citation.score.toFixed(2)}` : ""}
      </p>
      <p className="mt-1 text-sm font-body text-[var(--cm-text)]">{citation.snippet}</p>
    </div>
  );
}
