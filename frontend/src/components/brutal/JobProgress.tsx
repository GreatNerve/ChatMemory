import { JobSnapshot } from "@/lib/api/types";
import { formatEta } from "@/lib/formatEta";

const STEP_LABELS: Record<string, string> = {
  queued: "Queued",
  parsing: "Parsing WhatsApp export",
  extracting_people: "Finding speakers",
  chunking: "Preparing search index",
  embedding: "Embedding messages",
  saving_index: "Writing search index",
  finalizing: "Saving workspace",
  validating: "Validating",
  refreshing_samples: "Refreshing samples",
  style_profile: "Building style profile",
  activating: "Activating persona",
  done: "Complete",
};

function labelFor(job: JobSnapshot | null, connecting: boolean): string {
  if (connecting && !job) return "Connecting to job…";
  if (!job) return "Starting…";
  if (job.message) return job.message;
  if (job.step && STEP_LABELS[job.step]) return STEP_LABELS[job.step];
  if (job.status === "queued") return "Queued — starting soon";
  if (job.status === "running") return "Processing…";
  if (job.status === "done") return "Complete";
  if (job.status === "error") return "Failed";
  return job.status;
}

export function JobProgress({
  job,
  connecting = false,
  title = "Job progress",
}: {
  job: JobSnapshot | null;
  connecting?: boolean;
  title?: string;
}) {
  const percent = job?.percent ?? 0;
  const label = labelFor(job, connecting);
  const eta =
    job?.status === "running" && job.etaSeconds != null && job.etaSeconds > 0
      ? formatEta(job.etaSeconds)
      : null;

  return (
    <div className="flex flex-col gap-2 border-2 border-[var(--cm-border-muted)] bg-[var(--cm-bg)] p-3">
      <p className="font-mono text-[10px] uppercase tracking-widest text-[var(--cm-accent)]">
        {title}
      </p>
      <div className="flex justify-between gap-3 font-mono text-xs uppercase tracking-widest text-[var(--cm-text-muted)]">
        <span className="min-w-0 flex-1 truncate">{label}</span>
        <span className="shrink-0 text-right">
          {percent}%
          {eta ? (
            <>
              <span className="mx-1 text-[var(--cm-border-muted)]">·</span>
              <span className="normal-case tracking-normal">{eta}</span>
            </>
          ) : null}
        </span>
      </div>
      <div className="h-3 border-2 border-[var(--cm-border)] bg-[var(--cm-surface)]">
        <div
          className="h-full bg-[var(--cm-accent)] transition-[width] duration-300"
          style={{ width: `${Math.max(percent, job ? 2 : 0)}%` }}
        />
      </div>
      {job?.error ? (
        <p className="text-sm text-[var(--cm-error)] font-body whitespace-pre-wrap">{job.error}</p>
      ) : null}
    </div>
  );
}
