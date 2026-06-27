import { ActivityBucket } from "@/lib/api/types";

export function ActivityBars({
  buckets,
  emptyLabel = "No activity data",
}: {
  buckets: ActivityBucket[];
  emptyLabel?: string;
}) {
  if (!buckets.length) {
    return (
      <p className="font-mono text-xs text-[var(--cm-text-muted)]">{emptyLabel}</p>
    );
  }

  const max = Math.max(...buckets.map((b) => b.count), 1);

  return (
    <ul className="flex flex-col gap-2">
      {buckets.map((bucket) => (
        <li key={bucket.key} className="grid grid-cols-[4.5rem_1fr_2.5rem] items-center gap-2">
          <span className="font-mono text-[10px] uppercase text-[var(--cm-text-muted)]">
            {bucket.label}
          </span>
          <div className="h-3 border border-[var(--cm-border-muted)] bg-[var(--cm-bg)]">
            <div
              className="h-full bg-[var(--cm-accent)]"
              style={{ width: `${Math.max(4, (bucket.count / max) * 100)}%` }}
            />
          </div>
          <span className="text-right font-mono text-xs">{bucket.count}</span>
        </li>
      ))}
    </ul>
  );
}
