"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { PersonAnalytics } from "@/lib/api/types";

// Colors matching the brutal accent palette
const SPEAKER_COLORS = [
  "#e8ff00",
  "#00ff88",
  "#ffaa00",
  "#00ccff",
  "#ff3333",
  "#cc88ff",
  "#ff88cc",
];

function BrutalTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ value: number | null; name: string }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  const val = payload[0].value;
  return (
    <div
      className="border-2 border-[var(--cm-border)] bg-[var(--cm-surface)] p-2"
      style={{ fontFamily: "monospace" }}
    >
      <p className="text-[10px] uppercase tracking-widest text-[var(--cm-text-muted)]">{label}</p>
      <p className="mt-1 text-sm text-[var(--cm-accent)]">
        {val != null ? `${Math.round(val)}s avg` : "—"}
      </p>
    </div>
  );
}

/** Avg response time per speaker — horizontal bars via recharts */
export function ResponseTimeChart({ people }: { people: PersonAnalytics[] }) {
  const withData = people.filter((p) => p.avgResponseSeconds != null);

  if (!withData.length) {
    return (
      <p className="font-mono text-xs text-[var(--cm-text-muted)]">
        Not enough reply data to show response times.
      </p>
    );
  }

  // Recharts data row
  const chartData = withData.map((p) => ({
    name: p.displayName,
    seconds: p.avgResponseSeconds,
  }));

  return (
    <ResponsiveContainer width="100%" height={Math.max(120, chartData.length * 36)}>
      <BarChart
        layout="vertical"
        data={chartData}
        margin={{ top: 4, right: 40, left: 0, bottom: 4 }}
        barCategoryGap="25%"
      >
        <CartesianGrid horizontal={false} stroke="#333333" strokeDasharray="4 4" />
        <XAxis
          type="number"
          tick={{ fill: "#a3a3a3", fontFamily: "monospace", fontSize: 9 }}
          axisLine={{ stroke: "#444444" }}
          tickLine={false}
          tickFormatter={(v) => `${Math.round(v)}s`}
        />
        <YAxis
          type="category"
          dataKey="name"
          width={80}
          tick={{ fill: "#a3a3a3", fontFamily: "monospace", fontSize: 9 }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip content={<BrutalTooltip />} cursor={{ fill: "#ffffff10" }} />
        <Bar dataKey="seconds" radius={0} isAnimationActive={false}>
          {chartData.map((_, i) => (
            <Cell key={i} fill={SPEAKER_COLORS[i % SPEAKER_COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Per-person response-time bucket histogram (collapsed) */
export function ResponseTimeBucketBars({ person }: { person: PersonAnalytics }) {
  const buckets = person.responseTimeBuckets ?? [];
  const nonEmpty = buckets.filter((b) => b.count > 0);
  if (!nonEmpty.length) {
    return (
      <p className="font-mono text-[10px] text-[var(--cm-text-muted)]">No reply data.</p>
    );
  }

  const max = Math.max(...buckets.map((b) => b.count), 1);

  return (
    <ul className="flex flex-col gap-1">
      {buckets.map((b) => (
        <li key={b.label} className="grid grid-cols-[3.5rem_1fr_2.5rem] items-center gap-2">
          <span className="font-mono text-[9px] uppercase text-[var(--cm-text-muted)]">
            {b.label}
          </span>
          <div className="h-2.5 border border-[var(--cm-border-muted)] bg-[var(--cm-bg)]">
            <div
              className="h-full bg-[var(--cm-success)]"
              style={{ width: `${Math.max(b.count > 0 ? 4 : 0, (b.count / max) * 100)}%` }}
            />
          </div>
          <span className="text-right font-mono text-[10px] text-[var(--cm-text-muted)]">
            {b.count}
          </span>
        </li>
      ))}
    </ul>
  );
}
