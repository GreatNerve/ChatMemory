"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { WeeklyPoint } from "@/lib/api/types";

type ViewMode = "week" | "month";

/**
 * Decide how many ticks to skip so the X-axis doesn't get crowded.
 * Monthly mode has fewer bars so the thresholds are lower.
 */
function computeTickInterval(length: number, mode: ViewMode): number {
  if (mode === "month") {
    if (length <= 12) return 0; // show all months
    if (length <= 24) return 1; // every 2nd month
    return 2;                   // every 3rd month
  }
  // Weekly mode — original thresholds
  if (length <= 12) return 0; // show all
  if (length <= 26) return 1; // every 2nd
  if (length <= 52) return 3; // every 4th
  return 7;                   // every 8th
}

function BrutalTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ value: number }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="border-2 border-[var(--cm-border)] bg-[var(--cm-surface)] p-2"
      style={{ fontFamily: "monospace" }}
    >
      <p className="text-[10px] uppercase tracking-widest text-[var(--cm-text-muted)]">
        {label}
      </p>
      <p className="mt-1 text-sm text-[var(--cm-accent)]">{payload[0].value} msgs</p>
    </div>
  );
}

export function ConversationGrowthChart({
  series,
  mode = "week",
}: {
  series: WeeklyPoint[];
  /** Controls tick-interval calculation: "week" (default) or "month". */
  mode?: ViewMode;
}) {
  if (!series.length) {
    return (
      <p className="font-mono text-xs text-[var(--cm-text-muted)]">No time-series data.</p>
    );
  }

  const interval = computeTickInterval(series.length, mode);

  return (
    /* Recharts needs an explicit pixel height inside ResponsiveContainer */
    <ResponsiveContainer width="100%" height={180}>
      <BarChart data={series} margin={{ top: 4, right: 4, left: -20, bottom: 0 }} barCategoryGap="20%">
        <CartesianGrid
          vertical={false}
          stroke="#333333"
          strokeDasharray="4 4"
        />
        <XAxis
          dataKey="label"
          interval={interval}
          tick={{ fill: "#a3a3a3", fontFamily: "monospace", fontSize: 9 }}
          axisLine={{ stroke: "#444444" }}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: "#a3a3a3", fontFamily: "monospace", fontSize: 9 }}
          axisLine={false}
          tickLine={false}
          width={38}
        />
        <Tooltip content={<BrutalTooltip />} cursor={{ fill: "#ffffff10" }} />
        <Bar
          dataKey="count"
          fill="#e8ff00"
          radius={0}  /* no rounded corners — brutal */
          isAnimationActive={false}
        />
      </BarChart>
    </ResponsiveContainer>
  );
}
