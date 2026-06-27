"use client";

import { HeatmapCell } from "@/lib/api/types";

const DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

/** Sparse hour labels shown along the x-axis */
const HOUR_TICKS = [0, 3, 6, 9, 12, 15, 18, 21];
function hourLabel(h: number) {
  if (h === 0) return "12A";
  if (h < 12) return `${h}A`;
  if (h === 12) return "12P";
  return `${h - 12}P`;
}

/** Map a normalised 0–1 ratio to a CSS background color */
function heatColor(ratio: number): string {
  if (ratio === 0) return "#1a1a1a"; // surface-raised — empty cell
  if (ratio < 0.2) return "#2a3300";
  if (ratio < 0.4) return "#4d6600";
  if (ratio < 0.65) return "#8ab800";
  if (ratio < 0.85) return "#c8e600";
  return "#e8ff00"; // accent — hottest
}

export function HeatmapGrid({ cells }: { cells: HeatmapCell[] }) {
  if (!cells.length) {
    return (
      <p className="font-mono text-xs text-[var(--cm-text-muted)]">No heatmap data.</p>
    );
  }

  // Build a quick lookup: grid[day][hour] = count
  const grid: number[][] = Array.from({ length: 7 }, () => new Array(24).fill(0));
  let maxVal = 0;
  for (const cell of cells) {
    grid[cell.day][cell.hour] = cell.count;
    if (cell.count > maxVal) maxVal = cell.count;
  }

  return (
    <div className="overflow-x-auto">
      <div style={{ minWidth: 480 }}>
        {/* Hour tick labels across the top */}
        <div className="mb-1 flex" style={{ paddingLeft: "3rem" }}>
          {Array.from({ length: 24 }, (_, h) => (
            <div
              key={h}
              className="flex-1 text-center font-mono text-[8px] text-[var(--cm-text-muted)]"
            >
              {HOUR_TICKS.includes(h) ? hourLabel(h) : ""}
            </div>
          ))}
        </div>

        {/* Rows: one per day */}
        {DAY_LABELS.map((day, d) => (
          <div key={d} className="mb-0.5 flex items-center gap-1">
            {/* Day label */}
            <span
              className="w-10 shrink-0 text-right font-mono text-[9px] uppercase text-[var(--cm-text-muted)]"
            >
              {day}
            </span>

            {/* 24 cells */}
            <div className="flex flex-1 gap-px">
              {Array.from({ length: 24 }, (_, h) => {
                const count = grid[d][h];
                const ratio = maxVal > 0 ? count / maxVal : 0;
                const bg = heatColor(ratio);
                return (
                  <div
                    key={h}
                    title={`${day} ${hourLabel(h)}: ${count} msgs`}
                    className="h-4 flex-1 border border-[#2a2a2a]"
                    style={{ backgroundColor: bg }}
                  />
                );
              })}
            </div>
          </div>
        ))}

        {/* Legend */}
        <div className="mt-2 flex items-center gap-2 pl-11">
          <span className="font-mono text-[9px] uppercase text-[var(--cm-text-muted)]">Low</span>
          {["#2a3300", "#4d6600", "#8ab800", "#c8e600", "#e8ff00"].map((c) => (
            <div key={c} className="h-3 w-6 border border-[#333]" style={{ backgroundColor: c }} />
          ))}
          <span className="font-mono text-[9px] uppercase text-[var(--cm-text-muted)]">High</span>
        </div>
      </div>
    </div>
  );
}
