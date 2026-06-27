"use client";

import { PersonAnalytics } from "@/lib/api/types";

// Accent palette cycles for multiple speakers
const SPEAKER_COLORS = [
  "#e8ff00", // accent yellow
  "#00ff88", // success green
  "#ffaa00", // warning orange
  "#00ccff", // cyan
  "#ff3333", // red
  "#cc88ff", // purple-ish
  "#ff88cc", // pink
  "#88ffcc", // mint
];

export function SpeakerShareChart({ people }: { people: PersonAnalytics[] }) {
  if (!people.length) {
    return (
      <p className="font-mono text-xs text-[var(--cm-text-muted)]">No speaker data.</p>
    );
  }

  // Sort descending by share so largest is on top
  const sorted = [...people].sort((a, b) => b.sharePercent - a.sharePercent);

  return (
    <ul className="flex flex-col gap-2">
      {sorted.map((person, i) => {
        const color = SPEAKER_COLORS[i % SPEAKER_COLORS.length];
        return (
          <li key={person.personId} className="grid grid-cols-[8rem_1fr_3.5rem] items-center gap-2">
            {/* Name */}
            <span
              className="truncate font-mono text-[10px] uppercase"
              style={{ color }}
              title={person.displayName}
            >
              {person.displayName}
            </span>

            {/* Bar track */}
            <div className="h-4 border border-[var(--cm-border-muted)] bg-[var(--cm-bg)]">
              <div
                className="h-full"
                style={{
                  width: `${Math.max(2, person.sharePercent)}%`,
                  backgroundColor: color,
                }}
              />
            </div>

            {/* Percent label */}
            <span className="text-right font-mono text-xs text-[var(--cm-text-muted)]">
              {person.sharePercent}%
            </span>
          </li>
        );
      })}
    </ul>
  );
}
