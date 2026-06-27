"use client";

import Link from "next/link";
import { useState } from "react";
import { BrutalButton } from "@/components/brutal/BrutalButton";
import { BrutalPanel } from "@/components/brutal/BrutalPanel";
import { ActivityBars } from "@/components/modules/overview/ActivityBars";
import { ConversationGrowthChart } from "@/components/modules/overview/ConversationGrowthChart";
import { HeatmapGrid } from "@/components/modules/overview/HeatmapGrid";
import { SpeakerShareChart } from "@/components/modules/overview/SpeakerShareChart";
import { ResponseTimeChart, ResponseTimeBucketBars } from "@/components/modules/overview/ResponseTimeChart";
import { ApiError } from "@/lib/api/client";
import {
  GroupAnalytics,
  PairAnalytics,
  PersonAnalytics,
  WeeklyPoint,
  WorkspaceAnalytics,
} from "@/lib/api/types";

function StatCell({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <p className="font-mono text-[10px] uppercase tracking-widest text-[var(--cm-text-muted)]">
        {label}
      </p>
      <p className="mt-1 font-mono text-sm">{value}</p>
    </div>
  );
}

function ConnectionBar({ score }: { score: number }) {
  return (
    <div className="flex items-center gap-2">
      <div className="h-2 flex-1 border border-[var(--cm-border-muted)] bg-[var(--cm-bg)]">
        <div
          className="h-full bg-[var(--cm-accent)]"
          style={{ width: `${Math.min(100, score)}%` }}
        />
      </div>
      <span className="w-10 text-right font-mono text-xs">{score}%</span>
    </div>
  );
}

function PanelTitle({ children }: { children: React.ReactNode }) {
  return (
    <p className="mb-4 font-mono text-xs uppercase tracking-widest text-[var(--cm-text-muted)]">
      {children}
    </p>
  );
}

function ChatStats({ group, isGroup }: { group: GroupAnalytics; isGroup: boolean }) {
  const strongest = group.strongestPair;

  return (
    <BrutalPanel>
      {/* Title adapts: "Group rhythm" for 3+ speakers, "Conversation rhythm" for 1-on-1 */}
      <PanelTitle>{isGroup ? "Group rhythm" : "Conversation rhythm"}</PanelTitle>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCell label="Typical reply" value={group.avgResponseLabel ?? "—"} />
        <StatCell label="Busiest hour" value={group.busiestHourLabel ?? "—"} />
        <StatCell label="Busiest day" value={group.busiestDay ?? "—"} />
        <StatCell label="Msgs / day" value={group.medianMessagesPerDay} />
      </div>

      {strongest ? (
        <p className="mt-4 border-t border-[var(--cm-border-muted)] pt-3 font-mono text-xs text-[var(--cm-text-muted)]">
          {/* "Strongest pair" meaningful for groups; for 1-on-1 it's the only pair */}
          {isGroup ? "Strongest pair" : "Connection"}:{" "}
          <span className="text-[var(--cm-text)]">
            {strongest.personAName} ↔ {strongest.personBName}
          </span>{" "}
          ({strongest.connectionScore}% connected · {strongest.exchanges} exchanges)
        </p>
      ) : null}

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <div>
          <p className="mb-2 font-mono text-[10px] uppercase tracking-widest text-[var(--cm-text-muted)]">
            Peak hours
          </p>
          <ActivityBars buckets={group.activeHours} />
        </div>
        <div>
          <p className="mb-2 font-mono text-[10px] uppercase tracking-widest text-[var(--cm-text-muted)]">
            Active days
          </p>
          <ActivityBars buckets={group.activeDays} />
        </div>
      </div>
    </BrutalPanel>
  );
}

/** ISO week key for today, e.g. "2026-W26". */
function currentWeekKey(): string {
  const now = new Date();
  // Compute ISO week number per ISO 8601 (week starts Monday, week 1 contains first Thursday)
  const d = new Date(Date.UTC(now.getFullYear(), now.getMonth(), now.getDate()));
  d.setUTCDate(d.getUTCDate() + 4 - (d.getUTCDay() || 7));
  const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
  const week = Math.ceil(((d.getTime() - yearStart.getTime()) / 86400000 + 1) / 7);
  return `${d.getUTCFullYear()}-W${String(week).padStart(2, "0")}`;
}

/** Current month key for today, e.g. "2026-06". */
function currentMonthKey(): string {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}`;
}

/**
 * Convert an ISO week key ("YYYY-WXX") to the Monday Date of that week.
 * Week 1 is defined as the week containing Jan 4 (ISO 8601).
 */
function weekKeyToDate(weekKey: string): Date {
  const [yearStr, weekStr] = weekKey.split("-W");
  const year = parseInt(yearStr, 10);
  const week = parseInt(weekStr, 10);
  // Jan 4 is always in ISO week 1
  const jan4 = new Date(Date.UTC(year, 0, 4));
  // Rewind to the Monday of that week, then advance by (week-1) full weeks
  const monday = new Date(jan4);
  monday.setUTCDate(jan4.getUTCDate() - ((jan4.getUTCDay() || 7) - 1) + (week - 1) * 7);
  return monday;
}

/**
 * Aggregate a weekly series into monthly buckets.
 * Each returned point has week="YYYY-MM", label="Jan '24", count=sum.
 * Weeks that straddle a month boundary are attributed to the month that
 * contains the Monday (start) of the week.
 */
function aggregateToMonthly(series: WeeklyPoint[]): WeeklyPoint[] {
  const buckets = new Map<string, number>();

  for (const pt of series) {
    const date = weekKeyToDate(pt.week);
    const monthKey = `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, "0")}`;
    buckets.set(monthKey, (buckets.get(monthKey) ?? 0) + pt.count);
  }

  // Sort chronologically and build WeeklyPoint-compatible objects
  return Array.from(buckets.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([monthKey, count]) => {
      const [yearStr, monthStr] = monthKey.split("-");
      // Build a label like "Jan '24" using the locale API
      const date = new Date(parseInt(yearStr, 10), parseInt(monthStr, 10) - 1, 1);
      const shortMonth = date.toLocaleDateString("en-US", { month: "short" });
      const shortYear = String(date.getFullYear()).slice(-2);
      return { week: monthKey, label: `${shortMonth} '${shortYear}`, count };
    });
}

type ViewMode = "week" | "month";

function ConversationGrowth({ series }: { series: WeeklyPoint[] }) {
  const [view, setView] = useState<ViewMode>("week");

  // Week mode: drop future weeks, keep last 52
  const cutoff = currentWeekKey();
  const upToToday = series.filter((w) => w.week <= cutoff);
  const weekSeries = upToToday.slice(-52);

  // Month mode: aggregate all data up to today into calendar months
  const monthCutoff = currentMonthKey();
  // Filter future weekly points before aggregating so partial future months are excluded
  const monthSeries = aggregateToMonthly(upToToday).filter(
    (m) => m.week <= monthCutoff,
  );

  const displaySeries = view === "week" ? weekSeries : monthSeries;
  // Show "past 1 year" badge only in week mode when data was trimmed
  const showYearBadge = view === "week" && weekSeries.length < series.length;

  return (
    <BrutalPanel>
      <div className="mb-4 flex items-center justify-between gap-2">
        <p className="font-mono text-xs uppercase tracking-widest text-[var(--cm-text-muted)]">
          Conversation growth —{" "}
          {view === "week" ? "messages per week" : "messages per month"}
        </p>

        <div className="flex items-center gap-2">
          {/* "past 1 year" label in week mode when series was trimmed */}
          {showYearBadge && (
            <span className="font-mono text-[10px] uppercase text-[var(--cm-text-muted)]">
              past 1 year
            </span>
          )}
          {/* "all time" label in month mode */}
          {view === "month" && (
            <span className="font-mono text-[10px] uppercase text-[var(--cm-text-muted)]">
              all time
            </span>
          )}

          {/* W / M toggle — neo-brutalism style: no rounding, hard borders */}
          <div className="flex">
            <button
              type="button"
              onClick={() => setView("week")}
              className={`border border-[var(--cm-border)] px-2 py-0.5 font-mono text-xs uppercase transition-colors ${
                view === "week"
                  ? "bg-[var(--cm-accent)] text-[var(--cm-bg)]"
                  : "bg-transparent text-[var(--cm-text-muted)] hover:text-[var(--cm-text)]"
              }`}
            >
              W
            </button>
            {/* Collapse the shared border between the two buttons */}
            <button
              type="button"
              onClick={() => setView("month")}
              className={`-ml-px border border-[var(--cm-border)] px-2 py-0.5 font-mono text-xs uppercase transition-colors ${
                view === "month"
                  ? "bg-[var(--cm-accent)] text-[var(--cm-bg)]"
                  : "bg-transparent text-[var(--cm-text-muted)] hover:text-[var(--cm-text)]"
              }`}
            >
              M
            </button>
          </div>
        </div>
      </div>

      <ConversationGrowthChart series={displaySeries} mode={view} />
    </BrutalPanel>
  );
}

function ActivityHeatmap({ group }: { group: GroupAnalytics }) {
  return (
    <BrutalPanel>
      <PanelTitle>Activity heatmap — hour × day</PanelTitle>
      <HeatmapGrid cells={group.heatmap} />
    </BrutalPanel>
  );
}

function SpeakerShare({ people }: { people: PersonAnalytics[] }) {
  return (
    <BrutalPanel>
      <PanelTitle>Speaker share — % of total messages</PanelTitle>
      <SpeakerShareChart people={people} />
    </BrutalPanel>
  );
}

function ResponseTimeOverview({ people }: { people: PersonAnalytics[] }) {
  return (
    <BrutalPanel>
      <PanelTitle>Avg response time per speaker</PanelTitle>
      <ResponseTimeChart people={people} />
    </BrutalPanel>
  );
}

function TopActiveWeeks({ weeks }: { weeks: WeeklyPoint[] }) {
  if (!weeks.length) return null;
  const max = Math.max(...weeks.map((w) => w.count), 1);

  return (
    <BrutalPanel>
      <PanelTitle>Top 5 busiest weeks</PanelTitle>
      <ul className="flex flex-col gap-2">
        {weeks.map((w, i) => (
          <li key={w.week} className="grid grid-cols-[1rem_6rem_1fr_3rem] items-center gap-3">
            <span className="font-mono text-[10px] text-[var(--cm-text-muted)]">#{i + 1}</span>
            <span className="font-mono text-[10px] uppercase text-[var(--cm-text-muted)]">
              {w.label}
            </span>
            <div className="h-3 border border-[var(--cm-border-muted)] bg-[var(--cm-bg)]">
              <div
                className="h-full bg-[var(--cm-accent)]"
                style={{ width: `${(w.count / max) * 100}%` }}
              />
            </div>
            <span className="text-right font-mono text-xs">{w.count}</span>
          </li>
        ))}
      </ul>
    </BrutalPanel>
  );
}

function PersonCard({
  person,
  workspaceId,
  expanded,
  onToggle,
}: {
  person: PersonAnalytics;
  workspaceId: string;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <BrutalPanel className="flex flex-col gap-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <Link
            href={`/workspace/${workspaceId}/people/${person.personId}`}
            className="font-mono text-sm font-bold uppercase hover:text-[var(--cm-accent)]"
          >
            {person.displayName}
          </Link>
          <p className="mt-1 font-mono text-xs text-[var(--cm-text-muted)]">
            {person.messageCount} msgs · {person.sharePercent}% of chat
          </p>
        </div>
        <BrutalButton variant="ghost" type="button" onClick={onToggle}>
          {expanded ? "Less" : "Details"}
        </BrutalButton>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatCell label="Typical reply" value={person.avgResponseLabel ?? "—"} />
        <StatCell label="Peak hour" value={person.peakHourLabel ?? "—"} />
        <StatCell label="Starts convos" value={person.initiations} />
        <StatCell label="Avg length" value={`${person.avgMessageLength} chars`} />
      </div>

      {expanded ? (
        <div className="border-t border-[var(--cm-border-muted)] pt-4">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <StatCell
              label="Median reply (raw)"
              value={
                person.medianResponseSeconds != null
                  ? `${Math.round(person.medianResponseSeconds)}s`
                  : "—"
              }
            />
            <StatCell label="Replies given" value={person.repliesGiven} />
            <StatCell label="Replies received" value={person.repliesReceived} />
          </div>

          {/* Response time distribution — new */}
          <div className="mt-4">
            <p className="mb-2 font-mono text-[10px] uppercase tracking-widest text-[var(--cm-text-muted)]">
              Reply time distribution
            </p>
            <ResponseTimeBucketBars person={person} />
          </div>

          <div className="mt-4 grid gap-6 lg:grid-cols-2">
            <div>
              <p className="mb-2 font-mono text-[10px] uppercase tracking-widest text-[var(--cm-text-muted)]">
                When they text (hours)
              </p>
              <ActivityBars buckets={person.activeHours} />
            </div>
            <div>
              <p className="mb-2 font-mono text-[10px] uppercase tracking-widest text-[var(--cm-text-muted)]">
                When they text (days)
              </p>
              <ActivityBars buckets={person.activeDays} />
            </div>
          </div>
        </div>
      ) : null}
    </BrutalPanel>
  );
}

function PairsTable({
  pairs,
  workspaceId,
  isGroup,
}: {
  pairs: PairAnalytics[];
  workspaceId: string;
  /** True for 3+ speakers; false for a 1-on-1 conversation. */
  isGroup: boolean;
}) {
  if (!pairs.length) {
    return null;
  }

  return (
    <BrutalPanel>
      {/* For groups: "Pair connectivity"; for 1-on-1: "Conversation stats" (only one pair) */}
      <PanelTitle>{isGroup ? "Pair connectivity" : "Conversation stats"}</PanelTitle>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[36rem] border-collapse font-mono text-xs">
          <thead>
            <tr className="border-b-2 border-[var(--cm-border)] text-left text-[var(--cm-text-muted)]">
              <th className="pb-2 pr-4 font-normal uppercase">Pair</th>
              <th className="pb-2 pr-4 font-normal uppercase">Connection</th>
              <th className="pb-2 pr-4 font-normal uppercase">Exchanges</th>
              <th className="pb-2 pr-4 font-normal uppercase">A → B</th>
              <th className="pb-2 pr-4 font-normal uppercase">B → A</th>
              <th className="pb-2 font-normal uppercase">Typical reply</th>
            </tr>
          </thead>
          <tbody>
            {pairs.map((pair) => (
              <tr
                key={`${pair.personAId}-${pair.personBId}`}
                className="border-b border-[var(--cm-border-muted)]"
              >
                <td className="py-3 pr-4">
                  <Link
                    href={`/workspace/${workspaceId}/people/${pair.personAId}`}
                    className="hover:text-[var(--cm-accent)]"
                  >
                    {pair.personAName}
                  </Link>
                  <span className="text-[var(--cm-text-muted)]"> ↔ </span>
                  <Link
                    href={`/workspace/${workspaceId}/people/${pair.personBId}`}
                    className="hover:text-[var(--cm-accent)]"
                  >
                    {pair.personBName}
                  </Link>
                </td>
                <td className="py-3 pr-4">
                  <ConnectionBar score={pair.connectionScore} />
                </td>
                <td className="py-3 pr-4">{pair.exchanges}</td>
                <td className="py-3 pr-4">{pair.aToBReplies}</td>
                <td className="py-3 pr-4">{pair.bToAReplies}</td>
                <td className="py-3">{pair.avgResponseLabel ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </BrutalPanel>
  );
}

export function WorkspaceAnalyticsPanel({
  workspaceId,
  analytics,
  isLoading,
  error,
  onRefresh,
  isRefreshing,
}: {
  workspaceId: string;
  analytics: WorkspaceAnalytics | undefined;
  isLoading: boolean;
  error: Error | null;
  onRefresh: () => void;
  isRefreshing: boolean;
}) {
  const [expandedPersonId, setExpandedPersonId] = useState<string | null>(null);

  if (isLoading) {
    return (
      <BrutalPanel>
        <p className="font-mono text-xs uppercase text-[var(--cm-text-muted)]">
          Computing chat stats…
        </p>
      </BrutalPanel>
    );
  }

  if (error) {
    const message =
      error instanceof ApiError ? error.message : "Could not load analytics";
    return (
      <BrutalPanel>
        <p className="text-[var(--cm-error)]">{message}</p>
        <BrutalButton className="mt-3" type="button" onClick={onRefresh} disabled={isRefreshing}>
          {isRefreshing ? "Refreshing…" : "Compute stats"}
        </BrutalButton>
      </BrutalPanel>
    );
  }

  if (!analytics) {
    return null;
  }

  const computedLabel = new Date(analytics.computedAt).toLocaleString();

  // Derive chat type from speaker count — avoids needing an extra prop from the parent.
  // 2 people = 1-on-1 conversation; 3+ = group chat.
  const isGroup = analytics.people.length > 2;

  return (
    <div className="flex flex-col gap-6">
      {/* Header row */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="font-mono text-[10px] uppercase tracking-widest text-[var(--cm-text-muted)]">
          Stats as of {computedLabel}
        </p>
        <BrutalButton
          variant="ghost"
          type="button"
          onClick={onRefresh}
          disabled={isRefreshing}
        >
          {isRefreshing ? "Refreshing…" : "Refresh stats"}
        </BrutalButton>
      </div>

      {/* ① Chat/Group rhythm stats */}
      <ChatStats group={analytics.group} isGroup={isGroup} />

      {/* ② Conversation growth over time — weekly bars */}
      {analytics.group.weeklySeries?.length > 0 && (
        <ConversationGrowth series={analytics.group.weeklySeries} />
      )}

      {/* ③ Top 5 busiest weeks */}
      {analytics.group.topActiveWeeks?.length > 0 && (
        <TopActiveWeeks weeks={analytics.group.topActiveWeeks} />
      )}

      {/* ④ Activity heatmap (hour × day) */}
      {analytics.group.heatmap?.length > 0 && (
        <ActivityHeatmap group={analytics.group} />
      )}

      {/* ⑤ Speaker share breakdown */}
      {analytics.people.length > 1 && (
        <SpeakerShare people={analytics.people} />
      )}

      {/* ⑥ Avg response time per speaker */}
      {analytics.people.some((p) => p.avgResponseSeconds != null) && (
        <ResponseTimeOverview people={analytics.people} />
      )}

      {/* ⑦ Per-person expandable cards */}
      <div>
        <p className="mb-3 font-mono text-xs uppercase tracking-widest text-[var(--cm-text-muted)]">
          Per person
        </p>
        <div className="flex flex-col gap-3">
          {analytics.people.map((person) => (
            <PersonCard
              key={person.personId}
              person={person}
              workspaceId={workspaceId}
              expanded={expandedPersonId === person.personId}
              onToggle={() =>
                setExpandedPersonId((id) =>
                  id === person.personId ? null : person.personId,
                )
              }
            />
          ))}
        </div>
      </div>

      {/* ⑧ Pair connectivity table */}
      <PairsTable pairs={analytics.pairs} workspaceId={workspaceId} isGroup={isGroup} />
    </div>
  );
}
