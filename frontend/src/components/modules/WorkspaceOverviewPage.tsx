"use client";

import Link from "next/link";
import { useState } from "react";
import { AppShell } from "@/components/AppShell";
import { BrutalButton } from "@/components/brutal/BrutalButton";
import { BrutalPanel } from "@/components/brutal/BrutalPanel";
import { IngestBadge } from "@/components/brutal/BrutalBadge";
import { WorkspaceAnalyticsPanel } from "@/components/modules/overview/WorkspaceAnalyticsPanel";
import {
  useWorkspaceAnalyticsQuery,
  useWorkspaceQuery,
} from "@/react-query/queries/useWorkspacesQuery";

export function WorkspaceOverviewPage({ workspaceId }: { workspaceId: string }) {
  const { data, isLoading, error } = useWorkspaceQuery(workspaceId);
  const {
    data: analytics,
    isLoading: analyticsLoading,
    error: analyticsError,
    refreshAnalytics,
  } = useWorkspaceAnalyticsQuery(workspaceId);
  const [isRefreshing, setIsRefreshing] = useState(false);

  async function handleRefreshAnalytics() {
    setIsRefreshing(true);
    try {
      await refreshAnalytics();
    } finally {
      setIsRefreshing(false);
    }
  }

  if (isLoading) {
    return (
      <AppShell workspaceId={workspaceId}>
        <p className="font-mono text-xs uppercase text-[var(--cm-text-muted)]">Loading…</p>
      </AppShell>
    );
  }

  if (error || !data) {
    return (
      <AppShell workspaceId={workspaceId}>
        <p className="text-[var(--cm-error)]">Workspace not found</p>
      </AppShell>
    );
  }

  return (
    <AppShell workspaceId={workspaceId} workspaceName={data.name}>
      <div className="flex flex-col gap-6">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="font-mono text-2xl font-bold uppercase">{data.name}</h1>
          <IngestBadge status={data.ingestStatus} />
        </div>

        <div className="grid gap-3 sm:grid-cols-3">
          {[
            { label: "Messages", value: data.messageCount },
            { label: "Speakers", value: data.speakerCount },
            {
              label: "Range",
              value: data.dateFrom
                ? `${data.dateFrom.slice(0, 10)} → ${data.dateTo?.slice(0, 10) ?? "?"}`
                : "—",
            },
          ].map((stat) => (
            <BrutalPanel key={stat.label}>
              <p className="font-mono text-[10px] uppercase tracking-widest text-[var(--cm-text-muted)]">
                {stat.label}
              </p>
              <p className="mt-1 font-mono text-xl">{stat.value}</p>
            </BrutalPanel>
          ))}
        </div>

        <div className="flex flex-wrap gap-2">
          <Link href={`/workspace/${workspaceId}/ask`}>
            <BrutalButton>Ask past</BrutalButton>
          </Link>
          <Link href={`/workspace/${workspaceId}/people`}>
            <BrutalButton variant="ghost">People</BrutalButton>
          </Link>
        </div>

        {data.ingestStatus === "done" ? (
          <WorkspaceAnalyticsPanel
            workspaceId={workspaceId}
            analytics={analytics}
            isLoading={analyticsLoading}
            error={analyticsError}
            onRefresh={handleRefreshAnalytics}
            isRefreshing={isRefreshing}
          />
        ) : (
          <BrutalPanel>
            <p className="font-mono text-xs text-[var(--cm-text-muted)]">
              Chat stats appear after ingest finishes.
            </p>
          </BrutalPanel>
        )}
      </div>
    </AppShell>
  );
}
