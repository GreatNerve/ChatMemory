"use client";

import Link from "next/link";
import { AppShell } from "@/components/AppShell";
import { PersonaBadge } from "@/components/brutal/BrutalBadge";
import { BrutalPanel } from "@/components/brutal/BrutalPanel";
import { usePeopleQuery, useWorkspaceQuery } from "@/react-query/queries/useWorkspacesQuery";

export function PeoplePage({ workspaceId }: { workspaceId: string }) {
  const { data: workspace } = useWorkspaceQuery(workspaceId);
  const { data, isLoading, error } = usePeopleQuery(workspaceId);

  const loading = isLoading && !data;

  return (
    <AppShell workspaceId={workspaceId} workspaceName={workspace?.name}>
      <div className="flex flex-col gap-6">
        <h1 className="font-mono text-2xl font-bold uppercase">People</h1>

        {loading ? (
          <p className="font-mono text-xs uppercase text-[var(--cm-text-muted)]">Loading…</p>
        ) : error ? (
          <p className="text-[var(--cm-error)]">Failed to load speakers</p>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            {(data?.people ?? []).map((person) => (
              <Link key={person.id} href={`/workspace/${workspaceId}/people/${person.id}`}>
                <BrutalPanel className="hover:border-[var(--cm-accent)] transition-colors">
                  <div className="flex items-start justify-between gap-2">
                    <p className="font-mono text-lg uppercase">{person.displayName}</p>
                    <PersonaBadge status={person.personaStatus} />
                  </div>
                  <p className="mt-2 font-mono text-xs text-[var(--cm-text-muted)]">
                    {person.messageCount} messages
                  </p>
                </BrutalPanel>
              </Link>
            ))}
          </div>
        )}
      </div>
    </AppShell>
  );
}
