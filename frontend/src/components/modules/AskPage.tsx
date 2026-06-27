"use client";

import { FormEvent, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { BrutalButton } from "@/components/brutal/BrutalButton";
import { BrutalInput } from "@/components/brutal/BrutalInput";
import { BrutalPanel } from "@/components/brutal/BrutalPanel";
import { BrutalTextarea } from "@/components/brutal/BrutalTextarea";
import { CitationBlock } from "@/components/brutal/CitationBlock";
import { useAskMutation } from "@/react-query/mutations/useWorkspaceMutations";
import { usePeopleQuery, useWorkspaceQuery } from "@/react-query/queries/useWorkspacesQuery";
import { askSchema } from "@/schema/workspace";
import { AskResponse } from "@/lib/api/types";
import { ApiError } from "@/lib/api/client";

export function AskPage({ workspaceId }: { workspaceId: string }) {
  const { data: workspace } = useWorkspaceQuery(workspaceId);
  const { data: peopleData } = usePeopleQuery(workspaceId);
  const askMutation = useAskMutation(workspaceId);

  const [question, setQuestion] = useState("");
  const [speaker, setSpeaker] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setResult(null);
    const parsed = askSchema.safeParse({
      question,
      speaker: speaker || undefined,
      dateFrom: dateFrom || undefined,
      dateTo: dateTo || undefined,
    });
    if (!parsed.success) {
      setError(parsed.error.issues[0]?.message ?? "Invalid");
      return;
    }
    try {
      const res = await askMutation.mutateAsync(parsed.data);
      setResult(res);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Ask failed");
    }
  }

  return (
    <AppShell workspaceId={workspaceId} workspaceName={workspace?.name}>
      <div className="flex flex-col gap-6">
        <h1 className="font-mono text-2xl font-bold uppercase">Ask the past</h1>

        <BrutalPanel>
          <form onSubmit={onSubmit} className="flex flex-col gap-4">
            <div className="flex flex-col gap-2">
              <label className="font-mono text-xs uppercase tracking-widest">Question</label>
              <BrutalTextarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="When did we plan the Goa trip?"
              />
            </div>
            <div className="grid gap-4 md:grid-cols-3">
              <div className="flex flex-col gap-2">
                <label className="font-mono text-xs uppercase tracking-widest">Speaker</label>
                <select
                  value={speaker}
                  onChange={(e) => setSpeaker(e.target.value)}
                  className="border-2 border-[var(--cm-border)] bg-[var(--cm-bg)] px-3 py-2 text-sm"
                >
                  <option value="">All</option>
                  {(peopleData?.people ?? []).map((p) => (
                    <option key={p.id} value={p.displayName}>
                      {p.displayName}
                    </option>
                  ))}
                </select>
              </div>
              <div className="flex flex-col gap-2">
                <label className="font-mono text-xs uppercase tracking-widest">From</label>
                <BrutalInput type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} />
              </div>
              <div className="flex flex-col gap-2">
                <label className="font-mono text-xs uppercase tracking-widest">To</label>
                <BrutalInput type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} />
              </div>
            </div>
            {error ? <p className="text-sm text-[var(--cm-error)]">{error}</p> : null}
            <BrutalButton type="submit" disabled={askMutation.isPending}>
              {askMutation.isPending ? "Searching…" : "Search"}
            </BrutalButton>
          </form>
        </BrutalPanel>

        {result?.status === "not_found" ? (
          <BrutalPanel className="border-[var(--cm-error)]">
            <p className="font-mono text-sm uppercase text-[var(--cm-error)]">Not in this chat</p>
            <p className="mt-2 text-sm font-body text-[var(--cm-text-muted)]">{result.reason}</p>
            {result.nearMisses.length > 0 ? (
              <div className="mt-4 flex flex-col gap-2">
                <p className="font-mono text-xs uppercase text-[var(--cm-text-muted)]">Near misses</p>
                {result.nearMisses.map((c) => (
                  <CitationBlock key={c.messageId} citation={c} />
                ))}
              </div>
            ) : null}
          </BrutalPanel>
        ) : null}

        {result?.status === "answered" && result.answer ? (
          <BrutalPanel>
            <p className="mb-3 font-mono text-xs uppercase tracking-widest text-[var(--cm-accent)]">
              Answer
            </p>
            <p className="text-base font-body leading-relaxed">{result.answer}</p>
            {result.citations.length > 0 ? (
              <div className="mt-4 flex flex-col gap-2">
                <p className="font-mono text-xs uppercase text-[var(--cm-text-muted)]">Citations</p>
                {result.citations.map((c) => (
                  <CitationBlock key={c.messageId} citation={c} />
                ))}
              </div>
            ) : null}
          </BrutalPanel>
        ) : null}
      </div>
    </AppShell>
  );
}
