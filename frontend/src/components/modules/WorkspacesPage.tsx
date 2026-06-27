"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { FormEvent, useCallback, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { BrutalButton } from "@/components/brutal/BrutalButton";
import { IngestBadge } from "@/components/brutal/BrutalBadge";
import { BrutalInput } from "@/components/brutal/BrutalInput";
import { BrutalPanel } from "@/components/brutal/BrutalPanel";
import { JobProgress } from "@/components/brutal/JobProgress";
import { useJobStream } from "@/hooks/useJobStream";
import {
  useCreateWorkspaceMutation,
  useDeleteWorkspaceMutation,
} from "@/react-query/mutations/useWorkspaceMutations";
import { useWorkspacesQuery } from "@/react-query/queries/useWorkspacesQuery";
import { createWorkspaceSchema } from "@/schema/workspace";
import { ApiError } from "@/lib/api/client";

export function WorkspacesPage() {
  const router = useRouter();
  const { data, isLoading, error, refetch } = useWorkspacesQuery();
  const createMutation = useCreateWorkspaceMutation();
  const deleteMutation = useDeleteWorkspaceMutation();

  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [formError, setFormError] = useState<string | null>(null);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [pendingWorkspaceId, setPendingWorkspaceId] = useState<string | null>(null);

  const onIngestDone = useCallback(() => {
    refetch();
    if (pendingWorkspaceId) {
      router.push(`/workspace/${pendingWorkspaceId}`);
    }
  }, [pendingWorkspaceId, refetch, router]);

  const { job, error: jobError } = useJobStream(activeJobId, onIngestDone);

  const isUploading = createMutation.isPending;
  const isProcessing =
    activeJobId !== null && job?.status !== "done" && job?.status !== "error";

  function submitLabel() {
    if (isUploading) return "Uploading file…";
    if (isProcessing) return `Processing… ${job?.percent ?? 0}%`;
    return "Ingest";
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setFormError(null);
    if (!file) {
      setFormError("Select a WhatsApp .txt export");
      return;
    }
    const parsed = createWorkspaceSchema.safeParse({ name, file });
    if (!parsed.success) {
      setFormError(parsed.error.issues[0]?.message ?? "Invalid form");
      return;
    }
    try {
      const result = await createMutation.mutateAsync({ name, file });
      setActiveJobId(result.jobId);
      setPendingWorkspaceId(result.workspace.id);
      setName("");
      setFile(null);
    } catch (err) {
      setFormError(err instanceof ApiError ? err.message : "Upload failed");
    }
  }

  return (
    <AppShell>
      <div className="flex flex-col gap-6">
        <div>
          <h1 className="font-mono text-3xl font-bold uppercase tracking-tight">Workspaces</h1>
          <p className="mt-2 text-sm font-body text-[var(--cm-text-muted)]">
            Upload a WhatsApp export (.txt) — group or 1-on-1 — to index messages and build personas.
          </p>
        </div>

        <BrutalPanel className="p-6">
          <p className="mb-4 font-mono text-xs uppercase tracking-widest text-[var(--cm-text-muted)]">
            New workspace
          </p>
          <form onSubmit={onSubmit} className="flex flex-col gap-4">
            <div className="flex flex-col gap-2">
              <label className="font-mono text-xs uppercase tracking-widest">Name</label>
              <BrutalInput
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="College gang"
                disabled={isUploading || isProcessing}
              />
            </div>
            <div className="flex flex-col gap-2">
              <label className="font-mono text-xs uppercase tracking-widest">WhatsApp .txt</label>
              <input
                type="file"
                accept=".txt"
                disabled={isUploading || isProcessing}
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                className="border-2 border-dashed border-[var(--cm-border)] bg-[var(--cm-bg)] p-4 font-mono text-xs uppercase disabled:opacity-50"
              />
            </div>
            {formError ? <p className="text-sm text-[var(--cm-error)]">{formError}</p> : null}
            <BrutalButton type="submit" disabled={isUploading || isProcessing}>
              {submitLabel()}
            </BrutalButton>
          </form>
          {activeJobId ? (
            <div className="mt-4 flex flex-col gap-2">
              <JobProgress job={job} connecting={!job && !jobError} title="Ingest progress" />
              {jobError ? <p className="text-sm text-[var(--cm-error)]">{jobError}</p> : null}
              {job?.status === "done" ? (
                <p className="font-mono text-xs uppercase text-[var(--cm-success)]">
                  Ingest complete — opening workspace…
                </p>
              ) : null}
            </div>
          ) : null}
        </BrutalPanel>

        {isLoading ? (
          <p className="font-mono text-xs uppercase text-[var(--cm-text-muted)]">Loading…</p>
        ) : error ? (
          <p className="text-sm text-[var(--cm-error)]">Cannot reach API — start backend on :8000</p>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            {(data?.workspaces ?? []).map((ws) => (
              <BrutalPanel key={ws.id} className="flex flex-col gap-3">
                <div className="flex items-start justify-between gap-2">
                  <Link
                    href={`/workspace/${ws.id}`}
                    className="font-mono text-lg uppercase hover:text-[var(--cm-accent)]"
                  >
                    {ws.name}
                  </Link>
                  <IngestBadge status={ws.ingestStatus} />
                </div>
                <p className="font-mono text-xs text-[var(--cm-text-muted)]">
                  {ws.messageCount} msgs · {ws.speakerCount} people
                </p>
                <div className="flex gap-2">
                  <Link href={`/workspace/${ws.id}`}>
                    <BrutalButton variant="ghost">Open</BrutalButton>
                  </Link>
                  <BrutalButton
                    variant="danger"
                    disabled={deleteMutation.isPending}
                    onClick={() => deleteMutation.mutate(ws.id)}
                  >
                    Delete
                  </BrutalButton>
                </div>
              </BrutalPanel>
            ))}
          </div>
        )}
      </div>
    </AppShell>
  );
}
