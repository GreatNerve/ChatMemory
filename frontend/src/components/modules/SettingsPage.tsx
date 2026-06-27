"use client";

import { AppShell } from "@/components/AppShell";
import { BrutalPanel } from "@/components/brutal/BrutalPanel";
import { useHealthQuery, useSettingsQuery } from "@/react-query/queries/useWorkspacesQuery";

export function SettingsPage() {
  const { data: settings, isLoading, error } = useSettingsQuery();
  const { data: health } = useHealthQuery();

  return (
    <AppShell>
      <div className="flex flex-col gap-6">
        <h1 className="font-mono text-2xl font-bold uppercase">Settings</h1>

        {isLoading ? (
          <p className="font-mono text-xs uppercase text-[var(--cm-text-muted)]">Loading…</p>
        ) : error ? (
          <p className="text-[var(--cm-error)]">Cannot reach API</p>
        ) : settings ? (
          <div className="flex flex-col gap-4">
            <BrutalPanel>
              <dl className="flex flex-col gap-3 font-mono text-sm">
                <div className="flex justify-between gap-4 border-b border-[var(--cm-border-muted)] pb-2">
                  <dt className="text-[var(--cm-text-muted)]">Data root</dt>
                  <dd>{settings.dataRoot}</dd>
                </div>
                <div className="flex justify-between gap-4 border-b border-[var(--cm-border-muted)] pb-2">
                  <dt className="text-[var(--cm-text-muted)]">Embed model</dt>
                  <dd className="text-right text-xs">{settings.embedModel}</dd>
                </div>
                <div className="flex justify-between gap-4 border-b border-[var(--cm-border-muted)] pb-2">
                  <dt className="text-[var(--cm-text-muted)]">Embed device</dt>
                  <dd>{settings.embedDevice ?? "cpu"}</dd>
                </div>
                <div className="flex justify-between gap-4 border-b border-[var(--cm-border-muted)] pb-2">
                  <dt className="text-[var(--cm-text-muted)]">Vector store</dt>
                  <dd>{settings.vectorStore ?? "chroma"}</dd>
                </div>
                <div className="flex justify-between gap-4 border-b border-[var(--cm-border-muted)] pb-2">
                  <dt className="text-[var(--cm-text-muted)]">Gemini</dt>
                  <dd>
                    {settings.geminiConfigured ? `OK · ${settings.geminiModel}` : "NOT CONFIGURED"}
                  </dd>
                </div>
                <div className="flex justify-between gap-4">
                  <dt className="text-[var(--cm-text-muted)]">GPU</dt>
                  <dd>
                    {settings.gpuBusy ? "Busy" : settings.gpuAvailable ? "Idle" : "CPU only"}
                    {settings.activeJobId ? ` (${settings.activeJobId})` : ""}
                  </dd>
                </div>
              </dl>
            </BrutalPanel>

            {health ? (
              <BrutalPanel>
                <p className="mb-2 font-mono text-xs uppercase text-[var(--cm-text-muted)]">Health</p>
                <p className="font-mono text-sm">
                  API {health.status} · Gemini {health.geminiConfigured ? "OK" : "NOT SET"} · Embed{" "}
                  {health.embedReady ? "READY" : "COLD"} · ML{" "}
                  {health.mlStackAvailable === false ? "BLOCKED" : "OK"} · Disk{" "}
                  {health.dataRootWritable ? "OK" : "ERR"}
                </p>
                {health.mlStackError ? (
                  <p className="mt-2 text-xs font-body text-[var(--cm-warning)]">
                    {health.mlStackError}
                  </p>
                ) : null}
                {health.geminiConfigured === false ? (
                  <p className="mt-2 text-xs font-body text-[var(--cm-warning)]">
                    Set GEMINI_API_KEY in backend/.env for Q&amp;A and persona chat
                  </p>
                ) : null}
              </BrutalPanel>
            ) : null}
          </div>
        ) : null}
      </div>
    </AppShell>
  );
}
