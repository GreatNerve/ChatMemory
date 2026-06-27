"use client";

import { FormEvent, useCallback, useRef, useState } from "react";
import { AppShell } from "@/components/AppShell";
import { BrutalButton } from "@/components/brutal/BrutalButton";
import { PersonaBadge } from "@/components/brutal/BrutalBadge";
import { BrutalPanel } from "@/components/brutal/BrutalPanel";
import { PersonaChatPanel } from "@/components/modules/PersonaChatPanel";
import { JobProgress } from "@/components/brutal/JobProgress";
import { useJobStream } from "@/hooks/useJobStream";
import {
  useCancelTrainMutation,
  useTrainPersonaMutation,
} from "@/react-query/mutations/useWorkspaceMutations";
import { usePersonQuery, useWorkspaceQuery } from "@/react-query/queries/useWorkspacesQuery";
import { trainPersonaSchema } from "@/schema/workspace";
import { ApiError, streamPersonaChat, summarizePersonaChat } from "@/lib/api/client";
import { ChatMessage } from "@/lib/api/types";
import { KEEP_RECENT, SUMMARIZE_THRESHOLD } from "@/lib/personaChatConstants";

export function PersonDetailPage({
  workspaceId,
  personId,
}: {
  workspaceId: string;
  personId: string;
}) {
  const { data: workspace } = useWorkspaceQuery(workspaceId);
  const { data: person, refetch } = usePersonQuery(workspaceId, personId);
  const trainMutation = useTrainPersonaMutation(workspaceId, personId);
  const cancelMutation = useCancelTrainMutation(workspaceId, personId);

  const [consent, setConsent] = useState(false);
  const [forceThin, setForceThin] = useState(false);
  const [forceRebuild, setForceRebuild] = useState(false);
  const [trainJobId, setTrainJobId] = useState<string | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);

  const [chatInput, setChatInput] = useState("");
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [chatError, setChatError] = useState<string | null>(null);
  const [chatLoading, setChatLoading] = useState(false);
  // Each element is one burst bubble currently being streamed; last element is active.
  const [streamingBursts, setStreamingBursts] = useState<string[]>([]);
  const [lastInteractionId, setLastInteractionId] = useState<string | null>(null);
  const [conversationSummary, setConversationSummary] = useState<string | null>(null);
  const [summarizeError, setSummarizeError] = useState<string | null>(null);
  const [chatFullscreen, setChatFullscreen] = useState(false);
  const chatAbortRef = useRef<AbortController | null>(null);

  const activeJobId = trainJobId ?? person?.lastTrainJobId ?? null;

  const onTrainDone = useCallback(() => {
    refetch();
    setTrainJobId(null);
  }, [refetch]);

  const { job, error: jobStreamError } = useJobStream(activeJobId, onTrainDone);

  const isTraining =
    job?.status === "queued" ||
    job?.status === "running" ||
    (person?.personaStatus === "training" && Boolean(activeJobId) && !job);

  async function onCancelTrain() {
    setTrainError(null);
    try {
      await cancelMutation.mutateAsync();
      setTrainJobId(null);
      await refetch();
    } catch (err) {
      setTrainError(err instanceof ApiError ? err.message : "Cancel failed");
    }
  }

  async function onTrain(e: FormEvent) {
    e.preventDefault();
    setTrainError(null);
    const parsed = trainPersonaSchema.safeParse({
      consent,
      forceThin,
      forceRetrain: forceRebuild,
    });
    if (!parsed.success) {
      setTrainError(parsed.error.issues[0]?.message ?? "Invalid");
      return;
    }
    try {
      const res = await trainMutation.mutateAsync(parsed.data);
      setTrainJobId(res.jobId);
      await refetch();
    } catch (err) {
      setTrainError(err instanceof ApiError ? err.message : "Train failed");
    }
  }

  async function maybeSummarizeHistory(nextHistory: ChatMessage[]): Promise<ChatMessage[]> {
    if (nextHistory.length <= SUMMARIZE_THRESHOLD) {
      return nextHistory;
    }

    const older = nextHistory.slice(0, -KEEP_RECENT);
    // When re-summarizing, include prior summary so Gemini can merge context.
    const historyForSummarize: ChatMessage[] = conversationSummary
      ? [
          { role: "user", content: "Context from earlier in this conversation (already summarized):" },
          { role: "assistant", content: conversationSummary },
          ...older,
        ]
      : older;

    setSummarizeError(null);
    try {
      const result = await summarizePersonaChat(workspaceId, personId, {
        history: historyForSummarize,
        keepRecent: KEEP_RECENT,
      });
      setConversationSummary(result.summary);
      // Trimmed history + reset interaction chain — server-side summary carries older context.
      setLastInteractionId(null);
      return nextHistory.slice(-KEEP_RECENT);
    } catch (err) {
      setSummarizeError(err instanceof ApiError ? err.message : "Summarization failed");
      return nextHistory;
    }
  }

  async function onChat(e: FormEvent) {
    e.preventDefault();
    if (!chatInput.trim() || chatLoading) return;
    setChatError(null);
    const userMsg: ChatMessage = { role: "user", content: chatInput.trim() };
    const priorHistory = history;
    setHistory([...priorHistory, userMsg]);
    setChatInput("");
    setChatLoading(true);
    setStreamingBursts([]);

    chatAbortRef.current?.abort();
    const controller = new AbortController();
    chatAbortRef.current = controller;

    // burstParts is a local mutable array so event callbacks always see latest state.
    const burstParts: string[] = [""];
    let nextInteractionId = lastInteractionId;
    try {
      await streamPersonaChat(
        workspaceId,
        personId,
        {
          message: userMsg.content,
          history: priorHistory,
          previousInteractionId: lastInteractionId,
          conversationSummary,
        },
        (ev) => {
          if ("status" in ev && ev.status === "thinking") return;
          if ("msg_break" in ev) {
            // Commit current bubble; next tokens start a new one.
            burstParts.push("");
            setStreamingBursts([...burstParts]);
            return;
          }
          if ("token" in ev && ev.token) {
            burstParts[burstParts.length - 1] += ev.token;
            setStreamingBursts([...burstParts]);
          }
          if ("done" in ev && ev.done) {
            // Always update nextInteractionId from the done event.
            // If interactionId is absent (Gemini didn't return one), reset to null
            // so the next turn falls back to full-history flattening rather than
            // referencing a stale chain ID.
            nextInteractionId = ev.interactionId ?? null;
          }
          if ("error" in ev && ev.error) {
            throw new ApiError(502, ev.error);
          }
        },
        controller.signal,
      );
      const completedBursts = burstParts.filter((b) => b.trim());
      if (!completedBursts.length) {
        throw new ApiError(502, "Empty response from persona — try again");
      }
      setLastInteractionId(nextInteractionId);
      let nextHistory: ChatMessage[] = [
        ...priorHistory,
        userMsg,
        ...completedBursts.map((b) => ({ role: "assistant" as const, content: b })),
      ];
      nextHistory = await maybeSummarizeHistory(nextHistory);
      setHistory(nextHistory);
      setStreamingBursts([]);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setChatError(err instanceof ApiError ? err.message : "Chat failed");
      setHistory(priorHistory);
      setStreamingBursts([]);
    } finally {
      setChatLoading(false);
    }
  }

  if (!person) {
    return (
      <AppShell workspaceId={workspaceId}>
        <p className="font-mono text-xs uppercase text-[var(--cm-text-muted)]">Loading…</p>
      </AppShell>
    );
  }

  return (
    <AppShell workspaceId={workspaceId} workspaceName={workspace?.name}>
      <div className="flex flex-col gap-6">
        <div className="flex flex-wrap items-center gap-3">
          <h1 className="font-mono text-2xl font-bold uppercase">{person.displayName}</h1>
          <PersonaBadge status={person.personaStatus} />
        </div>

        <div className="grid gap-3 sm:grid-cols-3">
          <BrutalPanel>
            <p className="font-mono text-[10px] uppercase text-[var(--cm-text-muted)]">Messages</p>
            <p className="font-mono text-xl">{person.messageCount}</p>
          </BrutalPanel>
          <BrutalPanel>
            <p className="font-mono text-[10px] uppercase text-[var(--cm-text-muted)]">Emoji rate</p>
            <p className="font-mono text-xl">{person.styleProfile.emojiRate.toFixed(2)}</p>
          </BrutalPanel>
          <BrutalPanel>
            <p className="font-mono text-[10px] uppercase text-[var(--cm-text-muted)]">Hinglish</p>
            <p className="font-mono text-xl">{person.styleProfile.hinglishRatio.toFixed(2)}</p>
          </BrutalPanel>
        </div>

        <BrutalPanel>
          <p className="mb-3 font-mono text-xs uppercase tracking-widest text-[var(--cm-text-muted)]">
            Sample messages
          </p>
          <ul className="flex flex-col gap-2">
            {person.sampleMessages.map((m) => (
              <li key={m.messageId ?? m.timestamp} className="border-b border-[var(--cm-border-muted)] py-2 text-sm font-body">
                <span className="font-mono text-[10px] text-[var(--cm-text-muted)]">
                  {new Date(m.timestamp).toLocaleString()}
                </span>
                {/* Clamp to 3 lines so very long messages don't break the layout */}
                <p className="mt-1 line-clamp-3">{m.text}</p>
              </li>
            ))}
          </ul>
        </BrutalPanel>

        <BrutalPanel>
          <p className="mb-3 font-mono text-xs uppercase tracking-widest">Activate persona</p>
          <p className="mb-3 text-xs text-[var(--cm-text-muted)]">
            Fast build via Google Gemini — refreshes samples and style profile (no GPU training)
          </p>
          {person.trainWarning ? (
            <p
              className={`mb-3 text-sm ${
                person.trainEligible
                  ? "text-[var(--cm-warning)]"
                  : "text-[var(--cm-error)]"
              }`}
            >
              {person.trainWarning}
            </p>
          ) : null}
          <form onSubmit={onTrain} className="flex flex-col gap-4">
            <label className="flex items-start gap-2 text-sm font-body">
              <input
                type="checkbox"
                checked={consent}
                onChange={(e) => setConsent(e.target.checked)}
                disabled={isTraining}
                className="mt-1"
              />
              I have permission to mimic this person for personal use
            </label>
            {person.personaStatus === "thin" ? (
              <label className="flex items-center gap-2 text-sm font-body">
                <input
                  type="checkbox"
                  checked={forceThin}
                  onChange={(e) => setForceThin(e.target.checked)}
                  disabled={isTraining}
                />
                Force activate with thin data
              </label>
            ) : null}
            {person.personaStatus === "ready_model" ? (
              <label className="flex items-start gap-2 text-sm font-body">
                <input
                  type="checkbox"
                  checked={forceRebuild}
                  onChange={(e) => setForceRebuild(e.target.checked)}
                  disabled={isTraining}
                  className="mt-1"
                />
                <span>Rebuild persona (refresh samples and style profile)</span>
              </label>
            ) : null}
            {trainError ? <p className="text-sm text-[var(--cm-error)]">{trainError}</p> : null}
            <div className="flex flex-wrap gap-2">
              <BrutalButton
                type="submit"
                disabled={
                  !person.trainEligible ||
                  trainMutation.isPending ||
                  isTraining ||
                  (person.personaStatus === "ready_model" && !forceRebuild)
                }
              >
                {trainMutation.isPending
                  ? "Starting…"
                  : person.personaStatus === "ready_model"
                    ? "Rebuild persona"
                    : "Activate persona"}
              </BrutalButton>
              {isTraining ? (
                <BrutalButton
                  type="button"
                  variant="danger"
                  disabled={cancelMutation.isPending}
                  onClick={onCancelTrain}
                >
                  {cancelMutation.isPending ? "Cancelling…" : "Cancel activation"}
                </BrutalButton>
              ) : null}
            </div>
          </form>
          {activeJobId ? (
            <div className="mt-4">
              <JobProgress
                job={job}
                connecting={!job && !jobStreamError}
                title="Activation progress"
              />
              {jobStreamError ? (
                <p className="mt-2 text-sm text-[var(--cm-error)]">{jobStreamError}</p>
              ) : null}
            </div>
          ) : null}
        </BrutalPanel>

        {person.personaStatus === "ready_model" ? (
          <PersonaChatPanel
            workspaceName={workspace?.name ?? "Workspace"}
            displayName={person.displayName}
            history={history}
            streamingBursts={streamingBursts}
            chatLoading={chatLoading}
            chatError={chatError ?? summarizeError}
            chatInput={chatInput}
            onChatInputChange={setChatInput}
            onSubmit={onChat}
            isFullscreen={chatFullscreen}
            onToggleFullscreen={() => setChatFullscreen((v) => !v)}
            hasConversationSummary={Boolean(conversationSummary)}
          />
        ) : null}
      </div>
    </AppShell>
  );
}
