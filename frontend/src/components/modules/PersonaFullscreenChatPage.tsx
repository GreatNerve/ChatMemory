"use client";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { BrutalButton } from "@/components/brutal/BrutalButton";
import { BrutalInput } from "@/components/brutal/BrutalInput";
import { ApiError, streamPersonaChat, summarizePersonaChat } from "@/lib/api/client";
import { ChatMessage, StageEvent } from "@/lib/api/types";
import { usePersonQuery, useSettingsQuery, useWorkspaceQuery } from "@/react-query/queries/useWorkspacesQuery";
import { KEEP_RECENT, SUMMARIZE_THRESHOLD } from "@/lib/personaChatConstants";
import { downloadPersonaChatPdf } from "@/lib/personaChatPdf";
import { ThinkingPanel } from "@/components/ui/ThinkingPanel";

// localStorage key for user's explicit output-only preference.
const LS_THINKING_KEY = "chatmemory_thinking_output_only";

// ── Message bubbles ─────────────────────────────────────────────────────────

function UserBubble({ content }: { content: string }) {
  return (
    <div className="self-end max-w-[80%] border-2 border-[var(--cm-border)] bg-[var(--cm-surface-raised)] px-4 py-2.5">
      <p className="font-mono text-[10px] uppercase text-[var(--cm-text-muted)]">You</p>
      <p className="mt-1 whitespace-pre-wrap text-sm font-body">{content}</p>
    </div>
  );
}

function AssistantBubble({
  displayName,
  content,
  streaming = false,
  active = false,
  stages = [],
  outputOnly,
  onToggleOutputOnly,
}: {
  displayName: string;
  content: string;
  streaming?: boolean;
  active?: boolean;
  /** Accumulated stage events for this message (empty for messages predating this feature). */
  stages?: StageEvent[];
  outputOnly: boolean;
  onToggleOutputOnly: () => void;
}) {
  return (
    <div
      className={`self-start max-w-[80%] ${streaming ? "opacity-90" : ""}`}
    >
      <div className="border-l-4 border-[var(--cm-accent)] pl-3">
        <p className="font-mono text-[10px] uppercase text-[var(--cm-text-muted)]">{displayName}</p>
        <p className="mt-1 whitespace-pre-wrap text-sm font-body">
          {content}
          {active && <span className="animate-pulse">▌</span>}
        </p>
      </div>
      {/* ThinkingPanel: live while streaming, collapsed toggle after completion */}
      <ThinkingPanel
        stages={stages}
        isStreaming={streaming}
        outputOnly={outputOnly}
        onToggleOutputOnly={onToggleOutputOnly}
      />
    </div>
  );
}

function TypingIndicator({ displayName }: { displayName: string }) {
  return (
    <div className="self-start max-w-[80%] border-l-4 border-[var(--cm-accent)] pl-3 opacity-70">
      <p className="font-mono text-[10px] uppercase text-[var(--cm-text-muted)]">{displayName}</p>
      <p className="mt-1 text-sm font-body text-[var(--cm-text-muted)]">
        typing<span className="animate-pulse">…</span>
      </p>
    </div>
  );
}

// ── Loading skeleton ─────────────────────────────────────────────────────────

function HeaderSkeleton() {
  return (
    <div className="flex items-center gap-3">
      <div className="h-10 w-10 animate-pulse border-2 border-[var(--cm-border-muted)] bg-[var(--cm-surface-raised)]" />
      <div className="flex flex-col gap-1.5">
        <div className="h-4 w-32 animate-pulse bg-[var(--cm-surface-raised)]" />
        <div className="h-2.5 w-20 animate-pulse bg-[var(--cm-surface-raised)]" />
      </div>
    </div>
  );
}

// ── Main component ───────────────────────────────────────────────────────────

export function PersonaFullscreenChatPage({
  workspaceId,
  personId,
}: {
  workspaceId: string;
  personId: string;
}) {
  const { data: workspace } = useWorkspaceQuery(workspaceId);
  const { data: person, isLoading: personLoading, error: personError } = usePersonQuery(
    workspaceId,
    personId,
  );
  // Settings are used to seed the outputOnly default from the server config.
  const { data: settings } = useSettingsQuery();

  const [chatInput, setChatInput] = useState("");
  const [history, setHistory] = useState<ChatMessage[]>([]);
  // Parallel stages array — one entry per history item (empty array for user messages / old messages).
  const [stagesHistory, setStagesHistory] = useState<StageEvent[][]>([]);
  const [chatError, setChatError] = useState<string | null>(null);
  const [chatLoading, setChatLoading] = useState(false);
  const [streamingBursts, setStreamingBursts] = useState<string[]>([]);
  // Live stage events accumulating for the currently-streaming turn.
  const [streamingStages, setStreamingStages] = useState<StageEvent[]>([]);
  const [lastInteractionId, setLastInteractionId] = useState<string | null>(null);
  const [conversationSummary, setConversationSummary] = useState<string | null>(null);
  const [summarizeError, setSummarizeError] = useState<string | null>(null);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfError, setPdfError] = useState<string | null>(null);
  // Output-only toggle: hides the INPUT section in ThinkingPanel.
  // Priority: localStorage explicit preference > server THINKING_SHOW_INPUT config > true (safe default).
  const [outputOnly, setOutputOnly] = useState<boolean>(() => {
    try {
      const stored = localStorage.getItem(LS_THINKING_KEY);
      // If user has an explicit saved preference, honour it immediately.
      if (stored !== null) return stored === "true";
    } catch {
      // Ignore storage errors (private browsing, etc.)
    }
    // No explicit preference yet — default to output-only until the server config loads.
    return true;
  });

  // Whether we've finished seeding outputOnly from the server config (used to avoid overriding localStorage).
  const serverDefaultAppliedRef = useRef(false);

  // Once settings arrive, apply the server default only when the user has no explicit localStorage pref.
  useEffect(() => {
    if (serverDefaultAppliedRef.current) return;
    try {
      if (localStorage.getItem(LS_THINKING_KEY) !== null) {
        serverDefaultAppliedRef.current = true;
        return; // user has an explicit preference — leave it alone
      }
    } catch {
      // ignore
    }
    if (settings !== undefined) {
      // thinkingShowInput=false (default) → outputOnly=true; thinkingShowInput=true → outputOnly=false
      setOutputOnly(!(settings.thinkingShowInput ?? false));
      serverDefaultAppliedRef.current = true;
    }
  }, [settings]);

  const messagesRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const prevLoadingRef = useRef(chatLoading);
  const chatAbortRef = useRef<AbortController | null>(null);

  // Auto-scroll to bottom when messages or stage events change.
  useEffect(() => {
    const el = messagesRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [history, streamingBursts, streamingStages, chatLoading]);

  // Restore focus to input when loading ends.
  useEffect(() => {
    if (prevLoadingRef.current && !chatLoading) {
      inputRef.current?.focus();
    }
    prevLoadingRef.current = chatLoading;
  }, [chatLoading]);

  const backHref = `/workspace/${workspaceId}/people/${personId}`;

  // PDF export — only show button when there are committed messages to download.
  const exportableHistory = history.filter((m) => m.content.trim());
  const canDownloadPdf = exportableHistory.length > 0 && !pdfLoading;

  async function onDownloadPdf() {
    if (!canDownloadPdf) return;
    setPdfError(null);
    setPdfLoading(true);
    try {
      await downloadPersonaChatPdf({
        workspaceName: workspace?.name ?? "Workspace",
        displayName: person?.displayName ?? "Persona",
        history: exportableHistory,
      });
    } catch (err) {
      setPdfError(err instanceof Error ? err.message : "PDF export failed");
    } finally {
      setPdfLoading(false);
    }
  }

  function handleToggleOutputOnly() {
    setOutputOnly((prev) => {
      const next = !prev;
      try {
        // Persist explicit user preference so it survives page reloads and overrides the server default.
        localStorage.setItem(LS_THINKING_KEY, String(next));
      } catch {
        // ignore storage errors
      }
      return next;
    });
  }

  async function maybeSummarizeHistory(
    nextHistory: ChatMessage[],
    nextStages: StageEvent[][],
  ): Promise<[ChatMessage[], StageEvent[][]]> {
    if (nextHistory.length <= SUMMARIZE_THRESHOLD) return [nextHistory, nextStages];

    const older = nextHistory.slice(0, -KEEP_RECENT);
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
      setLastInteractionId(null);
      return [nextHistory.slice(-KEEP_RECENT), nextStages.slice(-KEEP_RECENT)];
    } catch (err) {
      setSummarizeError(err instanceof ApiError ? err.message : "Summarization failed");
      return [nextHistory, nextStages];
    }
  }

  const onChat = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      if (!chatInput.trim() || chatLoading) return;
      setChatError(null);

      const userMsg: ChatMessage = { role: "user", content: chatInput.trim() };
      const priorHistory = history;
      const priorStages = stagesHistory;
      setHistory([...priorHistory, userMsg]);
      setStagesHistory([...priorStages, []]);
      setChatInput("");
      setChatLoading(true);
      setStreamingBursts([]);
      setStreamingStages([]);

      chatAbortRef.current?.abort();
      const controller = new AbortController();
      chatAbortRef.current = controller;

      const burstParts: string[] = [""];
      // Local accumulator for stage events — avoids stale-closure issues with state.
      const stagesList: StageEvent[] = [];
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
            // Thinking status — ignore, we already show a typing indicator.
            if ("status" in ev && ev.status === "thinking") return;

            // Stage events — accumulate and update live ThinkingPanel.
            if ("type" in ev && ev.type === "stage") {
              stagesList.push(ev as StageEvent);
              setStreamingStages([...stagesList]);
              return;
            }

            if ("msg_break" in ev) {
              burstParts.push("");
              setStreamingBursts([...burstParts]);
              return;
            }
            if ("token" in ev && ev.token) {
              burstParts[burstParts.length - 1] += ev.token;
              setStreamingBursts([...burstParts]);
            }
            if ("done" in ev && ev.done) {
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

        // Capture stages accumulated during this turn for commit into history.
        const capturedStages: StageEvent[] = [...stagesList];

        let nextHistory: ChatMessage[] = [
          ...priorHistory,
          userMsg,
          ...completedBursts.map((b) => ({ role: "assistant" as const, content: b })),
        ];
        // Assign the same stage events to all burst messages in this turn.
        let nextStages: StageEvent[][] = [
          ...priorStages,
          [], // user message has no stages
          ...completedBursts.map(() => capturedStages),
        ];

        [nextHistory, nextStages] = await maybeSummarizeHistory(nextHistory, nextStages);
        setHistory(nextHistory);
        setStagesHistory(nextStages);
        setStreamingBursts([]);
        setStreamingStages([]);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setChatError(err instanceof ApiError ? err.message : "Chat failed");
        setHistory(priorHistory);
        setStagesHistory(priorStages);
        setStreamingBursts([]);
        setStreamingStages([]);
      } finally {
        setChatLoading(false);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [chatInput, chatLoading, history, stagesHistory, lastInteractionId, conversationSummary, workspaceId, personId],
  );

  // ── Render ─────────────────────────────────────────────────────────────────

  const displayName = person?.displayName ?? "Persona";
  const initials = displayName
    .split(" ")
    .map((w) => w[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();

  // Error loading person
  if (personError) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 bg-[var(--cm-bg)] p-8">
        <p className="font-mono text-sm text-[var(--cm-error)]">Failed to load persona</p>
        <Link href={backHref}>
          <BrutalButton type="button" variant="ghost">← Back</BrutalButton>
        </Link>
      </div>
    );
  }

  // Persona not ready
  if (!personLoading && person && person.personaStatus !== "ready_model") {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4 bg-[var(--cm-bg)] p-8">
        <p className="font-mono text-sm uppercase text-[var(--cm-warning)]">
          Persona not yet activated
        </p>
        <p className="text-sm font-body text-[var(--cm-text-muted)]">
          Activate the persona from the person detail page first.
        </p>
        <Link href={backHref}>
          <BrutalButton type="button" variant="ghost">← Back to {displayName}</BrutalButton>
        </Link>
      </div>
    );
  }

  const hasMessages = history.length > 0 || streamingBursts.some((b) => b) || chatLoading;

  return (
    <div className="flex h-screen flex-col bg-[var(--cm-bg)] text-[var(--cm-text)]">
      {/* ── Fixed header ── */}
      <header className="flex shrink-0 items-center gap-3 border-b-4 border-[var(--cm-border)] bg-[var(--cm-surface)] px-4 py-3">
        <Link href={backHref} aria-label="Back to person detail">
          <BrutalButton type="button" variant="ghost" className="px-2 py-1">
            ←
          </BrutalButton>
        </Link>

        {/* Avatar initials */}
        {personLoading ? (
          <HeaderSkeleton />
        ) : (
          <>
            <div
              className="flex h-9 w-9 shrink-0 items-center justify-center border-2 border-[var(--cm-accent)] bg-[var(--cm-surface-raised)] font-mono text-sm font-bold text-[var(--cm-accent)]"
              aria-hidden="true"
            >
              {initials}
            </div>
            <div className="min-w-0">
              <p className="truncate font-mono text-sm font-bold uppercase tracking-tight">
                {displayName}
              </p>
              <p className="font-mono text-[10px] uppercase text-[var(--cm-text-muted)]">
                {workspace?.name ?? "Workspace"} · persona chat
              </p>
            </div>
          </>
        )}

        {/* Right-side header controls */}
        <div className="ml-auto flex items-center gap-3">
          {conversationSummary ? (
            <span className="font-mono text-[10px] uppercase text-[var(--cm-text-muted)]">
              Earlier summarized
            </span>
          ) : null}
          {/* Only render the PDF button once there are messages */}
          {exportableHistory.length > 0 ? (
            <BrutalButton
              type="button"
              variant="ghost"
              onClick={onDownloadPdf}
              disabled={!canDownloadPdf}
              aria-label="Download chat as PDF"
            >
              {pdfLoading ? "Generating…" : "Download PDF"}
            </BrutalButton>
          ) : null}
        </div>
      </header>
      {/* PDF error banner shown below header */}
      {pdfError ? (
        <p className="shrink-0 bg-[var(--cm-surface)] px-4 py-1 font-mono text-xs text-[var(--cm-error)]" role="alert">
          {pdfError}
        </p>
      ) : null}

      {/* ── Scrollable message thread ── */}
      <div
        ref={messagesRef}
        className="flex flex-1 flex-col gap-4 overflow-y-auto px-4 py-6"
        aria-label="Chat messages"
        aria-live="polite"
      >
        {!hasMessages ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center">
            <div
              className="mb-2 flex h-16 w-16 items-center justify-center border-4 border-[var(--cm-accent)] font-mono text-2xl font-bold text-[var(--cm-accent)]"
              aria-hidden="true"
            >
              {initials}
            </div>
            <p className="font-mono text-sm uppercase tracking-widest text-[var(--cm-text-muted)]">
              Start chatting with {displayName}
            </p>
            <p className="max-w-xs text-xs font-body text-[var(--cm-text-muted)]">
              Replies via Gemini using their real messages as style context
            </p>
          </div>
        ) : (
          <>
            {/* Committed history messages */}
            {history.map((msg, i) =>
              msg.content.trim() ? (
                msg.role === "user" ? (
                  <UserBubble key={`h-${i}`} content={msg.content} />
                ) : (
                  <AssistantBubble
                    key={`h-${i}`}
                    displayName={displayName}
                    content={msg.content}
                    stages={stagesHistory[i] ?? []}
                    outputOnly={outputOnly}
                    onToggleOutputOnly={handleToggleOutputOnly}
                  />
                )
              ) : null,
            )}

            {/* Streaming burst bubbles — live ThinkingPanel shown on first bubble */}
            {streamingBursts.map((burst, i) => {
              const isActive = i === streamingBursts.length - 1;
              if (!burst && !isActive) return null;
              if (!burst) return <TypingIndicator key={`sb-${i}`} displayName={displayName} />;
              return (
                <AssistantBubble
                  key={`sb-${i}`}
                  displayName={displayName}
                  content={burst}
                  streaming
                  active={isActive}
                  // Only attach live stages to the first burst bubble to avoid duplicates.
                  stages={i === 0 ? streamingStages : []}
                  outputOnly={outputOnly}
                  onToggleOutputOnly={handleToggleOutputOnly}
                />
              );
            })}

            {/* Show live ThinkingPanel on the typing indicator when stages are arriving
                but the first token hasn't come through yet. */}
            {chatLoading && streamingBursts.length === 0 && streamingStages.length > 0 && (
              <div className="self-start max-w-[80%]">
                <div className="border-l-4 border-[var(--cm-accent)] pl-3">
                  <p className="font-mono text-[10px] uppercase text-[var(--cm-text-muted)]">{displayName}</p>
                  <ThinkingPanel
                    stages={streamingStages}
                    isStreaming
                    outputOnly={outputOnly}
                    onToggleOutputOnly={handleToggleOutputOnly}
                  />
                </div>
              </div>
            )}

            {/* Initial typing indicator before first token — only when no stage events yet */}
            {chatLoading && streamingBursts.length === 0 && streamingStages.length === 0 && (
              <TypingIndicator displayName={displayName} />
            )}
          </>
        )}
      </div>

      {/* ── Fixed bottom area ── */}
      <div className="shrink-0 border-t-4 border-[var(--cm-border)] bg-[var(--cm-surface)] px-4 py-3">
        {/* Error + summarize error banners */}
        {(chatError || summarizeError) && (
          <p className="mb-2 font-mono text-xs text-[var(--cm-error)]" role="alert">
            {chatError || summarizeError}
          </p>
        )}

        <form onSubmit={onChat} className="flex gap-2">
          <BrutalInput
            ref={inputRef}
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            placeholder={`Message ${displayName}…`}
            disabled={chatLoading || personLoading}
            aria-label="Chat message"
          />
          <BrutalButton
            type="submit"
            disabled={chatLoading || personLoading || !chatInput.trim()}
          >
            {chatLoading ? "…" : "Send"}
          </BrutalButton>
        </form>

        <p className="mt-1.5 font-mono text-[10px] text-[var(--cm-text-muted)]">
          Replies may not reflect the real person. ⚙ thinking shows live pipeline stages.
        </p>
      </div>
    </div>
  );
}
