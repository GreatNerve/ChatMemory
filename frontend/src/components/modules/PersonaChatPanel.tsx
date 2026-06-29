"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { BrutalButton } from "@/components/brutal/BrutalButton";
import { BrutalInput } from "@/components/brutal/BrutalInput";
import { BrutalPanel } from "@/components/brutal/BrutalPanel";
import { ChatMessage } from "@/lib/api/types";
import { downloadPersonaChatPdf } from "@/lib/personaChatPdf";

interface PersonaChatPanelProps {
  workspaceName: string;
  displayName: string;
  history: ChatMessage[];
  /** Each element is one burst bubble currently streaming; last element is the active one. */
  streamingBursts: string[];
  chatLoading: boolean;
  chatError: string | null;
  chatInput: string;
  onChatInputChange: (value: string) => void;
  onSubmit: (e: FormEvent) => void;
  hasConversationSummary?: boolean;
  /** If provided, shows an "open full screen" icon button that opens this URL in a new tab. */
  fullscreenUrl?: string;
}

function UserBubble({ content }: { content: string }) {
  return (
    <div className="self-end max-w-[85%] border-2 border-[var(--cm-border)] bg-[var(--cm-surface-raised)] px-3 py-2">
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
}: {
  displayName: string;
  content: string;
  streaming?: boolean;
  active?: boolean;
}) {
  return (
    <div
      className={`self-start max-w-[85%] border-l-4 border-[var(--cm-accent)] pl-3 ${
        streaming ? "opacity-90" : ""
      }`}
    >
      <p className="font-mono text-[10px] uppercase text-[var(--cm-text-muted)]">{displayName}</p>
      <p className="mt-1 whitespace-pre-wrap text-sm font-body">
        {content}
        {active && <span className="animate-pulse">▌</span>}
      </p>
    </div>
  );
}

function TypingIndicator({ displayName }: { displayName: string }) {
  return (
    <div className="self-start max-w-[85%] border-l-4 border-[var(--cm-accent)] pl-3 opacity-70">
      <p className="font-mono text-[10px] uppercase text-[var(--cm-text-muted)]">{displayName}</p>
      <p className="mt-1 text-sm font-body text-[var(--cm-text-muted)]">
        typing<span className="animate-pulse">…</span>
      </p>
    </div>
  );
}

function ChatMessages({
  displayName,
  history,
  streamingBursts,
  chatLoading,
}: Pick<
  PersonaChatPanelProps,
  "displayName" | "history" | "streamingBursts" | "chatLoading"
>) {
  const hasContent = history.length > 0 || streamingBursts.some((b) => b) || chatLoading;

  if (!hasContent) {
    return (
      <p className="text-sm text-[var(--cm-text-muted)]">Say something in Hinglish or English…</p>
    );
  }

  // Completed bubbles committed to history already contain burst messages as separate entries.
  const historyBubbles = history.map((m, i) =>
    m.content.trim() ? (
      m.role === "user" ? (
        <UserBubble key={`h-${i}`} content={m.content} />
      ) : (
        <AssistantBubble key={`h-${i}`} displayName={displayName} content={m.content} />
      )
    ) : null,
  );

  // Burst bubbles currently streaming.
  const streamingBubbles = streamingBursts.map((burst, i) => {
    const isActive = i === streamingBursts.length - 1;
    if (!burst && !isActive) return null;
    // Empty non-active parts (gaps between bursts) show as typing indicator.
    if (!burst) {
      return <TypingIndicator key={`sb-${i}`} displayName={displayName} />;
    }
    return (
      <AssistantBubble
        key={`sb-${i}`}
        displayName={displayName}
        content={burst}
        streaming
        active={isActive}
      />
    );
  });

  // Show typing indicator at the very start before any tokens arrive.
  const showInitialTyping = chatLoading && streamingBursts.length === 0;

  return (
    <>
      {historyBubbles}
      {streamingBubbles}
      {showInitialTyping ? <TypingIndicator displayName={displayName} /> : null}
    </>
  );
}

export function PersonaChatPanel({
  workspaceName,
  displayName,
  history,
  streamingBursts,
  chatLoading,
  chatError,
  chatInput,
  onChatInputChange,
  onSubmit,
  hasConversationSummary = false,
  fullscreenUrl,
}: PersonaChatPanelProps) {
  const messagesRef = useRef<HTMLDivElement>(null);
  // Ref on the chat input so we can restore focus after each response.
  const inputRef = useRef<HTMLInputElement>(null);
  // Track previous chatLoading to detect the loading→idle transition.
  const prevChatLoadingRef = useRef(chatLoading);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfError, setPdfError] = useState<string | null>(null);

  const exportableHistory = history.filter((m) => m.content.trim());
  const canDownloadPdf = exportableHistory.length > 0 && !pdfLoading;

  async function onDownloadPdf() {
    if (!canDownloadPdf) return;
    setPdfError(null);
    setPdfLoading(true);
    try {
      await downloadPersonaChatPdf({
        workspaceName,
        displayName,
        history: exportableHistory,
      });
    } catch (err) {
      setPdfError(err instanceof Error ? err.message : "PDF export failed");
    } finally {
      setPdfLoading(false);
    }
  }

  const headerActions = (
    <div className="flex flex-wrap items-center gap-2">
      {/* Fullscreen link — opens the dedicated chat page in a new tab */}
      {fullscreenUrl ? (
        <Link
          href={fullscreenUrl}
          target="_blank"
          rel="noopener noreferrer"
          title="Open full screen"
          aria-label="Open full screen chat"
          className="inline-flex h-8 w-8 items-center justify-center border-2 border-(--cm-border) bg-(--cm-surface) font-mono text-sm text-(--cm-text) transition-colors hover:bg-(--cm-surface-raised) active:translate-y-px"
        >
          ↗
        </Link>
      ) : null}
      <BrutalButton
        type="button"
        variant="ghost"
        onClick={onDownloadPdf}
        disabled={!canDownloadPdf}
        aria-label="Download chat as PDF"
      >
        {pdfLoading ? "Generating…" : "Download PDF"}
      </BrutalButton>
    </div>
  );

  const header = (
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div>
        <p className="font-mono text-xs uppercase tracking-widest">Persona chat</p>
        <p className="mt-1 text-xs text-[var(--cm-text-muted)]">
          Replies via Google Gemini using their real messages as style context
        </p>
        {hasConversationSummary ? (
          <p className="mt-1 font-mono text-[10px] uppercase text-[var(--cm-text-muted)]">
            Earlier messages summarized
          </p>
        ) : null}
        {pdfError ? (
          <p className="mt-1 text-xs text-[var(--cm-error)]">{pdfError}</p>
        ) : null}
      </div>
      {headerActions}
    </div>
  );

  useEffect(() => {
    const el = messagesRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [history, streamingBursts, chatLoading]);

  // Re-focus the input whenever chatLoading transitions from true → false.
  // This covers both the success path (stream done) and the error path.
  useEffect(() => {
    if (prevChatLoadingRef.current && !chatLoading) {
      inputRef.current?.focus();
    }
    prevChatLoadingRef.current = chatLoading;
  }, [chatLoading]);

  const inputForm = (
    <form onSubmit={onSubmit} className="flex gap-2">
      <BrutalInput
        ref={inputRef}
        value={chatInput}
        onChange={(e) => onChatInputChange(e.target.value)}
        placeholder="Message…"
        disabled={chatLoading}
      />
      <BrutalButton type="submit" disabled={chatLoading}>
        {chatLoading ? "…" : "Send"}
      </BrutalButton>
    </form>
  );

  const errorBlock = chatError ? (
    <p className="mt-2 whitespace-pre-wrap text-sm text-[var(--cm-error)]">{chatError}</p>
  ) : null;

  return (
    <BrutalPanel>
      {header}
      <div
        ref={messagesRef}
        className="mb-4 mt-3 flex max-h-80 flex-col gap-3 overflow-y-auto border-2 border-[var(--cm-border-muted)] p-3"
      >
        <ChatMessages
          displayName={displayName}
          history={history}
          streamingBursts={streamingBursts}
          chatLoading={chatLoading}
        />
      </div>
      {inputForm}
      {errorBlock}
    </BrutalPanel>
  );
}
