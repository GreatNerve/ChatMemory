"use client";

import { useEffect, useRef, useState } from "react";
import { StageEvent } from "@/lib/api/types";

// ── Types ────────────────────────────────────────────────────────────────────

interface ThinkingPanelProps {
  /** Accumulated stage events for this message (both running + done). */
  stages: StageEvent[];
  /** True while the reply is still streaming. */
  isStreaming: boolean;
  /** When true, hides the INPUT section and shows only OUTPUT. */
  outputOnly: boolean;
  onToggleOutputOnly: () => void;
}

// Pipeline order for consistent display — matches the graph execution order.
const STAGE_ORDER: StageEvent["stage"][] = [
  "route",
  "classify",
  "rewrite",
  "retrieve",
  "generate",
];

// Human-readable labels for each stage name.
const STAGE_LABELS: Record<StageEvent["stage"], string> = {
  route: "ROUTE",
  classify: "CLASSIFY",
  rewrite: "REWRITE",
  retrieve: "RETRIEVE",
  generate: "GENERATE",
};

// ── Internal helpers ─────────────────────────────────────────────────────────

/** Derive a compact one-line output summary for the collapsed stage row. */
function stageSummary(stage: StageEvent["stage"], output: Record<string, unknown> | null): string {
  if (!output) return "";
  switch (stage) {
    case "route":
      // e.g. → ambiguous
      return `→ ${output.route ?? "?"}`;
    case "classify": {
      // Show only the true flags + query count: → needs_history · needs_rewrite · 4 queries
      const sq = Array.isArray(output.search_queries) ? output.search_queries.length : 0;
      const flags: string[] = [];
      if (output.needs_history) flags.push("needs_history");
      if (output.needs_rewrite) flags.push("needs_rewrite");
      flags.push(`${sq} quer${sq === 1 ? "y" : "ies"}`);
      return `→ ${flags.join(" · ")}`;
    }
    case "rewrite": {
      // e.g. → "internship location city…"
      const rq = typeof output.rewritten_query === "string" ? output.rewritten_query : "";
      return rq ? `→ "${rq.slice(0, 40)}${rq.length > 40 ? "…" : ""}"` : "→ skipped";
    }
    case "retrieve": {
      // e.g. → 5 blocks · top: "she was working at EY as an intern…"
      const blocks = typeof output.blocks_retrieved === "number" ? output.blocks_retrieved : 0;
      const snip = typeof output.top_snippet === "string" ? output.top_snippet : null;
      if (snip) {
        const truncated = snip.slice(0, 40) + (snip.length > 40 ? "…" : "");
        return `→ ${blocks} block${blocks !== 1 ? "s" : ""} · top: "${truncated}"`;
      }
      return `→ ${blocks} block${blocks !== 1 ? "s" : ""}`;
    }
    case "generate":
      // e.g. → 50 chars
      return `→ ${output.response_length ?? 0} chars`;
    default:
      return "";
  }
}

/** Stable JSON stringify for display — handles undefined, rounds floats. */
function prettyJson(obj: Record<string, unknown> | null): string {
  if (!obj) return "null";
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

interface StageState {
  stage: StageEvent["stage"];
  running: boolean;
  done: boolean;
  input: Record<string, unknown> | null;
  output: Record<string, unknown> | null;
}

function buildStageList(events: StageEvent[]): StageState[] {
  const stageMap = new Map<string, StageState>();
  for (const ev of events) {
    const existing: StageState = stageMap.get(ev.stage) ?? {
      stage: ev.stage,
      running: false,
      done: false,
      input: null,
      output: null,
    };
    if (ev.status === "running") {
      existing.running = true;
      existing.input = ev.input;
    } else if (ev.status === "done") {
      existing.done = true;
      existing.output = ev.output;
      // Keep input from running event if already captured; fall back to done input.
      if (!existing.input) existing.input = ev.input;
    }
    stageMap.set(ev.stage, existing);
  }
  // Return in pipeline order, only stages that actually appeared.
  return STAGE_ORDER.filter((s) => stageMap.has(s)).map((s) => stageMap.get(s)!);
}

// ── Stage row ────────────────────────────────────────────────────────────────

function SpinnerIcon() {
  return (
    <span
      className="inline-block animate-spin"
      aria-label="Running"
      style={{ display: "inline-block", fontSize: 10 }}
    >
      ⟳
    </span>
  );
}

function StageRow({
  stage,
  outputOnly,
}: {
  stage: StageState;
  outputOnly: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const summary = stageSummary(stage.stage, stage.output);

  return (
    <div
      style={{
        borderBottom: "1px solid #222",
        paddingBottom: 4,
        marginBottom: 4,
      }}
    >
      {/* Row header — click to expand input/output */}
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          width: "100%",
          background: "none",
          border: "none",
          cursor: "pointer",
          padding: "2px 0",
          textAlign: "left",
          color: "inherit",
          fontFamily: "inherit",
          fontSize: "inherit",
        }}
        aria-expanded={expanded}
      >
        {/* Stage name */}
        <span
          style={{
            width: 70,
            flexShrink: 0,
            color: stage.done ? "#a0ff80" : stage.running ? "#ffd080" : "#555",
            letterSpacing: "0.08em",
          }}
        >
          {STAGE_LABELS[stage.stage]}
        </span>

        {/* Status icon */}
        <span style={{ width: 16, flexShrink: 0, color: "#888" }}>
          {stage.done ? (
            <span style={{ color: "#60d060" }}>✓</span>
          ) : stage.running ? (
            <SpinnerIcon />
          ) : null}
        </span>

        {/* Output summary */}
        <span style={{ color: "#777", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {summary}
        </span>

        {/* Expand caret */}
        {(stage.input || stage.output) ? (
          <span style={{ color: "#444", flexShrink: 0 }}>{expanded ? "▲" : "▼"}</span>
        ) : null}
      </button>

      {/* Expanded input/output panel */}
      {expanded && (
        <div
          style={{
            marginTop: 4,
            paddingLeft: 8,
            borderLeft: "2px solid #333",
          }}
        >
          {!outputOnly && stage.input !== null && (
            <div style={{ marginBottom: 6 }}>
              <div style={{ color: "#555", marginBottom: 2, letterSpacing: "0.06em" }}>INPUT:</div>
              <pre
                style={{
                  margin: 0,
                  color: "#888",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-all",
                  fontSize: 11,
                }}
              >
                {prettyJson(stage.input)}
              </pre>
            </div>
          )}
          {stage.output !== null && (
            <div>
              <div style={{ color: "#555", marginBottom: 2, letterSpacing: "0.06em" }}>OUTPUT:</div>
              <pre
                style={{
                  margin: 0,
                  color: "#aaa",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-all",
                  fontSize: 11,
                }}
              >
                {prettyJson(stage.output)}
              </pre>
            </div>
          )}
          {stage.running && !stage.done && (
            <div style={{ color: "#555", fontStyle: "italic" }}>running…</div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main component ───────────────────────────────────────────────────────────

export function ThinkingPanel({
  stages,
  isStreaming,
  outputOnly,
  onToggleOutputOnly,
}: ThinkingPanelProps) {
  const [manuallyExpanded, setManuallyExpanded] = useState(false);
  const prevStreamingRef = useRef(isStreaming);

  // Auto-collapse when streaming ends; auto-expand when a new stream starts.
  useEffect(() => {
    if (!prevStreamingRef.current && isStreaming) {
      // New stream started — expand to show live progress.
      setManuallyExpanded(true);
    } else if (prevStreamingRef.current && !isStreaming) {
      // Stream finished — collapse to the summary line.
      setManuallyExpanded(false);
    }
    prevStreamingRef.current = isStreaming;
  }, [isStreaming]);

  const stageList = buildStageList(stages);
  const doneCount = stageList.filter((s) => s.done).length;

  // Nothing to show yet — hide the panel entirely.
  if (stageList.length === 0 && !isStreaming) return null;

  const isExpanded = isStreaming || manuallyExpanded;

  return (
    <div
      style={{
        marginTop: 6,
        border: "2px solid #333",
        background: "#0d0d0d",
        fontFamily: "monospace",
        fontSize: 12,
      }}
      role="region"
      aria-label="Thinking process"
    >
      {/* Panel header — always visible */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "4px 8px",
          borderBottom: isExpanded ? "1px solid #222" : "none",
          gap: 8,
        }}
      >
        {/* Toggle expand/collapse (after streaming ends) */}
        <button
          type="button"
          onClick={() => !isStreaming && setManuallyExpanded((v) => !v)}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            background: "none",
            border: "none",
            cursor: isStreaming ? "default" : "pointer",
            padding: 0,
            color: "#555",
            fontFamily: "inherit",
            fontSize: "inherit",
            letterSpacing: "0.08em",
          }}
          aria-expanded={isExpanded}
          aria-label="Toggle thinking panel"
        >
          <span style={{ color: "#444" }}>⚙</span>
          <span>
            {isStreaming ? (
              <>
                thinking<span style={{ animation: "pulse 1s infinite" }}>…</span>
              </>
            ) : (
              `thinking · ${doneCount} stage${doneCount !== 1 ? "s" : ""}`
            )}
          </span>
          {!isStreaming && (
            <span style={{ color: "#333" }}>{manuallyExpanded ? "▲" : "▼"}</span>
          )}
        </button>

        {/* output-only toggle — persists to localStorage */}
        {isExpanded && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onToggleOutputOnly();
            }}
            style={{
              background: "none",
              border: "1px solid #333",
              cursor: "pointer",
              padding: "1px 6px",
              color: "#555",
              fontFamily: "inherit",
              fontSize: 10,
              letterSpacing: "0.05em",
            }}
            title={outputOnly ? "Currently showing output only — click to also show input" : "Currently showing input+output — click to show output only"}
            aria-pressed={outputOnly}
          >
            {outputOnly ? "output only" : "input+output"}
          </button>
        )}
      </div>

      {/* Expanded stage list */}
      {isExpanded && (
        <div style={{ padding: "6px 8px" }}>
          {stageList.length === 0 ? (
            <div style={{ color: "#444", fontStyle: "italic" }}>waiting for stages…</div>
          ) : (
            stageList.map((s) => (
              <StageRow key={s.stage} stage={s} outputOnly={outputOnly} />
            ))
          )}
        </div>
      )}
    </div>
  );
}
