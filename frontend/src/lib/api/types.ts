export type IngestStatus = "pending" | "running" | "done" | "error";
export type PersonaStatus =
  | "not_enough"
  | "thin"
  | "ready"
  | "training"
  | "ready_model"
  | "error";
export type JobStatus = "queued" | "running" | "done" | "error";

export interface WorkspaceSummary {
  id: string;
  name: string;
  createdAt: string;
  messageCount: number;
  speakerCount: number;
  dateFrom: string | null;
  dateTo: string | null;
  ingestStatus: IngestStatus;
  /** True when speakerCount > 2 (group chat); false for a 1-on-1 conversation. Computed by backend. */
  isGroup: boolean;
}

export interface TopSpeaker {
  personId: string;
  displayName: string;
  messageCount: number;
}

export interface WorkspaceDetail extends WorkspaceSummary {
  topSpeakers: TopSpeaker[];
}

export interface PersonSummary {
  id: string;
  displayName: string;
  messageCount: number;
  firstSeen: string | null;
  lastSeen: string | null;
  personaStatus: PersonaStatus;
}

export interface StyleProfile {
  avgMessageLength: number;
  emojiRate: number;
  hinglishRatio: number;
}

export interface SampleMessage {
  messageId?: string;
  timestamp: string;
  text: string;
}

export interface PersonDetail extends PersonSummary {
  ollamaModelName: string | null;
  styleProfile: StyleProfile;
  sampleMessages: SampleMessage[];
  trainEligible: boolean;
  trainWarning: string | null;
  lastTrainJobId?: string | null;
}

export interface Citation {
  messageId: string;
  speaker: string;
  timestamp: string;
  snippet: string;
  score?: number | null;
}

export interface AskResponse {
  status: "answered" | "not_found";
  answer: string | null;
  reason?: string | null;
  citations: Citation[];
  nearMisses: Citation[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface PersonaChatRequest {
  message: string;
  history: ChatMessage[];
  previousInteractionId?: string | null;
  conversationSummary?: string | null;
}

export interface PersonaSummarizeRequest {
  history: ChatMessage[];
  keepRecent?: number;
}

export interface PersonaSummarizeResponse {
  summary: string;
  summarizedTurnCount: number;
}

export interface PersonaChatResponse {
  reply: string;
  model: string;
  interactionId?: string | null;
}

export interface JobSnapshot {
  id: string;
  type: "ingest" | "persona_train";
  workspaceId?: string | null;
  personId?: string | null;
  status: JobStatus;
  step?: string | null;
  percent: number;
  message?: string | null;
  error?: string | null;
  result?: Record<string, unknown> | null;
  etaSeconds?: number | null;
}

export interface Settings {
  dataRoot: string;
  embedModel: string;
  activeEmbedBackend?: string;
  embedDevice?: string;
  vectorStore?: string;
  gpuAvailable: boolean;
  gpuBusy: boolean;
  activeJobId: string | null;
  geminiConfigured?: boolean;
  geminiModel?: string;
}

export interface Health {
  status: "ok" | "degraded";
  dataRootWritable: boolean;
  mlStackAvailable?: boolean;
  mlStackError?: string | null;
  geminiConfigured?: boolean;
  embedReady?: boolean;
}

export interface ActivityBucket {
  key: number;
  label: string;
  count: number;
}

/** One point in the weekly message-count time series. */
export interface WeeklyPoint {
  week: string;   // ISO week key e.g. "2024-W03"
  label: string;  // Short display label e.g. "Jan 15"
  count: number;
}

/** Non-zero cell in the hour×day message-frequency heatmap. */
export interface HeatmapCell {
  hour: number; // 0–23
  day: number;  // 0 Mon … 6 Sun
  count: number;
}

/** One bar in the per-person response-time histogram. */
export interface ResponseTimeBucket {
  label: string; // e.g. "<1m", "1–5m"
  count: number;
}

export interface PersonAnalytics {
  personId: string;
  displayName: string;
  messageCount: number;
  sharePercent: number;
  avgMessageLength: number;
  /** @deprecated Use typicalPickupReply — now reflects session-aware pickup timing */
  avgResponseSeconds: number | null;
  /** @deprecated Use typicalPickupReply */
  medianResponseSeconds: number | null;
  /** @deprecated Use typicalPickupReplyLabel */
  avgResponseLabel: string | null;
  /** Median first-reply time after a session break (the "how long to pick up the phone" stat) */
  typicalPickupReply: number | null;
  typicalPickupReplyLabel: string | null;
  /** Median reply time during active back-and-forth within a session */
  typicalBurstReply: number | null;
  typicalBurstReplyLabel: string | null;
  repliesGiven: number;
  repliesReceived: number;
  initiations: number;
  peakHour: number | null;
  peakHourLabel: string | null;
  activeHours: ActivityBucket[];
  activeDays: ActivityBucket[];
  responseTimeBuckets: ResponseTimeBucket[];
}

export interface PairAnalytics {
  personAId: string;
  personAName: string;
  personBId: string;
  personBName: string;
  exchanges: number;
  aToBReplies: number;
  bToAReplies: number;
  /** @deprecated Use typicalPickupReply */
  avgResponseSeconds: number | null;
  /** @deprecated Use typicalPickupReplyLabel */
  avgResponseLabel: string | null;
  /** Median first-reply time after a session break for this pair */
  typicalPickupReply: number | null;
  typicalPickupReplyLabel: string | null;
  /** Median within-session reply time for this pair */
  typicalBurstReply: number | null;
  typicalBurstReplyLabel: string | null;
  connectionScore: number;
}

export interface GroupAnalytics {
  busiestHour: number | null;
  busiestHourLabel: string | null;
  busiestDay: string | null;
  avgResponseSeconds: number | null;
  avgResponseLabel: string | null;
  medianMessagesPerDay: number;
  activeHours: ActivityBucket[];
  activeDays: ActivityBucket[];
  strongestPair: PairAnalytics | null;
  weeklySeries: WeeklyPoint[];
  topActiveWeeks: WeeklyPoint[];
  heatmap: HeatmapCell[];
}

export interface WorkspaceAnalytics {
  computedAt: string;
  group: GroupAnalytics;
  people: PersonAnalytics[];
  pairs: PairAnalytics[];
}
