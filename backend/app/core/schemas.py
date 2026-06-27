from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field
from pydantic.alias_generators import to_camel


class ApiModel(BaseModel):
    """HTTP JSON uses camelCase; Python fields stay snake_case."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
        ser_json_by_alias=True,
    )


class ErrorBody(ApiModel):
    code: str
    message: str
    field_errors: dict[str, str] | None = None


class ErrorResponse(ApiModel):
    error: ErrorBody


JobStatus = Literal["queued", "running", "done", "error"]
JobType = Literal["ingest", "persona_train"]
IngestStatus = Literal["pending", "running", "done", "error"]
PersonaStatus = Literal["not_enough", "thin", "ready", "training", "ready_model", "error"]


class StyleProfile(ApiModel):
    avg_message_length: float = 0
    emoji_rate: float = 0
    hinglish_ratio: float = 0


class SampleMessage(ApiModel):
    message_id: str | None = None
    timestamp: str
    text: str


class WorkspaceSummary(ApiModel):
    id: str
    name: str
    created_at: datetime
    message_count: int = 0
    speaker_count: int = 0
    date_from: str | None = None
    date_to: str | None = None
    ingest_status: IngestStatus = "pending"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_group(self) -> bool:
        """True when there are more than 2 speakers (group chat); False for 1-on-1."""
        return self.speaker_count > 2


class TopSpeaker(ApiModel):
    person_id: str
    display_name: str
    message_count: int


class WorkspaceDetail(WorkspaceSummary):
    top_speakers: list[TopSpeaker] = Field(default_factory=list)


class ActivityBucket(ApiModel):
    key: int
    label: str
    count: int


class WeeklyPoint(ApiModel):
    """One data point in the conversation-growth time series."""

    week: str  # ISO week key e.g. "2024-W03"
    label: str  # Short display label e.g. "Jan 15"
    count: int


class HeatmapCell(ApiModel):
    """Non-zero cell in the hour×day message-frequency heatmap."""

    hour: int  # 0–23
    day: int  # 0 Mon … 6 Sun
    count: int


class ResponseTimeBucket(ApiModel):
    """One bar in the per-person response-time histogram."""

    label: str  # e.g. "<1m", "1–5m"
    count: int


class PersonAnalytics(ApiModel):
    person_id: str
    display_name: str
    message_count: int
    share_percent: float = 0
    avg_message_length: float = 0
    avg_response_seconds: float | None = None
    median_response_seconds: float | None = None
    avg_response_label: str | None = None
    replies_given: int = 0
    replies_received: int = 0
    initiations: int = 0
    peak_hour: int | None = None
    peak_hour_label: str | None = None
    active_hours: list[ActivityBucket] = Field(default_factory=list)
    active_days: list[ActivityBucket] = Field(default_factory=list)
    response_time_buckets: list[ResponseTimeBucket] = Field(default_factory=list)


class PairAnalytics(ApiModel):
    person_a_id: str
    person_a_name: str
    person_b_id: str
    person_b_name: str
    exchanges: int = 0
    a_to_b_replies: int = 0
    b_to_a_replies: int = 0
    avg_response_seconds: float | None = None
    avg_response_label: str | None = None
    connection_score: float = 0


class GroupAnalytics(ApiModel):
    busiest_hour: int | None = None
    busiest_hour_label: str | None = None
    busiest_day: str | None = None
    avg_response_seconds: float | None = None
    avg_response_label: str | None = None
    median_messages_per_day: float = 0
    active_hours: list[ActivityBucket] = Field(default_factory=list)
    active_days: list[ActivityBucket] = Field(default_factory=list)
    strongest_pair: PairAnalytics | None = None
    # New: time-series and heatmap analytics
    weekly_series: list[WeeklyPoint] = Field(default_factory=list)
    top_active_weeks: list[WeeklyPoint] = Field(default_factory=list)
    heatmap: list[HeatmapCell] = Field(default_factory=list)


class WorkspaceAnalytics(ApiModel):
    computed_at: str
    group: GroupAnalytics
    people: list[PersonAnalytics] = Field(default_factory=list)
    pairs: list[PairAnalytics] = Field(default_factory=list)


class PersonSummary(ApiModel):
    id: str
    display_name: str
    message_count: int = 0
    first_seen: str | None = None
    last_seen: str | None = None
    persona_status: PersonaStatus = "not_enough"


class PersonDetail(PersonSummary):
    ollama_model_name: str | None = None
    style_profile: StyleProfile = Field(default_factory=StyleProfile)
    sample_messages: list[SampleMessage] = Field(default_factory=list)
    train_eligible: bool = False
    train_warning: str | None = None
    last_train_job_id: str | None = None
    # Extracted at persona-build time by Gemini; None for personas built before this feature.
    personality_notes: str | None = None
    # HOW the person types: casing, punctuation, abbreviations, emoji patterns, sentence structure.
    # Extracted separately from personality_notes so it can be injected verbatim into the system prompt.
    writing_style_notes: str | None = None
    # Deep multi-call analysis of messaging patterns: vocabulary, topics, emotional tone, dynamics.
    # Extracted from the full message corpus via chunked Gemini calls during persona build.
    chat_analysis: str | None = None


class Citation(ApiModel):
    message_id: str
    speaker: str
    timestamp: str
    snippet: str
    score: float | None = None


class AskRequest(ApiModel):
    question: str
    speaker: str | None = None
    date_from: str | None = None
    date_to: str | None = None


class AskResponse(ApiModel):
    status: Literal["answered", "not_found"]
    answer: str | None = None
    reason: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    near_misses: list[Citation] = Field(default_factory=list)


class TrainRequest(ApiModel):
    consent: bool = False
    force_thin: bool = False
    force_retrain: bool = False


class ChatMessage(ApiModel):
    role: Literal["user", "assistant"]
    content: str


class PersonaChatRequest(ApiModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    previous_interaction_id: str | None = None
    conversation_summary: str | None = None


class PersonaSummarizeRequest(ApiModel):
    history: list[ChatMessage] = Field(default_factory=list)
    keep_recent: int = 10


class PersonaSummarizeResponse(ApiModel):
    summary: str
    summarized_turn_count: int


class PersonaChatResponse(ApiModel):
    reply: str
    model: str
    interaction_id: str | None = None


class JobSnapshot(ApiModel):
    id: str
    type: JobType
    workspace_id: str | None = None
    person_id: str | None = None
    status: JobStatus
    step: str | None = None
    percent: int = 0
    message: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    eta_seconds: int | None = None


class SettingsResponse(ApiModel):
    data_root: str
    embed_model: str
    active_embed_backend: str = "local"
    embed_device: str = "cpu"
    vector_store: str = "chroma"
    gpu_available: bool = True
    gpu_busy: bool = False
    active_job_id: str | None = None
    gemini_configured: bool = False
    gemini_model: str = "gemini-3.5-flash"


class SettingsUpdate(ApiModel):
    data_root: str | None = None


class HealthResponse(ApiModel):
    status: Literal["ok", "degraded"]
    data_root_writable: bool
    ml_stack_available: bool = True
    ml_stack_error: str | None = None
    gemini_configured: bool = False
    embed_ready: bool = False
