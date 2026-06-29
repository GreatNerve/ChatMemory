from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_root: Path = Field(default=Path("../data"))
    embed_model: str = "intfloat/multilingual-e5-large"
    vector_store: str = "chroma"  # chroma | file | auto

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Persona activation message gates (legacy env names kept for compatibility)
    lora_min_messages: int = 200
    lora_thin_min_messages: int = 50

    qa_semantic_top_k: int = 20
    qa_bm25_top_k: int = 20
    qa_rerank_top_k: int = 8
    qa_grade_threshold: float = 0.6
    # Lower threshold used when 2+ query variants are generated (cross-language path).
    # A chunk found by multiple phrasings is likely relevant even if individual scores
    # fall below the standard 0.6 cutoff.
    qa_multi_query_grade_threshold: float = 0.45
    qa_min_passing_chunks: int = 2
    # Context window expansion for Q&A: how many messages to include before/after each matched hit.
    # More messages after than before — the answer typically follows the question in chat.
    qa_context_window_before: int = 3
    qa_context_window_after: int = 4

    persona_retrieve_top_k: int = 8
    persona_retrieve_weak_threshold: float = 0.32
    persona_retrieve_min_strong_hits: int = 2
    persona_retrieve_strong_score: float = 0.25
    persona_memory_window_before: int = 3
    # Bumped from 2 → 4 so Q&A replies that follow the hit are captured.
    persona_memory_window_after: int = 4
    persona_memory_max_blocks: int = 5
    # Minimum cosine similarity score a retrieved hit must reach before its memory
    # block is injected into the persona system prompt.  Hits below this threshold
    # are discarded entirely so weakly-related context never reaches the model.
    persona_memory_inject_min_score: float = 0.35
    # Lower gate used when retrieval runs in multi-query / cross-language mode
    # (i.e. the classify step returned 2+ search queries covering Hinglish and
    # English phrasings).  Cross-language embedding similarity is structurally
    # lower even for semantically identical content, so a tighter threshold would
    # silently discard valid matches like "intern lag gayi" ↔ "interning at EY".
    # Hits that appear in 2+ query result sets get an additional discount (×0.65)
    # on top of this threshold to reward multi-signal corroboration.
    persona_memory_inject_min_score_cross_lang: float = 0.22

    embed_batch_size: int = 32
    embed_device: str = "auto"  # auto | cuda | cpu

    min_workspace_messages: int = 50

    # When False the frontend defaults to output-only mode in the ThinkingPanel
    # (INPUT section hidden); user can still toggle locally via the UI button.
    thinking_show_input: bool = False

    # Hugging Face token — forwarded to os.environ so transformers picks it up
    hf_token: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def workspaces_dir(self) -> Path:
        return self.data_root / "workspaces"

    @property
    def jobs_dir(self) -> Path:
        return self.data_root / "jobs"

    @property
    def config_path(self) -> Path:
        return self.data_root / "config.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
