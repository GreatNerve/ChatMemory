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
    qa_min_passing_chunks: int = 2

    embed_batch_size: int = 32
    embed_device: str = "auto"  # auto | cuda | cpu

    min_workspace_messages: int = 50

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
