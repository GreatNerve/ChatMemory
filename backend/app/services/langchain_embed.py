"""LangChain Embeddings adapter over ChatMemory embed service."""

from __future__ import annotations

from app.services import embed as embed_service
from langchain_core.embeddings import Embeddings


class ChatMemoryEmbeddings(Embeddings):
    """Wrap local sentence-transformers embeddings as LangChain Embeddings."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return embed_service.embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed_service.embed_query(text)


def get_embeddings() -> ChatMemoryEmbeddings:
    return ChatMemoryEmbeddings()
