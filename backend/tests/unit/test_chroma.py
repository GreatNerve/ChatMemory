from datetime import datetime

import pytest

from app.services import chroma as chroma_service
from app.services.parser.whatsapp import Message


@pytest.fixture
def mock_embed(monkeypatch):
    def fake_embed_texts(texts, batch_size=None):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def fake_embed_query(text):
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr("app.services.embed.embed_texts", fake_embed_texts)
    monkeypatch.setattr("app.services.embed.embed_query", fake_embed_query)


def _sample_messages() -> list[Message]:
    return [
        Message(
            id="msg-a",
            timestamp=datetime(2024, 3, 12, 18, 22),
            sender="Rahul",
            text="Goa trip final kar dete hain March end",
        ),
        Message(
            id="msg-b",
            timestamp=datetime(2024, 3, 13, 9, 0),
            sender="Priya",
            text="Flight book kar li?",
        ),
    ]


def test_chroma_upsert_and_semantic_search(tmp_path, monkeypatch, mock_embed):
    ws_id = "ws-chroma-test"
    ws_dir = tmp_path / ws_id
    ws_dir.mkdir(parents=True)

    monkeypatch.setattr("app.services.chroma.workspace_path", lambda _id: ws_dir)
    chroma_service.clear_store_cache()

    messages = _sample_messages()
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    person_map = {"Rahul": "person-1", "Priya": "person-2"}

    count = chroma_service.upsert_messages(ws_id, messages, embeddings, person_map)
    assert count == 2

    hits = chroma_service.semantic_search(ws_id, [1.0, 0.0, 0.0], top_k=2)
    assert len(hits) >= 1
    assert hits[0]["message_id"] == "msg-a"
    assert hits[0]["speaker"] == "Rahul"

    by_person = chroma_service.messages_for_person(ws_id, "person-2")
    assert len(by_person) == 1
    assert by_person[0]["message_id"] == "msg-b"

    chroma_service.clear_store_cache()


def test_active_vector_store_defaults_to_chroma(monkeypatch):
    from app.core.config import get_settings
    from app.services.vector_index import active_vector_store_mode

    get_settings.cache_clear()
    monkeypatch.setenv("VECTOR_STORE", "chroma")
    get_settings.cache_clear()
    monkeypatch.setattr("app.services.vector_index.chroma_available", lambda: True)
    assert active_vector_store_mode() == "chroma"
    get_settings.cache_clear()
