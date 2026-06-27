import json
from unittest.mock import patch

from app.services.vector_index import resolve_vector_store


def test_resolve_vector_store_prefers_chunks_json(tmp_path, monkeypatch):
    ws_id = "ws-test-1234"
    ws_dir = tmp_path / ws_id
    (ws_dir / "vectors").mkdir(parents=True)
    (ws_dir / "vectors" / "chunks.json").write_text("[{}]", encoding="utf-8")
    (ws_dir / "meta.json").write_text(
        json.dumps({"id": ws_id, "vectorStore": "file"}), encoding="utf-8"
    )

    monkeypatch.setattr("app.services.vector_index.workspace_path", lambda _id: ws_dir)
    with patch("app.services.vector_index.using_file_store", return_value=False):
        assert resolve_vector_store(ws_id) == "file"


def test_resolve_vector_store_detects_legacy_file_index(tmp_path, monkeypatch):
    ws_id = "ws-legacy-12"
    ws_dir = tmp_path / ws_id
    (ws_dir / "vectors").mkdir(parents=True)
    (ws_dir / "vectors" / "chunks.json").write_text(
        json.dumps([{"messageId": "1", "personId": "p1", "text": "hi"}]),
        encoding="utf-8",
    )
    (ws_dir / "meta.json").write_text(json.dumps({"id": ws_id}), encoding="utf-8")

    monkeypatch.setattr("app.services.vector_index.workspace_path", lambda _id: ws_dir)
    with patch("app.services.vector_index.using_file_store", return_value=False):
        assert resolve_vector_store(ws_id) == "file"
