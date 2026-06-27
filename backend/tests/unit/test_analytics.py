from pathlib import Path

from app.services.analytics import compute_analytics
from app.services.parser.whatsapp import parse_whatsapp_export


def test_compute_analytics_on_fixture(tmp_path, monkeypatch):
    fixture = (
        Path(__file__).parent.parent / "fixtures" / "whatsapp" / "android_group.txt"
    )
    text = fixture.read_text(encoding="utf-8")
    ws_id = "test-ws"
    ws_dir = tmp_path / ws_id
    (ws_dir / "people").mkdir(parents=True)
    parsed = parse_whatsapp_export(text)
    senders = {m.sender for m in parsed.messages if not m.is_system}
    for i, sender in enumerate(sorted(senders)):
        (ws_dir / "people" / f"p{i}.json").write_text(
            f'{{"id":"p{i}","displayName":"{sender}"}}', encoding="utf-8"
        )
    (ws_dir / "export.txt").write_text(text, encoding="utf-8")
    monkeypatch.setattr(
        "app.services.analytics.workspace_path", lambda _id: ws_dir
    )
    data = compute_analytics(ws_id)
    assert len(data["people"]) >= 1
    assert "group" in data
    if len(data["people"]) >= 2:
        assert len(data["pairs"]) >= 1
