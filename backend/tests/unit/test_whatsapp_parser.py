from datetime import datetime
from pathlib import Path

import pytest

from app.services.parser.whatsapp import Message, is_noise_message, non_system_messages, parse_whatsapp_export


def _msg(text: str, *, is_system: bool = False) -> Message:
    """Construct a minimal Message for unit tests."""
    return Message(id="test-id", timestamp=datetime(2024, 1, 1), sender="Alice", text=text, is_system=is_system)

FIXTURE = """12/03/2024, 18:22 - Rahul: yaar kal meeting hai kya?
12/03/2024, 18:23 - Priya: haan 5 baje
12/03/2024, 18:24 - Rahul: location yeh hai
maps link here
"""


def test_parse_android_multiline():
    result = parse_whatsapp_export(FIXTURE)
    assert result.format == "android"
    assert len(result.messages) == 3
    assert "maps link" in result.messages[-1].text


def test_parse_ios_format():
    ios = "[12/03/2024, 18:22:15] Rahul: hello bhai"
    result = parse_whatsapp_export(ios)
    assert len(result.messages) == 1
    assert result.messages[0].sender == "Rahul"


def test_parse_us_date_ampm_narrow_space():
    # U+202F before AM/PM — common on US Android exports
    line = "9/21/23, 8:23\u202fPM - Rahul: hello"
    result = parse_whatsapp_export(line)
    assert len(result.messages) == 1
    assert result.messages[0].timestamp.year == 2023
    assert result.messages[0].timestamp.month == 9
    assert result.messages[0].timestamp.day == 21
    assert result.messages[0].sender == "Rahul"


def test_parse_group_name_with_commas():
    text = (
        "6/24/24, 12:32\u202fPM - Test Group, Sample Workspace: Hi bob\n"
        "6/24/24, 12:32\u202fPM - Test Group, Sample Workspace is a contact\n"
        "9/21/23, 8:23\u202fPM - Messages and calls are end-to-end encrypted. Learn more\n"
    )
    result = parse_whatsapp_export(text)
    assert len(result.messages) == 3
    assert result.messages[0].sender == "Test Group, Sample Workspace"
    assert result.messages[0].text == "Hi bob"
    assert result.messages[1].is_system
    assert "contact" in result.messages[1].text
    assert result.messages[2].is_system
    assert "encrypted" in result.messages[2].text


def test_workspace_export_file():
    path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "workspaces"
        / "5f2978b6-c6fd-4a10-8f3e-b8741c840b71"
        / "export.txt"
    )
    if not path.exists():
        return
    result = parse_whatsapp_export(path.read_text(encoding="utf-8"))
    assert len(result.messages) >= 50
    assert len({m.sender for m in result.messages if not m.is_system}) >= 2


def test_fixture_file_if_present():
    path = Path(__file__).resolve().parent.parent / "fixtures" / "whatsapp" / "android_group.txt"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        result = parse_whatsapp_export(text)
        assert len(result.messages) >= 50


class TestIsNoiseMessage:
    def test_media_omitted(self):
        assert is_noise_message(_msg("<Media omitted>"))

    def test_media_omitted_case_insensitive(self):
        assert is_noise_message(_msg("<media omitted>"))

    def test_deleted_placeholder(self):
        # preprocess converts "This message was deleted" → "[message deleted]"
        assert is_noise_message(_msg("[message deleted]"))

    def test_raw_deleted_variants(self):
        # belt-and-suspenders: raw forms before preprocessing
        assert is_noise_message(_msg("This message was deleted"))
        assert is_noise_message(_msg("This message was deleted."))
        assert is_noise_message(_msg("message was deleted"))

    @pytest.mark.parametrize("kind", ["image", "video", "audio", "document", "sticker", "GIF"])
    def test_attachment_omitted_variants(self, kind: str):
        # LRM (U+200E) is stripped by preprocessor, leaving bare "<kind> omitted"
        assert is_noise_message(_msg(f"{kind} omitted"))
        assert is_noise_message(_msg(f"{kind.lower()} omitted"))

    def test_blank_messages(self):
        assert is_noise_message(_msg(""))
        assert is_noise_message(_msg("   "))
        assert is_noise_message(_msg("\t\n"))

    def test_real_text_not_noise(self):
        assert not is_noise_message(_msg("hello world"))
        assert not is_noise_message(_msg("yaar kal meeting hai kya?"))
        # Partial match must NOT be flagged
        assert not is_noise_message(_msg("Media is great"))
        assert not is_noise_message(_msg("I omitted the details"))
        assert not is_noise_message(_msg("The image looks nice"))

    def test_noise_with_surrounding_whitespace(self):
        assert is_noise_message(_msg("  <Media omitted>  "))
        assert is_noise_message(_msg("  [message deleted]  "))


class TestNonSystemMessages:
    def test_excludes_system_messages(self):
        msgs = [_msg("hello"), _msg("system event", is_system=True)]
        assert len(non_system_messages(msgs)) == 1

    def test_excludes_noise_placeholders(self):
        noise = [
            _msg("<Media omitted>"),
            _msg("[message deleted]"),
            _msg("image omitted"),
            _msg("video omitted"),
            _msg("audio omitted"),
            _msg("document omitted"),
            _msg("sticker omitted"),
            _msg("GIF omitted"),
            _msg(""),
            _msg("   "),
        ]
        assert non_system_messages(noise) == []

    def test_keeps_real_messages(self):
        msgs = [
            _msg("hello bhai"),
            _msg("<Media omitted>"),
            _msg("[message deleted]"),
            _msg("real content here"),
            _msg("system", is_system=True),
        ]
        result = non_system_messages(msgs)
        assert len(result) == 2
        assert result[0].text == "hello bhai"
        assert result[1].text == "real content here"

    def test_end_to_end_noise_filtered_from_parse(self):
        """Parsed noise messages must not appear in non_system_messages output."""
        raw = (
            "12/03/2024, 18:22 - Rahul: hello bhai\n"
            "12/03/2024, 18:23 - Priya: <Media omitted>\n"
            "12/03/2024, 18:24 - Rahul: image omitted\n"
            "12/03/2024, 18:25 - Priya: real reply here\n"
        )
        result = parse_whatsapp_export(raw)
        usable = non_system_messages(result.messages)
        texts = [m.text for m in usable]
        assert "hello bhai" in texts
        assert "real reply here" in texts
        assert "<Media omitted>" not in texts
        assert "image omitted" not in texts
