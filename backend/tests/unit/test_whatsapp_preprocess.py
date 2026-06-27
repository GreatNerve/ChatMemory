from pathlib import Path

from app.services.parser.preprocess import (
    DELETED_PLACEHOLDER,
    preprocess_whatsapp_export,
)
from app.services.parser.whatsapp import parse_whatsapp_export

# Snippets derived from no-push export patterns (not committed).
_TEST_GROUP = "Test Group, Sample Workspace"
_NBSP_TIME = "\u202f"


def test_normalize_crlf_and_bom():
    raw = "\ufeff6/24/24, 12:32 PM - Rahul: hi\r\n6/24/24, 12:33 PM - Priya: ok\r"
    cleaned = preprocess_whatsapp_export(raw)
    assert "\r" not in cleaned
    assert cleaned.startswith("6/24/24")
    result = parse_whatsapp_export(cleaned)
    assert len(result.messages) == 2


def test_strip_zero_width_and_narrow_nbsp_before_ampm():
    raw = f"9/21/23, 8:23\u200e{_NBSP_TIME}PM - Rahul: hello"
    cleaned = preprocess_whatsapp_export(raw)
    assert "\u200e" not in cleaned
    assert "\u202f" not in cleaned
    result = parse_whatsapp_export(cleaned)
    assert len(result.messages) == 1
    assert result.messages[0].sender == "Rahul"


def test_strip_edited_suffix_on_header_and_continuation():
    raw = (
        f"8/6/25, 8:29 PM - {_TEST_GROUP}: Bhai kaam ke time no mazak "
        f"<This message was edited>\n"
        "Ig inko inform kr de abhi <This message was edited>"
    )
    cleaned = preprocess_whatsapp_export(raw)
    assert "<This message was edited>" not in cleaned
    result = parse_whatsapp_export(cleaned)
    assert len(result.messages) == 1
    assert "mazak" in result.messages[0].text
    assert "Ig inko inform" in result.messages[0].text
    assert "<This message was edited>" not in result.messages[0].text


def test_tag_deleted_message_body():
    raw = f"4/6/25, 6:15 PM - {_TEST_GROUP}: This message was deleted"
    cleaned = preprocess_whatsapp_export(raw)
    result = parse_whatsapp_export(cleaned)
    assert result.messages[0].text == DELETED_PLACEHOLDER


def test_group_name_with_commas_and_apostrophe():
    raw = (
        f"6/24/24, 12:32{_NBSP_TIME}PM - {_TEST_GROUP}: Hi bob\n"
        "Alice here (team mate)\n"
        "Tech me what is the current scenario?"
    )
    cleaned = preprocess_whatsapp_export(raw)
    result = parse_whatsapp_export(cleaned)
    assert result.messages[0].sender == _TEST_GROUP
    assert "Alice here" in result.messages[0].text
    assert "scenario" in result.messages[0].text


def test_collapse_blank_lines_inside_multiline_message():
    raw = (
        f"10/26/25, 8:18 PM - {_TEST_GROUP}: Yr inka department decided nhi h abhi\n"
        "\n"
        "General pi lena hoga and usme hi pooch lenge"
    )
    cleaned = preprocess_whatsapp_export(raw)
    assert "\n\n" not in cleaned
    result = parse_whatsapp_export(cleaned)
    assert len(result.messages) == 1
    assert "General pi lena" in result.messages[0].text


def test_system_contact_and_encryption_lines_unchanged():
    raw = (
        f"9/21/23, 8:23{_NBSP_TIME}PM - Messages and calls are end-to-end encrypted. "
        "*Learn more*\n"
        f"6/24/24, 12:32{_NBSP_TIME}PM - {_TEST_GROUP} is a contact"
    )
    cleaned = preprocess_whatsapp_export(raw)
    result = parse_whatsapp_export(cleaned)
    assert len(result.messages) == 2
    assert result.messages[0].is_system
    assert "encrypted" in result.messages[0].text
    assert result.messages[1].is_system
    assert "contact" in result.messages[1].text


def test_media_omitted_preserved():
    raw = "1/24/25, 10:34 PM - Bob: <Media omitted>"
    cleaned = preprocess_whatsapp_export(raw)
    result = parse_whatsapp_export(cleaned)
    assert result.messages[0].text == "<Media omitted>"


def test_group_patterns_fixture():
    path = Path(__file__).resolve().parent.parent / "fixtures" / "whatsapp" / "group_patterns.txt"
    raw = path.read_text(encoding="utf-8")
    cleaned = preprocess_whatsapp_export(raw)
    result = parse_whatsapp_export(cleaned)
    assert len(result.messages) >= 6
    assert any(m.sender == _TEST_GROUP for m in result.messages)
    assert "<This message was edited>" not in cleaned


def test_local_export_file_if_present():
    """Optional integration against local no-push sample (not committed)."""
    path = (
        Path(__file__).resolve().parents[3]
        / "no-push"
        / "WhatsApp Chat with Test Group, Sample Workspace.txt"
    )
    if not path.exists():
        return
    raw = path.read_text(encoding="utf-8")
    cleaned = preprocess_whatsapp_export(raw)
    result = parse_whatsapp_export(cleaned)
    assert len(result.messages) >= 50
    edited = [m for m in result.messages if "<This message was edited>" in m.text]
    assert edited == []
    assert any(m.sender == _TEST_GROUP for m in result.messages)
