from pathlib import Path

from app.services.parser.whatsapp import parse_whatsapp_export

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
