# WhatsApp export format

Parser spec for MVP ingest. Supports **Android** and **iOS** WhatsApp "Export chat" `.txt` files (without media).

**Implementation:** `backend/app/services/parser/preprocess.py` → `whatsapp.py`

---

## Preprocess (before parse)

Raw export text passes through `preprocess_whatsapp_export()` in the ingest graph **before** `parse_whatsapp_export()`.

| Step | What |
|------|------|
| Line endings | CRLF / CR → LF |
| Invisible chars | Strip BOM, zero-width, and bidi control marks (LRM/RLM/LRE/RLE…) |
| Spaces | U+202F (narrow NBSP) and U+00A0 → ASCII space |
| Edited tag | Remove trailing `<This message was edited>` from bodies |
| Deleted body | Replace standalone `This message was deleted` with `[message deleted]` |
| Multiline repair | Drop blank lines between continuation lines (inside one message) |

System lines (encryption notice, `X is a contact`) and `<Media omitted>` pass through unchanged; the parser marks system vs user messages.

---

## Export options (user instruction)

Tell operators to export from WhatsApp:

1. Open group chat → menu → **Export chat**
2. Choose **Without media**
3. Save `.txt` file
4. Upload in ChatMemory workspace create flow

---

## Line patterns

### Android (common)

```
M/D/YY, H:MM AM/PM - Sender Name: message text
DD/MM/YYYY, HH:MM - Sender Name: message text
```

US exports often use **2-digit years** and **12-hour AM/PM**. WhatsApp may insert a **narrow no-break space (U+202F)** before `AM`/`PM` — parser normalizes this.

**Group names with commas/apostrophes:** `Test Group, Sample Workspace: Hi` — split on first `: ` after the dash.

**System lines** (no `: ` after dash): encryption notice, `X is a contact`, etc.

**System messages** (no sender colon pattern):

```
DD/MM/YYYY, HH:MM - Messages and calls are end-to-end encrypted
```

### iOS (common)

```
[DD/MM/YYYY, HH:MM:SS] Sender Name: message text
```

Brackets around datetime. May include seconds.

### Variants to handle

| Variant | Notes |
|---------|-------|
| US date `MM/DD/YYYY` | Detect by heuristic or WhatsApp locale in first lines |
| 12-hour `AM/PM` | Parse with `%I:%M %p` |
| Sender with colon in name | Rare; use last `: ` before message body pattern |
| Multiline messages | Continuation lines lack date prefix — append to previous message |
| `<Media omitted>` | Store as text placeholder or skip embed (store for count) |
| Deleted message | `This message was deleted` — index or skip (recommend: skip embed, keep count optional |

---

## Parser algorithm

```
1. preprocess_whatsapp_export(text)  — normalize chars, strip edited tags
2. Read as UTF-8 (fallback latin-1 with warning at upload)
3. Detect format: android | ios from first substantive line regex
4. For each line:
   a. If matches message header regex → push new Message
   b. Else if current message → append line to current message body (trim)
5. Normalize sender name (strip whitespace)
6. Parse timestamp to UTC ISO (assume local TZ or configurable in settings later)
7. Assign messageId UUID per message
```

---

## Message model

```python
@dataclass
class Message:
    id: str              # UUID
    timestamp: datetime
    sender: str          # display name as in export
    text: str
    is_system: bool
```

---

## Speaker normalization

- `sender` string is display key
- Different spellings = different people in MVP (no fuzzy merge)
- Future global library may add alias merge UI

---

## Validation gates

| Check | Min | On fail |
|-------|-----|---------|
| Total messages (non-system) | 50 | Reject ingest |
| Unique speakers | 1 | Reject ingest |
| File encoding readable | — | Reject with encoding error |
| File size | — | Warn if &gt; 50MB; still try |

---

## Test fixtures

```
backend/tests/fixtures/whatsapp/
  android_group.txt
  group_patterns.txt
  ios_group.txt
  hinglish_sample.txt
  multiline_messages.txt
```

`hinglish_sample.txt` includes code-mixed lines for embed QA tests.

---

## Example lines

**Android:**

```
12/03/2024, 18:22 - Alice: yaar kal meeting hai kya?
12/03/2024, 18:23 - Bob: haan 5 baje
```

**iOS:**

```
[12/03/2024, 18:22:15] Alice: yaar kal meeting hai kya?
[12/03/2024, 18:23:01] Bob: haan 5 baje
```

**Multiline:**

```
12/03/2024, 18:24 - Alice: location yeh hai
maps.google.com/...
```

Second line attaches to Alice's message.

---

## Related

- [../langgraph/ingest.md](../langgraph/ingest.md)
- [../data-layout.md](../data-layout.md)
