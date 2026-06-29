from app.core.config import get_settings
from app.core.paths import ensure_data_dirs, person_path, workspace_path
from app.core.schemas import (
    PersonaStatus,
    PersonDetail,
    PersonSummary,
    SampleMessage,
    StyleProfile,
    TopSpeaker,
    WorkspaceDetail,
    WorkspaceSummary,
)
from app.prompts.persona_build import (
    persona_extract_chat_analysis,
    persona_extract_chat_analysis_consolidate,
    persona_extract_listening_style,
    persona_extract_personality,
    persona_extract_relationship_emotional,
    persona_extract_response_patterns_topic_map,
    persona_extract_typing_fingerprint,
    persona_extract_writing_style,
    persona_label_voice_samples,
)
from app.services import gemini as gemini_service
from app.services import jobs as job_service
from app.services.parser.preprocess import preprocess_whatsapp_export
from app.services.parser.whatsapp import (
    Message,
    is_noise_message,
    non_system_messages,
    parse_whatsapp_export,
)
from app.services.rate_limit import GeminiRateLimiter, estimate_tokens
import asyncio
from datetime import datetime, timezone
import json
import logging
import os
import re
import shutil
import threading
from typing import Any
import uuid

logger = logging.getLogger("chatmemory.workspace")

# Shared rate limiter for all build-time Gemini calls in this module.
# Persona builds run behind the GPU mutex (one at a time), so this limiter
# mainly guards against multi-call bursts within a single build run.
_rate_limiter = GeminiRateLimiter(max_rpm=14, max_tpm=100_000)

# Module-level timeline cache keyed by workspace_id → (mtime, messages).
# Avoids re-parsing large export.txt files (100k+ messages) on every Q&A
# query or persona chat turn.  Cache is invalidated when the file's mtime changes.
_timeline_cache: dict[str, tuple[float, list[Message]]] = {}

# Per-file threading locks for person JSON records.
# Serialises concurrent read-merge-write cycles so training threads and
# concurrent API reads (reconcile, get_person) cannot interleave on the same file.
_person_file_locks: dict[str, threading.Lock] = {}
_person_file_locks_mutex = threading.Lock()


def _get_person_lock(path: str) -> threading.Lock:
    """Return (creating if needed) a per-path Lock for a person JSON file."""
    with _person_file_locks_mutex:
        if path not in _person_file_locks:
            _person_file_locks[path] = threading.Lock()
        return _person_file_locks[path]


def load_export_timeline(workspace_id: str) -> list[Message]:
    """All non-system messages in the workspace export, chronological.

    Results are cached in memory keyed by (workspace_id, export.txt mtime).
    The cache is invalidated automatically when the file is re-ingested.
    """
    export_path = workspace_path(workspace_id) / "export.txt"
    if not export_path.exists():
        raise FileNotFoundError("export.txt")

    mtime = export_path.stat().st_mtime
    cached = _timeline_cache.get(workspace_id)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    raw = export_path.read_text(encoding="utf-8")
    parsed = parse_whatsapp_export(preprocess_whatsapp_export(raw))
    msgs = non_system_messages(parsed.messages)
    msgs.sort(key=lambda m: m.timestamp)

    _timeline_cache[workspace_id] = (mtime, msgs)
    return msgs


def _persona_status_for_count(count: int) -> PersonaStatus:
    settings = get_settings()
    if count < settings.lora_thin_min_messages:
        return "not_enough"
    if count < settings.lora_min_messages:
        return "thin"
    return "ready"


def _hinglish_ratio(text: str) -> float:
    if not text:
        return 0.0
    # Devanagari or common Roman Hindi tokens
    devanagari = len(re.findall(r"[\u0900-\u097F]", text))
    roman_hints = len(
        re.findall(
            r"\b(yaar|bhai|kya|hai|nahi|haan|acha|theek|kal|aaj|kyu|kaise)\b",
            text,
            re.I,
        )
    )
    words = max(len(text.split()), 1)
    return min(1.0, (devanagari + roman_hints * 3) / words)


def _emoji_rate(text: str) -> float:
    if not text:
        return 0.0
    emojis = len(re.findall(r"[\U0001F300-\U0001FAFF]", text))
    return emojis / max(len(text), 1)


def _load_meta(workspace_id: str) -> dict[str, Any]:
    path = workspace_path(workspace_id) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(workspace_id)
    return json.loads(path.read_text(encoding="utf-8"))


def _save_meta(workspace_id: str, meta: dict[str, Any]) -> None:
    path = workspace_path(workspace_id) / "meta.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def list_workspaces() -> list[WorkspaceSummary]:
    ensure_data_dirs()
    root = get_settings().workspaces_dir
    if not root.exists():
        return []

    items: list[WorkspaceSummary] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        meta_file = child / "meta.json"
        if not meta_file.exists():
            continue
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        items.append(WorkspaceSummary.model_validate(_meta_to_summary(meta)))
    return items


def _meta_to_summary(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": meta["id"],
        "name": meta["name"],
        "created_at": meta["createdAt"],
        "message_count": meta.get("messageCount", 0),
        "speaker_count": meta.get("speakerCount", 0),
        "date_from": meta.get("dateFrom"),
        "date_to": meta.get("dateTo"),
        "ingest_status": meta.get("ingestStatus", "pending"),
    }


def get_workspace(workspace_id: str) -> WorkspaceDetail:
    meta = _load_meta(workspace_id)
    people_dir = workspace_path(workspace_id) / "people"
    top: list[TopSpeaker] = []
    if people_dir.exists():
        people: list[tuple[int, TopSpeaker]] = []
        for pf in people_dir.glob("*.json"):
            pdata = json.loads(pf.read_text(encoding="utf-8"))
            people.append(
                (
                    pdata.get("messageCount", 0),
                    TopSpeaker(
                        person_id=pdata["id"],
                        display_name=pdata["displayName"],
                        message_count=pdata.get("messageCount", 0),
                    ),
                )
            )
        people.sort(key=lambda x: x[0], reverse=True)
        top = [p for _, p in people[:5]]

    base = _meta_to_summary(meta)
    return WorkspaceDetail(**base, top_speakers=top)


def create_workspace_record(name: str, export_filename: str) -> dict[str, Any]:
    ensure_data_dirs()
    workspace_id = str(uuid.uuid4())
    ws_dir = workspace_path(workspace_id)
    (ws_dir / "people").mkdir(parents=True, exist_ok=True)
    (ws_dir / "chroma").mkdir(parents=True, exist_ok=True)
    (ws_dir / "bm25").mkdir(parents=True, exist_ok=True)

    meta = {
        "id": workspace_id,
        "name": name,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "ingestStatus": "pending",
        "ingestJobId": None,
        "messageCount": 0,
        "speakerCount": 0,
        "dateFrom": None,
        "dateTo": None,
        "exportFilename": export_filename,
    }
    _save_meta(workspace_id, meta)
    return meta


def get_speaker_count(workspace_id: str) -> int:
    """Return speaker count from workspace metadata (fast, no timeline load)."""
    meta = _load_meta(workspace_id)
    return meta.get("speakerCount", 0)


def delete_workspace(workspace_id: str) -> None:
    import shutil
    import time

    from app.services import bm25 as bm25_service
    from app.services import chroma as chroma_service

    ws_dir = workspace_path(workspace_id)
    if not ws_dir.exists():
        raise FileNotFoundError(workspace_id)

    # Release all in-memory handles before touching the filesystem.
    # On Windows, chromadb holds open file locks on SQLite + HNSW binaries;
    # BM25 may hold a reference to the loaded index.  Both must be released
    # before rmtree or we get WinError 32.
    chroma_service.clear_store_cache(workspace_id)
    bm25_service.clear_index_cache(workspace_id)
    _timeline_cache.pop(workspace_id, None)

    # Force GC so Python drops Chroma/chromadb objects and file handles
    # before rmtree. On Windows GC is not deterministic — this is required.
    import gc
    gc.collect()
    for attempt in range(5):
        try:
            shutil.rmtree(ws_dir)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.5 * (attempt + 1))


def build_people_from_messages(
    workspace_id: str, messages: list[Message]
) -> dict[str, dict[str, Any]]:
    """Group messages by sender; write people/*.json; return person map."""
    by_sender: dict[str, list[Message]] = {}
    for msg in non_system_messages(messages):
        by_sender.setdefault(msg.sender, []).append(msg)

    people: dict[str, dict[str, Any]] = {}
    people_dir = workspace_path(workspace_id) / "people"
    people_dir.mkdir(parents=True, exist_ok=True)

    for sender, msgs in by_sender.items():
        person_id = str(uuid.uuid4())
        msgs.sort(key=lambda m: m.timestamp)
        texts = [m.text for m in msgs]
        profile = StyleProfile(
            avg_message_length=sum(len(t) for t in texts) / max(len(texts), 1),
            emoji_rate=sum(_emoji_rate(t) for t in texts) / max(len(texts), 1),
            hinglish_ratio=sum(_hinglish_ratio(t) for t in texts) / max(len(texts), 1),
        )
        samples = _pick_samples(msgs)
        record = {
            "id": person_id,
            "workspaceId": workspace_id,
            "displayName": sender,
            "aliases": [],
            "messageCount": len(msgs),
            "firstSeen": msgs[0].timestamp.isoformat(),
            "lastSeen": msgs[-1].timestamp.isoformat(),
            "personaStatus": _persona_status_for_count(len(msgs)),
            "ollamaModelName": None,
            "lastTrainJobId": None,
            "lastTrainAt": None,
            "styleProfile": profile.model_dump(by_alias=True),
            "sampleMessages": samples,
        }
        (people_dir / f"{person_id}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        people[person_id] = record

    return people


def _spread_evenly(msgs: list[Message], n: int) -> list[Message]:
    """Pick up to n messages evenly spread across msgs (chronological order preserved)."""
    if n <= 0:
        return []
    if len(msgs) <= n:
        return msgs
    if n == 1:
        # Single pick: take the middle message as a representative sample.
        return [msgs[len(msgs) // 2]]
    step = max((len(msgs) - 1) // (n - 1), 1)
    indices = list(range(0, len(msgs), step))
    if indices[-1] != len(msgs) - 1:
        indices.append(len(msgs) - 1)
    return [msgs[i] for i in sorted(set(indices))][:n]


def _recency_weighted_sample(msgs: list[Message], limit: int) -> list[Message]:
    """Sample up to `limit` messages, giving 60% of slots to the most recent third.

    Assumes msgs is sorted chronologically (oldest first).
    The returned list preserves chronological order.
    Recent messages reflect *current* communication habits better than old ones,
    so we intentionally over-represent them.
    """
    if len(msgs) <= limit:
        return msgs

    # Divide into older 2/3 and newer 1/3.
    split = max(1, len(msgs) * 2 // 3)
    older = msgs[:split]
    recent = msgs[split:]

    # 60 % of slots to recent third, 40 % to older portion; clamp to actual bucket sizes.
    n_recent = min(round(limit * 0.6), len(recent))
    n_older = min(limit - n_recent, len(older))

    picked = _spread_evenly(older, n_older) + _spread_evenly(recent, n_recent)
    picked.sort(key=lambda m: m.timestamp)
    return picked[:limit]


def _build_paired_exchanges(
    timeline: list[Message],
    person_name: str,
    target_msgs: list[Message],
) -> list[str]:
    """Build paired conversational exchanges for relational-context training.

    For each target message, looks backward in the chronological *timeline* (up to
    8 positions) to find the most-recent message from a *different* sender.  When
    found, the pair is rendered as:

        [Other]: <their message>
        [Name]: <target response>

    Standalone messages (no eligible preceding message within 8 positions) are
    rendered as:

        [Name]: <message>

    Falls back gracefully to standalone format for single-speaker workspaces.
    The order of ``target_msgs`` does not matter — lookups always use the
    chronological ``timeline`` index.
    """
    id_to_pos: dict[str, int] = {m.id: i for i, m in enumerate(timeline)}

    exchanges: list[str] = []
    for msg in target_msgs:
        pos = id_to_pos.get(msg.id)
        preceding: Message | None = None

        if pos is not None and pos > 0:
            # Scan back up to 8 positions for the nearest other-sender message.
            for prev_idx in range(pos - 1, max(pos - 9, -1), -1):
                prev = timeline[prev_idx]
                if (
                    prev.sender != person_name
                    and prev.text.strip()
                    and not is_noise_message(prev)
                ):
                    preceding = prev
                    break

        if preceding:
            exchanges.append(
                f"[{preceding.sender}]: {preceding.text}\n"
                f"[{person_name}]: {msg.text}"
            )
        else:
            exchanges.append(f"[{person_name}]: {msg.text}")

    return exchanges


def _pick_samples(msgs: list[Message], limit: int = 20) -> list[dict[str, str]]:
    """Pick sample messages, preferring recency.

    Strategy: take one message per calendar month scanning newest-first, then
    fill remaining slots by spreading evenly over older messages.  This ensures
    recent activity is always represented while still showing some history.
    """
    if not msgs:
        return []
    if len(msgs) <= limit:
        picked = list(msgs)
    else:
        # Sort newest-first for the recency pass.
        by_recency = sorted(msgs, key=lambda m: m.timestamp, reverse=True)

        # Phase 1: one representative message per calendar month.
        seen_months: set[str] = set()
        monthly: list[Message] = []
        for m in by_recency:
            month_key = m.timestamp.strftime("%Y-%m")
            if month_key not in seen_months:
                seen_months.add(month_key)
                monthly.append(m)
            if len(monthly) >= limit:
                break

        if len(monthly) >= limit:
            picked = sorted(monthly[:limit], key=lambda m: m.timestamp)
        else:
            # Phase 2: fill remaining slots evenly from messages not yet picked.
            picked_ids = {m.id for m in monthly}
            older = [m for m in msgs if m.id not in picked_ids]
            remaining = limit - len(monthly)
            extras = _spread_evenly(older, remaining)
            picked = sorted(monthly + extras, key=lambda m: m.timestamp)[:limit]

    return [
        {
            "messageId": m.id,
            "timestamp": m.timestamp.isoformat(),
            "text": m.text[:280],
        }
        for m in picked
    ]


def finalize_workspace_stats(workspace_id: str, messages: list[Message], people_count: int) -> None:
    meta = _load_meta(workspace_id)
    usable = non_system_messages(messages)
    meta["messageCount"] = len(usable)
    meta["speakerCount"] = people_count
    if usable:
        meta["dateFrom"] = min(m.timestamp for m in usable).isoformat()
        meta["dateTo"] = max(m.timestamp for m in usable).isoformat()
    meta["ingestStatus"] = "done"
    _save_meta(workspace_id, meta)


def set_vector_store(workspace_id: str, store: str) -> None:
    meta = _load_meta(workspace_id)
    meta["vectorStore"] = store
    _save_meta(workspace_id, meta)


def set_ingest_status(workspace_id: str, status: str, job_id: str | None = None) -> None:
    meta = _load_meta(workspace_id)
    meta["ingestStatus"] = status
    if job_id:
        meta["ingestJobId"] = job_id
    _save_meta(workspace_id, meta)


def list_people(workspace_id: str) -> list[PersonSummary]:
    people_dir = workspace_path(workspace_id) / "people"
    if not people_dir.exists():
        return []
    items: list[PersonSummary] = []
    for pf in sorted(people_dir.glob("*.json")):
        pdata = json.loads(pf.read_text(encoding="utf-8"))
        items.append(
            PersonSummary(
                id=pdata["id"],
                display_name=pdata["displayName"],
                message_count=pdata.get("messageCount", 0),
                first_seen=pdata.get("firstSeen"),
                last_seen=pdata.get("lastSeen"),
                persona_status=pdata.get("personaStatus", "not_enough"),
            )
        )
    items.sort(key=lambda p: p.message_count, reverse=True)
    return items


def _restore_persona_status(pdata: dict[str, Any]) -> PersonaStatus:
    if pdata.get("ollamaModelName"):
        return "ready_model"
    return _persona_status_for_count(pdata.get("messageCount", 0))


def reconcile_person_training(workspace_id: str, person_id: str) -> dict[str, Any]:
    """Fix stuck personaStatus=training and recover lastTrainJobId for UI polling."""
    path = person_path(workspace_id, person_id)
    pdata = json.loads(path.read_text(encoding="utf-8"))
    if pdata.get("personaStatus") != "training":
        return pdata

    job_id = pdata.get("lastTrainJobId") or job_service.find_latest_train_job(
        workspace_id, person_id
    )
    if not job_id:
        pdata["personaStatus"] = _restore_persona_status(pdata)
        path.write_text(json.dumps(pdata, indent=2), encoding="utf-8")
        return pdata

    if job_id != pdata.get("lastTrainJobId"):
        pdata["lastTrainJobId"] = job_id

    job = job_service.get_job(job_id)
    if job is None or job.status in ("done", "error"):
        pdata["personaStatus"] = (
            "error" if job and job.status == "error" else _restore_persona_status(pdata)
        )
        if job and job.status == "done" and job.result and job.result.get("ollamaModelName"):
            pdata["personaStatus"] = "ready_model"
            pdata["ollamaModelName"] = job.result["ollamaModelName"]

    path.write_text(json.dumps(pdata, indent=2), encoding="utf-8")
    return pdata


def cancel_person_training(workspace_id: str, person_id: str) -> None:
    path = person_path(workspace_id, person_id)
    pdata = json.loads(path.read_text(encoding="utf-8"))
    job_id = pdata.get("lastTrainJobId") or job_service.find_latest_train_job(
        workspace_id, person_id
    )
    if job_id:
        job_service.cancel_job(job_id, "Training cancelled")
    pdata["personaStatus"] = _restore_persona_status(pdata)
    path.write_text(json.dumps(pdata, indent=2), encoding="utf-8")


def training_is_active(workspace_id: str, person_id: str) -> bool:
    path = person_path(workspace_id, person_id)
    if not path.exists():
        return False
    pdata = json.loads(path.read_text(encoding="utf-8"))
    if pdata.get("personaStatus") != "training":
        return False
    job_id = pdata.get("lastTrainJobId") or job_service.find_latest_train_job(
        workspace_id, person_id
    )
    if not job_id:
        return False
    job = job_service.get_job(job_id)
    return job is not None and job.status in ("queued", "running")


def get_person(workspace_id: str, person_id: str) -> PersonDetail:
    path = person_path(workspace_id, person_id)
    if not path.exists():
        raise FileNotFoundError(person_id)
    pdata = reconcile_person_training(workspace_id, person_id)
    settings = get_settings()
    count = pdata.get("messageCount", 0)
    train_eligible = count >= settings.lora_thin_min_messages
    gemini_ok, gemini_err = gemini_service.config_status()
    if not gemini_ok:
        train_eligible = False
    warning = None
    if not gemini_ok and gemini_err:
        warning = gemini_err
    elif settings.lora_thin_min_messages <= count < settings.lora_min_messages:
        warning = f"Only {count} messages; recommend {settings.lora_min_messages}+ for a stronger persona."

    return PersonDetail(
        id=pdata["id"],
        display_name=pdata["displayName"],
        message_count=count,
        first_seen=pdata.get("firstSeen"),
        last_seen=pdata.get("lastSeen"),
        persona_status=pdata.get("personaStatus", "not_enough"),
        ollama_model_name=pdata.get("ollamaModelName"),
        style_profile=StyleProfile.model_validate(pdata.get("styleProfile", {})),
        sample_messages=[SampleMessage.model_validate(s) for s in pdata.get("sampleMessages", [])],
        train_eligible=train_eligible,
        train_warning=warning,
        last_train_job_id=pdata.get("lastTrainJobId"),
        # Legacy fields (backward compat — kept but no longer generated by new training runs)
        personality_notes=pdata.get("personalityNotes"),
        writing_style_notes=pdata.get("writingStyleNotes"),
        chat_analysis=pdata.get("chatAnalysis"),
        active_listening_style=pdata.get("activeListeningStyle"),
        # New v2 persona fields
        relationship_dynamic=pdata.get("relationshipDynamic"),
        typing_fingerprint=pdata.get("typingFingerprint"),
        topic_map=pdata.get("topicMap"),
        response_patterns=pdata.get("responsePatterns"),
        voice_samples=pdata.get("voiceSamples"),
        emotional_profile=pdata.get("emotionalProfile"),
    )


def refresh_person_samples(workspace_id: str, person_id: str) -> None:
    """Re-pick sample messages from the full export timeline."""
    person = get_person(workspace_id, person_id)
    timeline = load_export_timeline(workspace_id)
    person_msgs = [m for m in timeline if m.sender == person.display_name and m.text.strip()]
    samples = _pick_samples(person_msgs, limit=20)
    update_person_record(workspace_id, person_id, {"sampleMessages": samples})


def refresh_person_style_profile(workspace_id: str, person_id: str) -> None:
    """Recompute style metrics from the full export timeline."""
    person = get_person(workspace_id, person_id)
    timeline = load_export_timeline(workspace_id)
    person_msgs = [m for m in timeline if m.sender == person.display_name and m.text.strip()]
    texts = [m.text for m in person_msgs]
    profile = StyleProfile(
        avg_message_length=sum(len(t) for t in texts) / max(len(texts), 1),
        emoji_rate=sum(_emoji_rate(t) for t in texts) / max(len(texts), 1),
        hinglish_ratio=sum(_hinglish_ratio(t) for t in texts) / max(len(texts), 1),
    )
    update_person_record(
        workspace_id,
        person_id,
        {"styleProfile": profile.model_dump(by_alias=True)},
    )


def update_person_record(workspace_id: str, person_id: str, updates: dict[str, Any]) -> None:
    """Read-merge-write a person JSON record with full race-condition protection.

    Three-layer defence:
    1. Per-file threading.Lock — serialises all concurrent callers on the same
       person file so no two threads can interleave their read-merge-write cycles.
    2. Empty-file fallback — if the file is empty (truncated mid-write by a prior
       crash) we attempt to recover from the adjacent .bak file before failing.
    3. Atomic write — new content is written to a .tmp file first, then the
       current .json is snapshotted to .bak, and finally os.replace() swaps .tmp
       into place.  os.replace() is atomic on Windows (NTFS) and POSIX, so
       concurrent readers always see a complete file, never a partial write.
    """
    path = person_path(workspace_id, person_id)
    lock = _get_person_lock(str(path))
    with lock:
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        if not raw.strip():
            # File is empty — possible mid-write truncation by a prior crash.
            bak = path.with_suffix(".bak")
            if bak.exists():
                logger.warning(
                    "Person record %s is empty; recovering from .bak backup", path.name
                )
                raw = bak.read_text(encoding="utf-8")
            else:
                raise ValueError(
                    f"Person record {path.name} is empty and no .bak backup exists. "
                    "The file may be corrupted — please re-ingest the workspace."
                )

        pdata = json.loads(raw)
        pdata.update(updates)

        # Write to .tmp first so the live .json is never partially overwritten.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(pdata, indent=2), encoding="utf-8")

        # Snapshot current good .json to .bak before replacing it.
        bak = path.with_suffix(".bak")
        if path.exists():
            shutil.copy2(path, bak)

        # Atomic swap: .tmp → .json (replaces the old file in one OS call).
        os.replace(tmp, path)


# Noise filtering for personality/style/analysis passes is handled by
# is_noise_message() from the parser, which covers the full set of noise patterns.
# load_export_timeline already returns a noise-free timeline via non_system_messages,
# so these per-function filters are a belt-and-suspenders guard only.

# ---------------------------------------------------------------------------
# v2 persona training helpers
# ---------------------------------------------------------------------------

def _parse_gemini_json(text: str) -> dict:
    """Strip markdown code fences (if any) and parse JSON from a Gemini response.

    Gemini sometimes wraps JSON in triple-backtick code blocks despite prompt
    instructions. This helper strips the fences before parsing so callers get
    a clean dict regardless of response format.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return json.loads(text.strip())


# Keyword sets for topic-bucket classification used by _select_voice_samples.
_VOICE_SAMPLE_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "project_work": [
        "website", "deploy", "code", "github", "server", "hosting",
        "backend", "frontend", "api", "bug", "build", "merge", "push",
        "database", "feature", "pr", "commit",
    ],
    "scheduling": [
        "kal", "timing", "meet", "time", "free", "kab", "schedule",
        "bje", "when", "date", "aaj", "abhi", "baad", "slot",
    ],
    "recruitment": [
        "interview", "candidate", "select", "panel", "hire", "liyo",
        "selected", "reject", "shortlist", "result",
    ],
    "creative_design": [
        "design", "figma", "ui", "ux", "banner", "poster", "logo",
        "creative", "graphic", "artwork",
    ],
    "social_casual": [
        "noice", "lol", "haha", "cool", "nice", "maza", "chill",
        "fun", "yaar", "bhai", "bruh", "bro", "yaar",
    ],
    "emotional_support": [
        "bura", "theek", "health", "rest", "tired", "soja", "sad",
        "sorry", "tension", "stressed", "worried", "pareshaan",
    ],
    "academic": [
        "paper", "exam", "result", "marks", "assignment", "college",
        "class", "lecture", "notes", "subject",
    ],
}


def _classify_topic(text: str) -> str:
    """Return the best-matching topic bucket key for the given lowercased text."""
    text_lower = text.lower()
    best_topic = "general"
    best_count = 0
    for topic, keywords in _VOICE_SAMPLE_TOPIC_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > best_count:
            best_count = count
            best_topic = topic
    return best_topic


def _is_pure_ack(text: str) -> bool:
    """True if the message is a low-effort acknowledgement with no substantive content.

    Uses message STRUCTURE rather than a hardcoded word list — language-agnostic.
    A message is treated as an ack when:
    - It is ≤4 characters (covers "ok", "hn", "ya", "k", "hm" in any language).
    - It is ≤8 characters AND the portion after the 4th character contains no spaces,
      meaning it is a single short token with no multi-word action content.
    """
    t = text.strip()
    if len(t) <= 4:
        return True
    if len(t) <= 8 and " " not in t[4:]:
        return True
    return False


def _score_target_exchange(person_name: str, exchange: list[dict]) -> int:
    """Score an exchange based ONLY on the TARGET PERSON's message quality.

    Scoring rubric:
      +2  target has ≥ 2 messages in the exchange
      +1  per target message with character length > 15
      +1  back-and-forth present (both target and another sender)
      +2  target's messages are NOT all low-effort acknowledgements
    """
    target_msgs = [m["text"] for m in exchange if m["sender"] == person_name]
    other_present = any(m["sender"] != person_name for m in exchange)

    score = 0
    if len(target_msgs) >= 2:
        score += 2
    for text in target_msgs:
        if len(text.strip()) > 15:
            score += 1
    if other_present:
        score += 1
    if target_msgs and not all(_is_pure_ack(t) for t in target_msgs):
        score += 2
    return score


def _label_voice_samples(
    exchanges: list[list[dict]],
    person_name: str,
) -> list[str]:
    """Call Gemini once to assign a 2–4 word context label to each selected exchange.

    One round-trip labels all exchanges together, producing natural descriptions of
    what the target person is doing (e.g. 'driving task urgency', 'offering emotional
    support') in any language or style — no hardcoded keyword lists required.

    Returns a list of label strings in the same order as ``exchanges``.
    Falls back to a generic label on any Gemini or parse error so selection is
    never blocked by a failed labeling call.
    """
    short_name = person_name.split(",")[0].strip()
    fallback = [f"{short_name} speaking"] * len(exchanges)

    if not exchanges:
        return fallback

    # Format each exchange as a numbered block for the prompt.
    blocks: list[str] = []
    for i, exc in enumerate(exchanges, 1):
        lines = "\n".join(f"[{m['sender']}]: {m['text']}" for m in exc)
        blocks.append(f"Exchange {i}:\n{lines}")
    exchanges_text = "\n\n".join(blocks)

    prompt = persona_label_voice_samples(person_name, exchanges_text)
    _rate_limiter.acquire(estimate_tokens(prompt))
    try:
        raw = gemini_service.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
        ).strip()
        _rate_limiter.record(estimate_tokens(prompt))
    except Exception as exc:
        _rate_limiter.record(estimate_tokens(prompt))
        logger.warning("Voice sample labeling failed: %s", exc)
        return fallback

    if not raw:
        return fallback

    # Strip markdown fences that Gemini sometimes adds despite prompt instructions.
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        labels = json.loads(raw.strip())
        if not isinstance(labels, list) or len(labels) != len(exchanges):
            raise ValueError(f"Expected {len(exchanges)} labels, got {len(labels) if isinstance(labels, list) else type(labels)}")
        return [str(label) for label in labels]
    except Exception as exc:
        logger.warning("Failed to parse voice sample labels: %s — raw: %.200s", exc, raw)
        return fallback


def _exchange_text_similarity(exc1: list[dict], exc2: list[dict]) -> float:
    """Jaccard similarity between two exchanges measured by message text.

    Returns a value in [0.0, 1.0]. Two exchanges are considered near-duplicates
    when the returned value exceeds 0.6 (>60% overlap by Jaccard).
    """
    texts1 = {m["text"] for m in exc1}
    texts2 = {m["text"] for m in exc2}
    if not texts1 and not texts2:
        return 1.0
    intersection = texts1 & texts2
    union = texts1 | texts2
    return len(intersection) / len(union)


async def _select_voice_samples(
    timeline: list[Message],
    person_name: str,
    min_samples: int = 5,
    max_samples: int = 8,
) -> list[dict]:
    """Algorithmically select voice sample exchanges, then label them via Gemini.

    Algorithm:
    1. Slide a window over the timeline collecting 4-turn exchange candidates
       where both the target person and at least one other person send messages.
    2. Pre-filter: drop any candidate containing a 1-word message (too sparse).
    3. Drop candidates where the target has no substantive message
       (needs ≥ 1 message with >10 chars that is not a pure acknowledgement).
    4. Score each candidate exclusively by the TARGET PERSON's message quality
       using _score_target_exchange (ignores the other person's length).
    5. Cluster by topic keyword bucket; pick the highest-scored candidate per bucket.
    6. Cross-bucket deduplication — discard any exchange that shares >60% of its
       message texts with an already-selected higher-scored exchange.
    7. Pad to min_samples from unused candidates (also deduplicated) if needed.
    8. One Gemini call labels ALL selected exchanges at once via _label_voice_samples,
       producing natural context descriptions (no keyword lists).
    9. Return up to max_samples ordered by target-person score descending.
    """
    from collections import defaultdict

    candidates: list[tuple[str, list[dict], int]] = []  # (topic, exchange, target_score)
    n = len(timeline)

    for i in range(n):
        anchor = timeline[i]
        if anchor.sender != person_name or not anchor.text.strip() or is_noise_message(anchor):
            continue

        # Build a 4-turn window starting 1 message before the anchor (captures the
        # stimulus that prompted this message), scanning forward.
        start = max(0, i - 1)
        window: list[Message] = []
        senders_in_window: set[str] = set()

        for j in range(start, min(n, start + 10)):
            m = timeline[j]
            if not m.text.strip() or is_noise_message(m):
                continue
            # Restrict to exactly 2 distinct senders once the second appears.
            if len(senders_in_window) >= 2 and m.sender not in senders_in_window:
                break
            senders_in_window.add(m.sender)
            window.append(m)
            if len(window) >= 4:
                break

        # Must involve both the target person and at least one other sender.
        if person_name not in {m.sender for m in window}:
            continue
        if not any(m.sender != person_name for m in window):
            continue

        # Pre-filter: 1-word messages are too sparse to show authentic voice.
        if any(len(m.text.strip().split()) <= 1 for m in window):
            continue

        # Require the target to have at least one substantive message
        # (longer than 10 chars AND not a pure low-effort acknowledgement).
        target_window_msgs = [m for m in window if m.sender == person_name]
        if not any(
            len(m.text.strip()) > 10 and not _is_pure_ack(m.text.strip())
            for m in target_window_msgs
        ):
            continue

        exchange = [{"sender": m.sender, "text": m.text.strip()} for m in window]
        all_text = " ".join(m["text"].lower() for m in exchange)
        topic = _classify_topic(all_text)
        target_score = _score_target_exchange(person_name, exchange)
        candidates.append((topic, exchange, target_score))

    if not candidates:
        return []

    # Group by topic; pick the candidate with the highest target-person score.
    topic_groups: dict[str, list[tuple[list[dict], int]]] = defaultdict(list)
    for topic, exc, target_score in candidates:
        topic_groups[topic].append((exc, target_score))

    # Build initial result — one exchange per topic bucket.
    pre_dedup: list[dict] = []
    used_fingerprints: set[str] = set()

    for topic, group in topic_groups.items():
        best_exc, best_score = max(group, key=lambda x: x[1])
        fp = json.dumps(best_exc, ensure_ascii=False)
        if fp in used_fingerprints:
            continue
        used_fingerprints.add(fp)
        pre_dedup.append({"exchange": best_exc, "_score": best_score})

    # Cross-bucket deduplication — keep only the higher-scored exchange when two
    # selected exchanges share >60% of their message texts (Jaccard).
    # Process in descending score order so higher-scored exchanges are kept first.
    pre_dedup.sort(key=lambda x: x["_score"], reverse=True)
    result: list[dict] = []
    for item in pre_dedup:
        is_dupe = any(
            _exchange_text_similarity(item["exchange"], kept["exchange"]) > 0.6
            for kept in result
        )
        if not is_dupe:
            result.append(item)

    # Pad to min_samples from unused candidates (also cross-deduplicated).
    if len(result) < min_samples:
        extras_sorted = sorted(
            [(exc, score) for _, exc, score in candidates],
            key=lambda x: x[1],
            reverse=True,
        )
        existing_exchanges = [item["exchange"] for item in result]
        for exc, _ in extras_sorted:
            if len(result) >= min_samples:
                break
            fp = json.dumps(exc, ensure_ascii=False)
            if fp in used_fingerprints:
                continue
            is_dupe = any(
                _exchange_text_similarity(exc, kept) > 0.6
                for kept in existing_exchanges
            )
            if is_dupe:
                continue
            used_fingerprints.add(fp)
            result.append({"exchange": exc, "_score": 0})
            existing_exchanges.append(exc)

    result = result[:max_samples]

    # Label all selected exchanges with one Gemini call — no keyword lists needed.
    if result:
        all_exchanges = [item["exchange"] for item in result]
        labels = await asyncio.to_thread(_label_voice_samples, all_exchanges, person_name)
        for item, label in zip(result, labels):
            item["context"] = label

    return [{"context": item.get("context", f"{person_name} speaking"), "exchange": item["exchange"]} for item in result]


# How many of the person's messages to feed into v2 relationship/typing/pattern extraction.
_V2_SAMPLE_LIMIT = 60


def refresh_person_relationship_emotional(workspace_id: str, person_id: str) -> None:
    """Gemini Call 1 — extract relationshipDynamic + emotionalProfile together.

    Uses recency-weighted paired exchanges so the LLM sees the relational context.
    Stores results as ``relationshipDynamic`` and ``emotionalProfile`` in the person
    JSON record. Skips gracefully if Gemini is not configured or no usable messages.
    """
    person = get_person(workspace_id, person_id)
    timeline = load_export_timeline(workspace_id)

    candidate_msgs = [
        m
        for m in timeline
        if m.sender == person.display_name and m.text.strip() and not is_noise_message(m)
    ]
    if not candidate_msgs:
        return

    sampled = _recency_weighted_sample(candidate_msgs, _V2_SAMPLE_LIMIT)
    exchanges = _build_paired_exchanges(timeline, person.display_name, sampled)
    messages_text = "\n\n".join(exchanges)

    prompt = persona_extract_relationship_emotional(person.display_name, messages_text)
    _rate_limiter.acquire(estimate_tokens(prompt))
    try:
        raw = gemini_service.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
        ).strip()
        _rate_limiter.record(estimate_tokens(prompt))
    except Exception as exc:
        _rate_limiter.record(estimate_tokens(prompt))
        logger.warning("Relationship/emotional extraction failed: %s", exc)
        return

    if not raw:
        return
    try:
        data = _parse_gemini_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse relationship/emotional JSON: %s — raw: %.200s", exc, raw)
        return

    updates: dict = {}
    if data.get("relationshipDynamic"):
        updates["relationshipDynamic"] = data["relationshipDynamic"]
    if data.get("emotionalProfile"):
        updates["emotionalProfile"] = data["emotionalProfile"]
    if updates:
        update_person_record(workspace_id, person_id, updates)


def refresh_person_typing_fingerprint(workspace_id: str, person_id: str) -> None:
    """Gemini Call 2 — extract typingFingerprint as a structured JSON object.

    Uses only the target person's own messages (no paired context needed for
    surface-pattern extraction). Stores result as ``typingFingerprint`` in the
    person JSON record. Skips gracefully if Gemini is not configured.
    """
    person = get_person(workspace_id, person_id)
    timeline = load_export_timeline(workspace_id)

    candidate_msgs = [
        m
        for m in timeline
        if m.sender == person.display_name and m.text.strip() and not is_noise_message(m)
    ]
    if not candidate_msgs:
        return

    # Use the same recency-weighted sample, but provide solo messages (no pairs).
    sampled = _recency_weighted_sample(candidate_msgs, _V2_SAMPLE_LIMIT)
    solo_text = "\n".join(m.text.strip() for m in sampled)

    prompt = persona_extract_typing_fingerprint(person.display_name, solo_text)
    _rate_limiter.acquire(estimate_tokens(prompt))
    try:
        raw = gemini_service.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
        ).strip()
        _rate_limiter.record(estimate_tokens(prompt))
    except Exception as exc:
        _rate_limiter.record(estimate_tokens(prompt))
        logger.warning("Typing fingerprint extraction failed: %s", exc)
        return

    if not raw:
        return
    try:
        data = _parse_gemini_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse typing fingerprint JSON: %s — raw: %.200s", exc, raw)
        return

    if data:
        update_person_record(workspace_id, person_id, {"typingFingerprint": data})


def refresh_person_response_patterns(workspace_id: str, person_id: str) -> None:
    """Gemini Call 3 — extract responsePatterns + topicMap together.

    Uses recency-weighted paired exchanges. Stores results as ``responsePatterns``
    and ``topicMap`` in the person JSON record. topicMap is stored for human
    analysis only and is NOT injected into the persona prompt.
    Skips gracefully if Gemini is not configured or no usable messages.
    """
    person = get_person(workspace_id, person_id)
    timeline = load_export_timeline(workspace_id)

    candidate_msgs = [
        m
        for m in timeline
        if m.sender == person.display_name and m.text.strip() and not is_noise_message(m)
    ]
    if not candidate_msgs:
        return

    sampled = _recency_weighted_sample(candidate_msgs, _V2_SAMPLE_LIMIT)
    exchanges = _build_paired_exchanges(timeline, person.display_name, sampled)
    messages_text = "\n\n".join(exchanges)

    prompt = persona_extract_response_patterns_topic_map(person.display_name, messages_text)
    _rate_limiter.acquire(estimate_tokens(prompt))
    try:
        raw = gemini_service.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
        ).strip()
        _rate_limiter.record(estimate_tokens(prompt))
    except Exception as exc:
        _rate_limiter.record(estimate_tokens(prompt))
        logger.warning("Response patterns extraction failed: %s", exc)
        return

    if not raw:
        return
    try:
        data = _parse_gemini_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse response patterns JSON: %s — raw: %.200s", exc, raw)
        return

    updates: dict = {}
    if data.get("responsePatterns"):
        updates["responsePatterns"] = data["responsePatterns"]
    if data.get("topicMap"):
        updates["topicMap"] = data["topicMap"]
    if updates:
        update_person_record(workspace_id, person_id, updates)


async def refresh_person_voice_samples(workspace_id: str, person_id: str) -> None:
    """Call 4 — algorithmically select voice samples, then label them via Gemini.

    Uses _select_voice_samples to cluster paired exchanges by topic bucket and
    pick the richest example from each, then makes one Gemini call to assign
    natural context labels (e.g. 'driving task urgency') to each selected exchange.
    Stores result as ``voiceSamples`` in the person JSON record.
    """
    person = get_person(workspace_id, person_id)
    try:
        timeline = load_export_timeline(workspace_id)
    except FileNotFoundError:
        logger.warning("export.txt not found — skipping voice sample selection")
        return

    samples = await _select_voice_samples(timeline, person.display_name)
    if samples:
        update_person_record(workspace_id, person_id, {"voiceSamples": samples})


# How many of the person's messages to feed into personality extraction.
_PERSONALITY_SAMPLE_LIMIT = 60


def refresh_person_personality(workspace_id: str, person_id: str) -> None:
    """Use Gemini to extract a concise personality keynote from the person's real messages.

    The result is stored in the person's JSON record as ``personalityNotes`` (string).
    Skips gracefully if Gemini is not configured or if there are no usable messages.
    """
    person = get_person(workspace_id, person_id)
    timeline = load_export_timeline(workspace_id)

    # Filter to this person's non-empty, non-media, non-deleted messages.
    candidate_msgs = [
        m
        for m in timeline
        if m.sender == person.display_name and m.text.strip() and not is_noise_message(m)
    ]

    if not candidate_msgs:
        return

    # Recency-weighted sample: 60 % from the most recent third of messages, 40 % from older.
    # This gives the model a stronger signal about how the person communicates *now*.
    candidate_msgs = _recency_weighted_sample(candidate_msgs, _PERSONALITY_SAMPLE_LIMIT)

    # Build paired exchanges so the LLM sees the relational context — how the person
    # responds TO the other person's messages, not just isolated outgoing messages.
    exchanges = _build_paired_exchanges(timeline, person.display_name, candidate_msgs)
    messages_text = "\n\n".join(exchanges)

    prompt = persona_extract_personality(person.display_name, messages_text)

    # Low temperature for factual, grounded extraction.
    # Rate-limit before the call; record after to stay under 14 RPM / 100k TPM.
    _rate_limiter.acquire(estimate_tokens(prompt))
    notes = gemini_service.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.4,
    ).strip()
    _rate_limiter.record(estimate_tokens(prompt))

    if notes:
        update_person_record(workspace_id, person_id, {"personalityNotes": notes})


# How many messages to feed into writing-style extraction.
_WRITING_STYLE_SAMPLE_LIMIT = 60


def refresh_person_writing_style(workspace_id: str, person_id: str) -> None:
    """Use Gemini to extract a concise description of HOW the person types.

    Captures surface-level typing patterns (casing, punctuation, abbreviations,
    emoji usage, sentence structure) rather than personality.  The result is
    stored as ``writingStyleNotes`` in the person's JSON record and later injected
    verbatim into the system prompt so the model can mirror the exact style.

    Uses recency-weighted sampling so recent habits dominate.
    Skips gracefully if Gemini is not configured or there are no usable messages.
    """
    person = get_person(workspace_id, person_id)
    timeline = load_export_timeline(workspace_id)

    # Filter to this person's non-empty, non-media, non-deleted messages.
    candidate_msgs = [
        m
        for m in timeline
        if m.sender == person.display_name and m.text.strip() and not is_noise_message(m)
    ]

    if not candidate_msgs:
        return

    # Prefer recent messages — writing habits evolve over time.
    sampled = _recency_weighted_sample(candidate_msgs, _WRITING_STYLE_SAMPLE_LIMIT)

    # Build paired exchanges for context; the prompt directs the LLM to focus
    # only on the target's lines for surface-pattern extraction.
    exchanges = _build_paired_exchanges(timeline, person.display_name, sampled)
    messages_text = "\n\n".join(exchanges)

    prompt = persona_extract_writing_style(person.display_name, messages_text)

    # Very low temperature — we want factual, grounded style extraction.
    # Rate-limit before the call; record after to stay under 14 RPM / 100k TPM.
    _rate_limiter.acquire(estimate_tokens(prompt))
    notes = gemini_service.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.3,
    ).strip()
    _rate_limiter.record(estimate_tokens(prompt))

    if notes:
        update_person_record(workspace_id, person_id, {"writingStyleNotes": notes})


# Chunk size for deep analysis: up to this many messages per Gemini call.
_CHAT_ANALYSIS_CHUNK_SIZE = 200
# Soft token ceiling per chunk; keeps prompts well under the rate limit.
_CHAT_ANALYSIS_CHUNK_TOKEN_LIMIT = 30_000


def refresh_person_chat_analysis(workspace_id: str, person_id: str) -> None:
    """Deep multi-call analysis of ALL of the person's messaging patterns.

    Splits the full message corpus into ~200-message chunks (each under ~30k
    estimated tokens), runs a structured Gemini observation call per chunk
    (respecting the shared rate limiter), then consolidates all observations
    into a coherent ``chatAnalysis`` paragraph (5–10 sentences).

    Stores the result as ``chatAnalysis`` in the person's JSON record.
    Skips gracefully when Gemini is not configured or there are no messages.
    """
    person = get_person(workspace_id, person_id)
    timeline = load_export_timeline(workspace_id)

    # All non-media, non-deleted messages for this person, newest-first.
    candidate_msgs = [
        m
        for m in timeline
        if m.sender == person.display_name and m.text.strip() and not is_noise_message(m)
    ]

    if not candidate_msgs:
        return

    candidate_msgs.sort(key=lambda m: m.timestamp, reverse=True)

    chunks: list[list[Message]] = []
    current: list[Message] = []
    current_tokens = 0

    for msg in candidate_msgs:
        # Multiply token estimate by 2 to account for the paired-exchange overhead:
        # each chunk item may include the preceding other-person message as context.
        tok = estimate_tokens(msg.text) * 2
        if current and (
            len(current) >= _CHAT_ANALYSIS_CHUNK_SIZE
            or current_tokens + tok > _CHAT_ANALYSIS_CHUNK_TOKEN_LIMIT
        ):
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(msg)
        current_tokens += tok

    if current:
        chunks.append(current)

    if not chunks:
        return

    logger.info(
        "Chat analysis: %d chunk(s) over %d messages (person=%s)",
        len(chunks),
        len(candidate_msgs),
        person_id,
    )

    chunk_observations: list[str] = []
    for i, chunk in enumerate(chunks):
        # Build paired exchanges for this chunk — each target message is paired
        # with the immediately preceding message from the other person (if any).
        exchanges = _build_paired_exchanges(timeline, person.display_name, chunk)
        messages_text = "\n\n".join(exchanges)
        prompt = persona_extract_chat_analysis(
            person.display_name, messages_text, i + 1, len(chunks)
        )

        _rate_limiter.acquire(estimate_tokens(prompt))
        try:
            obs = gemini_service.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
            ).strip()
            _rate_limiter.record(estimate_tokens(prompt))
        except Exception as exc:
            _rate_limiter.record(estimate_tokens(prompt))
            logger.warning("Chat analysis chunk %d/%d failed: %s", i + 1, len(chunks), exc)
            continue

        if obs:
            chunk_observations.append(obs)

    if not chunk_observations:
        return

    combined_obs = "\n\n---\n\n".join(chunk_observations)
    consolidation_prompt = persona_extract_chat_analysis_consolidate(
        person.display_name, combined_obs, len(chunk_observations)
    )

    _rate_limiter.acquire(estimate_tokens(consolidation_prompt))
    try:
        analysis = gemini_service.chat(
            [{"role": "user", "content": consolidation_prompt}],
            temperature=0.3,
        ).strip()
        _rate_limiter.record(estimate_tokens(consolidation_prompt))
    except Exception as exc:
        _rate_limiter.record(estimate_tokens(consolidation_prompt))
        logger.warning("Chat analysis consolidation call failed: %s", exc)
        return

    if analysis:
        update_person_record(workspace_id, person_id, {"chatAnalysis": analysis})


# How many messages to sample for listening-style extraction (same as writing style).
_LISTENING_STYLE_SAMPLE_LIMIT = 60


def refresh_person_listening_style(workspace_id: str, person_id: str) -> None:
    """Use Gemini to extract how this person listens and responds when others share problems or news.

    Unlike personality or writing style, this captures the *reactive* behavioural pattern —
    what they do when someone vents, shares news, or asks for support. The result is stored
    as ``activeListeningStyle`` in the person's JSON record and injected into the system prompt
    so the persona mirrors their actual listening habits in conversation.

    Uses recency-weighted sampling (same strategy as writing style).
    Skips gracefully if Gemini is not configured or there are no usable messages.
    """
    person = get_person(workspace_id, person_id)
    timeline = load_export_timeline(workspace_id)

    # Filter to this person's non-empty, non-media, non-deleted messages.
    candidate_msgs = [
        m
        for m in timeline
        if m.sender == person.display_name and m.text.strip() and not is_noise_message(m)
    ]

    if not candidate_msgs:
        return

    # Prefer recent messages — listening habits may shift over time.
    sampled = _recency_weighted_sample(candidate_msgs, _LISTENING_STYLE_SAMPLE_LIMIT)

    # Build paired exchanges — listening style extraction needs the other person's
    # messages as stimulus so the LLM can analyse the actual response patterns.
    exchanges = _build_paired_exchanges(timeline, person.display_name, sampled)
    messages_text = "\n\n".join(exchanges)

    prompt = persona_extract_listening_style(person.display_name, messages_text)

    # Low temperature for factual, grounded extraction.
    # Rate-limit before the call; record after to stay under 14 RPM / 100k TPM.
    _rate_limiter.acquire(estimate_tokens(prompt))
    notes = gemini_service.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.3,
    ).strip()
    _rate_limiter.record(estimate_tokens(prompt))

    if notes:
        update_person_record(workspace_id, person_id, {"activeListeningStyle": notes})
