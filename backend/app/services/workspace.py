import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

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
from app.services import gemini as gemini_service
from app.services import jobs as job_service
from app.services.parser.preprocess import preprocess_whatsapp_export
from app.services.parser.whatsapp import Message, is_noise_message, non_system_messages, parse_whatsapp_export
from app.services.rate_limit import GeminiRateLimiter, estimate_tokens

logger = logging.getLogger("chatmemory.workspace")

# Shared rate limiter for all build-time Gemini calls in this module.
# Persona builds run behind the GPU mutex (one at a time), so this limiter
# mainly guards against multi-call bursts within a single build run.
_rate_limiter = GeminiRateLimiter(max_rpm=14, max_tpm=100_000)


def load_export_timeline(workspace_id: str) -> list[Message]:
    """All non-system messages in the workspace export, chronological."""
    export_path = workspace_path(workspace_id) / "export.txt"
    if not export_path.exists():
        raise FileNotFoundError("export.txt")
    raw = export_path.read_text(encoding="utf-8")
    parsed = parse_whatsapp_export(preprocess_whatsapp_export(raw))
    msgs = non_system_messages(parsed.messages)
    msgs.sort(key=lambda m: m.timestamp)
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

    ws_dir = workspace_path(workspace_id)
    if not ws_dir.exists():
        raise FileNotFoundError(workspace_id)
    shutil.rmtree(ws_dir)


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
        personality_notes=pdata.get("personalityNotes"),
        writing_style_notes=pdata.get("writingStyleNotes"),
        chat_analysis=pdata.get("chatAnalysis"),
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
    path = person_path(workspace_id, person_id)
    pdata = json.loads(path.read_text(encoding="utf-8"))
    pdata.update(updates)
    path.write_text(json.dumps(pdata, indent=2), encoding="utf-8")


# Noise filtering for personality/style/analysis passes is handled by
# is_noise_message() from the parser, which covers the full set of noise patterns.
# load_export_timeline already returns a noise-free timeline via non_system_messages,
# so these per-function filters are a belt-and-suspenders guard only.

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
        if m.sender == person.display_name
        and m.text.strip()
        and not is_noise_message(m)
    ]

    if not candidate_msgs:
        return

    # Recency-weighted sample: 60 % from the most recent third of messages, 40 % from older.
    # This gives the model a stronger signal about how the person communicates *now*.
    candidate_msgs = _recency_weighted_sample(candidate_msgs, _PERSONALITY_SAMPLE_LIMIT)

    messages_text = "\n".join(f"- {m.text}" for m in candidate_msgs)

    prompt = (
        f"You are analysing WhatsApp messages sent by one person named {person.display_name}.\n"
        f"Based only on these messages, write a concise personality profile in 3–6 sentences.\n"
        f"Cover: communication style, vocabulary and slang, humour, recurring themes or interests, "
        f"emotional tone, and how they relate to others in the conversation.\n"
        f"Be specific and factual — ground every claim in the messages. Write in third person.\n\n"
        f"Messages:\n{messages_text}"
    )

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
        if m.sender == person.display_name
        and m.text.strip()
        and not is_noise_message(m)
    ]

    if not candidate_msgs:
        return

    # Prefer recent messages — writing habits evolve over time.
    sampled = _recency_weighted_sample(candidate_msgs, _WRITING_STYLE_SAMPLE_LIMIT)

    messages_text = "\n".join(f"- {m.text}" for m in sampled)

    prompt = (
        f"You are analysing WhatsApp messages sent by one person named {person.display_name}.\n"
        f"Based ONLY on these messages, describe their writing style in 3–5 plain sentences.\n"
        f"Focus on HOW they type — NOT on personality or topics. Cover these points:\n"
        f"1. Capitalisation: do they capitalise sentence-starts? ALL CAPS? Mostly lowercase?\n"
        f"2. Punctuation: do they use periods, question marks, commas — or skip them?\n"
        f"3. Abbreviations and shorthand they actually use (list the specific ones you see).\n"
        f"4. Emoji: do they use them? How frequently? Which ones appear most?\n"
        f"5. Sentence structure: short fragments vs complete sentences; terse or verbose?\n"
        f"Be specific. Quote or paraphrase actual examples from the messages where helpful.\n"
        f"Do NOT infer mood, personality, or intent — only describe surface typing patterns.\n\n"
        f"Messages:\n{messages_text}"
    )

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


# ── Deep chat analysis ────────────────────────────────────────────────────────

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
        if m.sender == person.display_name
        and m.text.strip()
        and not is_noise_message(m)
    ]

    if not candidate_msgs:
        return

    candidate_msgs.sort(key=lambda m: m.timestamp, reverse=True)

    # ── Build token-bounded chunks ──────────────────────────────────────────
    chunks: list[list[Message]] = []
    current: list[Message] = []
    current_tokens = 0

    for msg in candidate_msgs:
        tok = estimate_tokens(msg.text)
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

    # ── Per-chunk observation calls ─────────────────────────────────────────
    chunk_observations: list[str] = []
    for i, chunk in enumerate(chunks):
        messages_text = "\n".join(f"- {m.text}" for m in chunk)
        prompt = (
            f"Analyse these WhatsApp messages from {person.display_name}.\n"
            f"This is chunk {i + 1} of {len(chunks)} (newest messages listed first).\n"
            f"Write 3–5 specific bullet-point observations about:\n"
            f"  • vocabulary patterns and recurring phrases\n"
            f"  • recurring topics or interests\n"
            f"  • emotional tone and patterns\n"
            f"  • relationship dynamics visible from context\n"
            f"  • any time-of-day or frequency patterns\n"
            f"Ground every observation in the messages. Be specific and factual.\n\n"
            f"Messages:\n{messages_text}"
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

    # ── Consolidation call ──────────────────────────────────────────────────
    combined_obs = "\n\n---\n\n".join(chunk_observations)
    consolidation_prompt = (
        f"You have structured observations about {person.display_name}'s WhatsApp messaging "
        f"patterns from {len(chunk_observations)} analysis chunk(s).\n"
        f"Synthesise these into a coherent chat-pattern analysis of 5–10 sentences covering:\n"
        f"vocabulary patterns, recurring topics, emotional patterns, relationship dynamics, "
        f"and any notable time-of-day or activity patterns.\n"
        f"Write in third person. Be specific — ground every claim in the observations.\n\n"
        f"Observations:\n{combined_obs}"
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
