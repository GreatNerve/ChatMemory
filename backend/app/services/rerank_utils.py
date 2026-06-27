"""Shared rerank score parsing for Gemini LLM backends."""

from __future__ import annotations

import json
import re
from typing import Any


def coerce_rerank_score(raw: Any) -> float | None:
    """Normalize LLM rerank scores — small models often return lists or strings."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return max(0.0, min(1.0, float(raw)))
    if isinstance(raw, str):
        cleaned = raw.strip().rstrip("%")
        try:
            value = float(cleaned)
            if value > 1.0 and value <= 100.0:
                value /= 100.0
            return max(0.0, min(1.0, value))
        except ValueError:
            return None
    if isinstance(raw, list) and raw:
        return coerce_rerank_score(raw[0])
    if isinstance(raw, dict):
        for key in ("score", "relevance", "value"):
            if key in raw:
                return coerce_rerank_score(raw[key])
    return None


def parse_rerank_scores(raw: str, chunk_count: int) -> dict[int, float]:
    """Extract id→score map from model output; empty dict triggers fallback ranking."""
    text = raw.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text, re.I)
        if match:
            text = match.group(1)

    start = text.find("[")
    end = text.rfind("]") + 1
    if start < 0 or end <= start:
        return {}

    try:
        parsed = json.loads(text[start:end])
    except json.JSONDecodeError:
        return {}

    if not isinstance(parsed, list):
        return {}

    score_by_id: dict[int, float] = {}
    for item in parsed:
        if not isinstance(item, dict) or "id" not in item:
            continue
        try:
            row_id = int(item["id"])
        except (TypeError, ValueError):
            continue
        if row_id < 1 or row_id > chunk_count:
            continue
        score = coerce_rerank_score(item.get("score"))
        if score is not None:
            score_by_id[row_id] = score
    return score_by_id
