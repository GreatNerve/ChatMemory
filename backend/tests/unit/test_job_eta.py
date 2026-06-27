from datetime import datetime, timedelta, timezone

from app.services.jobs import estimate_eta_seconds


def test_estimate_eta_seconds_needs_minimum_progress():
    now = datetime.now(timezone.utc)
    raw = {
        "startedAt": (now - timedelta(seconds=30)).isoformat(),
        "createdAt": now.isoformat(),
    }
    assert estimate_eta_seconds(raw, 1) is None
    eta = estimate_eta_seconds(raw, 10)
    assert eta is not None
    assert eta > 200


def test_estimate_eta_seconds_zero_at_complete():
    raw = {"startedAt": datetime.now(timezone.utc).isoformat()}
    assert estimate_eta_seconds(raw, 100) is None


def test_training_step_skips_percent_based_eta():
    from app.services.jobs import _read_raw, _write_raw, create_job, update_job, get_job

    job = create_job("persona_train", workspace_id="ws", person_id="p")
    update_job(job.id, status="running", percent=40, step="training", message="Training")
    snapshot = get_job(job.id)
    assert snapshot is not None
    # 40% with 30s elapsed would be ~45s — must not apply during training step.
    raw = _read_raw(job.id)
    assert raw is not None
    raw["startedAt"] = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
    _write_raw(job.id, raw)
    update_job(job.id, percent=40, step="training", message="Training step 1/1322")
    snapshot = get_job(job.id)
    assert snapshot is not None
    assert snapshot.eta_seconds is None or snapshot.eta_seconds > 300
