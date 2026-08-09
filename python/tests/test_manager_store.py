"""Fast tests for manager path safety and durable run summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nextocr_manager.store import RunStore, atomic_write_json


def test_run_store_rejects_path_traversal_and_absolute_ids(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    for run_id in ("../escape", "/absolute", "nested/../../escape", "bad\\path"):
        with pytest.raises(ValueError, match="invalid runId|escapes"):
            store.validate_run_id(run_id, must_exist=False)


def test_run_creation_versions_never_overwrite_and_summary_tracks_total(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    first_id, first_dir = store.create_run(
        label="Mortier école", target_total_timesteps=10_000_000
    )
    second_id, second_dir = store.create_run(
        label="Mortier école", target_total_timesteps=20_000_000
    )

    assert first_id != second_id
    assert first_dir != second_dir
    assert store.summary(first_id)["version"] == 1
    assert store.summary(first_id)["targetTotalTimesteps"] == 10_000_000
    assert store.summary(second_id)["version"] == 2


def test_summary_uses_absolute_target_not_additional_steps_after_resume(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    run_id, run_dir = store.create_run(label="Daily", target_total_timesteps=10_000_000)
    atomic_write_json(
        run_dir / "config.json",
        {"timesteps_this_attempt": 9_800_000, "seed": 42, "deck_profile": "mortar_self_play_v1"},
    )
    atomic_write_json(
        run_dir / "manifest.json",
        {
            "status": "paused",
            "finalTimesteps": 200_000,
            "latestCheckpoint": "checkpoints/step_000000200000.zip",
            "attempts": [{"startTimesteps": 0}, {"startTimesteps": 100_000}],
        },
    )
    checkpoint = run_dir / "checkpoints" / "step_000000200000.zip"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    atomic_write_json(
        run_dir / "metrics.json",
        {
            "training": {"episodes": 3, "wins": 2, "losses": 1, "draws": 0},
            "checkpoints": [{"timesteps": 200_000, "path": str(checkpoint)}],
        },
    )

    summary = store.summary(run_id)

    assert summary["currentTimesteps"] == 200_000
    assert summary["durableTimesteps"] == 200_000
    assert summary["observedTimesteps"] == 0
    assert summary["targetTotalTimesteps"] == 10_000_000
    assert summary["remainingTimesteps"] == 9_800_000
    assert summary["progress"] == pytest.approx(0.02)
    assert store.checkpoint_from_value(run_id, "latest") == checkpoint.resolve()


def test_paused_summary_never_counts_uncheckpointed_observed_steps(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    run_id, run_dir = store.create_run(label="Crash", target_total_timesteps=10_000_000)
    checkpoint = run_dir / "checkpoints" / "step_000000500000_periodic.zip"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"model")
    atomic_write_json(
        run_dir / "manifest.json",
        {
            "status": "running",
            "latestCheckpoint": str(checkpoint.relative_to(run_dir)),
            "attempts": [{"status": "running", "startTimesteps": 0}],
        },
    )
    atomic_write_json(
        run_dir / "metrics.json",
        {
            "training": {"lastTimesteps": 507_780},
            "checkpoints": [{"timesteps": 500_000}],
        },
    )

    stopped = store.summary(run_id, live=False)
    running = store.summary(run_id, live=True)

    assert stopped["currentTimesteps"] == stopped["durableTimesteps"] == 500_000
    assert stopped["observedTimesteps"] == 507_780
    assert stopped["remainingTimesteps"] == 9_500_000
    assert running["currentTimesteps"] == 507_780


def test_detail_exposes_checkpoints_league_and_combined_logs(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    run_id, run_dir = store.create_run(label="UI", target_total_timesteps=100_000)
    atomic_write_json(
        run_dir / "metrics.json",
        {"training": {}, "checkpoints": [{"timesteps": 12}], "evaluations": [{"winRate": 1.0}]},
    )
    atomic_write_json(run_dir / "league" / "league.json", {"entries": [{"id": "initial"}]})
    (run_dir / "trainer.out.log").write_text("one\ntwo\n", encoding="utf-8")
    (run_dir / "metrics.csv").write_text(
        "timestamp_utc,event,timesteps,episode,reward,length,outcome\n"
        "2026-08-09T00:00:00+00:00,train_episode,12,1,3.5,8,win\n",
        encoding="utf-8",
    )

    detail = store.detail(run_id)
    logs = store.tail_logs(run_id, 2)

    assert detail["checkpoints"] == [{"timesteps": 12}]
    assert detail["league"]["entries"][0]["id"] == "initial"
    assert logs["lines"][-1].endswith("two")
    metrics = store.metrics(run_id)
    assert metrics["latest"] == {
        "timestampUtc": "2026-08-09T00:00:00+00:00",
        "timesteps": 12,
        "episode": 1,
        "reward": 3.5,
        "length": 8,
        "outcome": "win",
    }
