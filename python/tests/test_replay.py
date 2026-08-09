"""Tests for deterministic replay metadata, persistence, and safety boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crforge_gym.replay import (
    REPLAY_SCHEMA_VERSION,
    ReplayRequest,
    _checkpoint_timesteps,
    _frame_payload,
    _observed_training_timesteps,
    _resolve_checkpoint,
    _resolve_league_opponent,
    _validate_request,
    list_replays,
    load_replay,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _run_files(root: Path, *, status: str = "running") -> tuple[dict, dict]:
    config = {
        "endpoint": "tcp://localhost:9876",
        "eval_endpoint": "tcp://localhost:9877",
        "deck_profile": "mortar_self_play_v1",
        "ticks_per_step": 15,
    }
    manifest = {"runId": "unit-run", "status": status}
    _write_json(root / "config.json", config)
    _write_json(root / "manifest.json", manifest)
    return config, manifest


def test_active_run_requires_third_loopback_bridge(tmp_path: Path) -> None:
    config, manifest = _run_files(tmp_path)

    with pytest.raises(ValueError, match="training endpoint"):
        _validate_request(
            ReplayRequest(tmp_path, "tcp://127.0.0.1:9876"),
            tmp_path,
            config,
            manifest,
        )
    with pytest.raises(ValueError, match="periodic-evaluation endpoint"):
        _validate_request(
            ReplayRequest(tmp_path, "tcp://127.0.0.1:9877"),
            tmp_path,
            config,
            manifest,
        )

    _validate_request(
        ReplayRequest(tmp_path, "tcp://127.0.0.1:9888", replay_id="safe-id"),
        tmp_path,
        config,
        manifest,
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "tcp://0.0.0.0:9888",
        "tcp://192.0.2.1:9888",
        "http://127.0.0.1:9888",
        "tcp://localhost",
    ],
)
def test_remote_or_malformed_replay_endpoint_is_rejected(
    tmp_path: Path, endpoint: str
) -> None:
    config, manifest = _run_files(tmp_path)
    with pytest.raises(ValueError, match="loopback"):
        _validate_request(ReplayRequest(tmp_path, endpoint), tmp_path, config, manifest)


def test_checkpoint_must_remain_inside_run(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _run_files(run, status="paused")
    outside = tmp_path / "outside.zip"
    outside.write_bytes(b"model")

    with pytest.raises(ValueError, match="contained"):
        _resolve_checkpoint(run, outside)


def test_latest_checkpoint_step_and_observed_training_level(tmp_path: Path) -> None:
    _run_files(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "latest.zip"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"model")
    _write_json(
        tmp_path / "metrics.json",
        {
            "checkpoints": [
                {"path": "checkpoints/step_000000000100_periodic.zip", "timesteps": 100},
                {"path": "checkpoints/step_000000000200_periodic.zip", "timesteps": 200},
            ],
            "training": {"lastTimesteps": 237},
            "leagueSelections": {"last": {"timesteps": 241}},
            "evaluations": [],
        },
    )

    assert _checkpoint_timesteps(tmp_path, checkpoint, model_timesteps=200) == 200
    assert _observed_training_timesteps(tmp_path, 200) == 241


def test_league_resolution_is_explicit_and_contained(tmp_path: Path) -> None:
    _run_files(tmp_path)
    league = tmp_path / "league"
    old = league / "old.zip"
    new = league / "new.zip"
    league.mkdir()
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    _write_json(
        league / "league.json",
        {
            "initial": {
                "opponent_id": "old",
                "path": "old.zip",
                "checkpoint_step": 0,
                "category": "initial",
                "created_index": 0,
            },
            "recent": [
                {
                    "opponent_id": "new",
                    "path": "new.zip",
                    "checkpoint_step": 50,
                    "category": "recent",
                    "created_index": 1,
                }
            ],
            "historical": [],
        },
    )

    entry, path = _resolve_league_opponent(tmp_path, None)
    assert entry["opponent_id"] == "new"
    assert path == new.resolve()
    entry, path = _resolve_league_opponent(tmp_path, "old")
    assert entry["opponent_id"] == "old"
    assert path == old.resolve()
    with pytest.raises(ValueError, match="not active"):
        _resolve_league_opponent(tmp_path, "missing")


def test_frame_payload_is_compact_abstract_arena_state() -> None:
    raw = {
        "frame": 15,
        "gameTimeSeconds": 0.75,
        "isOvertime": False,
        "elixirMultiplier": 1,
        "bluePlayer": {
            "elixir": 4.25,
            "crowns": 0,
            "hand": [{"id": "mortar", "name": "Mortar", "type": "BUILDING", "cost": 4}],
            "towers": [
                {"id": 1, "type": "crown", "x": 9, "y": 3, "hp": 4000, "maxHp": 4000, "alive": True}
            ],
        },
        "redPlayer": {"elixir": 5, "crowns": 0, "hand": [], "towers": []},
        "entities": [
            {
                "id": 1,
                "name": "King Tower",
                "team": "BLUE",
                "entityType": "TOWER",
                "movementType": "BUILDING",
            },
            {
                "id": 7,
                "name": "Minion",
                "team": "BLUE",
                "entityType": "TROOP",
                "movementType": "AIR",
                "x": 5.25,
                "y": 11.75,
                "hp": 90,
                "maxHp": 100,
                "shield": 0,
                "raged": True,
            },
        ],
    }
    frame = _frame_payload(raw, step=1, reward=0.125)

    assert frame["time"] == 0.75
    assert frame["players"]["blue"]["hand"][0]["id"] == "mortar"
    assert frame["towers"][0]["team"] == "BLUE"
    assert [entity["name"] for entity in frame["entities"]] == ["Minion"]
    assert frame["entities"][0]["statuses"] == ["raged"]
    assert frame["reward"] == 0.125


def test_list_and_load_replays_do_not_accept_paths(tmp_path: Path) -> None:
    _run_files(tmp_path, status="paused")
    document = {
        "schemaVersion": REPLAY_SCHEMA_VERSION,
        "replayId": "replay-one",
        "createdAtUtc": "2026-08-09T18:00:00+00:00",
        "trainingLevel": {
            "runId": "unit-run",
            "checkpointTimesteps": 100,
            "currentTimesteps": 110,
        },
        "evaluation": {
            "seed": 42,
            "opponent": {"type": "rule_based"},
            "outcome": "win",
            "totalReward": 3.5,
            "steps": 12,
        },
        "frames": [],
    }
    _write_json(tmp_path / "replays" / "replay-one.json", document)
    _write_json(tmp_path / "replays" / "foreign.json", {"unrelated": True})

    assert [item["replayId"] for item in list_replays(tmp_path)] == ["replay-one"]
    assert load_replay(tmp_path, "replay-one")["evaluation"]["outcome"] == "win"
    with pytest.raises(ValueError, match="replay_id"):
        load_replay(tmp_path, "../manifest")

