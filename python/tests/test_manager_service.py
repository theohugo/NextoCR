"""Lifecycle contract tests for the local manager without spawning Java."""

from __future__ import annotations

from pathlib import Path

import pytest

from nextocr_manager.service import ManagerConfig, ManagerError, ManagerService
from nextocr_manager.store import atomic_write_json


def _service(tmp_path: Path) -> ManagerService:
    project = tmp_path / "project"
    (project / "manager-ui" / "static").mkdir(parents=True)
    config = ManagerConfig(
        project_root=project,
        runs_root=project / "runs",
        state_root=project / ".local" / "nextocr-manager",
        static_root=project / "manager-ui" / "static",
        port=8765,
    )
    service = ManagerService(config)
    service._live_run_map = lambda force=False: {}  # type: ignore[method-assign]
    return service


def _checkpointed_run(service: ManagerService, *, current: int = 200_000) -> tuple[str, Path]:
    run_id, run_dir = service.store.create_run(
        label="Mortier", target_total_timesteps=10_000_000
    )
    checkpoint = run_dir / "checkpoints" / f"step_{current:012d}.zip"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"model")
    config = service._default_launch_config()
    atomic_write_json(run_dir / "config.json", {**config, "timesteps_this_attempt": 10_000_000})
    atomic_write_json(
        run_dir / "manifest.json",
        {
            "status": "paused",
            "finalTimesteps": current,
            "latestCheckpoint": str(checkpoint.relative_to(run_dir)),
            "attempts": [{"startTimesteps": 0}],
        },
    )
    atomic_write_json(
        run_dir / "metrics.json",
        {"training": {}, "checkpoints": [{"timesteps": current}]},
    )
    service.store.update_manager(run_id, launchConfig=config)
    return run_id, checkpoint


def test_manager_config_refuses_non_loopback_and_outside_paths(tmp_path: Path) -> None:
    project = tmp_path / "project"
    with pytest.raises(ValueError, match="127.0.0.1"):
        ManagerConfig(project, project / "runs", project / ".local", project / "ui", host="0.0.0.0").validated()
    with pytest.raises(ValueError, match="inside"):
        ManagerConfig(project, tmp_path / "outside", project / ".local", project / "ui").validated()


def test_create_run_creates_and_starts_in_one_operation(tmp_path: Path) -> None:
    service = _service(tmp_path)
    launched = []
    service._launch_training = lambda run_id, **kwargs: launched.append((run_id, kwargs))  # type: ignore[method-assign]

    result = service.create_and_start_run(
        {"label": "Session matin", "targetTotalTimesteps": 10_000_000}
    )

    assert launched[0][0] == result["runId"]
    assert launched[0][1]["resume_from"] is None
    assert result["targetTotalTimesteps"] == 10_000_000


def test_resume_passes_only_remaining_absolute_target_steps(tmp_path: Path) -> None:
    service = _service(tmp_path)
    run_id, checkpoint = _checkpointed_run(service, current=200_000)
    launched = []
    service._launch_training = lambda selected, **kwargs: launched.append((selected, kwargs))  # type: ignore[method-assign]

    service.resume(run_id)

    # _launch_training owns the exact target-minus-current calculation; make
    # that invariant visible through the durable summary used by it.
    assert service.store.summary(run_id)["remainingTimesteps"] == 9_800_000
    assert launched == [(run_id, {"resume_from": checkpoint.resolve(), "config": service._launch_config_for_run(run_id)})]


def test_resume_gate_uses_durable_checkpoint_when_observed_metrics_exceed_target(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    run_id, checkpoint = _checkpointed_run(service, current=500_000)
    run_dir = service.store.validate_run_id(run_id)
    service.store.update_manager(run_id, targetTotalTimesteps=600_000)
    atomic_write_json(
        run_dir / "metrics.json",
        {
            "training": {"lastTimesteps": 700_000},
            "checkpoints": [{"timesteps": 500_000}],
        },
    )
    launched = []
    service._launch_training = lambda selected, **kwargs: launched.append((selected, kwargs))  # type: ignore[method-assign]

    service.resume(run_id)

    assert launched[0][1]["resume_from"] == checkpoint.resolve()
    assert service.store.summary(run_id)["durableTimesteps"] == 500_000


def test_launch_calculates_remaining_from_checkpoint_not_newer_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    service = _service(tmp_path)
    run_id, checkpoint = _checkpointed_run(service, current=500_000)
    run_dir = service.store.validate_run_id(run_id)
    metrics = {
        "training": {"lastTimesteps": 507_780},
        "checkpoints": [{"timesteps": 500_000}],
    }
    atomic_write_json(run_dir / "metrics.json", metrics)
    captured = {}
    service._ensure_bridge_distribution = lambda: None  # type: ignore[method-assign]
    service._reserve_port_block = lambda count: [10_101, 10_102]  # type: ignore[method-assign]

    def capture_command(_run_dir, *, remaining, ports, config, resume_from):
        captured.update(remaining=remaining, resume_from=resume_from)
        return ["trainer"]

    service._trainer_command = capture_command  # type: ignore[method-assign]
    service._bridge_launcher = lambda: Path("bridge")  # type: ignore[method-assign]
    service._wait_for_port = lambda *args, **kwargs: None  # type: ignore[method-assign]

    class _FakeProcess:
        _next_pid = 100

        def __init__(self):
            self.pid = self._next_pid
            type(self)._next_pid += 1

        def poll(self):
            return None

    service._spawn = lambda *args, **kwargs: _FakeProcess()  # type: ignore[method-assign]

    class _NoStartThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr("nextocr_manager.service.threading.Thread", _NoStartThread)

    service._launch_training(
        run_id,
        resume_from=checkpoint,
        config=service._launch_config_for_run(run_id),
    )

    assert captured["resume_from"] == checkpoint
    assert captured["remaining"] == 9_500_000
    bundle = service._bundles[run_id]
    assert bundle.start_timesteps == 500_000
    for handle in bundle.log_handles:
        handle.close()


def test_new_version_has_parent_and_source_without_overwriting_parent(tmp_path: Path) -> None:
    service = _service(tmp_path)
    parent_id, checkpoint = _checkpointed_run(service)
    launched = []
    service._launch_training = lambda selected, **kwargs: launched.append((selected, kwargs))  # type: ignore[method-assign]

    child = service.create_version(
        parent_id,
        {"label": "Mortier v2", "targetTotalTimesteps": 12_000_000},
    )

    assert child["runId"] != parent_id
    assert child["parentRunId"] == parent_id
    assert Path(child["sourceCheckpoint"]) == checkpoint.resolve()
    assert child["currentTimesteps"] == 200_000
    assert child["remainingTimesteps"] == 11_800_000
    assert launched[0][1]["resume_from"] == checkpoint.resolve()
    assert service.store.summary(parent_id)["targetTotalTimesteps"] == 10_000_000


def test_new_version_uses_the_explicitly_selected_checkpoint(tmp_path: Path) -> None:
    service = _service(tmp_path)
    parent_id, latest = _checkpointed_run(service, current=500_000)
    parent_dir = service.store.validate_run_id(parent_id)
    older = parent_dir / "checkpoints" / "step_000000200000_periodic.zip"
    older.write_bytes(b"older-model")
    service._launch_training = lambda *args, **kwargs: None  # type: ignore[method-assign]

    child = service.create_version(
        parent_id,
        {
            "label": "Branche 200k",
            "checkpoint": str(older.relative_to(parent_dir)).replace("\\", "/"),
            "targetTotalTimesteps": 1_000_000,
        },
    )

    assert Path(child["sourceCheckpoint"]) == older.resolve()
    assert child["currentTimesteps"] == 200_000
    assert child["remainingTimesteps"] == 800_000
    assert Path(service.store.summary(parent_id)["latestCheckpoint"]) == latest.resolve()


def test_replay_latest_sentinel_is_validated_before_job(tmp_path: Path, monkeypatch) -> None:
    service = _service(tmp_path)
    run_id, checkpoint = _checkpointed_run(service)
    started = []

    class _ImmediateThread:
        def __init__(self, *, target, args, **_kwargs):
            self.target = target
            self.args = args

        def start(self):
            started.append(self.args[0])

    monkeypatch.setattr("nextocr_manager.service.threading.Thread", _ImmediateThread)
    job = service.create_replay(
        run_id,
        {"checkpoint": "latest", "seed": 123, "opponent": "rule_based"},
    )

    assert job["status"] == "queued"
    assert job["checkpoint"] == str(checkpoint.relative_to(service.store.validate_run_id(run_id)))
    assert started == [job["jobId"]]


def test_invalid_mutation_values_are_rejected(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(ManagerError, match="targetTotalTimesteps"):
        service.create_and_start_run({"targetTotalTimesteps": True})
