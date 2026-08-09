# SPDX-License-Identifier: Apache-2.0
"""Training lifecycle orchestration for the localhost NextoCR dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any
import uuid

from .store import RunStore, atomic_write_json, read_json, utc_now


DEFAULT_TARGET_TIMESTEPS = 10_000_000
MIN_TARGET_TIMESTEPS = 10_000
MAX_TARGET_TIMESTEPS = 1_000_000_000
SAFE_COMMAND_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,95}$")
ALLOWED_OPPONENTS = frozenset({"noop", "random", "rule_based", "self_play"})
ALLOWED_REPLAY_OPPONENTS = frozenset({"rule_based", "league"})


class ManagerError(RuntimeError):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass(frozen=True)
class ManagerConfig:
    project_root: Path
    runs_root: Path
    state_root: Path
    static_root: Path
    host: str = "127.0.0.1"
    port: int = 8765

    @classmethod
    def discover(cls, *, port: int = 8765) -> "ManagerConfig":
        project_root = Path(__file__).resolve().parents[2]
        return cls(
            project_root=project_root,
            runs_root=project_root / "runs",
            state_root=project_root / ".local" / "nextocr-manager",
            static_root=project_root / "manager-ui" / "static",
            host="127.0.0.1",
            port=port,
        )

    def validated(self) -> "ManagerConfig":
        project = self.project_root.resolve()
        runs = self.runs_root.resolve()
        state = self.state_root.resolve()
        static = self.static_root.resolve()
        if self.host != "127.0.0.1":
            raise ValueError("the manager must bind exactly to 127.0.0.1")
        if not (1024 <= int(self.port) <= 65535):
            raise ValueError("manager port must be between 1024 and 65535")
        for child in (runs, state, static):
            try:
                child.relative_to(project)
            except ValueError as exc:
                raise ValueError("manager paths must stay inside the project workspace") from exc
        return ManagerConfig(project, runs, state, static, self.host, int(self.port))


@dataclass
class ProcessBundle:
    run_id: str
    trainer: subprocess.Popen
    bridges: list[subprocess.Popen]
    log_handles: list[Any]
    ports: list[int]
    start_timesteps: int
    started_monotonic: float


class ManagerService:
    """Own the run registry, trainer processes and replay jobs."""

    def __init__(self, config: ManagerConfig):
        self.config = config.validated()
        self.config.state_root.mkdir(parents=True, exist_ok=True)
        self.store = RunStore(self.config.runs_root)
        self.session_token = secrets.token_urlsafe(32)
        self._state_path = self.config.state_root / "state.json"
        self._state = read_json(self._state_path, {}) or {
            "schemaVersion": 1,
            "selectedRunId": None,
            "processes": {},
            "replayJobs": {},
        }
        self._lock = threading.RLock()
        self._bundles: dict[str, ProcessBundle] = {}
        self._replay_threads: dict[str, threading.Thread] = {}
        self._bridge_built = False
        self._external_cache: tuple[float, dict[str, list[int]]] = (0.0, {})
        self._closing = False
        self._normalize_persisted_state()

    # ----- Read API -----------------------------------------------------

    def session(self) -> dict[str, Any]:
        return {
            "token": self.session_token,
            "header": "X-NextoCR-Token",
            "expires": "when the local manager process exits",
        }

    def state(self) -> dict[str, Any]:
        live = self._live_run_map()
        runs = [self._enrich_summary(self.store.summary(run_id, live=run_id in live))
                for run_id in self.store.discover()]
        runs.sort(key=lambda item: item.get("updatedAtUtc") or "", reverse=True)
        selected = self._state.get("selectedRunId")
        if selected not in {item["runId"] for item in runs}:
            selected = next((item["runId"] for item in runs if item["live"]), None)
            selected = selected or (runs[0]["runId"] if runs else None)
        replay_jobs = [self._public_replay_job(job) for job in self._state["replayJobs"].values()]
        replay_jobs.sort(key=lambda item: item.get("createdAtUtc", ""), reverse=True)
        return {
            "manager": {
                "status": "shutting_down" if self._closing else "ready",
                "host": self.config.host,
                "port": self.config.port,
                "projectRoot": str(self.config.project_root),
                "runsRoot": str(self.config.runs_root),
            },
            "selectedRunId": selected,
            "activeRunIds": sorted(live),
            "runs": runs,
            "replayJobs": replay_jobs,
        }

    def runs(self) -> list[dict[str, Any]]:
        return self.state()["runs"]

    def run_detail(self, run_id: str) -> dict[str, Any]:
        live = run_id in self._live_run_map()
        detail = self.store.detail(run_id, live=live)
        detail.update(self._enrich_summary(detail))
        return detail

    def metrics(self, run_id: str) -> dict[str, Any]:
        return self.store.metrics(run_id)

    def logs(self, run_id: str, *, tail: int = 200) -> dict[str, Any]:
        return self.store.tail_logs(run_id, tail)

    # ----- Training mutations -----------------------------------------

    def create_and_start_run(self, body: dict[str, Any]) -> dict[str, Any]:
        self._require_no_active_training()
        label = body.get("label", "Mortier self-play")
        target = self._validate_target(body.get("targetTotalTimesteps", DEFAULT_TARGET_TIMESTEPS))
        run_id, _ = self.store.create_run(label=label, target_total_timesteps=target)
        launch_config = self._default_launch_config()
        self.store.update_manager(run_id, launchConfig=launch_config, status="starting")
        try:
            self._launch_training(run_id, resume_from=None, config=launch_config)
        except Exception:
            self.store.update_manager(run_id, status="failed_to_start")
            raise
        return self.run_detail(run_id)

    def checkpoint(self, run_id: str, *, timeout: float = 90.0) -> dict[str, Any]:
        self._require_live(run_id)
        return self._send_training_command(run_id, "checkpoint", timeout=timeout)

    def pause(self, run_id: str, *, timeout: float = 120.0) -> dict[str, Any]:
        self._require_live(run_id)
        response = self._send_training_command(run_id, "pause", timeout=timeout)
        try:
            self._wait_for_run_exit(run_id, timeout=45.0)
        except ManagerError:
            bundle = self._bundles.get(run_id)
            if bundle is None:
                raise
            self._terminate_owned_process(bundle.trainer)
        finally:
            self._stop_run_bridges(run_id)
        self.store.update_manager(run_id, status="paused")
        return {"run": self.run_detail(run_id), "command": response}

    def resume(self, run_id: str) -> dict[str, Any]:
        self.store.validate_run_id(run_id)
        self._require_no_active_training()
        summary = self.store.summary(run_id)
        checkpoint = self.store.latest_checkpoint(run_id)
        checkpoint_timesteps = self._checkpoint_timesteps(
            checkpoint,
            fallback=summary["durableTimesteps"],
        )
        remaining = int(summary["targetTotalTimesteps"]) - checkpoint_timesteps
        if remaining <= 0:
            raise ManagerError("target_reached", "this run already reached its total target", 409)
        config = self._launch_config_for_run(run_id)
        self.store.update_manager(run_id, status="starting")
        self._launch_training(run_id, resume_from=checkpoint, config=config)
        return self.run_detail(run_id)

    def create_version(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        self.store.validate_run_id(run_id)
        self._require_no_active_training()
        checkpoint = self.store.checkpoint_from_value(run_id, body.get("checkpoint"))
        parent = self.store.summary(run_id)
        checkpoint_timesteps = self._checkpoint_timesteps(checkpoint, fallback=parent["currentTimesteps"])
        requested_target = body.get("targetTotalTimesteps", parent["targetTotalTimesteps"])
        target = self._validate_target(requested_target)
        if target <= checkpoint_timesteps:
            raise ManagerError(
                "invalid_target",
                "the new version target must exceed the source checkpoint timesteps",
            )
        label = body.get("label", f"{parent['label']} – nouvelle version")
        new_id, _ = self.store.create_run(
            label=label,
            target_total_timesteps=target,
            parent_run_id=run_id,
            source_checkpoint=checkpoint,
            source_checkpoint_timesteps=checkpoint_timesteps,
        )
        launch_config = self._launch_config_for_run(run_id)
        self.store.update_manager(new_id, launchConfig=launch_config, status="starting")
        try:
            self._launch_training(new_id, resume_from=checkpoint, config=launch_config)
        except Exception:
            self.store.update_manager(new_id, status="failed_to_start")
            raise
        return self.run_detail(new_id)

    # ----- Replay mutations -------------------------------------------

    def create_replay(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        self.store.validate_run_id(run_id)
        checkpoint = self.store.checkpoint_from_value(run_id, body.get("checkpoint"))
        seed_value = body.get("seed")
        seed = self.store.summary(run_id)["seed"] + 30_000_000 if seed_value is None else seed_value
        if isinstance(seed, bool) or not isinstance(seed, int) or not (0 <= seed <= 2_147_483_647):
            raise ManagerError("invalid_seed", "seed must be an integer between 0 and 2147483647")
        opponent = body.get("opponent", "rule_based")
        if opponent not in ALLOWED_REPLAY_OPPONENTS:
            raise ManagerError("invalid_opponent", "replay opponent must be rule_based or league")
        with self._lock:
            if any(job.get("status") in ("queued", "running")
                   for job in self._state["replayJobs"].values()):
                raise ManagerError("replay_busy", "another replay job is already running", 409)
            job_id = f"replay-{uuid.uuid4().hex[:16]}"
            job = {
                "jobId": job_id,
                "replayId": job_id,
                "runId": run_id,
                "status": "queued",
                "checkpoint": str(checkpoint.relative_to(self.store.validate_run_id(run_id))),
                "seed": seed,
                "opponent": opponent,
                "createdAtUtc": utc_now(),
            }
            self._state["replayJobs"][job_id] = job
            self._save_state()
            thread = threading.Thread(
                target=self._run_replay_job,
                args=(job_id,),
                name=f"NextoCR-{job_id}",
                daemon=True,
            )
            self._replay_threads[job_id] = thread
            thread.start()
        return self._public_replay_job(job)

    def replay(self, run_id: str, replay_id: str) -> dict[str, Any]:
        self.store.validate_run_id(run_id)
        if not SAFE_COMMAND_ID.fullmatch(replay_id):
            raise ManagerError("invalid_replay_id", "invalid replay id")
        job = self._state["replayJobs"].get(replay_id)
        if job is None or job.get("runId") != run_id:
            # Completed replay files survive manager state resets.
            replay_path = self.store.validate_run_id(run_id) / "replays" / f"{replay_id}.json"
            data = read_json(replay_path, None)
            if data is None:
                raise FileNotFoundError(f"unknown replay: {replay_id}")
            return {"jobId": replay_id, "replayId": replay_id, "runId": run_id,
                    "status": "ready", "replay": data,
                    **self._replay_api_payload(data)}
        public = self._public_replay_job(job)
        if job.get("status") == "ready":
            replay_path = self.store.validate_run_id(run_id) / "replays" / f"{replay_id}.json"
            data = read_json(replay_path, None)
            if data is None:
                public.update(status="failed", error="replay artifact is missing")
            else:
                public["replay"] = data
                public.update(self._replay_api_payload(data))
        return public

    # ----- Lifecycle ---------------------------------------------------

    def close(self, *, pause_timeout: float = 30.0) -> None:
        self._closing = True
        # Pause trainers launched by this manager, including a process recovered
        # from its durable state after the dashboard itself was restarted.
        managed_run_ids = set(self._bundles) | set(self._state.get("processes", {}))
        for run_id in managed_run_ids:
            bundle = self._bundles.get(run_id)
            live = run_id in self._live_run_map(force=True)
            if not live:
                self._stop_run_bridges(run_id)
                continue
            try:
                self._send_training_command(run_id, "pause", timeout=pause_timeout)
                self._wait_for_run_exit(run_id, timeout=15.0)
            except Exception as exc:
                self.store.update_manager(
                    run_id,
                    status="shutdown_error",
                    shutdownError=f"{type(exc).__name__}: {exc}",
                )
                if bundle is not None:
                    self._signal_trainer(bundle)
                    try:
                        bundle.trainer.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        self._terminate_owned_process(bundle.trainer)
                else:
                    self._terminate_recovered_trainer(run_id)
            self._stop_run_bridges(run_id)
        self._save_state()

    # ----- Internals ---------------------------------------------------

    def _launch_training(
        self,
        run_id: str,
        *,
        resume_from: Path | None,
        config: dict[str, Any],
    ) -> None:
        run_dir = self.store.validate_run_id(run_id)
        summary = self.store.summary(run_id)
        current = (
            self._checkpoint_timesteps(
                resume_from,
                fallback=summary["durableTimesteps"],
            )
            if resume_from is not None
            else 0
        )
        target = int(summary["targetTotalTimesteps"])
        remaining = target - current
        if remaining <= 0:
            raise ManagerError("target_reached", "run target is already reached", 409)
        self._validate_launch_config(config)
        self._ensure_bridge_distribution()
        ports = self._reserve_port_block(2)
        handles: list[Any] = []
        bridges: list[subprocess.Popen] = []
        trainer: subprocess.Popen | None = None
        try:
            for port in ports:
                stdout = (run_dir / f"bridge-{port}.out.log").open("a", encoding="utf-8")
                stderr = (run_dir / f"bridge-{port}.err.log").open("a", encoding="utf-8")
                handles.extend([stdout, stderr])
                bridge = self._spawn(
                    [str(self._bridge_launcher()), str(port)],
                    stdout=stdout,
                    stderr=stderr,
                )
                bridges.append(bridge)
            for bridge, port in zip(bridges, ports):
                self._wait_for_port(port, bridge, timeout=35.0)

            stdout = (run_dir / "trainer.out.log").open("a", encoding="utf-8")
            stderr = (run_dir / "trainer.err.log").open("a", encoding="utf-8")
            handles.extend([stdout, stderr])
            command = self._trainer_command(
                run_dir,
                remaining=remaining,
                ports=ports,
                config=config,
                resume_from=resume_from,
            )
            environment = {**os.environ, "PYTHONHASHSEED": str(config["seed"])}
            trainer = self._spawn(command, stdout=stdout, stderr=stderr, env=environment)
            bundle = ProcessBundle(
                run_id=run_id,
                trainer=trainer,
                bridges=bridges,
                log_handles=handles,
                ports=ports,
                start_timesteps=current,
                started_monotonic=time.monotonic(),
            )
            with self._lock:
                self._bundles[run_id] = bundle
                self._state["selectedRunId"] = run_id
                self._state["processes"][run_id] = {
                    "trainerPid": trainer.pid,
                    "bridgePids": [process.pid for process in bridges],
                    "ports": ports,
                    "startTimesteps": current,
                    "startedAtUtc": utc_now(),
                    "runDir": str(run_dir),
                }
                self._save_state()
            self.store.update_manager(
                run_id,
                status="running",
                launchConfig=config,
                targetTotalTimesteps=target,
                lastLaunch={
                    "startedAtUtc": utc_now(),
                    "startTimesteps": current,
                    "requestedAdditionalTimesteps": remaining,
                    "resumeFrom": str(resume_from) if resume_from else None,
                    "ports": ports,
                },
            )
            threading.Thread(
                target=self._monitor_bundle,
                args=(run_id, bundle),
                name=f"NextoCR-monitor-{run_id.replace('/', '-')}",
                daemon=True,
            ).start()
        except Exception:
            if trainer is not None and trainer.poll() is None:
                self._terminate_owned_process(trainer)
            for bridge in bridges:
                self._terminate_owned_process(bridge)
            for handle in handles:
                handle.close()
            raise

    def _trainer_command(
        self,
        run_dir: Path,
        *,
        remaining: int,
        ports: list[int],
        config: dict[str, Any],
        resume_from: Path | None,
    ) -> list[str]:
        script = (self.config.project_root / "python" / "examples" / "train_ppo.py").resolve()
        script.relative_to(self.config.project_root)
        command = [
            sys.executable,
            str(script),
            "--run-dir", str(run_dir),
            "--timesteps", str(remaining),
            "--endpoint", f"tcp://127.0.0.1:{ports[0]}",
            "--eval-endpoint", f"tcp://127.0.0.1:{ports[1]}",
            "--seed", str(config["seed"]),
            "--eval-seed", str(config["eval_seed"]),
            "--deck-profile", config["deck_profile"],
            "--opponent", config["opponent"],
            "--eval-opponent", config["evaluation_opponent"],
            "--ticks-per-step", str(config["ticks_per_step"]),
            "--checkpoint-freq", str(config["checkpoint_freq"]),
            "--eval-freq", str(config["eval_freq"]),
            "--eval-episodes", str(config["eval_episodes"]),
            "--episode-log-interval", str(config["episode_log_interval"]),
            "--self-play-interval", str(config["self_play_interval"]),
            "--league-max-recent", str(config["league_max_recent"]),
            "--league-max-historical", str(config["league_max_historical"]),
            "--league-model-cache", str(config["league_model_cache"]),
            "--league-initial-weight", str(config["league_initial_weight"]),
            "--league-recent-weight", str(config["league_recent_weight"]),
            "--league-historical-weight", str(config["league_historical_weight"]),
            "--learning-rate", str(config["learning_rate"]),
            "--n-steps", str(config["n_steps"]),
            "--batch-size", str(config["batch_size"]),
            "--n-epochs", str(config["n_epochs"]),
            "--gamma", str(config["gamma"]),
            "--gae-lambda", str(config["gae_lambda"]),
            "--clip-range", str(config["clip_range"]),
            "--ent-coef", str(config["ent_coef"]),
            "--vf-coef", str(config["vf_coef"]),
            "--max-grad-norm", str(config["max_grad_norm"]),
            "--net-arch", ",".join(str(width) for width in config["net_arch"]),
            "--device", config["device"],
        ]
        if not config["deterministic_torch"]:
            command.append("--no-deterministic-torch")
        if resume_from is not None:
            command.extend(["--resume", str(resume_from)])
        return command

    def _default_launch_config(self) -> dict[str, Any]:
        return {
            "seed": 42,
            "eval_seed": 10_000_042,
            "deck_profile": "mortar_self_play_v1",
            "opponent": "self_play",
            "evaluation_opponent": "rule_based",
            "ticks_per_step": 15,
            "checkpoint_freq": 100_000,
            "eval_freq": 250_000,
            "eval_episodes": 20,
            "episode_log_interval": 50,
            "self_play_interval": 50_000,
            "league_max_recent": 4,
            "league_max_historical": 8,
            "league_model_cache": 2,
            "league_initial_weight": 0.10,
            "league_recent_weight": 0.60,
            "league_historical_weight": 0.30,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 512,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.005,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "net_arch": [512, 256],
            "device": "cpu",
            "deterministic_torch": True,
        }

    def _launch_config_for_run(self, run_id: str) -> dict[str, Any]:
        run_dir = self.store.validate_run_id(run_id)
        manager = read_json(run_dir / "manager.json", {}) or {}
        stored = manager.get("launchConfig")
        if isinstance(stored, dict):
            config = {**self._default_launch_config(), **stored}
            self._validate_launch_config(config)
            return config
        source = read_json(run_dir / "config.json", {}) or {}
        config = self._default_launch_config()
        aliases = {
            "seed": "seed", "eval_seed": "eval_seed", "deck_profile": "deck_profile",
            "opponent": "opponent", "evaluation_opponent": "evaluation_opponent",
            "ticks_per_step": "ticks_per_step", "checkpoint_freq": "checkpoint_freq",
            "eval_freq": "eval_freq", "eval_episodes": "eval_episodes",
            "self_play_interval": "self_play_interval", "league_max_recent": "league_max_recent",
            "league_max_historical": "league_max_historical", "league_model_cache": "league_model_cache",
            "league_initial_weight": "league_initial_weight", "league_recent_weight": "league_recent_weight",
            "league_historical_weight": "league_historical_weight", "learning_rate": "learning_rate",
            "n_steps": "n_steps", "batch_size": "batch_size", "n_epochs": "n_epochs",
            "gamma": "gamma", "gae_lambda": "gae_lambda", "clip_range": "clip_range",
            "ent_coef": "ent_coef", "vf_coef": "vf_coef", "max_grad_norm": "max_grad_norm",
            "net_arch": "net_arch", "device": "device", "deterministic_torch": "deterministic_torch",
        }
        for source_key, target_key in aliases.items():
            if source_key in source:
                config[target_key] = source[source_key]
        self._validate_launch_config(config)
        return config

    def _validate_launch_config(self, config: dict[str, Any]) -> None:
        if config.get("deck_profile") != "mortar_self_play_v1":
            raise ManagerError("unsupported_deck", "manager supports the mortar_self_play_v1 deck only")
        if config.get("opponent") != "self_play" or config.get("evaluation_opponent") != "rule_based":
            raise ManagerError("unsupported_opponent", "manager requires self-play with rule-based evaluation")
        if config.get("device") not in ("cpu", "cuda", "auto"):
            raise ManagerError("invalid_config", "invalid training device")
        if not isinstance(config.get("net_arch"), list) or not config["net_arch"]:
            raise ManagerError("invalid_config", "network architecture is invalid")
        numeric_positive = (
            "ticks_per_step", "checkpoint_freq", "eval_freq", "eval_episodes",
            "episode_log_interval", "self_play_interval", "league_max_recent",
            "league_model_cache", "n_steps", "batch_size", "n_epochs",
        )
        for key in numeric_positive:
            value = config.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ManagerError("invalid_config", f"{key} must be a positive integer")
        if int(config["batch_size"]) > int(config["n_steps"]):
            raise ManagerError("invalid_config", "batch_size cannot exceed n_steps")

    def _send_training_command(self, run_id: str, command_type: str, *, timeout: float) -> dict[str, Any]:
        run_dir = self.store.validate_run_id(run_id)
        command_id = f"manager-{command_type}-{uuid.uuid4().hex[:20]}"
        if not SAFE_COMMAND_ID.fullmatch(command_id):
            raise AssertionError("generated unsafe command id")
        request_dir = run_dir / "control" / "requests"
        response_dir = run_dir / "control" / "responses"
        request_dir.mkdir(parents=True, exist_ok=True)
        response_dir.mkdir(parents=True, exist_ok=True)
        destination = request_dir / f"{command_id}.json"
        atomic_write_json(
            destination,
            {
                "schemaVersion": 1,
                "id": command_id,
                "type": command_type,
                "createdAtUtc": utc_now(),
            },
        )
        response_path = response_dir / f"{command_id}.json"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = read_json(response_path, None)
            if isinstance(response, dict):
                if response.get("status") != "completed":
                    raise ManagerError("command_failed", str(response.get("error", "command failed")), 500)
                return response
            if run_id not in self._live_run_map(force=True):
                destination.unlink(missing_ok=True)
                raise ManagerError("trainer_stopped", "trainer stopped before acknowledging the command", 409)
            time.sleep(0.1)
        # Cancel only if the trainer has not claimed this exact request yet.
        destination.unlink(missing_ok=True)
        raise ManagerError("command_timeout", "trainer did not acknowledge the command in time", 504)

    def _run_replay_job(self, job_id: str) -> None:
        bridge: subprocess.Popen | None = None
        handles: list[Any] = []
        try:
            with self._lock:
                job = self._state["replayJobs"][job_id]
                job["status"] = "running"
                job["startedAtUtc"] = utc_now()
                self._save_state()
            run_id = job["runId"]
            run_dir = self.store.validate_run_id(run_id)
            self._ensure_bridge_distribution()
            active_ports = {
                port
                for process in self._state.get("processes", {}).values()
                for port in process.get("ports", [])
                if isinstance(port, int)
            }
            run_config = read_json(run_dir / "config.json", {}) or {}
            for endpoint_key in ("endpoint", "eval_endpoint"):
                endpoint = run_config.get(endpoint_key)
                if isinstance(endpoint, str):
                    match = re.fullmatch(r"tcp://(?:127\.0\.0\.1|localhost):(\d+)", endpoint)
                    if match:
                        active_ports.add(int(match.group(1)))
            port = self._reserve_port_block(1, excluded=active_ports)[0]
            stdout = (run_dir / "replays" / f"{job_id}.bridge.out.log")
            stderr = (run_dir / "replays" / f"{job_id}.bridge.err.log")
            stdout.parent.mkdir(parents=True, exist_ok=True)
            out_handle = stdout.open("a", encoding="utf-8")
            err_handle = stderr.open("a", encoding="utf-8")
            handles.extend([out_handle, err_handle])
            bridge = self._spawn(
                [str(self._bridge_launcher()), str(port)],
                stdout=out_handle,
                stderr=err_handle,
            )
            self._wait_for_port(port, bridge, timeout=35.0)
            script = self.config.project_root / "python" / "examples" / "generate_replay.py"
            command = [
                sys.executable,
                str(script),
                "--run-dir", str(run_dir),
                "--endpoint", f"tcp://127.0.0.1:{port}",
                "--checkpoint", job["checkpoint"],
                "--seed", str(job["seed"]),
                "--opponent", job["opponent"],
                "--replay-id", job_id,
            ]
            result = subprocess.run(
                command,
                cwd=self.config.project_root,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "replay CLI failed")
            summary = json.loads(result.stdout.strip().splitlines()[-1])
            replay_path = run_dir / "replays" / f"{job_id}.json"
            if not replay_path.is_file():
                raise FileNotFoundError("replay CLI did not create the expected artifact")
            with self._lock:
                job.update(
                    status="ready",
                    completedAtUtc=utc_now(),
                    summary=summary,
                    path=str(replay_path.relative_to(run_dir)),
                )
                self._save_state()
        except Exception as exc:
            with self._lock:
                job = self._state["replayJobs"].get(job_id, {"jobId": job_id})
                job.update(
                    status="failed",
                    completedAtUtc=utc_now(),
                    error=f"{type(exc).__name__}: {exc}",
                )
                self._state["replayJobs"][job_id] = job
                self._save_state()
        finally:
            if bridge is not None:
                self._terminate_owned_process(bridge)
            for handle in handles:
                handle.close()
            self._replay_threads.pop(job_id, None)

    def _monitor_bundle(self, run_id: str, bundle: ProcessBundle) -> None:
        return_code = bundle.trainer.wait()
        self._stop_bundle_bridges(bundle)
        for handle in bundle.log_handles:
            handle.close()
        with self._lock:
            if self._bundles.get(run_id) is bundle:
                self._bundles.pop(run_id, None)
            self._state.get("processes", {}).pop(run_id, None)
            self._save_state()
        manifest = read_json(self.store.validate_run_id(run_id) / "manifest.json", {}) or {}
        manifest_status = manifest.get("status")
        status = manifest_status or ("stopped" if return_code == 0 else "failed")
        self.store.update_manager(run_id, status=status, lastExitCode=return_code)

    def _live_run_map(self, *, force: bool = False) -> dict[str, list[int]]:
        now = time.monotonic()
        cached_at, cached = self._external_cache
        if not force and now - cached_at < 1.0:
            live = dict(cached)
        else:
            live = self._scan_external_trainers()
            self._external_cache = (now, live)
        with self._lock:
            for run_id, bundle in list(self._bundles.items()):
                if bundle.trainer.poll() is None:
                    live.setdefault(run_id, []).append(bundle.trainer.pid)
        return live

    def _scan_external_trainers(self) -> dict[str, list[int]]:
        live: dict[str, list[int]] = {}
        records: list[tuple[int, str]] = []
        if os.name == "nt":
            command = (
                "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*train_ppo.py*' } "
                "| Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
            )
            try:
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", command],
                    capture_output=True, text=True, timeout=5, check=False,
                )
                payload = json.loads(result.stdout or "[]")
                if isinstance(payload, dict):
                    payload = [payload]
                records = [(int(item["ProcessId"]), str(item.get("CommandLine") or ""))
                           for item in payload]
            except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired):
                records = []
        elif Path("/proc").is_dir():
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    command_line = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
                except OSError:
                    continue
                if "train_ppo.py" in command_line:
                    records.append((int(entry.name), command_line))
        pattern = re.compile(r"(?:^|\s)--run-dir(?:=|\s+)(?:\"([^\"]+)\"|'([^']+)'|(\S+))")
        for pid, command_line in records:
            match = pattern.search(command_line)
            if not match:
                continue
            value = next((group for group in match.groups() if group is not None), "")
            try:
                run_id = self.store.run_id(Path(value).expanduser().resolve())
                self.store.validate_run_id(run_id)
            except (ValueError, FileNotFoundError, OSError):
                continue
            live.setdefault(run_id, []).append(pid)
        return live

    def _enrich_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        result = dict(summary)
        run_id = result["runId"]
        process = self._state.get("processes", {}).get(run_id, {})
        throughput = 0.0
        bundle = self._bundles.get(run_id)
        if bundle is not None and bundle.trainer.poll() is None:
            elapsed = max(0.001, time.monotonic() - bundle.started_monotonic)
            throughput = max(0, int(result["currentTimesteps"]) - bundle.start_timesteps) / elapsed
        if throughput <= 0:
            manifest = read_json(self.store.validate_run_id(run_id) / "manifest.json", {}) or {}
            seconds = float(manifest.get("trainingSeconds", 0.0) or 0.0)
            attempts = manifest.get("attempts", [])
            if seconds > 0 and attempts:
                last = attempts[-1]
                steps = int(last.get("finalTimesteps", 0) or 0) - int(last.get("startTimesteps", 0) or 0)
                throughput = max(0, steps) / seconds
        remaining = int(result["remainingTimesteps"])
        result["throughputStepsPerSecond"] = throughput
        result["etaSeconds"] = (remaining / throughput if throughput > 0 else None)
        result["process"] = {
            "trainerPid": process.get("trainerPid"),
            "bridgePids": process.get("bridgePids", []),
            "ports": process.get("ports", []),
            "startedAtUtc": process.get("startedAtUtc"),
        }
        result["actions"] = {
            "canCheckpoint": bool(result["live"]),
            "canPause": bool(result["live"]),
            "canResume": (
                not result["live"]
                and remaining > 0
                and (
                    result["latestCheckpoint"] is not None
                    or result.get("manifestStatus") == "created"
                )
            ),
            "canVersion": not result["live"] and result["latestCheckpoint"] is not None,
            "canReplay": result["latestCheckpoint"] is not None,
        }
        return result

    def _require_no_active_training(self) -> None:
        active = self._live_run_map(force=True)
        if active:
            raise ManagerError(
                "training_active",
                f"pause the active run first: {', '.join(sorted(active))}",
                409,
            )

    def _require_live(self, run_id: str) -> None:
        self.store.validate_run_id(run_id)
        if run_id not in self._live_run_map(force=True):
            raise ManagerError("trainer_not_running", "this run is not currently training", 409)

    @staticmethod
    def _validate_target(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ManagerError("invalid_target", "targetTotalTimesteps must be an integer")
        if not MIN_TARGET_TIMESTEPS <= value <= MAX_TARGET_TIMESTEPS:
            raise ManagerError(
                "invalid_target",
                f"targetTotalTimesteps must be between {MIN_TARGET_TIMESTEPS} and {MAX_TARGET_TIMESTEPS}",
            )
        return value

    def _ensure_bridge_distribution(self) -> None:
        with self._lock:
            if self._bridge_built and self._bridge_launcher().is_file():
                return
            wrapper = self.config.project_root / ("gradlew.bat" if os.name == "nt" else "gradlew")
            result = subprocess.run(
                [str(wrapper), ":gym-bridge:installDist", "-q"],
                cwd=self.config.project_root,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            if result.returncode != 0 or not self._bridge_launcher().is_file():
                raise ManagerError(
                    "bridge_build_failed",
                    (result.stderr or result.stdout or "bridge distribution was not created").strip(),
                    500,
                )
            self._bridge_built = True

    def _bridge_launcher(self) -> Path:
        filename = "gym-bridge.bat" if os.name == "nt" else "gym-bridge"
        path = (
            self.config.project_root / "gym-bridge" / "build" / "install" /
            "gym-bridge" / "bin" / filename
        ).resolve()
        path.relative_to(self.config.project_root)
        return path

    def _reserve_port_block(self, count: int, *, excluded: set[int] | None = None) -> list[int]:
        excluded = excluded or set()
        for first in range(9876, 12_000 - count):
            ports = list(range(first, first + count))
            if any(port in excluded for port in ports):
                continue
            sockets: list[socket.socket] = []
            try:
                for port in ports:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.bind(("127.0.0.1", port))
                    sockets.append(sock)
                return ports
            except OSError:
                continue
            finally:
                for sock in sockets:
                    sock.close()
        raise ManagerError("no_bridge_port", "no local bridge port is available", 503)

    @staticmethod
    def _wait_for_port(port: int, process: subprocess.Popen, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"bridge process exited with code {process.returncode}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.1)
        raise TimeoutError(f"bridge on port {port} did not become ready")

    def _spawn(self, command: list[str], *, stdout: Any, stderr: Any,
               env: dict[str, str] | None = None) -> subprocess.Popen:
        kwargs: dict[str, Any] = {
            "cwd": self.config.project_root,
            "stdout": stdout,
            "stderr": stderr,
            "stdin": subprocess.DEVNULL,
            "env": env,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
        return subprocess.Popen(command, **kwargs)

    def _wait_for_run_exit(self, run_id: str, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if run_id not in self._live_run_map(force=True):
                return
            time.sleep(0.2)
        raise ManagerError("trainer_exit_timeout", "trainer saved its checkpoint but is still exiting", 504)

    def _stop_run_bridges(self, run_id: str) -> None:
        bundle = self._bundles.get(run_id)
        if bundle is not None:
            self._stop_bundle_bridges(bundle)
        persisted = self._state.get("processes", {}).get(run_id, {})
        bridge_pids = persisted.get("bridgePids", [])
        ports = persisted.get("ports", [])
        if isinstance(bridge_pids, list) and isinstance(ports, list):
            for pid, port in zip(bridge_pids, ports):
                self._terminate_recovered_bridge(pid, port)

    def _stop_bundle_bridges(self, bundle: ProcessBundle) -> None:
        for bridge in bundle.bridges:
            self._terminate_owned_process(bridge)

    @staticmethod
    def _signal_trainer(bundle: ProcessBundle) -> None:
        if bundle.trainer.poll() is not None:
            return
        if os.name == "nt" and hasattr(signal, "CTRL_BREAK_EVENT"):
            bundle.trainer.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            bundle.trainer.send_signal(signal.SIGTERM)

    @staticmethod
    def _terminate_owned_process(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    def _terminate_recovered_bridge(self, pid: Any, port: Any) -> None:
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            return
        if not isinstance(port, int) or isinstance(port, bool) or not (9876 <= port < 12_000):
            return
        command = self._process_command_line(pid)
        if command is None:
            return
        project_marker = str(self.config.project_root).lower()
        lowered = command.lower()
        if project_marker not in lowered or "gym-bridge" not in lowered or str(port) not in command:
            return
        self._kill_verified_pid(pid)

    def _terminate_recovered_trainer(self, run_id: str) -> None:
        process = self._state.get("processes", {}).get(run_id, {})
        pid = process.get("trainerPid")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            return
        command = self._process_command_line(pid)
        if command is None:
            return
        run_dir = str(self.store.validate_run_id(run_id)).lower()
        lowered = command.lower()
        if "train_ppo.py" not in lowered or run_dir not in lowered:
            return
        self._kill_verified_pid(pid)

    @staticmethod
    def _process_command_line(pid: int) -> str | None:
        if os.name == "nt":
            script = (
                f"Get-CimInstance Win32_Process -Filter \"ProcessId = {pid}\" "
                "| Select-Object -ExpandProperty CommandLine"
            )
            try:
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", script],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                return None
            return result.stdout.strip() or None
        try:
            return (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(
                b"\0", b" "
            ).decode()
        except OSError:
            return None

    @staticmethod
    def _kill_verified_pid(pid: int) -> None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
        else:
            try:
                os.killpg(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                return

    def _public_replay_job(self, job: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in job.items() if key not in {"internal"}}

    @staticmethod
    def _replay_api_payload(data: dict[str, Any]) -> dict[str, Any]:
        training = data.get("trainingLevel", {})
        evaluation = data.get("evaluation", {})
        arena = data.get("arena", {})
        steps = int(evaluation.get("steps", 0) or 0)
        ticks_per_step = int(evaluation.get("ticksPerStep", 0) or 0)
        ticks_per_second = int(arena.get("ticksPerSecond", 20) or 20)
        return {
            "metadata": {
                "checkpoint": training.get("checkpoint"),
                "timesteps": training.get("checkpointTimesteps"),
                "currentTimesteps": training.get("currentTimesteps"),
                "seed": evaluation.get("seed"),
                "opponent": evaluation.get("opponent"),
                "outcome": evaluation.get("outcome"),
                "reward": evaluation.get("totalReward"),
                "steps": steps,
                "durationSeconds": (
                    steps * ticks_per_step / ticks_per_second
                    if ticks_per_second > 0
                    else None
                ),
            },
            "frames": data.get("frames", []),
        }

    @staticmethod
    def _checkpoint_timesteps(checkpoint: Path, *, fallback: Any) -> int:
        match = re.search(r"step_(\d+)", checkpoint.name)
        if match:
            return int(match.group(1))
        try:
            return max(0, int(fallback))
        except (TypeError, ValueError):
            return 0

    def _normalize_persisted_state(self) -> None:
        self._state.setdefault("schemaVersion", 1)
        self._state.setdefault("selectedRunId", None)
        self._state.setdefault("processes", {})
        self._state.setdefault("replayJobs", {})
        # Jobs cannot still be running after this manager process restarted.
        for job in self._state["replayJobs"].values():
            if job.get("status") in ("queued", "running"):
                job.update(
                    status="failed",
                    completedAtUtc=utc_now(),
                    error="manager restarted before replay generation completed",
                )
        self._save_state()

    def _save_state(self) -> None:
        atomic_write_json(self._state_path, self._state)
