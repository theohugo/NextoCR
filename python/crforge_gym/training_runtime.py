# SPDX-License-Identifier: Apache-2.0
"""Reproducible experiment utilities for the NextoCR PPO trainer.

This module is intentionally separate from the simulator API.  Import it only
through the optional ``train``/``dev`` dependency sets.
"""

from __future__ import annotations

from collections import deque
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import signal
import subprocess
import sys
from typing import Any, Callable

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from nextocr_fsio import atomic_write_json, replace_with_retry


ACTION_SCHEMA = "exact_discrete_v1"
ACTION_COUNT = 41
SCHEMA_VERSION = 1
EPISODE_RANK_SEED_STRIDE = 1_000_000_000_000
_RESUME_CRITICAL_FIELDS = (
    "action_schema",
    "observation_preprocessing",
    "deck_profile",
    "deck",
    # These change the observation width or the reward being optimised, so a
    # checkpoint trained under different values cannot be resumed.
    "deck_mode",
    "match_memory",
    "elixir_shaping_budget",
    "curriculum",
    "seed",
    "eval_seed",
    "episode_seed_strategy",
    "backend",
    "num_envs",
    "opponent",
    "evaluation_opponent",
    "ticks_per_step",
    "n_steps",
    "batch_size",
    "n_epochs",
    "learning_rate",
    "gamma",
    "gae_lambda",
    "clip_range",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "net_arch",
    "league_max_recent",
    "league_max_historical",
    "league_model_cache",
    "league_initial_weight",
    "league_recent_weight",
    "league_historical_weight",
)
_CSV_FIELDS = (
    "timestamp_utc",
    "event",
    "timesteps",
    "episode",
    "reward",
    "length",
    "outcome",
    "mean_reward",
    "std_reward",
    "mean_length",
    "win_rate",
    "loss_rate",
    "draw_rate",
    "throughput_steps_per_second",
    "path",
    "details",
)


def utc_now() -> str:
    """Return a stable UTC timestamp suitable for JSON artifacts."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_run_dir(seed: int, now: datetime | None = None) -> Path:
    """Create a collision-resistant default path without touching the disk."""
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / "nextocr" / f"{timestamp}-seed{seed}-pid{os.getpid()}"


def seed_everything(seed: int, deterministic_torch: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy and Torch and return reproducibility metadata."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_torch, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic_torch

    return {
        "seed": seed,
        "pythonHashSeed": os.environ["PYTHONHASHSEED"],
        "torchDeterministicAlgorithms": deterministic_torch,
        "torchVersion": torch.__version__,
        "cudaAvailable": torch.cuda.is_available(),
    }


def derive_episode_seed(base_seed: int, *, rank: int, checkpoint_timesteps: int) -> int:
    """Derive a non-replaying episode seed namespace for an attempt and env rank."""
    if rank < 0 or checkpoint_timesteps < 0:
        raise ValueError("rank and checkpoint_timesteps must be non-negative")
    if checkpoint_timesteps >= EPISODE_RANK_SEED_STRIDE:
        raise ValueError("checkpoint timestep exceeds the episode seed namespace stride")
    return int(base_seed) + rank * EPISODE_RANK_SEED_STRIDE + int(checkpoint_timesteps)


class DeterministicEpisodeSeedWrapper(gym.Wrapper):
    """Assign a deterministic, increasing seed to every episode reset."""

    def __init__(self, env: gym.Env, base_seed: int):
        super().__init__(env)
        self.set_seed_sequence(base_seed)

    def set_seed_sequence(self, base_seed: int) -> None:
        self._next_seed = int(base_seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._next_seed = int(seed)
        episode_seed = self._next_seed
        self._next_seed += 1
        return self.env.reset(seed=episode_seed, options=options)


def validate_exact_action_space(action_space: spaces.Space[Any]) -> None:
    """Reject legacy factorized checkpoints with an actionable error."""
    if not isinstance(action_space, spaces.Discrete) or action_space.n != ACTION_COUNT:
        raise ValueError(
            "This checkpoint uses the legacy factorized MultiDiscrete action schema. "
            "NextoCR training requires exact_discrete_v1 (Discrete(41)); legacy "
            "checkpoints cannot be resumed safely and must be retrained or migrated."
        )


def resolve_resume_checkpoint(value: str | os.PathLike[str]) -> Path:
    """Resolve a checkpoint file, extensionless SB3 path, or prior run directory."""
    candidate = Path(value).expanduser().resolve()
    if candidate.is_file():
        return candidate
    zip_candidate = Path(f"{candidate}.zip")
    if zip_candidate.is_file():
        return zip_candidate
    if not candidate.is_dir():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {candidate}")

    manifest_path = candidate / "manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        latest = manifest.get("latestCheckpoint")
        if latest:
            latest_path = Path(latest)
            if not latest_path.is_absolute():
                latest_path = candidate / latest_path
            if latest_path.is_file():
                return latest_path.resolve()

    preferred = [
        candidate / "checkpoints" / "latest.zip",
        candidate / "final_model.zip",
    ]
    for path in preferred:
        if path.is_file():
            return path.resolve()

    checkpoints = sorted((candidate / "checkpoints").glob("step_*.zip"))
    if checkpoints:
        return checkpoints[-1].resolve()
    raise FileNotFoundError(f"No SB3 checkpoint was found under run directory: {candidate}")


def infer_run_dir_from_checkpoint(checkpoint: Path) -> Path | None:
    """Return the owning run directory when its config is discoverable."""
    candidates = [checkpoint.parent, checkpoint.parent.parent]
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate
    return None


def runtime_metadata() -> dict[str, Any]:
    """Capture non-secret runtime versions for the manifest."""
    packages: dict[str, str] = {}
    for name in ("nextocr-gym", "gymnasium", "numpy", "torch", "stable-baselines3", "sb3-contrib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
        "git": _git_metadata(),
    }


def _git_metadata() -> dict[str, Any]:
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"root": root, "commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"root": None, "commit": None, "dirty": None}


class RunArtifacts:
    """Own the durable config, manifest, metrics and checkpoint files for one run."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root).expanduser().resolve()
        self.checkpoint_dir = self.root / "checkpoints"
        self.attempt_dir = self.root / "attempts"
        self.config_path = self.root / "config.json"
        self.manifest_path = self.root / "manifest.json"
        self.metrics_csv_path = self.root / "metrics.csv"
        self.metrics_json_path = self.root / "metrics.json"
        self._manifest: dict[str, Any] = {}
        self._metrics: dict[str, Any] = {}
        self._attempt_index: int | None = None

    def initialize(
        self,
        config: dict[str, Any],
        *,
        resume_from: Path | None,
        start_timesteps: int,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.attempt_dir.mkdir(parents=True, exist_ok=True)

        existing_config = _read_json(self.config_path) if self.config_path.is_file() else None
        if existing_config is not None:
            if resume_from is None:
                raise FileExistsError(
                    f"Run directory already contains config.json: {self.root}. "
                    "Choose a new --run-dir or resume its checkpoint."
                )
            mismatches = _critical_config_mismatches(existing_config, config)
            if mismatches:
                details = ", ".join(
                    f"{key}: {old!r} -> {new!r}" for key, old, new in mismatches
                )
                raise ValueError(
                    "Resume configuration changes reproducibility-critical fields: " + details
                )
        else:
            _atomic_write_json(self.config_path, config)

        if self.manifest_path.is_file():
            self._manifest = _read_json(self.manifest_path)
        else:
            created = utc_now()
            self._manifest = {
                "schemaVersion": SCHEMA_VERSION,
                "runId": self.root.name,
                "actionSchema": ACTION_SCHEMA,
                "action_schema": config.get("action_schema", ACTION_SCHEMA),
                "observation_preprocessing": config.get("observation_preprocessing"),
                "deck_profile": config.get("deck_profile"),
                "createdAtUtc": created,
                "updatedAtUtc": created,
                "status": "created",
                "config": "config.json",
                "metricsCsv": "metrics.csv",
                "metricsJson": "metrics.json",
                "curriculum": config.get(
                    "curriculum",
                    {"stage": "mirror", "opponent_catalog_snapshot": None},
                ),
                "attempts": [],
                "runtime": runtime_metadata(),
            }

        # A process can disappear after a durable checkpoint (power loss or a
        # legacy trainer migration) without getting to close its attempt.  A
        # subsequent explicit resume is proof that the prior process is gone;
        # close that stale record before opening the next attempt.
        previous_attempts = self._manifest.get("attempts", [])
        if previous_attempts and previous_attempts[-1].get("status") == "running":
            previous_attempts[-1].update(
                {
                    "status": "interrupted",
                    "finishedAtUtc": utc_now(),
                    "finalTimesteps": int(start_timesteps),
                    "recoveredByResume": True,
                }
            )

        self._metrics = (
            _read_json(self.metrics_json_path)
            if self.metrics_json_path.is_file()
            else {
                "schemaVersion": SCHEMA_VERSION,
                "training": {
                    "episodes": 0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "unknownOutcomes": 0,
                    "rewardSum": 0.0,
                    "lengthSum": 0,
                },
                "evaluations": [],
                "checkpoints": [],
                "leagueSelections": {
                    "episodes": 0,
                    "byOpponent": {},
                    "last": None,
                },
            }
        )
        # ``lastTimesteps`` is a gauge, not an all-time maximum.  Reset it to
        # the checkpoint actually loaded so a previous process that advanced
        # beyond its last durable save cannot make a resumed model look newer
        # than it really is.
        self._metrics.setdefault("training", {})["lastTimesteps"] = int(start_timesteps)

        attempt_number = len(self._manifest.get("attempts", [])) + 1
        attempt_file = self.attempt_dir / f"attempt_{attempt_number:03d}.json"
        attempt_config = {
            "schemaVersion": SCHEMA_VERSION,
            "startedAtUtc": utc_now(),
            "resumeFrom": str(resume_from) if resume_from else None,
            "startTimesteps": int(start_timesteps),
            "config": config,
        }
        _atomic_write_json(attempt_file, attempt_config)
        attempt = {
            "attempt": attempt_number,
            "config": str(attempt_file.relative_to(self.root)),
            "startedAtUtc": attempt_config["startedAtUtc"],
            "resumeFrom": attempt_config["resumeFrom"],
            "startTimesteps": int(start_timesteps),
            "status": "running",
        }
        self._manifest.setdefault("attempts", []).append(attempt)
        self._attempt_index = len(self._manifest["attempts"]) - 1
        self._manifest["status"] = "running"
        self._manifest["updatedAtUtc"] = utc_now()
        _atomic_write_json(self.manifest_path, self._manifest)
        _atomic_write_json(self.metrics_json_path, self._metrics)

    def record_episode(
        self, *, timesteps: int, episode: int, reward: float, length: int, outcome: str
    ) -> None:
        training = self._metrics["training"]
        training["episodes"] += 1
        training["rewardSum"] += float(reward)
        training["lengthSum"] += int(length)
        outcome_key = {"win": "wins", "loss": "losses", "draw": "draws"}.get(
            outcome, "unknownOutcomes"
        )
        training[outcome_key] += 1
        training["lastTimesteps"] = int(timesteps)
        training["meanReward"] = training["rewardSum"] / training["episodes"]
        training["meanLength"] = training["lengthSum"] / training["episodes"]
        self._append_csv(
            event="train_episode",
            timesteps=timesteps,
            episode=episode,
            reward=reward,
            length=length,
            outcome=outcome,
        )
        self._flush_metrics()

    def record_evaluation(self, *, timesteps: int, result: "EvaluationResult") -> None:
        entry = {
            "timestampUtc": utc_now(),
            "timesteps": int(timesteps),
            "episodes": result.episodes,
            "meanReward": result.mean_reward,
            "stdReward": result.std_reward,
            "meanLength": result.mean_length,
            "winRate": result.win_rate,
            "lossRate": result.loss_rate,
            "drawRate": result.draw_rate,
            "unknownOutcomes": result.unknown_outcomes,
        }
        self._metrics["evaluations"].append(entry)
        self._append_csv(
            event="evaluation",
            timesteps=timesteps,
            mean_reward=result.mean_reward,
            std_reward=result.std_reward,
            mean_length=result.mean_length,
            win_rate=result.win_rate,
            loss_rate=result.loss_rate,
            draw_rate=result.draw_rate,
            details={"episodes": result.episodes, "unknownOutcomes": result.unknown_outcomes},
        )
        self._flush_metrics()

    def record_league_selection(
        self,
        *,
        timesteps: int,
        episode: int,
        opponent_id: str,
        category: str,
        checkpoint_step: int,
        phase: str = "train",
    ) -> None:
        selections = self._metrics.setdefault(
            "leagueSelections", {"episodes": 0, "byOpponent": {}, "last": None}
        )
        selections["episodes"] += 1
        selections["byOpponent"][opponent_id] = (
            selections["byOpponent"].get(opponent_id, 0) + 1
        )
        selections["last"] = {
            "timestampUtc": utc_now(),
            "timesteps": int(timesteps),
            "episode": int(episode),
            "opponentId": opponent_id,
            "category": category,
            "checkpointStep": int(checkpoint_step),
            "phase": phase,
        }
        self._append_csv(
            event="league_opponent",
            timesteps=timesteps,
            episode=episode,
            details={
                "opponentId": opponent_id,
                "category": category,
                "checkpointStep": checkpoint_step,
                "phase": phase,
            },
        )
        self._flush_metrics()

    def save_checkpoint(self, model: Any, *, timesteps: int, reason: str) -> Path:
        safe_reason = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in reason)
        base = self.checkpoint_dir / f"step_{int(timesteps):012d}_{safe_reason}"
        model.save(str(base))
        checkpoint = _saved_model_path(base)
        latest = self.checkpoint_dir / "latest.zip"
        temporary = self.checkpoint_dir / f".latest.zip.{os.getpid()}.tmp"
        shutil.copy2(checkpoint, temporary)
        # ``latest.zip`` is the file the dashboard and the replay viewer open,
        # so it is the most contended artefact of the run.  The step-numbered
        # checkpoint above is already durable, hence tolerating a miss here.
        if not replace_with_retry(temporary, latest, tolerate_failure=True):
            temporary.unlink(missing_ok=True)

        relative = str(checkpoint.relative_to(self.root))
        entry = {
            "timestampUtc": utc_now(),
            "timesteps": int(timesteps),
            "reason": reason,
            "path": relative,
        }
        self._metrics["checkpoints"].append(entry)
        self._manifest["latestCheckpoint"] = relative
        self._manifest["updatedAtUtc"] = utc_now()
        self._append_csv(
            event="checkpoint", timesteps=timesteps, path=relative, details={"reason": reason}
        )
        self._flush_metrics()
        _atomic_write_json(self.manifest_path, self._manifest, tolerate_failure=True)
        return checkpoint

    def save_final_model(self, model: Any, path: str | os.PathLike[str]) -> Path:
        base = Path(path).expanduser().resolve()
        base.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(base))
        saved = _saved_model_path(base)
        try:
            manifest_path = str(saved.relative_to(self.root))
        except ValueError:
            manifest_path = str(saved)
        self._manifest["finalModel"] = manifest_path
        self._manifest["latestCheckpoint"] = manifest_path
        self._manifest["updatedAtUtc"] = utc_now()
        _atomic_write_json(self.manifest_path, self._manifest)
        return saved

    def finish(
        self,
        status: str,
        *,
        timesteps: int,
        training_seconds: float,
        attempt_steps: int | None = None,
        error: str | None = None,
    ) -> None:
        now = utc_now()
        self._manifest["status"] = status
        self._manifest["updatedAtUtc"] = now
        self._manifest["finalTimesteps"] = int(timesteps)
        self._manifest["trainingSeconds"] = float(training_seconds)
        if error:
            self._manifest["error"] = error
        if self._attempt_index is not None:
            attempt = self._manifest["attempts"][self._attempt_index]
            attempt.update(
                {
                    "status": status,
                    "finishedAtUtc": now,
                    "finalTimesteps": int(timesteps),
                    "trainingSeconds": float(training_seconds),
                }
            )
            if error:
                attempt["error"] = error
        self._append_csv(
            event=status,
            timesteps=timesteps,
            throughput_steps_per_second=(
                max(0, attempt_steps if attempt_steps is not None else timesteps)
                / training_seconds
                if training_seconds > 0
                else 0.0
            ),
            details={"error": error} if error else None,
        )
        _atomic_write_json(self.manifest_path, self._manifest, tolerate_failure=True)
        self._flush_metrics()

    def _append_csv(self, **values: Any) -> None:
        row = {field: "" for field in _CSV_FIELDS}
        row["timestamp_utc"] = utc_now()
        for key, value in values.items():
            if key not in row or value is None:
                continue
            row[key] = json.dumps(value, sort_keys=True) if key == "details" else value
        new_file = not self.metrics_csv_path.exists()
        with self.metrics_csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def _flush_metrics(self) -> None:
        self._metrics["updatedAtUtc"] = utc_now()
        # Flushed several times per second while the dashboard polls the same
        # file; a lost sample is reissued on the next flush, a raise is not.
        _atomic_write_json(self.metrics_json_path, self._metrics, tolerate_failure=True)


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    mean_reward: float
    std_reward: float
    mean_length: float
    win_rate: float
    loss_rate: float
    draw_rate: float
    unknown_outcomes: int


def evaluate_maskable_policy(
    model: Any,
    env: gym.Env,
    *,
    n_episodes: int,
    base_seed: int,
) -> EvaluationResult:
    """Evaluate with sb3-contrib's mask-aware evaluator and deterministic resets."""
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    setter = getattr(env, "set_seed_sequence", None)
    if setter is None:
        try:
            setter = env.get_wrapper_attr("set_seed_sequence")
        except (AttributeError, gym.error.Error):
            setter = None
    if setter is not None:
        setter(base_seed)

    from sb3_contrib.common.maskable.evaluation import evaluate_policy

    outcomes: list[str] = []

    def capture_outcomes(locals_: dict[str, Any], _globals: dict[str, Any]) -> None:
        infos = locals_.get("infos", [])
        dones = locals_.get("dones", [])
        for done, info in zip(dones, infos):
            if done:
                outcomes.append(info.get("game_outcome", "unknown"))

    rewards, lengths = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_episodes,
        deterministic=True,
        callback=capture_outcomes,
        return_episode_rewards=True,
        warn=False,
        use_masking=True,
    )
    wins = outcomes.count("win")
    losses = outcomes.count("loss")
    draws = outcomes.count("draw")
    known = wins + losses + draws
    unknown = max(0, n_episodes - known)
    return EvaluationResult(
        episodes=n_episodes,
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_length=float(np.mean(lengths)),
        win_rate=wins / n_episodes,
        loss_rate=losses / n_episodes,
        draw_rate=draws / n_episodes,
        unknown_outcomes=unknown,
    )


class EpisodeMetricsCallback(BaseCallback):
    """Persist episode outcomes and rolling metrics to TensorBoard/CSV/JSON."""

    def __init__(
        self,
        artifacts: RunArtifacts,
        *,
        target_timesteps: int,
        log_interval_episodes: int = 50,
        window_size: int = 100,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.artifacts = artifacts
        self.target_timesteps = target_timesteps
        self.log_interval_episodes = max(1, log_interval_episodes)
        self._outcomes: deque[str] = deque(maxlen=window_size)
        self._rewards: deque[float] = deque(maxlen=window_size)
        self._lengths: deque[int] = deque(maxlen=window_size)
        self._episodes = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if not episode:
                continue
            self._episodes += 1
            reward = float(episode["r"])
            length = int(episode["l"])
            outcome = info.get("game_outcome", "unknown")
            self._outcomes.append(outcome)
            self._rewards.append(reward)
            self._lengths.append(length)
            self.artifacts.record_episode(
                timesteps=self.num_timesteps,
                episode=self._episodes,
                reward=reward,
                length=length,
                outcome=outcome,
            )
            if self._episodes % self.log_interval_episodes == 0:
                self._log_window()
        return True

    def _on_training_end(self) -> None:
        if self._episodes:
            self._log_window()

    def _log_window(self) -> None:
        count = len(self._outcomes)
        if not count:
            return
        win_rate = self._outcomes.count("win") / count
        loss_rate = self._outcomes.count("loss") / count
        draw_rate = self._outcomes.count("draw") / count
        mean_reward = float(np.mean(self._rewards))
        mean_length = float(np.mean(self._lengths))
        self.logger.record("game/win_rate", win_rate)
        self.logger.record("game/loss_rate", loss_rate)
        self.logger.record("game/draw_rate", draw_rate)
        self.logger.record("game/ep_reward_mean", mean_reward)
        self.logger.record("game/ep_length_mean", mean_length)
        self.logger.record("game/total_episodes", self._episodes)
        if self.verbose:
            print(
                f"[{self.num_timesteps}/{self.target_timesteps} steps | ep {self._episodes}] "
                f"win={win_rate:.1%} loss={loss_rate:.1%} draw={draw_rate:.1%} "
                f"reward={mean_reward:.2f} len={mean_length:.1f}"
            )


class PeriodicCheckpointCallback(BaseCallback):
    """Write model checkpoints at absolute timestep intervals."""

    def __init__(self, artifacts: RunArtifacts, frequency: int, verbose: int = 1):
        super().__init__(verbose)
        self.artifacts = artifacts
        self.frequency = int(frequency)
        self._next_timestep = 0

    def _on_training_start(self) -> None:
        current = int(self.model.num_timesteps)
        self._next_timestep = ((current // self.frequency) + 1) * self.frequency

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_timestep:
            return True
        path = self.artifacts.save_checkpoint(
            self.model, timesteps=self.num_timesteps, reason="periodic"
        )
        while self._next_timestep <= self.num_timesteps:
            self._next_timestep += self.frequency
        if self.verbose:
            print(f"Checkpoint saved: {path}")
        return True


class PeriodicEvaluationCallback(BaseCallback):
    """Run deterministic, action-masked evaluation at timestep intervals."""

    def __init__(
        self,
        artifacts: RunArtifacts,
        eval_env: gym.Env,
        *,
        frequency: int,
        episodes: int,
        base_seed: int,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.artifacts = artifacts
        self.eval_env = eval_env
        self.frequency = int(frequency)
        self.episodes = int(episodes)
        self.base_seed = int(base_seed)
        self._next_timestep = 0
        self._evaluation_index = 0

    def _on_training_start(self) -> None:
        current = int(self.model.num_timesteps)
        self._next_timestep = ((current // self.frequency) + 1) * self.frequency

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_timestep:
            return True
        seed = self.base_seed + self._evaluation_index * self.episodes
        result = evaluate_maskable_policy(
            self.model, self.eval_env, n_episodes=self.episodes, base_seed=seed
        )
        self.artifacts.record_evaluation(timesteps=self.num_timesteps, result=result)
        self.logger.record("eval/mean_reward", result.mean_reward)
        self.logger.record("eval/std_reward", result.std_reward)
        self.logger.record("eval/mean_ep_length", result.mean_length)
        self.logger.record("eval/win_rate", result.win_rate)
        while self._next_timestep <= self.num_timesteps:
            self._next_timestep += self.frequency
        self._evaluation_index += 1
        if self.verbose:
            print(
                f"Evaluation at {self.num_timesteps}: reward={result.mean_reward:.3f} "
                f"+/- {result.std_reward:.3f}, win={result.win_rate:.1%}"
            )
        return True


class StopRequestedCallback(BaseCallback):
    """Stop learning at the next safe callback boundary after a signal."""

    def __init__(self, should_stop: Callable[[], bool]):
        super().__init__(verbose=0)
        self.should_stop = should_stop

    def _on_step(self) -> bool:
        return not self.should_stop()


class TrainingCommandChannel:
    """Durable, file-backed command queue for a single training run.

    Producers atomically place JSON requests in ``control/requests``.  The
    trainer claims them with ``os.replace`` before executing them and writes an
    atomic response in ``control/responses``.  Both a dashboard restarted after
    a crash and a trainer resumed after a power-off can therefore recover the
    last known command state without a network control socket.
    """

    _ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,95}$")
    _COMMAND_TYPES = frozenset({"checkpoint", "pause"})

    def __init__(self, run_dir: str | os.PathLike[str]):
        self.root = Path(run_dir).expanduser().resolve() / "control"
        self.requests_dir = self.root / "requests"
        self.processing_dir = self.root / "processing"
        self.responses_dir = self.root / "responses"
        for directory in (
            self.requests_dir,
            self.processing_dir,
            self.responses_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_command(cls, command: dict[str, Any]) -> tuple[str, str]:
        command_id = command.get("id")
        command_type = command.get("type")
        if not isinstance(command_id, str) or not cls._ID_PATTERN.fullmatch(command_id):
            raise ValueError("command id must contain 8-96 safe ASCII characters")
        if command_type not in cls._COMMAND_TYPES:
            raise ValueError(f"unsupported training command: {command_type!r}")
        return command_id, str(command_type)

    def submit(self, command: dict[str, Any]) -> Path:
        """Atomically enqueue one validated command and return its request path."""
        command_id, command_type = self.validate_command(command)
        response = self.response_path(command_id)
        if response.exists():
            raise FileExistsError(f"command already completed: {command_id}")
        if self._command_path(self.processing_dir, command_id).exists():
            raise FileExistsError(f"command already processing: {command_id}")
        destination = self._command_path(self.requests_dir, command_id)
        if destination.exists():
            raise FileExistsError(f"command already queued: {command_id}")
        payload = {
            "schemaVersion": SCHEMA_VERSION,
            "id": command_id,
            "type": command_type,
            "createdAtUtc": command.get("createdAtUtc") or utc_now(),
        }
        temporary = self.requests_dir / f".{command_id}.{os.getpid()}.tmp"
        _atomic_write_json(temporary, payload)
        # Never overwrite an existing request created by another manager.
        try:
            # A hard link publishes the already-fsynced inode atomically and,
            # unlike rename on POSIX, never overwrites a competing request.
            os.link(temporary, destination)
        except FileExistsError:
            raise
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def claim_next(self) -> tuple[Path, dict[str, Any]] | None:
        """Claim the oldest request, recovering an interrupted claim first."""
        processing = sorted(self.processing_dir.glob("*.json"))
        candidates = processing + sorted(self.requests_dir.glob("*.json"))
        for source in candidates:
            command_id = source.stem
            if not self._ID_PATTERN.fullmatch(command_id):
                source.unlink(missing_ok=True)
                continue
            response = self.response_path(command_id)
            if response.exists():
                source.unlink(missing_ok=True)
                continue
            claimed = self._command_path(self.processing_dir, command_id)
            if source.parent == self.requests_dir:
                try:
                    # A dashboard reading the request it just queued can hold it
                    # open; leaving the command queued for the next poll is the
                    # correct degradation, so tolerate a persistent lock.
                    if not replace_with_retry(source, claimed, tolerate_failure=True):
                        continue
                except FileNotFoundError:
                    continue
            try:
                command = _read_json(claimed)
                actual_id, _ = self.validate_command(command)
                if actual_id != command_id:
                    raise ValueError("command filename and payload id differ")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.complete(
                    claimed,
                    {
                        "schemaVersion": SCHEMA_VERSION,
                        "id": command_id,
                        "status": "failed",
                        "completedAtUtc": utc_now(),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                continue
            return claimed, command
        return None

    def complete(self, claimed_path: Path, response: dict[str, Any]) -> Path:
        command_id = claimed_path.stem
        if not self._ID_PATTERN.fullmatch(command_id):
            raise ValueError("unsafe claimed command filename")
        destination = self.response_path(command_id)
        _atomic_write_json(destination, response)
        claimed_path.unlink(missing_ok=True)
        return destination

    def response_path(self, command_id: str) -> Path:
        if not self._ID_PATTERN.fullmatch(command_id):
            raise ValueError("unsafe command id")
        return self._command_path(self.responses_dir, command_id)

    @staticmethod
    def _command_path(directory: Path, command_id: str) -> Path:
        return directory / f"{command_id}.json"


class TrainingControlCallback(BaseCallback):
    """Execute dashboard checkpoint/pause commands at a safe PPO boundary."""

    def __init__(
        self,
        artifacts: RunArtifacts,
        channel: TrainingCommandChannel,
        *,
        poll_steps: int = 8,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.artifacts = artifacts
        self.channel = channel
        self.poll_steps = max(1, int(poll_steps))
        self.stop_requested = False
        self.pause_checkpoint: Path | None = None

    def _on_step(self) -> bool:
        if self.n_calls % self.poll_steps != 0:
            return not self.stop_requested
        while not self.stop_requested:
            claimed = self.channel.claim_next()
            if claimed is None:
                break
            claimed_path, command = claimed
            command_id, command_type = self.channel.validate_command(command)
            try:
                reason = "paused" if command_type == "pause" else "manual"
                checkpoint = self.artifacts.save_checkpoint(
                    self.model,
                    timesteps=int(self.num_timesteps),
                    reason=f"{reason}-{command_id[-12:]}",
                )
                response = {
                    "schemaVersion": SCHEMA_VERSION,
                    "id": command_id,
                    "type": command_type,
                    "status": "completed",
                    "completedAtUtc": utc_now(),
                    "timesteps": int(self.num_timesteps),
                    "checkpoint": str(checkpoint),
                }
                self.channel.complete(claimed_path, response)
                if self.verbose:
                    print(f"Dashboard command {command_type!r} completed: {checkpoint}")
                if command_type == "pause":
                    self.stop_requested = True
                    self.pause_checkpoint = checkpoint
            except Exception as exc:
                self.channel.complete(
                    claimed_path,
                    {
                        "schemaVersion": SCHEMA_VERSION,
                        "id": command_id,
                        "type": command_type,
                        "status": "failed",
                        "completedAtUtc": utc_now(),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                if self.verbose:
                    print(f"Dashboard command {command_type!r} failed: {exc}")
        return not self.stop_requested


class SignalController:
    """Turn SIGINT/SIGTERM into a cooperative stop request."""

    def __init__(self):
        self.requested = False
        self.signal_name: str | None = None
        self._previous: dict[int, Any] = {}

    def install(self) -> None:
        signal_numbers = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGBREAK"):
            signal_numbers.append(signal.SIGBREAK)
        for signal_number in signal_numbers:
            self._previous[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, self._handle)

    def restore(self) -> None:
        for signal_number, previous in self._previous.items():
            signal.signal(signal_number, previous)
        self._previous.clear()

    def _handle(self, signal_number: int, _frame: Any) -> None:
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        try:
            self.signal_name = signal.Signals(signal_number).name
        except ValueError:
            self.signal_name = str(signal_number)
        print(
            f"\n{self.signal_name} received; stopping at the next safe step. "
            "Send the signal again to interrupt immediately."
        )


def _critical_config_mismatches(
    previous: dict[str, Any], current: dict[str, Any]
) -> list[tuple[str, Any, Any]]:
    return [
        (field, previous.get(field), current.get(field))
        for field in _RESUME_CRITICAL_FIELDS
        if previous.get(field) != current.get(field)
    ]


def _saved_model_path(base: Path) -> Path:
    if base.is_file():
        return base
    zipped = Path(f"{base}.zip")
    if zipped.is_file():
        return zipped
    raise FileNotFoundError(f"model.save() did not create {base} or {zipped}")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write_json(
    path: Path, value: dict[str, Any], *, tolerate_failure: bool = False
) -> bool:
    """Publish *value* at *path*, waiting out dashboard readers on Windows.

    Set *tolerate_failure* for bookkeeping the run can lose: a locked
    ``metrics.json`` must never terminate a multi-day training job.
    """
    return atomic_write_json(path, value, tolerate_failure=tolerate_failure)
