# SPDX-License-Identifier: Apache-2.0
"""Deterministic, browser-friendly replay generation for NextoCR runs.

The training bridge stays exclusively owned by the trainer.  A caller (the
local manager or the CLI) must provide a separate loopback ZMQ endpoint.  One
episode is evaluated from an immutable SB3 checkpoint and reduced to a compact
JSON document containing only abstract arena state; no game artwork is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Callable
from urllib.parse import urlparse
import uuid

import numpy as np

from nextocr_fsio import replace_with_retry

from crforge_gym.decks import get_deck_profile
from crforge_gym.env import ARENA_HEIGHT, ARENA_WIDTH, PLACEMENT_ZONES, CRForgeEnv
from crforge_gym.observation_preprocessing import (
    PreprocessedPolicyAdapter,
    StaticObservationPreprocessingWrapper,
    validate_model_preprocessing,
)
from crforge_gym.opponents import RuleBasedOpponent, SelfPlayOpponent
from crforge_gym.training_runtime import (
    resolve_resume_checkpoint,
    validate_exact_action_space,
)
from crforge_gym.wrappers import ExactDiscreteActionWrapper, FlattenedObsWrapper


REPLAY_SCHEMA_VERSION = 1
REPLAY_SOURCE = "nextocr_offline_simulator"
_REPLAY_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STEP_IN_CHECKPOINT_PATTERN = re.compile(r"step_(\d+)")
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


@dataclass(frozen=True)
class ReplayRequest:
    """Inputs required to evaluate and persist one replay."""

    run_dir: str | os.PathLike[str]
    endpoint: str
    checkpoint: str | os.PathLike[str] | None = None
    seed: int | None = None
    opponent: str = "rule_based"
    opponent_id: str | None = None
    replay_id: str | None = None
    max_steps: int = 2_000


@dataclass(frozen=True)
class ReplaySummary:
    """Small response returned to the manager after the full replay is saved."""

    replay_id: str
    path: str
    run_id: str
    checkpoint_timesteps: int
    current_timesteps: int
    seed: int
    opponent: dict[str, Any]
    outcome: str
    total_reward: float
    steps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "replayId": self.replay_id,
            "path": self.path,
            "runId": self.run_id,
            "checkpointTimesteps": self.checkpoint_timesteps,
            "currentTimesteps": self.current_timesteps,
            "seed": self.seed,
            "opponent": self.opponent,
            "outcome": self.outcome,
            "totalReward": self.total_reward,
            "steps": self.steps,
        }


class _RecordingOpponent:
    """Capture the exact red deployment submitted by an opponent policy."""

    def __init__(self, delegate: Any):
        self.delegate = delegate
        self.last_action: dict[str, Any] | None = None

    def act(
        self,
        obs_raw: dict[str, Any] | None,
        player: str = "red",
        obs_flat: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        action = self.delegate.act(obs_raw, player=player, obs_flat=obs_flat)
        self.last_action = dict(action) if action is not None else None
        return action

    def take_last_action(self) -> dict[str, Any] | None:
        action = self.last_action
        self.last_action = None
        return action


class _DeterministicModelAdapter:
    """Force deterministic inference even when SelfPlayOpponent asks otherwise."""

    def __init__(self, model: Any):
        self.model = model
        self.action_space = model.action_space

    def predict(self, observation: np.ndarray, *args: Any, **kwargs: Any):
        kwargs["deterministic"] = True
        return self.model.predict(observation, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


def generate_replay(
    request: ReplayRequest,
    *,
    model_loader: Callable[[Path], Any] | None = None,
) -> ReplaySummary:
    """Generate one deterministic evaluation episode and atomically save it.

    ``endpoint`` must be loopback-only and must not be the run's training
    endpoint.  While a run is active its configured periodic-evaluation endpoint
    is reserved too, ensuring this feature cannot stall or perturb training.
    The manager is responsible for launching a short-lived bridge on a third
    port before calling this function.
    """

    run_dir = Path(request.run_dir).expanduser().resolve()
    config = _read_required_json(run_dir / "config.json")
    manifest = _read_required_json(run_dir / "manifest.json")
    _validate_request(request, run_dir, config, manifest)

    checkpoint = _resolve_checkpoint(run_dir, request.checkpoint)
    loader = model_loader or _load_maskable_ppo
    model = loader(checkpoint)
    validate_exact_action_space(model.action_space)
    validate_model_preprocessing(model)

    checkpoint_timesteps = _checkpoint_timesteps(
        run_dir,
        checkpoint,
        model_timesteps=int(getattr(model, "num_timesteps", 0)),
    )
    observed_timesteps = _observed_training_timesteps(run_dir, checkpoint_timesteps)
    seed = int(request.seed if request.seed is not None else config.get("eval_seed", 10_000_042))
    ticks_per_step = int(config.get("ticks_per_step", 15))
    if ticks_per_step <= 0:
        raise ValueError("run config ticks_per_step must be positive")

    opponent, opponent_metadata = _make_opponent(
        run_dir,
        kind=request.opponent,
        opponent_id=request.opponent_id,
        seed=seed,
        model_loader=loader,
    )
    recording_opponent = _RecordingOpponent(opponent)
    deck_profile = get_deck_profile(str(config.get("deck_profile", "mortar_self_play_v1")))
    deck = list(deck_profile.simulator_card_ids)

    base_env = CRForgeEnv(
        endpoint=request.endpoint,
        blue_deck=deck,
        red_deck=deck,
        level=int(config.get("level", 11)),
        ticks_per_step=ticks_per_step,
        opponent=recording_opponent,
        binary_obs=False,
    )
    env = ExactDiscreteActionWrapper(
        StaticObservationPreprocessingWrapper(FlattenedObsWrapper(base_env))
    )

    frames: list[dict[str, Any]] = []
    total_reward = 0.0
    steps = 0
    outcome = "unknown"
    try:
        observation, _ = env.reset(seed=seed)
        initial_raw = _require_raw_observation(base_env)
        frames.append(_frame_payload(initial_raw, step=0, reward=0.0))

        terminated = False
        truncated = False
        while not (terminated or truncated):
            if steps >= request.max_steps:
                raise RuntimeError(
                    f"evaluation exceeded max_steps={request.max_steps}; "
                    "the simulator did not return a terminal transition"
                )

            before = _require_raw_observation(base_env)
            action_masks = env.action_masks()
            policy_action, _ = model.predict(
                observation,
                deterministic=True,
                action_masks=action_masks,
            )
            action_id = int(np.asarray(policy_action).reshape(-1)[0])
            blue_action = _blue_action_payload(action_id, before)

            observation, reward, terminated, truncated, info = env.step(policy_action)
            steps += 1
            total_reward += float(reward)
            raw = _require_raw_observation(base_env)
            red_engine_action = recording_opponent.take_last_action()
            red_action = _engine_action_payload("red", red_engine_action, before)
            if blue_action is not None:
                blue_action["accepted"] = not bool(info.get("action_failed", False))

            frames.append(
                _frame_payload(
                    raw,
                    step=steps,
                    reward=float(reward),
                    blue_action=blue_action,
                    red_action=red_action,
                )
            )
            outcome = str(info.get("game_outcome", "unknown"))

        if outcome not in {"win", "loss", "draw"}:
            raise RuntimeError(f"terminal replay has no canonical outcome: {outcome!r}")
    finally:
        env.close()

    replay_id = _validated_replay_id(request.replay_id or _new_replay_id(checkpoint_timesteps, seed))
    replay_dir = run_dir / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    destination = replay_dir / f"{replay_id}.json"
    if destination.exists():
        raise FileExistsError(f"replay already exists: {destination}")

    run_id = str(manifest.get("runId") or run_dir.name)
    target_timesteps = _target_timesteps(config, manifest)
    checkpoint_relative = str(checkpoint.relative_to(run_dir))
    document: dict[str, Any] = {
        "schemaVersion": REPLAY_SCHEMA_VERSION,
        "replayId": replay_id,
        "createdAtUtc": _utc_now(),
        "source": REPLAY_SOURCE,
        "arena": {
            "width": ARENA_WIDTH,
            "height": ARENA_HEIGHT,
            "ticksPerSecond": 20,
        },
        "trainingLevel": {
            "runId": run_id,
            "runStatus": str(manifest.get("status", "unknown")),
            "currentTimesteps": observed_timesteps,
            "targetTimesteps": target_timesteps,
            "progress": (
                min(1.0, observed_timesteps / target_timesteps)
                if target_timesteps and target_timesteps > 0
                else None
            ),
            "checkpointTimesteps": checkpoint_timesteps,
            "checkpointLagTimesteps": max(0, observed_timesteps - checkpoint_timesteps),
            "checkpoint": {
                "path": checkpoint_relative,
                "step": checkpoint_timesteps,
                "numTimesteps": int(getattr(model, "num_timesteps", checkpoint_timesteps)),
                "reason": _checkpoint_reason(run_dir, checkpoint, checkpoint_timesteps),
            },
        },
        "evaluation": {
            "seed": seed,
            "deterministic": True,
            "opponent": opponent_metadata,
            "outcome": outcome,
            "totalReward": round(total_reward, 6),
            "steps": steps,
            "ticksPerStep": ticks_per_step,
        },
        "decks": {
            "profileId": deck_profile.profile_id,
            "blue": deck,
            "red": deck,
        },
        "frames": frames,
    }
    _atomic_write_json(destination, document)

    return ReplaySummary(
        replay_id=replay_id,
        path=str(destination),
        run_id=run_id,
        checkpoint_timesteps=checkpoint_timesteps,
        current_timesteps=observed_timesteps,
        seed=seed,
        opponent=opponent_metadata,
        outcome=outcome,
        total_reward=round(total_reward, 6),
        steps=steps,
    )


def list_replays(run_dir: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Return newest-first replay summaries without loading frame arrays."""

    root = Path(run_dir).expanduser().resolve()
    _read_required_json(root / "manifest.json")
    replay_dir = root / "replays"
    if not replay_dir.is_dir():
        return []

    results: list[dict[str, Any]] = []
    for path in replay_dir.glob("*.json"):
        try:
            data = _read_required_json(path)
            evaluation = data["evaluation"]
            training = data["trainingLevel"]
            results.append(
                {
                    "replayId": data["replayId"],
                    "createdAtUtc": data["createdAtUtc"],
                    "runId": training["runId"],
                    "checkpointTimesteps": training["checkpointTimesteps"],
                    "currentTimesteps": training["currentTimesteps"],
                    "seed": evaluation["seed"],
                    "opponent": evaluation["opponent"],
                    "outcome": evaluation["outcome"],
                    "totalReward": evaluation["totalReward"],
                    "steps": evaluation["steps"],
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            # A partially copied or foreign JSON file is not a valid replay.
            continue
    return sorted(results, key=lambda item: item["createdAtUtc"], reverse=True)


def load_replay(run_dir: str | os.PathLike[str], replay_id: str) -> dict[str, Any]:
    """Load one replay by validated opaque ID (never by caller-supplied path)."""

    root = Path(run_dir).expanduser().resolve()
    _read_required_json(root / "manifest.json")
    safe_id = _validated_replay_id(replay_id)
    path = root / "replays" / f"{safe_id}.json"
    data = _read_required_json(path)
    if data.get("replayId") != safe_id or data.get("schemaVersion") != REPLAY_SCHEMA_VERSION:
        raise ValueError(f"invalid replay document: {path}")
    return data


def _validate_request(
    request: ReplayRequest,
    run_dir: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    _validate_loopback_endpoint(request.endpoint)
    if request.opponent not in {"rule_based", "league"}:
        raise ValueError("opponent must be 'rule_based' or 'league'")
    if request.opponent != "league" and request.opponent_id is not None:
        raise ValueError("opponent_id is only valid with opponent='league'")
    if request.max_steps <= 0 or request.max_steps > 20_000:
        raise ValueError("max_steps must be between 1 and 20000")
    if request.replay_id is not None:
        _validated_replay_id(request.replay_id)

    endpoint = _canonical_endpoint(request.endpoint)
    training_endpoint = config.get("endpoint")
    if training_endpoint and endpoint == _canonical_endpoint(str(training_endpoint)):
        raise ValueError("replay endpoint must not be the run's training endpoint")
    eval_endpoint = config.get("eval_endpoint")
    if (
        str(manifest.get("status", "")).lower() == "running"
        and eval_endpoint
        and endpoint == _canonical_endpoint(str(eval_endpoint))
    ):
        raise ValueError(
            "active run's periodic-evaluation endpoint is reserved; "
            "launch a dedicated replay bridge on another loopback port"
        )


def _make_opponent(
    run_dir: Path,
    *,
    kind: str,
    opponent_id: str | None,
    seed: int,
    model_loader: Callable[[Path], Any],
) -> tuple[Any, dict[str, Any]]:
    if kind == "rule_based":
        return RuleBasedOpponent(rng=np.random.default_rng(seed)), {"type": "rule_based"}

    entry, checkpoint = _resolve_league_opponent(run_dir, opponent_id)
    model = model_loader(checkpoint)
    validate_exact_action_space(model.action_space)
    validate_model_preprocessing(model)
    policy = _DeterministicModelAdapter(PreprocessedPolicyAdapter(model))
    return SelfPlayOpponent(policy), {
        "type": "league",
        "opponentId": entry["opponent_id"],
        "category": entry["category"],
        "checkpointStep": int(entry["checkpoint_step"]),
    }


def _resolve_league_opponent(
    run_dir: Path, opponent_id: str | None
) -> tuple[dict[str, Any], Path]:
    league_root = (run_dir / "league").resolve()
    state = _read_required_json(league_root / "league.json")
    entries: list[dict[str, Any]] = []
    if state.get("initial"):
        entries.append(state["initial"])
    entries.extend(state.get("recent", []))
    entries.extend(state.get("historical", []))
    if not entries:
        raise FileNotFoundError(f"self-play league is empty: {league_root}")

    if opponent_id is None:
        entry = max(
            entries,
            key=lambda item: (int(item["checkpoint_step"]), int(item["created_index"])),
        )
    else:
        matches = [item for item in entries if item.get("opponent_id") == opponent_id]
        if not matches:
            raise ValueError(f"league opponent is not active: {opponent_id}")
        entry = matches[0]

    checkpoint = (league_root / str(entry["path"])).resolve()
    if not checkpoint.is_relative_to(league_root) or not checkpoint.is_file():
        raise FileNotFoundError("league manifest points outside its root or to a missing file")
    return entry, checkpoint


def _resolve_checkpoint(
    run_dir: Path, requested: str | os.PathLike[str] | None
) -> Path:
    if requested is not None:
        candidate = Path(requested).expanduser()
        if not candidate.is_absolute():
            candidate = run_dir / candidate
        checkpoint = candidate.resolve()
        if not checkpoint.is_file() and checkpoint.suffix.lower() != ".zip":
            checkpoint = Path(f"{checkpoint}.zip")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    else:
        try:
            checkpoint = resolve_resume_checkpoint(run_dir)
        except FileNotFoundError:
            _, checkpoint = _resolve_league_opponent(run_dir, None)

    checkpoint = checkpoint.resolve()
    if not checkpoint.is_relative_to(run_dir):
        raise ValueError("checkpoint must be contained in the selected run directory")
    return checkpoint


def _checkpoint_timesteps(
    run_dir: Path, checkpoint: Path, *, model_timesteps: int
) -> int:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = _read_required_json(metrics_path)
        for entry in reversed(metrics.get("checkpoints", [])):
            path = (run_dir / str(entry.get("path", ""))).resolve()
            same_latest = checkpoint.name == "latest.zip" and path.name.startswith("step_")
            if path == checkpoint or same_latest:
                return int(entry.get("timesteps", model_timesteps))

    match = _STEP_IN_CHECKPOINT_PATTERN.search(checkpoint.name)
    return int(match.group(1)) if match else max(0, int(model_timesteps))


def _observed_training_timesteps(run_dir: Path, checkpoint_timesteps: int) -> int:
    values = [int(checkpoint_timesteps)]
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = _read_required_json(manifest_path)
        values.append(int(manifest.get("finalTimesteps", 0) or 0))
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = _read_required_json(metrics_path)
        training = metrics.get("training", {})
        values.append(int(training.get("lastTimesteps", 0) or 0))
        last_selection = metrics.get("leagueSelections", {}).get("last") or {}
        values.append(int(last_selection.get("timesteps", 0) or 0))
        values.extend(int(item.get("timesteps", 0) or 0) for item in metrics.get("checkpoints", []))
        values.extend(int(item.get("timesteps", 0) or 0) for item in metrics.get("evaluations", []))
    return max(values)


def _target_timesteps(config: dict[str, Any], manifest: dict[str, Any]) -> int | None:
    additional = config.get("timesteps_this_attempt")
    if additional is None:
        return None
    attempts = manifest.get("attempts", [])
    start = int((attempts[-1] if attempts else {}).get("startTimesteps", 0) or 0)
    target = start + int(additional)
    return target if target > 0 else None


def _checkpoint_reason(run_dir: Path, checkpoint: Path, step: int) -> str:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        metrics = _read_required_json(metrics_path)
        for entry in reversed(metrics.get("checkpoints", [])):
            if int(entry.get("timesteps", -1)) == step:
                return str(entry.get("reason", "checkpoint"))
    if checkpoint.parent.name == "league":
        return "league_snapshot"
    if checkpoint.name == "final_model.zip":
        return "final"
    return "checkpoint"


def _blue_action_payload(action_id: int, raw_before: dict[str, Any]) -> dict[str, Any]:
    if action_id == 0:
        return {"kind": "noop", "policyAction": 0}
    decoded = ExactDiscreteActionWrapper.decode(action_id)
    hand_index = int(decoded[1])
    zone = int(decoded[2])
    x, y = PLACEMENT_ZONES[zone]
    payload = _engine_action_payload(
        "blue",
        {"handIndex": hand_index, "x": x, "y": y},
        raw_before,
    )
    if payload is None:  # pragma: no cover - action ID guarantees a deployment
        raise RuntimeError("failed to decode a non-noop blue action")
    payload["policyAction"] = action_id
    payload["zone"] = zone
    return payload


def _engine_action_payload(
    player: str,
    action: dict[str, Any] | None,
    raw_before: dict[str, Any],
) -> dict[str, Any]:
    if action is None:
        return {"kind": "noop"}
    hand_index = int(action["handIndex"])
    player_key = "bluePlayer" if player == "blue" else "redPlayer"
    hand = raw_before.get(player_key, {}).get("hand", [])
    card = hand[hand_index] if 0 <= hand_index < len(hand) else {}
    return {
        "kind": "play",
        "handIndex": hand_index,
        "cardId": card.get("id"),
        "cardName": card.get("name"),
        "cardType": card.get("type"),
        "x": round(float(action["x"]), 3),
        "y": round(float(action["y"]), 3),
    }


def _frame_payload(
    raw: dict[str, Any],
    *,
    step: int,
    reward: float,
    blue_action: dict[str, Any] | None = None,
    red_action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blue = raw.get("bluePlayer", {})
    red = raw.get("redPlayer", {})
    towers: list[dict[str, Any]] = []
    for team, player in (("BLUE", blue), ("RED", red)):
        for tower in player.get("towers", []):
            towers.append(
                {
                    "id": int(tower.get("id", 0)),
                    "team": team,
                    "type": str(tower.get("type", "tower")),
                    "x": round(float(tower.get("x", 0.0)), 3),
                    "y": round(float(tower.get("y", 0.0)), 3),
                    "hp": int(tower.get("hp", 0)),
                    "maxHp": int(tower.get("maxHp", 0)),
                    "alive": bool(tower.get("alive", False)),
                }
            )

    entities: list[dict[str, Any]] = []
    for entity in raw.get("entities", []):
        if str(entity.get("entityType", "")).upper() == "TOWER":
            continue
        statuses = [
            name
            for name in ("stunned", "slowed", "raged", "frozen", "poisoned")
            if bool(entity.get(name, False))
        ]
        entities.append(
            {
                "id": int(entity.get("id", 0)),
                "name": str(entity.get("name", "unknown")),
                "team": str(entity.get("team", "NEUTRAL")),
                "type": str(entity.get("entityType", "UNKNOWN")),
                "movement": str(entity.get("movementType", "UNKNOWN")),
                "x": round(float(entity.get("x", 0.0)), 3),
                "y": round(float(entity.get("y", 0.0)), 3),
                "hp": int(entity.get("hp", 0)),
                "maxHp": int(entity.get("maxHp", 0)),
                "shield": int(entity.get("shield", 0)),
                "attackReady": round(float(entity.get("attackCooldownFraction", 0.0)), 3),
                "attacking": bool(entity.get("isAttacking", False)),
                "hasTarget": bool(entity.get("hasTarget", False)),
                "statuses": statuses,
                "lifetime": round(float(entity.get("lifetimeFraction", 0.0)), 3),
            }
        )

    return {
        "step": int(step),
        "frame": int(raw.get("frame", 0)),
        "time": round(float(raw.get("gameTimeSeconds", 0.0)), 3),
        "overtime": bool(raw.get("isOvertime", False)),
        "elixirMultiplier": int(raw.get("elixirMultiplier", 1)),
        "players": {
            "blue": _player_payload(blue),
            "red": _player_payload(red),
        },
        "towers": towers,
        "entities": entities,
        "blueAction": blue_action,
        "redAction": red_action,
        "reward": round(float(reward), 6),
    }


def _player_payload(player: dict[str, Any]) -> dict[str, Any]:
    return {
        "elixir": round(float(player.get("elixir", 0.0)), 3),
        "crowns": int(player.get("crowns", 0)),
        "hand": [
            {
                "id": card.get("id"),
                "name": card.get("name"),
                "type": card.get("type"),
                "cost": int(card.get("cost", 0)),
            }
            for card in player.get("hand", [])[:4]
        ],
    }


def _require_raw_observation(base_env: CRForgeEnv) -> dict[str, Any]:
    raw = base_env._last_obs_raw
    if not isinstance(raw, dict):
        raise RuntimeError("replay generation requires the bridge JSON observation mode")
    return raw


def _load_maskable_ppo(checkpoint: Path) -> Any:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:  # pragma: no cover - exercised by CLI environments
        raise RuntimeError('install replay dependencies with: pip install -e "python[train]"') from exc
    return MaskablePPO.load(str(checkpoint), device="cpu")


def _validate_loopback_endpoint(endpoint: str) -> None:
    parsed = urlparse(endpoint)
    if parsed.scheme != "tcp" or parsed.hostname not in _LOOPBACK_HOSTS or parsed.port is None:
        raise ValueError("replay endpoint must be tcp://127.0.0.1:PORT (loopback only)")


def _canonical_endpoint(endpoint: str) -> tuple[str, int]:
    _validate_loopback_endpoint(endpoint)
    parsed = urlparse(endpoint)
    host = "127.0.0.1" if parsed.hostname in _LOOPBACK_HOSTS else str(parsed.hostname)
    return host, int(parsed.port or 0)


def _validated_replay_id(replay_id: str) -> str:
    if not _REPLAY_ID_PATTERN.fullmatch(replay_id):
        raise ValueError(
            "replay_id must be 1-128 ASCII letters, digits, dots, underscores, or hyphens"
        )
    return replay_id


def _new_replay_id(checkpoint_timesteps: int, seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        f"replay-step-{checkpoint_timesteps:012d}-seed-{seed}-{timestamp}-"
        f"{uuid.uuid4().hex[:8]}"
    )


def _read_required_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
