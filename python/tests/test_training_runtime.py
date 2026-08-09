"""Fast tests for durable/reproducible PPO run infrastructure."""

from __future__ import annotations

import json
from pathlib import Path
import random
import signal

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
from stable_baselines3.common.monitor import Monitor

from crforge_gym.training_runtime import (
    ACTION_SCHEMA,
    DeterministicEpisodeSeedWrapper,
    EvaluationResult,
    RunArtifacts,
    SignalController,
    TrainingCommandChannel,
    TrainingControlCallback,
    derive_episode_seed,
    evaluate_maskable_policy,
    resolve_resume_checkpoint,
    seed_everything,
    validate_exact_action_space,
)


class _SeedCaptureEnv(gym.Env):
    observation_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    action_space = spaces.Discrete(2)

    def __init__(self):
        self.seeds: list[int | None] = []

    def reset(self, *, seed=None, options=None):
        self.seeds.append(seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, True, False, {}


class _SavedModel:
    def save(self, path: str) -> None:
        Path(f"{path}.zip").write_bytes(b"model")


def _config(seed: int = 7) -> dict:
    return {
        "action_schema": ACTION_SCHEMA,
        "observation_preprocessing": "nextocr_static_v1",
        "deck_profile": "mortar_self_play_v1",
        "deck": {"simulator_card_ids": ["card"] * 8, "revision": 1},
        "seed": seed,
        "eval_seed": seed + 10_000_000,
        "episode_seed_strategy": "base_plus_rank_stride_plus_checkpoint_timesteps_v1",
        "backend": "jpype",
        "num_envs": 1,
        "opponent": "rule_based",
        "ticks_per_step": 15,
        "n_steps": 32,
        "batch_size": 16,
        "n_epochs": 1,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.005,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "net_arch": [32],
        "league_max_recent": 2,
        "league_max_historical": 2,
        "league_model_cache": 1,
        "league_initial_weight": 0.1,
        "league_recent_weight": 0.6,
        "league_historical_weight": 0.3,
        "curriculum": {"stage": "mirror", "opponent_catalog_snapshot": None},
    }


def test_seed_everything_replays_python_numpy_and_torch() -> None:
    seed_everything(123)
    first = (random.random(), np.random.random(), torch.rand(1).item())
    seed_everything(123)
    second = (random.random(), np.random.random(), torch.rand(1).item())
    assert first == second


def test_episode_seed_wrapper_advances_and_can_restart_sequence() -> None:
    base = _SeedCaptureEnv()
    env = DeterministicEpisodeSeedWrapper(base, 20)

    env.reset()
    env.reset()
    env.reset(seed=90)
    env.reset()

    assert base.seeds == [20, 21, 90, 91]


def test_resume_seed_namespace_does_not_replay_prior_episode_seeds() -> None:
    first_attempt = derive_episode_seed(42, rank=0, checkpoint_timesteps=0)
    resumed_attempt = derive_episode_seed(42, rank=0, checkpoint_timesteps=10_000)
    other_rank = derive_episode_seed(42, rank=1, checkpoint_timesteps=10_000)

    assert resumed_attempt > first_attempt + 9_999
    assert other_rank > resumed_attempt


def test_legacy_action_space_is_rejected_clearly() -> None:
    validate_exact_action_space(spaces.Discrete(41))
    with pytest.raises(ValueError, match="legacy factorized MultiDiscrete"):
        validate_exact_action_space(spaces.MultiDiscrete([2, 4, 10]))


def test_run_artifacts_checkpoint_metrics_and_resume(tmp_path: Path) -> None:
    run = RunArtifacts(tmp_path / "run")
    run.initialize(_config(), resume_from=None, start_timesteps=0)
    run.record_episode(timesteps=12, episode=1, reward=3.5, length=8, outcome="win")
    run.record_league_selection(
        timesteps=12,
        episode=1,
        opponent_id="initial",
        category="initial",
        checkpoint_step=0,
    )
    run.record_evaluation(
        timesteps=12,
        result=EvaluationResult(2, 1.5, 0.5, 7.0, 0.5, 0.0, 0.5, 0),
    )
    checkpoint = run.save_checkpoint(_SavedModel(), timesteps=12, reason="periodic")
    run.finish("completed", timesteps=12, training_seconds=2.0, attempt_steps=12)

    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text("utf-8"))
    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text("utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["actionSchema"] == ACTION_SCHEMA
    assert manifest["action_schema"] == ACTION_SCHEMA
    assert manifest["observation_preprocessing"] == "nextocr_static_v1"
    assert manifest["deck_profile"] == "mortar_self_play_v1"
    assert manifest["curriculum"] == {
        "stage": "mirror",
        "opponent_catalog_snapshot": None,
    }
    assert metrics["training"]["wins"] == 1
    assert metrics["leagueSelections"]["byOpponent"] == {"initial": 1}
    assert resolve_resume_checkpoint(tmp_path / "run") == checkpoint.resolve()

    resumed = RunArtifacts(tmp_path / "run")
    resumed.initialize(_config(), resume_from=checkpoint, start_timesteps=12)
    recovered_manifest = json.loads((tmp_path / "run" / "manifest.json").read_text("utf-8"))
    # The completed attempt stays completed; recovery only touches a stale
    # attempt which still claims that its vanished process is running.
    assert recovered_manifest["attempts"][0]["status"] == "completed"
    with pytest.raises(ValueError, match="reproducibility-critical"):
        RunArtifacts(tmp_path / "run").initialize(
            _config(seed=8), resume_from=checkpoint, start_timesteps=12
        )
    changed_deck = _config()
    changed_deck["deck"] = {**changed_deck["deck"], "revision": 2}
    with pytest.raises(ValueError, match="deck"):
        RunArtifacts(tmp_path / "run").initialize(
            changed_deck, resume_from=checkpoint, start_timesteps=12
        )


class _MaskedEvalEnv(gym.Env):
    observation_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
    action_space = spaces.Discrete(41)

    def __init__(self):
        self.next_seed = 0
        self.seeds: list[int] = []
        self.steps = 0

    def set_seed_sequence(self, base_seed: int) -> None:
        self.next_seed = base_seed

    def reset(self, *, seed=None, options=None):
        actual = self.next_seed if seed is None else seed
        self.next_seed = int(actual) + 1
        self.seeds.append(int(actual))
        self.steps = 0
        return np.zeros(1, dtype=np.float32), {}

    def action_masks(self):
        mask = np.zeros(41, dtype=bool)
        mask[0] = True
        return mask

    def step(self, action):
        assert int(action) == 0
        self.steps += 1
        terminated = self.steps == 2
        info = {"game_outcome": "win"} if terminated else {}
        return np.zeros(1, dtype=np.float32), 1.0, terminated, False, info


class _MaskAwareModel:
    def __init__(self):
        self.saw_masks = False

    def predict(self, observations, *, action_masks=None, **kwargs):
        self.saw_masks = action_masks is not None
        return np.zeros(len(observations), dtype=np.int64), None


def test_evaluation_uses_maskable_evaluator_and_deterministic_seeds() -> None:
    raw_env = _MaskedEvalEnv()
    env = Monitor(raw_env)
    model = _MaskAwareModel()

    result = evaluate_maskable_policy(model, env, n_episodes=2, base_seed=700)

    assert model.saw_masks
    assert raw_env.seeds[:2] == [700, 701]
    assert result.mean_reward == 2.0
    assert result.mean_length == 2.0
    assert result.win_rate == 1.0


def test_signal_controller_requests_then_escalates() -> None:
    controller = SignalController()
    controller._handle(signal.SIGINT, None)
    assert controller.requested
    assert controller.signal_name == "SIGINT"
    with pytest.raises(KeyboardInterrupt):
        controller._handle(signal.SIGINT, None)


def test_training_command_channel_claims_and_acknowledges_atomically(tmp_path: Path) -> None:
    channel = TrainingCommandChannel(tmp_path / "run")
    command = {
        "id": "checkpoint-0001",
        "type": "checkpoint",
        "createdAtUtc": "2026-08-09T00:00:00+00:00",
    }

    request = channel.submit(command)
    assert request.is_file()
    claimed = channel.claim_next()
    assert claimed is not None
    claimed_path, payload = claimed
    assert claimed_path.parent.name == "processing"
    assert payload == {"schemaVersion": 1, **command}

    response_path = channel.complete(
        claimed_path,
        {"id": command["id"], "status": "completed", "timesteps": 12},
    )
    assert not claimed_path.exists()
    assert json.loads(response_path.read_text("utf-8"))["timesteps"] == 12
    assert channel.claim_next() is None
    with pytest.raises(FileExistsError, match="already completed"):
        channel.submit(command)


def test_training_command_channel_rejects_unsafe_or_unknown_commands(tmp_path: Path) -> None:
    channel = TrainingCommandChannel(tmp_path / "run")
    with pytest.raises(ValueError, match="command id"):
        channel.submit({"id": "../bad", "type": "pause"})
    with pytest.raises(ValueError, match="unsupported"):
        channel.submit({"id": "unknown-0001", "type": "shell"})


def test_resume_closes_a_stale_running_attempt(tmp_path: Path) -> None:
    run = RunArtifacts(tmp_path / "run")
    run.initialize(_config(), resume_from=None, start_timesteps=0)
    checkpoint = run.save_checkpoint(_SavedModel(), timesteps=100, reason="periodic")

    resumed = RunArtifacts(tmp_path / "run")
    resumed.initialize(_config(), resume_from=checkpoint, start_timesteps=100)

    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text("utf-8"))
    assert manifest["attempts"][0]["status"] == "interrupted"
    assert manifest["attempts"][0]["recoveredByResume"] is True
    assert manifest["attempts"][0]["finalTimesteps"] == 100
    assert manifest["attempts"][1]["status"] == "running"


def test_training_control_callback_saves_manual_checkpoint_and_pauses(tmp_path: Path) -> None:
    artifacts = RunArtifacts(tmp_path / "run")
    artifacts.initialize(_config(), resume_from=None, start_timesteps=0)
    channel = TrainingCommandChannel(tmp_path / "run")
    channel.submit({"id": "manual-command-01", "type": "checkpoint"})
    channel.submit({"id": "pause-command-001", "type": "pause"})
    callback = TrainingControlCallback(artifacts, channel, poll_steps=1, verbose=0)
    callback.model = _SavedModel()
    callback.num_timesteps = 24
    callback.n_calls = 1

    assert callback._on_step() is False
    assert callback.stop_requested
    assert callback.pause_checkpoint is not None
    manual = json.loads(
        channel.response_path("manual-command-01").read_text("utf-8")
    )
    paused = json.loads(
        channel.response_path("pause-command-001").read_text("utf-8")
    )
    assert manual["status"] == "completed"
    assert paused["status"] == "completed"
    assert manual["timesteps"] == paused["timesteps"] == 24
