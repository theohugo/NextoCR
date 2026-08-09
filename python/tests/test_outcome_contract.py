# Modified by NextoCR contributors; see NOTICE for attribution.
"""Canonical terminal-outcome protocol and Gym propagation tests."""

import struct

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from crforge_gym.bridge import _decode_binary_step
from crforge_gym.env import OBS_SIZE, CRForgeEnv, _blue_game_outcome
from crforge_gym.wrappers import EpisodeStatsWrapper


def _binary_packet(*, outcome_code: int | None, terminated: bool) -> bytes:
    header = struct.pack("<ffBBBB", 1.25, -1.25, terminated, False, True, False)
    observation = np.zeros(OBS_SIZE, dtype=np.float32).tobytes()
    trailer = b"" if outcome_code is None else struct.pack("<i", outcome_code)
    return header + observation + trailer


@pytest.mark.parametrize(
    ("outcome_code", "expected"),
    [(0, "ONGOING"), (1, "BLUE_WIN"), (2, "RED_WIN"), (3, "DRAW")],
)
def test_binary_outcome_trailer_is_append_only_and_decoded(outcome_code, expected):
    packet = _binary_packet(
        outcome_code=outcome_code,
        terminated=outcome_code != 0,
    )

    decoded = _decode_binary_step(packet)

    assert decoded[0].shape == (OBS_SIZE,)
    assert decoded[1:3] == (1.25, -1.25)
    assert decoded[5:7] == (True, False)
    assert decoded[7] == expected


def test_binary_step_without_trailer_is_supported_without_reward_inference():
    terminal = _decode_binary_step(_binary_packet(outcome_code=None, terminated=True))
    non_terminal = _decode_binary_step(
        _binary_packet(outcome_code=None, terminated=False)
    )

    assert terminal[7] == "UNKNOWN"
    assert non_terminal[7] == "ONGOING"


@pytest.mark.parametrize(
    ("canonical", "terminated", "expected"),
    [
        ("BLUE_WIN", True, "win"),
        ("RED_WIN", True, "loss"),
        ("DRAW", True, "draw"),
        ("ONGOING", False, "ongoing"),
        (None, True, "unknown"),
    ],
)
def test_blue_outcome_mapping_is_canonical(canonical, terminated, expected):
    assert _blue_game_outcome(canonical, terminated=terminated) == expected


class _BinaryStepClient:
    def __init__(self, outcome):
        self.outcome = outcome

    def step(self, **kwargs):
        terminal = self.outcome != "ONGOING"
        return (
            np.zeros(OBS_SIZE, dtype=np.float32),
            -999.0,
            999.0,
            terminal,
            False,
            False,
            False,
            self.outcome,
        )

    def close(self):
        pass


class _JsonStepClient:
    def __init__(self, outcome):
        self.outcome = outcome

    def step(self, **kwargs):
        return {
            "observation": {
                "schemaVersion": 2,
                "bluePlayer": {"hand": [], "towers": []},
                "redPlayer": {"hand": [], "towers": []},
                "entities": [],
            },
            "reward": {"blue": -999.0, "red": 999.0},
            "terminated": self.outcome != "ONGOING",
            "truncated": False,
            "outcome": self.outcome,
        }

    def close(self):
        pass


def _step_with_fake_client(*, binary: bool, outcome: str):
    env = CRForgeEnv(binary_obs=binary, opponent="noop")
    env._client = _BinaryStepClient(outcome) if binary else _JsonStepClient(outcome)
    env._connected = True
    try:
        return env.step(np.array([0, 0, 0], dtype=np.int64))
    finally:
        env.close()


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ("BLUE_WIN", "win"),
        ("RED_WIN", "loss"),
        ("DRAW", "draw"),
        ("ONGOING", "ongoing"),
    ],
)
def test_json_and_binary_envs_expose_same_outcome(outcome, expected):
    binary_step = _step_with_fake_client(binary=True, outcome=outcome)
    json_step = _step_with_fake_client(binary=False, outcome=outcome)

    assert binary_step[4]["game_outcome"] == expected
    assert json_step[4]["game_outcome"] == expected
    assert binary_step[2:4] == json_step[2:4]


class _TerminalInfoEnv(gym.Env):
    observation_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    action_space = spaces.Discrete(1)

    def __init__(self, *, reward: float, info: dict):
        self.reward = reward
        self.info = info

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, dtype=np.float32), {"game_outcome": "ongoing"}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), self.reward, True, False, dict(self.info)


def test_episode_stats_preserves_engine_outcome_even_when_reward_has_opposite_sign():
    env = EpisodeStatsWrapper(
        _TerminalInfoEnv(reward=-999.0, info={"game_outcome": "win"})
    )
    env.reset()

    _, _, _, _, info = env.step(0)

    assert info["game_outcome"] == "win"
    assert info["episode"]["r"] == -999.0


def test_episode_stats_marks_missing_terminal_outcome_unknown():
    env = EpisodeStatsWrapper(_TerminalInfoEnv(reward=999.0, info={}))
    env.reset()

    _, _, _, _, info = env.step(0)

    assert info["game_outcome"] == "unknown"
