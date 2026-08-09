# Modified by NextoCR contributors; see NOTICE for attribution.
"""Tests for the exact Discrete RL action contract."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from crforge_gym.env import (
    IDX_BLUE_ELIXIR,
    IDX_HAND_ALLOWS_ENEMY_PLACEMENT_START,
    IDX_HAND_COSTS_START,
    IDX_HAND_TYPES_START,
    IDX_OBSERVATION_SCHEMA_VERSION,
    NUM_ZONES,
    OBSERVATION_SCHEMA_VERSION,
    OBS_SIZE,
)
from crforge_gym.wrappers import ExactDiscreteActionWrapper


class _BinaryEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.MultiDiscrete([2, 4, NUM_ZONES])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.binary_obs = True
        self._last_obs_flat = None
        self.last_action = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._last_obs_flat = np.zeros(OBS_SIZE, dtype=np.float32)
        self._last_obs_flat[IDX_OBSERVATION_SCHEMA_VERSION] = OBSERVATION_SCHEMA_VERSION
        return self._last_obs_flat.copy(), {}

    def step(self, action):
        self.last_action = np.asarray(action)
        return self._last_obs_flat.copy(), 0.0, False, False, {}


def test_exact_action_round_trip():
    action_id = ExactDiscreteActionWrapper.encode(hand_index=2, zone=7)

    assert action_id == 28
    np.testing.assert_array_equal(
        ExactDiscreteActionWrapper.decode(action_id), np.array([1, 2, 7])
    )
    np.testing.assert_array_equal(
        ExactDiscreteActionWrapper.decode(0), np.array([0, 0, 0])
    )


@pytest.mark.parametrize("action", [-1, 41, np.array([1, 2])])
def test_exact_action_rejects_values_outside_contract(action):
    with pytest.raises(ValueError):
        ExactDiscreteActionWrapper.decode(action)


def test_exact_mask_is_conditional_on_the_selected_card():
    base = _BinaryEnv()
    env = ExactDiscreteActionWrapper(base)
    env.reset()

    base._last_obs_flat[IDX_BLUE_ELIXIR] = 4.0
    base._last_obs_flat[IDX_HAND_COSTS_START : IDX_HAND_COSTS_START + 4] = [0.3, 0.2, 0.5, 0.0]
    # troop, spell, building, troop
    base._last_obs_flat[IDX_HAND_TYPES_START : IDX_HAND_TYPES_START + 4] = [0.0, 1.0, 2.0, 0.0]
    # The V2 engine says slot 1 is anywhere-placement while slot 0 is own-side only.
    base._last_obs_flat[
        IDX_HAND_ALLOWS_ENEMY_PLACEMENT_START :
        IDX_HAND_ALLOWS_ENEMY_PLACEMENT_START + 4
    ] = [0.0, 1.0, 0.0, 0.0]

    mask = env.action_masks()

    assert mask.shape == (41,)
    assert mask[0]
    assert mask[ExactDiscreteActionWrapper.encode(0, 6)]
    assert not mask[ExactDiscreteActionWrapper.encode(0, 7)]
    assert mask[ExactDiscreteActionWrapper.encode(1, 9)]
    assert not mask[ExactDiscreteActionWrapper.encode(2, 0)]
    assert not mask[ExactDiscreteActionWrapper.encode(3, 0)]
    assert int(mask.sum()) == 18  # no-op + 7 troop zones + 10 spell zones


def test_wrapper_translates_policy_action_before_step():
    base = _BinaryEnv()
    env = ExactDiscreteActionWrapper(base)
    env.reset()

    env.step(ExactDiscreteActionWrapper.encode(3, 4))

    np.testing.assert_array_equal(base.last_action, np.array([1, 3, 4]))


def test_mask_fails_closed_before_first_observation():
    env = ExactDiscreteActionWrapper(_BinaryEnv())

    mask = env.action_masks()

    assert mask[0]
    assert int(mask.sum()) == 1
