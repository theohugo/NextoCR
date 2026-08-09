"""Regression tests for opponent routing in binary observations."""

import numpy as np
import pytest
from gymnasium import spaces

from crforge_gym import CRForgeEnv
from crforge_gym.env import OBS_SIZE
from crforge_gym.opponents import SelfPlayOpponent
from crforge_gym.wrappers import ExactDiscreteActionWrapper


class _FakeBinaryClient:
    def reset(self, seed=None):
        return np.zeros(OBS_SIZE, dtype=np.float32)

    def close(self):
        pass


def test_rule_based_opponent_receives_binary_observation():
    env = CRForgeEnv(opponent="rule_based", binary_obs=True)
    try:
        flat_observation = np.zeros(OBS_SIZE, dtype=np.float32)
        flat_observation[4] = 10.0  # Red elixir in the binary schema.
        env._last_obs_raw = None
        env._last_obs_flat = flat_observation

        action = env._rule_based_action()

        assert action is not None
        assert 0 <= action["handIndex"] < 4
        assert 0.0 <= action["x"] <= 18.0
        assert 16.0 <= action["y"] <= 32.0
    finally:
        env.close()


def test_binary_self_play_fails_instead_of_using_the_blue_hand():
    opponent = SelfPlayOpponent(model=object())

    with pytest.raises(ValueError, match="red hand"):
        opponent.act(None, obs_flat=np.zeros(OBS_SIZE, dtype=np.float32))


def test_json_self_play_supports_exact_discrete_action_schema():
    class _ExactModel:
        action_space = spaces.Discrete(41)

        def __init__(self):
            self.last_mask = None

        def predict(self, obs, *, action_masks, deterministic):
            self.last_mask = action_masks
            return ExactDiscreteActionWrapper.encode(0, 9), None

    model = _ExactModel()
    opponent = SelfPlayOpponent(model=model)
    raw_obs = {
        "bluePlayer": {"elixir": 0.0, "hand": [], "towers": []},
        "redPlayer": {
            "elixir": 4.0,
            "hand": [
                {"cost": 2, "type": "SPELL", "allowsEnemyPlacement": True},
                {"cost": 5, "type": "TROOP"},
                {"cost": 2, "type": "SPELL", "allowsEnemyPlacement": False},
                {"cost": 0, "type": "TROOP"},
            ],
            "towers": [],
        },
        "entities": [],
    }

    action = opponent.act(raw_obs)

    assert action == {"handIndex": 0, "x": 9.0, "y": 11.5}
    assert model.last_mask[ExactDiscreteActionWrapper.encode(0, 9)]
    assert not model.last_mask[ExactDiscreteActionWrapper.encode(1, 0)]
    assert model.last_mask[ExactDiscreteActionWrapper.encode(2, 6)]
    assert not model.last_mask[ExactDiscreteActionWrapper.encode(2, 7)]


def test_rule_based_opponent_replays_choices_after_same_seed_reset():
    env = CRForgeEnv(opponent="rule_based", binary_obs=True)
    env._client = _FakeBinaryClient()
    env._connected = True
    # The fake client has no init(); mark the session already initialised so
    # reset() does not try to send decks to it.
    env._needs_init = False
    try:
        first_actions = _actions_after_reset(env, seed=17)
        second_actions = _actions_after_reset(env, seed=17)

        assert first_actions == second_actions
    finally:
        env.close()


def _actions_after_reset(env, *, seed):
    env.reset(seed=seed)
    env._last_obs_flat[4] = 10.0
    return [env._rule_based_action() for _ in range(8)]
