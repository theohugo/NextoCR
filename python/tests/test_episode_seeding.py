# Modified by NextoCR contributors; see NOTICE for attribution.
"""Episode-seed sequence regressions that do not require a running Java bridge."""

import numpy as np

from crforge_gym import CRForgeEnv
from crforge_gym.env import (
    IDX_OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    OBS_SIZE,
)


class _FakeBinaryClient:
    def __init__(self):
        self.init_seeds = []
        self.reset_seeds = []

    def connect(self):
        pass

    def init(self, blue_deck, red_deck, *, level, ticks_per_step, seed):
        self.init_seeds.append(seed)

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        observation = np.zeros(OBS_SIZE, dtype=np.float32)
        observation[IDX_OBSERVATION_SCHEMA_VERSION] = OBSERVATION_SCHEMA_VERSION
        return observation

    def close(self):
        pass


def _episode_seed_series(initial_seed: int, auto_resets: int):
    env = CRForgeEnv(binary_obs=True)
    client = _FakeBinaryClient()
    env._client = client
    try:
        _, first_info = env.reset(seed=initial_seed)
        seeds = [first_info["episode_seed"]]
        for _ in range(auto_resets):
            _, info = env.reset()
            seeds.append(info["episode_seed"])
        return seeds, client
    finally:
        env.close()


def test_same_initial_seed_produces_same_episode_seed_sequence():
    first, first_client = _episode_seed_series(42, auto_resets=4)
    second, second_client = _episode_seed_series(42, auto_resets=4)

    expected_rng = np.random.default_rng(42)
    expected = [42] + [
        int(expected_rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
        for _ in range(4)
    ]
    assert first == expected
    assert second == expected
    assert first_client.init_seeds == [42]
    assert first_client.reset_seeds == expected
    assert second_client.reset_seeds == expected


def test_explicit_reseed_restarts_auto_reset_sequence_without_consuming_seed():
    env = CRForgeEnv(binary_obs=True)
    client = _FakeBinaryClient()
    env._client = client
    try:
        _, explicit = env.reset(seed=17)
        _, first_auto = env.reset()
        _, explicit_again = env.reset(seed=17)
        _, repeated_auto = env.reset()

        assert explicit["episode_seed"] == 17
        assert explicit_again["episode_seed"] == 17
        assert first_auto["episode_seed"] == repeated_auto["episode_seed"]
        assert client.reset_seeds[0] == 17
        assert client.reset_seeds[2] == 17
    finally:
        env.close()
