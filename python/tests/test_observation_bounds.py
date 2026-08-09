"""Observation-bound regressions for large simulated swarms."""

import numpy as np

from crforge_gym import CRForgeEnv, parse_observation
from crforge_gym.env import MAX_ENTITIES


def test_json_lane_count_advantage_is_clamped_and_entity_count_reports_cap():
    entities = [
        {
            "team": "BLUE",
            "entityType": "TROOP",
            "movementType": "GROUND",
            "x": 5.0,
            "y": 10.0,
            "hp": 1,
            "maxHp": 1,
        }
        for _ in range(MAX_ENTITIES + 6)
    ]
    raw = {
        "bluePlayer": {"elixir": 5.0, "crowns": 0, "hand": [], "towers": []},
        "redPlayer": {"elixir": 5.0, "crowns": 0, "hand": [], "towers": []},
        "entities": entities,
    }

    observation = parse_observation(raw)

    assert observation["num_entities"][0] == MAX_ENTITIES
    assert observation["lane_summary"][-1] == np.float32(1.0)

    env = CRForgeEnv(binary_obs=False)
    try:
        assert env.observation_space.contains(observation)
    finally:
        env.close()
