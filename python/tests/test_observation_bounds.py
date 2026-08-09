"""Observation-bound regressions for large simulated swarms."""

import numpy as np
import pytest

from crforge_gym import CRForgeEnv, parse_observation
from crforge_gym.env import (
    IDX_ENTITY_IDENTITIES_START,
    IDX_HAND_ALLOWS_ENEMY_PLACEMENT_START,
    IDX_HAND_IDENTITIES_START,
    IDX_NEXT_CARD_IDENTITY,
    IDX_OBSERVATION_SCHEMA_VERSION,
    MAX_ENTITIES,
    OBSERVATION_SCHEMA_VERSION,
    OBS_SIZE,
    OBS_V1_SIZE,
    STABLE_ID_MAX,
    _coerce_binary_observation,
    _stable_identity_id,
)
from crforge_gym.opponents import _OBS_KEYS as SELF_PLAY_OBS_KEYS
from crforge_gym.opponents import _flatten_dict_obs
from crforge_gym.wrappers import FlattenedObsWrapper


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


@pytest.mark.parametrize(
    ("namespace", "key", "expected"),
    [
        ("card", "knight", 12_097_159),
        ("card", "fireball", 243_388),
        ("card", "barblog", 12_535_573),
        ("entity", "Crown Tower", 14_238_056),
        ("entity", "Princess Tower", 5_362_128),
        ("entity", "Knight", 15_105_610),
    ],
)
def test_stable_identity_hash_matches_java_contract(namespace, key, expected):
    assert _stable_identity_id(namespace, key) == expected


def test_json_v2_identity_and_placement_fields_are_normalized_and_bounded():
    raw = {
        "schemaVersion": OBSERVATION_SCHEMA_VERSION,
        "bluePlayer": {
            "elixir": 5.0,
            "crowns": 0,
            "hand": [
                {
                    "id": "fireball",
                    "type": "SPELL",
                    "cost": 4,
                    "cardIndex": 239,
                    "identityId": STABLE_ID_MAX,
                    "allowsEnemyPlacement": True,
                },
                {
                    "id": "barblog",
                    "type": "SPELL",
                    "cost": 2,
                    "cardIndex": 120,
                    "identityId": 1,
                    "allowsEnemyPlacement": False,
                },
            ],
            "nextCard": {
                "id": "knight",
                "type": "TROOP",
                "cost": 3,
                "cardIndex": 0,
                "identityId": 12_097_159,
            },
            "towers": [],
        },
        "redPlayer": {"elixir": 5.0, "crowns": 0, "hand": [], "towers": []},
        "entities": [
            {
                "name": "Knight",
                "identityId": STABLE_ID_MAX,
                "team": "BLUE",
                "entityType": "TROOP",
                "movementType": "GROUND",
                "hp": 100,
                "maxHp": 100,
            }
        ],
    }

    observation = parse_observation(raw)

    assert observation["observation_schema_version"][0] == 2.0
    np.testing.assert_array_equal(
        observation["hand_allows_enemy_placement"][:2], [1.0, 0.0]
    )
    assert observation["hand_card_identities"][0] == 1.0
    assert observation["hand_card_identities"][1] == np.float32(1 / STABLE_ID_MAX)
    assert observation["entity_identities"][0] == 1.0
    assert observation["next_card_identity"][0] == np.float32(
        12_097_159 / STABLE_ID_MAX
    )

    env = CRForgeEnv(binary_obs=False)
    try:
        assert env.observation_space.contains(observation)
    finally:
        env.close()


def test_json_v1_fallback_hashes_known_names_without_raw_integer_ids():
    raw = {
        "bluePlayer": {
            "elixir": 5.0,
            "crowns": 0,
            "hand": [{"id": "knight", "type": "TROOP", "cost": 3}],
            "towers": [],
        },
        "redPlayer": {"elixir": 5.0, "crowns": 0, "hand": [], "towers": []},
        "entities": [{"name": "Crown Tower", "entityType": "TOWER"}],
    }

    observation = parse_observation(raw)

    assert observation["observation_schema_version"][0] == 1.0
    assert observation["hand_card_identities"][0] == np.float32(
        12_097_159 / STABLE_ID_MAX
    )
    assert observation["entity_identities"][0] == np.float32(
        14_238_056 / STABLE_ID_MAX
    )
    assert observation["hand_allows_enemy_placement"][0] == 0.0


def test_binary_v1_is_upgraded_without_changing_its_prefix():
    legacy = np.arange(OBS_V1_SIZE, dtype=np.float32)

    upgraded = _coerce_binary_observation(legacy)

    assert upgraded.shape == (OBS_SIZE,)
    np.testing.assert_array_equal(upgraded[:OBS_V1_SIZE], legacy)
    assert upgraded[IDX_OBSERVATION_SCHEMA_VERSION] == 1.0
    assert not upgraded[IDX_HAND_IDENTITIES_START:IDX_NEXT_CARD_IDENTITY].any()
    assert upgraded[IDX_NEXT_CARD_IDENTITY] == 0.0
    assert not upgraded[
        IDX_ENTITY_IDENTITIES_START:IDX_HAND_ALLOWS_ENEMY_PLACEMENT_START
    ].any()
    assert not upgraded[IDX_HAND_ALLOWS_ENEMY_PLACEMENT_START:].any()


def test_binary_observation_rejects_unknown_schema_length():
    with pytest.raises(ValueError, match="Unsupported binary observation size"):
        _coerce_binary_observation(np.zeros(42, dtype=np.float32))


def test_json_flattening_matches_binary_v2_extension_order():
    raw = {
        "schemaVersion": 2,
        "bluePlayer": {
            "elixir": 5.0,
            "crowns": 0,
            "hand": [
                {
                    "id": "fireball",
                    "identityId": 243_388,
                    "allowsEnemyPlacement": True,
                }
            ],
            "nextCard": {"id": "knight", "identityId": 12_097_159},
            "towers": [],
        },
        "redPlayer": {"elixir": 5.0, "crowns": 0, "hand": [], "towers": []},
        "entities": [{"name": "Crown Tower", "identityId": 14_238_056}],
    }
    parsed = parse_observation(raw)

    assert SELF_PLAY_OBS_KEYS == FlattenedObsWrapper._OBS_KEYS
    flat = _flatten_dict_obs(parsed)
    assert flat.shape == (OBS_SIZE,)
    assert flat[IDX_OBSERVATION_SCHEMA_VERSION] == 2.0
    assert flat[IDX_HAND_IDENTITIES_START] == np.float32(243_388 / STABLE_ID_MAX)
    assert flat[IDX_NEXT_CARD_IDENTITY] == np.float32(12_097_159 / STABLE_ID_MAX)
    assert flat[IDX_ENTITY_IDENTITIES_START] == np.float32(14_238_056 / STABLE_ID_MAX)
    assert flat[IDX_HAND_ALLOWS_ENEMY_PLACEMENT_START] == 1.0
