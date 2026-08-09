"""Parity and bounds tests for the static policy observation contract."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest

from crforge_gym.env import (
    IDX_ENTITY_IDENTITIES_START,
    IDX_HAND_IDENTITIES_START,
    IDX_OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    OBS_SIZE,
)
from crforge_gym.observation_preprocessing import (
    OBSERVATION_PREPROCESSING_SCHEMA,
    PreprocessedPolicyAdapter,
    StaticObservationPreprocessingWrapper,
    preprocess_flat_observation,
    tag_model_preprocessing,
    validate_model_preprocessing,
)


def _raw_observation() -> np.ndarray:
    observation = np.zeros(OBS_SIZE, dtype=np.float32)
    observation[0] = 3_000
    observation[1] = 150
    observation[3:5] = [5, 10]
    observation[5:7] = [1, 3]
    observation[11:15] = [0, 1, 2, 1]
    observation[15:19] = [5, 100, 4_095, -1]
    observation[20] = 2
    observation[21] = 777
    observation[46:49] = [1, 4, 2]
    observation[49:62] = 3.0  # malformed high values must be contained
    observation[1070] = 32
    observation[1071:1079] = np.linspace(-1, 1, 8)
    observation[IDX_OBSERVATION_SCHEMA_VERSION] = OBSERVATION_SCHEMA_VERSION
    observation[IDX_HAND_IDENTITIES_START : IDX_HAND_IDENTITIES_START + 4] = [
        0.1,
        0.2,
        0.3,
        0.4,
    ]
    observation[IDX_ENTITY_IDENTITIES_START] = 0.75
    return observation


class _OneObservationEnv(gym.Env):
    action_space = spaces.Discrete(1)
    observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(OBS_SIZE,), dtype=np.float32
    )

    def __init__(self, observation):
        self.value = observation

    def reset(self, *, seed=None, options=None):
        return self.value.copy(), {}

    def step(self, action):
        return self.value.copy(), 0.0, True, False, {}


class _CaptureModel:
    action_space = spaces.Discrete(41)

    def __init__(self):
        self.observation = None

    def predict(self, observation, *args, **kwargs):
        self.observation = observation
        return np.array(0), None


def test_blue_wrapper_and_red_policy_adapter_are_identical() -> None:
    raw = _raw_observation()
    blue_env = StaticObservationPreprocessingWrapper(_OneObservationEnv(raw))
    blue_observation, _ = blue_env.reset()
    capture = _CaptureModel()
    red_policy = PreprocessedPolicyAdapter(capture)

    red_policy.predict(raw.copy())

    np.testing.assert_array_equal(capture.observation, blue_observation)


def test_preprocessing_is_bounded_and_uses_v2_identities() -> None:
    transformed = preprocess_flat_observation(_raw_observation())

    assert transformed.dtype == np.float32
    assert np.min(transformed) >= -1.0
    assert np.max(transformed) <= 1.0
    assert transformed[0] == pytest.approx(0.5)
    assert transformed[1] == pytest.approx(0.5)
    assert transformed[3:5].tolist() == pytest.approx([0.5, 1.0])
    assert transformed[15:19].tolist() == [0.0] * 4
    assert transformed[21] == 0.0
    assert transformed[IDX_HAND_IDENTITIES_START] == pytest.approx(0.1)
    assert transformed[IDX_ENTITY_IDENTITIES_START] == pytest.approx(0.75)
    assert transformed[1070] == pytest.approx(0.5)


def test_v1_observation_is_rejected() -> None:
    raw = _raw_observation()
    raw[IDX_OBSERVATION_SCHEMA_VERSION] = 1
    with pytest.raises(ValueError, match="requires observation V2"):
        preprocess_flat_observation(raw)


def test_checkpoint_preprocessing_tag_is_required() -> None:
    model = _CaptureModel()
    with pytest.raises(ValueError, match="missing or incompatible"):
        validate_model_preprocessing(model)
    tag_model_preprocessing(model)
    validate_model_preprocessing(model)
    assert model.nextocr_observation_preprocessing == OBSERVATION_PREPROCESSING_SCHEMA
