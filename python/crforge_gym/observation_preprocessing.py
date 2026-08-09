# SPDX-License-Identifier: Apache-2.0
"""Versioned static observation preprocessing for NextoCR policies."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from crforge_gym.env import (
    ENTITY_FEATURES,
    IDX_OBSERVATION_SCHEMA_VERSION,
    MAX_ENTITIES,
    OBSERVATION_SCHEMA_VERSION,
    OBS_SIZE,
)


OBSERVATION_PREPROCESSING_SCHEMA = "nextocr_static_v1"
_MAX_MATCH_FRAMES = 6_000.0  # 180 s regulation + 120 s overtime at 20 TPS.
_MAX_MATCH_SECONDS = 300.0

# V1-compatible prefix offsets. V2 appends stable identities after index 1078.
_IDX_FRAME = 0
_IDX_GAME_TIME = 1
_ELIXIR = slice(3, 5)
_CROWNS = slice(5, 7)
_HAND_TYPES = slice(11, 15)
_LEGACY_HAND_CARD_IDS = slice(15, 19)
_IDX_NEXT_CARD_TYPE = 20
_IDX_LEGACY_NEXT_CARD_ID = 21
_ENTITIES_START = 46
_IDX_NUM_ENTITIES = 1070


def preprocess_flat_observation(observation: np.ndarray) -> np.ndarray:
    """Map a V2 flat observation to a bounded, MLP-friendly static schema.

    The output retains the same shape, so checkpoint compatibility is governed
    explicitly by ``OBSERVATION_PREPROCESSING_SCHEMA`` rather than by dimensions.
    Legacy insertion-ordered card indices are zeroed; normalized V2 stable
    identities remain in the appended identity fields.
    """
    source = np.asarray(observation, dtype=np.float32)
    if source.ndim < 1 or source.shape[-1] < OBS_SIZE:
        raise ValueError(
            f"{OBSERVATION_PREPROCESSING_SCHEMA} requires last dimension {OBS_SIZE}, "
            f"got {source.shape}"
        )
    if source.shape[-1] > OBS_SIZE:
        # Appended blocks such as match memory are already bounded features;
        # normalise the legacy prefix and pass the remainder through untouched.
        head = preprocess_flat_observation(source[..., :OBS_SIZE])
        return np.concatenate([head, source[..., OBS_SIZE:]], axis=-1)
    schema_versions = source[..., IDX_OBSERVATION_SCHEMA_VERSION]
    if np.any(schema_versions < OBSERVATION_SCHEMA_VERSION):
        raise ValueError(
            f"{OBSERVATION_PREPROCESSING_SCHEMA} requires observation V2 stable identities"
        )

    result = source.copy()
    result[..., _IDX_FRAME] = np.clip(result[..., _IDX_FRAME] / _MAX_MATCH_FRAMES, 0.0, 1.0)
    result[..., _IDX_GAME_TIME] = np.clip(
        result[..., _IDX_GAME_TIME] / _MAX_MATCH_SECONDS, 0.0, 1.0
    )
    result[..., _ELIXIR] = np.clip(result[..., _ELIXIR] / 10.0, 0.0, 1.0)
    result[..., _CROWNS] = np.clip(result[..., _CROWNS] / 3.0, 0.0, 1.0)
    result[..., _HAND_TYPES] = np.clip(result[..., _HAND_TYPES] / 2.0, 0.0, 1.0)
    result[..., _LEGACY_HAND_CARD_IDS] = 0.0
    result[..., _IDX_NEXT_CARD_TYPE] = np.clip(
        result[..., _IDX_NEXT_CARD_TYPE] / 2.0, 0.0, 1.0
    )
    result[..., _IDX_LEGACY_NEXT_CARD_ID] = 0.0

    entities = result[..., _ENTITIES_START:_IDX_NUM_ENTITIES].reshape(
        *result.shape[:-1], MAX_ENTITIES, ENTITY_FEATURES
    )
    entities[..., 0] = np.clip(entities[..., 0], 0.0, 1.0)  # team
    entities[..., 1] = np.clip(entities[..., 1] / 4.0, 0.0, 1.0)  # entity type
    entities[..., 2] = np.clip(entities[..., 2] / 2.0, 0.0, 1.0)  # movement type
    entities[..., 3:] = np.clip(entities[..., 3:], 0.0, 1.0)
    result[..., _IDX_NUM_ENTITIES] = np.clip(
        result[..., _IDX_NUM_ENTITIES] / MAX_ENTITIES, 0.0, 1.0
    )
    result[..., IDX_OBSERVATION_SCHEMA_VERSION] = 1.0

    # Every remaining field is already semantically normalized. Clipping also
    # contains malformed/upstream edge values without online running statistics.
    return np.clip(result, -1.0, 1.0).astype(np.float32, copy=False)


class StaticObservationPreprocessingWrapper(gym.ObservationWrapper):
    """Apply ``nextocr_static_v1`` to every blue-policy observation."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box) or env.observation_space.shape != (
            OBS_SIZE,
        ):
            raise ValueError(
                "StaticObservationPreprocessingWrapper requires a flat V2 observation"
            )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return preprocess_flat_observation(observation)


class PreprocessedPolicyAdapter:
    """Apply the same static transform before a red self-play policy predicts."""

    def __init__(self, model: Any):
        self.model = model
        self.action_space = model.action_space

    def predict(self, observation: np.ndarray, *args: Any, **kwargs: Any):
        return self.model.predict(preprocess_flat_observation(observation), *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


def tag_model_preprocessing(model: Any) -> None:
    """Persist the preprocessing contract inside SB3 checkpoint metadata."""
    model.nextocr_observation_preprocessing = OBSERVATION_PREPROCESSING_SCHEMA


def validate_model_preprocessing(model: Any) -> None:
    """Refuse raw/unknown observation checkpoints despite matching dimensions."""
    actual = getattr(model, "nextocr_observation_preprocessing", None)
    if actual != OBSERVATION_PREPROCESSING_SCHEMA:
        raise ValueError(
            "Checkpoint observation preprocessing is missing or incompatible: "
            f"expected {OBSERVATION_PREPROCESSING_SCHEMA!r}, got {actual!r}. "
            "Raw-observation checkpoints cannot be resumed safely."
        )
