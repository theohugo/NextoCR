# Modified by NextoCR contributors; see NOTICE for attribution.
"""
Wrappers for CRForge environments.

EpisodeStatsWrapper tracks per-episode reward, length, and game outcome
(win/loss/draw) and exposes them in the SB3-compatible info["episode"] format.

FlattenedObsWrapper converts the Dict observation space into a single flat
float32 vector, which works with SB3's simpler MlpPolicy and avoids the
complexity of MultiInputPolicy.

ActionMaskedWrapper adds action masking for use with sb3-contrib's
MaskablePPO, preventing the agent from exploring invalid actions (cards
it can't afford, placing on the enemy half of the arena).

ExactDiscreteActionWrapper exposes the 41 meaningful decisions as one
Discrete action: no-op plus every card-slot/placement-zone pair. Unlike a
factorized MultiDiscrete mask, it can reject a troop/enemy-zone combination
without also rejecting that zone for an affordable spell.
"""

import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EpisodeStatsWrapper(gym.Wrapper):
    """Tracks per-episode statistics for SB3 logging.

    On episode termination, adds info["episode"] with:
    - r: total episode reward
    - l: episode length (steps)
    - t: wall-clock time since wrapper creation

    Preserves the engine-provided ``info["game_outcome"]``. A terminal transition
    missing a canonical outcome is labelled ``"unknown"``; rewards are never used
    to infer match results.

    Wrap order: EpisodeStatsWrapper(CRForgeEnv(...))
    Place this innermost, before FlattenedObsWrapper and ActionMaskedWrapper.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._episode_reward = 0.0
        self._episode_length = 0
        self._start_time = time.time()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._episode_reward = 0.0
        self._episode_length = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_reward += reward
        self._episode_length += 1

        if terminated or truncated:
            info["episode"] = {
                "r": self._episode_reward,
                "l": self._episode_length,
                "t": time.time() - self._start_time,
            }
            if info.get("game_outcome") not in {"win", "loss", "draw"}:
                info["game_outcome"] = "unknown"

        return obs, reward, terminated, truncated, info


class FlattenedObsWrapper(gym.ObservationWrapper):
    """Flattens the Dict observation space into a single 1D float32 vector.

    This is the recommended wrapper for use with Stable Baselines 3, since
    MlpPolicy expects a flat Box observation space.

    The flattened vector concatenates all observation fields in a fixed order:
    The first 1079 values retain the V1/binary order. V2 appends schema version,
    stable card/entity identities, and exact enemy-side placement flags.

    Note: When using binary_obs=True (the default), the env already returns
    a flat observation and this wrapper is not needed.
    """

    # Fixed ordering of observation keys for consistent flattening
    _OBS_KEYS = [
        "frame", "game_time", "is_overtime",
        "elixir", "crowns",
        "hand_costs", "hand_types", "hand_card_ids",
        "next_card_cost", "next_card_type", "next_card_id",
        "towers", "entities", "num_entities",
        "lane_summary",
        "observation_schema_version",
        "hand_card_identities", "next_card_identity", "entity_identities",
        "hand_allows_enemy_placement",
    ]

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # Compute total flat size from the wrapped env's observation space
        total_size = 0
        for key in self._OBS_KEYS:
            space = env.observation_space[key]
            total_size += int(np.prod(space.shape))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float32
        )

    def observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Flatten the dict observation into a single vector."""
        parts = []
        for key in self._OBS_KEYS:
            arr = observation[key].astype(np.float32).flatten()
            parts.append(arr)
        return np.concatenate(parts)


class ActionMaskedWrapper(gym.Wrapper):
    """Adds action masking for use with sb3-contrib's MaskablePPO.

    Computes masks from the observation to prevent the agent from exploring
    invalid actions:
    - action_type: noop always valid; play only if at least one card is affordable
    - hand_index: only affordable cards are unmasked
    - zone: own-half zones (0-6) always valid; enemy-half zones (7-9) only
      when an affordable spell is in hand

    For MultiDiscrete([2, 4, 10]), the mask is a flat boolean array of
    length 2 + 4 + 10 = 16.

    Supports both binary and JSON observation modes. In binary mode, reads
    elixir and hand data directly from the flat observation array indices.

    Wrap order: ActionMaskedWrapper(FlattenedObsWrapper(CRForgeEnv(...)))
    or ActionMaskedWrapper(EpisodeStatsWrapper(CRForgeEnv(binary_obs=True)))
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._last_dict_obs: dict[str, np.ndarray] | None = None

        # Grab the inner CRForgeEnv to access the raw obs
        inner = env
        while hasattr(inner, "env"):
            inner = inner.env
        self._inner_env = inner
        self._binary_mode = getattr(inner, "binary_obs", False)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._capture_obs()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._capture_obs()
        return obs, reward, terminated, truncated, info

    def _capture_obs(self):
        """Capture observation data for action masking."""
        if self._binary_mode:
            # In binary mode, read directly from the flat observation array
            flat = self._inner_env._last_obs_flat
            if flat is not None:
                from crforge_gym.env import IDX_BLUE_ELIXIR, IDX_HAND_COSTS_START, IDX_HAND_COSTS_END, IDX_HAND_TYPES_START, IDX_HAND_TYPES_END
                elixir = float(flat[IDX_BLUE_ELIXIR])
                # Hand costs are normalized by /10, convert back to actual cost
                hand_costs_norm = flat[IDX_HAND_COSTS_START:IDX_HAND_COSTS_END]
                hand_costs = hand_costs_norm * 10.0
                # Hand types: 0=TROOP, 1=SPELL, 2=BUILDING
                hand_types_float = flat[IDX_HAND_TYPES_START:IDX_HAND_TYPES_END]
                hand_types = []
                for t in hand_types_float:
                    if t == 1.0:
                        hand_types.append("SPELL")
                    elif t == 2.0:
                        hand_types.append("BUILDING")
                    else:
                        hand_types.append("TROOP")
                self._last_dict_obs = {
                    "elixir": elixir,
                    "hand_costs": hand_costs.astype(np.float32),
                    "hand_types": hand_types,
                }
        else:
            # JSON mode: extract from raw observation dict
            raw = self._inner_env._last_obs_raw
            if raw is not None:
                blue = raw.get("bluePlayer", {})
                hand = blue.get("hand", [])
                elixir = blue.get("elixir", 0.0)

                hand_costs = np.zeros(4, dtype=np.float32)
                hand_types = []
                for i, card in enumerate(hand[:4]):
                    hand_costs[i] = card.get("cost", 99)
                    hand_types.append(card.get("type", "TROOP"))

                self._last_dict_obs = {
                    "elixir": elixir,
                    "hand_costs": hand_costs,
                    "hand_types": hand_types,
                }

    def action_masks(self) -> np.ndarray:
        """Return a flat boolean mask of length 16 for MultiDiscrete([2, 4, 10])."""
        from crforge_gym.env import NUM_ZONES, NUM_OWN_HALF_ZONES

        action_type_mask = np.array([True, True])  # [noop, play]
        hand_mask = np.array([True, True, True, True])
        zone_mask = np.ones(NUM_ZONES, dtype=bool)

        if self._last_dict_obs is not None:
            elixir = self._last_dict_obs["elixir"]
            hand_costs = self._last_dict_obs["hand_costs"]
            hand_types = self._last_dict_obs.get("hand_types", [])

            has_affordable_spell = False
            for i in range(4):
                affordable = hand_costs[i] <= elixir and hand_costs[i] > 0
                hand_mask[i] = affordable
                if affordable and i < len(hand_types) and hand_types[i] == "SPELL":
                    has_affordable_spell = True

            if not np.any(hand_mask):
                action_type_mask[1] = False

            # Enemy-half zones (7-9) only valid when a spell is affordable
            if not has_affordable_spell:
                zone_mask[NUM_OWN_HALF_ZONES:] = False

        return np.concatenate([action_type_mask, hand_mask, zone_mask])


class ExactDiscreteActionWrapper(gym.ActionWrapper):
    """Expose an exact, maskable Discrete action space for RL training.

    Action ``0`` is no-op. Actions ``1..40`` encode the Cartesian product of
    four hand slots and ten strategic zones::

        action_id = 1 + hand_index * NUM_ZONES + zone

    The wrapped :class:`CRForgeEnv` still receives its legacy
    ``MultiDiscrete([2, 4, 10])`` representation, so this wrapper is backwards
    compatible with the bridge protocol while removing factorized-mask leaks.
    """

    NUM_HAND_SLOTS = 4

    def __init__(self, env: gym.Env):
        super().__init__(env)

        from crforge_gym.env import NUM_ZONES

        expected = np.array([2, self.NUM_HAND_SLOTS, NUM_ZONES])
        if not isinstance(env.action_space, spaces.MultiDiscrete) or not np.array_equal(
            env.action_space.nvec, expected
        ):
            raise ValueError(
                "ExactDiscreteActionWrapper requires MultiDiscrete([2, 4, NUM_ZONES])"
            )

        self.action_space = spaces.Discrete(1 + self.NUM_HAND_SLOTS * NUM_ZONES)

        inner = env
        while hasattr(inner, "env"):
            inner = inner.env
        self._inner_env = inner

    @staticmethod
    def encode(hand_index: int, zone: int) -> int:
        """Encode a card-slot/zone pair into the exact Discrete action ID."""
        from crforge_gym.env import NUM_ZONES

        if not 0 <= hand_index < ExactDiscreteActionWrapper.NUM_HAND_SLOTS:
            raise ValueError(f"hand_index must be in [0, 3], got {hand_index}")
        if not 0 <= zone < NUM_ZONES:
            raise ValueError(f"zone must be in [0, {NUM_ZONES - 1}], got {zone}")
        return 1 + hand_index * NUM_ZONES + zone

    @staticmethod
    def decode(action: int | np.ndarray) -> np.ndarray:
        """Decode an exact Discrete action into ``[type, hand, zone]``."""
        from crforge_gym.env import NUM_ZONES

        action_array = np.asarray(action)
        if action_array.size != 1:
            raise ValueError(f"expected one Discrete action, got shape {action_array.shape}")
        action_id = int(action_array.reshape(-1)[0])
        max_action = ExactDiscreteActionWrapper.NUM_HAND_SLOTS * NUM_ZONES
        if not 0 <= action_id <= max_action:
            raise ValueError(f"action must be in [0, {max_action}], got {action_id}")
        if action_id == 0:
            return np.array([0, 0, 0], dtype=np.int64)

        card_zone = action_id - 1
        hand_index, zone = divmod(card_zone, NUM_ZONES)
        return np.array([1, hand_index, zone], dtype=np.int64)

    def action(self, action: int | np.ndarray) -> np.ndarray:
        """Translate the policy action to the bridge-compatible representation."""
        return self.decode(action)

    def action_masks(self) -> np.ndarray:
        """Return the exact validity mask for all 41 decisions.

        If the current observation is unavailable, fail closed by exposing only
        no-op. Normal Gym operation always calls this after ``reset`` or ``step``.
        """
        from crforge_gym.env import NUM_OWN_HALF_ZONES, NUM_ZONES

        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[0] = True
        context = self._action_context()
        if context is None:
            return mask

        elixir, hand_costs, hand_types, allows_enemy_placement = context
        for hand_index in range(self.NUM_HAND_SLOTS):
            cost = float(hand_costs[hand_index])
            if not 0 < cost <= elixir:
                continue

            zone_count = (
                NUM_ZONES
                if allows_enemy_placement[hand_index]
                else NUM_OWN_HALF_ZONES
            )
            start = self.encode(hand_index, 0)
            mask[start : start + zone_count] = True

        return mask

    def _action_context(
        self,
    ) -> tuple[float, np.ndarray, list[str], np.ndarray] | None:
        """Read elixir, costs, and types from the latest engine observation."""
        inner = self._inner_env
        if getattr(inner, "binary_obs", False):
            flat = getattr(inner, "_last_obs_flat", None)
            if flat is None:
                return None

            from crforge_gym.env import (
                IDX_BLUE_ELIXIR,
                IDX_HAND_ALLOWS_ENEMY_PLACEMENT_START,
                IDX_HAND_COSTS_END,
                IDX_HAND_COSTS_START,
                IDX_HAND_TYPES_END,
                IDX_HAND_TYPES_START,
                IDX_OBSERVATION_SCHEMA_VERSION,
                OBSERVATION_SCHEMA_VERSION,
            )

            elixir = float(flat[IDX_BLUE_ELIXIR])
            hand_costs = (
                np.asarray(flat[IDX_HAND_COSTS_START:IDX_HAND_COSTS_END], dtype=np.float32)
                * 10.0
            )
            encoded_types = flat[IDX_HAND_TYPES_START:IDX_HAND_TYPES_END]
            hand_types = [
                "SPELL" if value == 1.0 else "BUILDING" if value == 2.0 else "TROOP"
                for value in encoded_types
            ]
            if flat[IDX_OBSERVATION_SCHEMA_VERSION] >= OBSERVATION_SCHEMA_VERSION:
                allows_enemy_placement = np.asarray(
                    flat[
                        IDX_HAND_ALLOWS_ENEMY_PLACEMENT_START :
                        IDX_HAND_ALLOWS_ENEMY_PLACEMENT_START + self.NUM_HAND_SLOTS
                    ],
                    dtype=bool,
                )
            else:
                # V1 compatibility: fail safely for non-spells, while preserving
                # the legacy generic-spell behavior until the V2 bridge is rebuilt.
                allows_enemy_placement = np.asarray(
                    [card_type == "SPELL" for card_type in hand_types], dtype=bool
                )
            return elixir, hand_costs, hand_types, allows_enemy_placement

        raw = getattr(inner, "_last_obs_raw", None)
        if raw is None:
            return None
        blue = raw.get("bluePlayer", {})
        hand = blue.get("hand", [])
        hand_costs = np.zeros(self.NUM_HAND_SLOTS, dtype=np.float32)
        hand_types: list[str] = []
        allows_enemy_placement = np.zeros(self.NUM_HAND_SLOTS, dtype=bool)
        for index, card in enumerate(hand[: self.NUM_HAND_SLOTS]):
            hand_costs[index] = card.get("cost", 0)
            card_type = card.get("type", "TROOP")
            hand_types.append(card_type)
            allows_enemy_placement[index] = card.get(
                "allowsEnemyPlacement", card_type == "SPELL"
            )
        return (
            float(blue.get("elixir", 0.0)),
            hand_costs,
            hand_types,
            allows_enemy_placement,
        )
