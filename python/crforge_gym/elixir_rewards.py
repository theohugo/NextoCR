# Modified by NextoCR contributors; see NOTICE for attribution.
"""Reward the elixir economy that decides Clash Royale matches.

Crowns arrive once, at the end of a three-minute match, so a policy learning
only from them has almost no signal about *why* it lost. Human play is decided
far earlier, by two habits:

* defending for less elixir than the attacker spent, so the next push is
  answered from a surplus;
* committing to an attack while the opponent cannot answer at full strength.

Both are measurable here. Spending is read from each player's elixir falling,
and the value destroyed is read from enemy units leaving the arena, priced from
the card catalogue.

Shaping is deliberately kept subordinate to the match result: the per-episode
total is clipped, so an agent that farms trades while losing every tower still
scores worse than one that wins. Reward shaping that outranks the objective is
how agents learn to avoid winning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym

from crforge_gym.card_properties import CardFeatureTable, default_table

__all__ = [
    "ElixirEconomyTracker",
    "ElixirTradeRewardWrapper",
    "TradeRewardConfig",
]

_MAX_ELIXIR = 10.0


@dataclass(frozen=True)
class TradeRewardConfig:
    """Weights for the economic shaping terms."""

    trade_weight: float = 0.6
    tempo_weight: float = 0.25
    #: Hard ceiling on |shaping| accumulated over one episode, in reward units.
    #: The terminal win/loss bonus is far larger, which is the point.
    episode_shaping_limit: float = 12.0
    #: Below this, an opponent cannot answer a committed push at full strength.
    tempo_advantage_threshold: float = 3.0

    def __post_init__(self) -> None:
        if self.episode_shaping_limit < 0:
            raise ValueError("episode_shaping_limit must not be negative")


def _unit_cost_index(table: CardFeatureTable) -> dict[str, float]:
    """Price each unit by the card that summons it, split across its group.

    A single Skeleton is not worth the whole card, so a card that spawns four
    bodies prices each body at a quarter. Without this, trading a spell for a
    swarm would look like a huge win every time one body died.
    """
    costs: dict[str, float] = {}
    for card in table.cards.values():
        unit = card.get("unit")
        if not unit:
            continue
        count = float(card.get("count", 1) or 1)
        secondary_count = float(card.get("secondaryCount", 0) or 0)
        total = max(1.0, count + secondary_count)
        cost = float(card.get("cost", 0) or 0)
        share = cost / total
        # Keep the cheapest attribution when several cards spawn the same unit,
        # so a unit is never priced above what it can actually cost to field.
        if unit not in costs or share < costs[str(unit)]:
            costs[str(unit)] = share
        secondary = card.get("secondaryUnit")
        if secondary and (str(secondary) not in costs or share < costs[str(secondary)]):
            costs[str(secondary)] = share
    return costs


class ElixirEconomyTracker:
    """Turn consecutive observations into an elixir trade signal."""

    def __init__(
        self,
        config: TradeRewardConfig | None = None,
        table: CardFeatureTable | None = None,
    ) -> None:
        self.config = config or TradeRewardConfig()
        self.table = table or default_table()
        self._unit_costs = _unit_cost_index(self.table)
        self.reset()

    def reset(self) -> None:
        self._previous_own_elixir: float | None = None
        self._previous_enemy_elixir: float | None = None
        self._live_enemy: dict[int, float] = {}
        self._live_own: dict[int, float] = {}
        self.shaping_spent = 0.0

    def unit_value(self, unit_name: str | None) -> float:
        return self._unit_costs.get(str(unit_name), 0.0) if unit_name else 0.0

    @staticmethod
    def _elixir(observation: dict[str, Any], key: str) -> float:
        player = observation.get(key) or {}
        return float(player.get("elixir", 0.0) or 0.0)

    def _index_entities(self, observation: dict[str, Any], team: str) -> dict[int, float]:
        entities = observation.get("entities")
        # A parsed observation stores numpy arrays under the same keys, and the
        # trade signal can only be read from the raw entity records.
        if not isinstance(entities, list):
            return {}
        indexed: dict[int, float] = {}
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            if str(entity.get("team", "")).upper() != team:
                continue
            identifier = entity.get("id")
            if identifier is None:
                continue
            indexed[int(identifier)] = self.unit_value(entity.get("name"))
        return indexed

    def update(self, observation: dict[str, Any], *, agent_team: str = "BLUE") -> float:
        """Return the shaping reward for the transition into *observation*."""
        if not isinstance(observation, dict):
            return 0.0
        enemy_team = "RED" if agent_team.upper() == "BLUE" else "BLUE"
        own_key = "bluePlayer" if agent_team.upper() == "BLUE" else "redPlayer"
        enemy_key = "redPlayer" if agent_team.upper() == "BLUE" else "bluePlayer"

        own_elixir = self._elixir(observation, own_key)
        enemy_elixir = self._elixir(observation, enemy_key)
        enemy_now = self._index_entities(observation, enemy_team)
        own_now = self._index_entities(observation, agent_team.upper())

        reward = 0.0
        if self._previous_own_elixir is not None:
            # Elixir only falls when a card is played; regeneration raises it.
            own_spent = max(0.0, self._previous_own_elixir - own_elixir)
            enemy_spent = max(0.0, self._previous_enemy_elixir - enemy_elixir)

            destroyed = sum(
                value for key, value in self._live_enemy.items() if key not in enemy_now
            )
            lost = sum(value for key, value in self._live_own.items() if key not in own_now)

            # Positive when the opponent's committed elixir dies for less than
            # it cost us to answer it.
            trade = (destroyed - own_spent) - (lost - enemy_spent)
            reward += self.config.trade_weight * trade

            if own_spent > 0.0:
                # Committing while the opponent is short on elixir is tempo;
                # committing into a full bar is how a push gets punished.
                advantage = own_elixir - enemy_elixir
                if enemy_elixir <= self.config.tempo_advantage_threshold and advantage > 0:
                    reward += self.config.tempo_weight * (own_spent * advantage / _MAX_ELIXIR)

        self._previous_own_elixir = own_elixir
        self._previous_enemy_elixir = enemy_elixir
        self._live_enemy = enemy_now
        self._live_own = own_now
        return self._clip_to_budget(reward)

    def _clip_to_budget(self, reward: float) -> float:
        """Keep total shaping inside its episode budget, both signs."""
        limit = self.config.episode_shaping_limit
        if limit <= 0:
            return 0.0
        remaining = limit - abs(self.shaping_spent)
        if remaining <= 0:
            return 0.0
        clipped = max(-remaining, min(remaining, reward))
        self.shaping_spent += clipped
        return clipped


class ElixirTradeRewardWrapper(gym.Wrapper):
    """Add elixir-economy shaping on top of the environment's own reward.

    Must sit directly on the environment while observations are still the raw
    dictionary, before any flattening wrapper.
    """

    def __init__(
        self,
        env: gym.Env,
        config: TradeRewardConfig | None = None,
        table: CardFeatureTable | None = None,
    ) -> None:
        super().__init__(env)
        self.tracker = ElixirEconomyTracker(config, table)

    def _raw_observation(self) -> dict[str, Any]:
        """Read the untouched bridge payload.

        The observation handed to a wrapper is already parsed into numpy
        arrays, which loses the entity identities the trade signal is built
        from, so the environment's retained raw dictionary is used instead.
        """
        raw = getattr(self.env.unwrapped, "_last_obs_raw", None)
        return raw if isinstance(raw, dict) else {}

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        self.tracker.reset()
        self.tracker.update(self._raw_observation())
        return observation, info

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = self.env.step(action)
        shaping = self.tracker.update(self._raw_observation())
        if shaping:
            info = dict(info)
            info["elixir_shaping"] = shaping
        return observation, float(reward) + shaping, terminated, truncated, info
