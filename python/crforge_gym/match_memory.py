# Modified by NextoCR contributors; see NOTICE for attribution.
"""Remember, within a match, what has been played on both sides.

A single frame does not say which card is about to come back around, nor what
the opponent has already shown. Without that, a policy cannot hold a counter
back for the push it knows is coming, and cannot cycle to an answer on purpose:
it can only react to what is on screen right now.

Two memories are exposed as features:

* **own cycle** - every card of the agent's deck, with its properties and
  whether it is currently in hand. The four cards *not* in hand are precisely
  the ones coming back, which is what makes deliberate cycling possible.
* **opponent reveal** - the cards the opponent has actually played, aggregated
  by property. This answers the questions that decide a defence: do they have
  an answer to air, a big spell, a second win condition?

Explicit features are used rather than hoping a recurrent policy infers the
same thing: the information is cheap to compute exactly, and learning it from
scratch through a memory cell would cost far more samples.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from crforge_gym.card_properties import (
    CARD_FEATURE_COUNT,
    CardFeatureTable,
    default_table,
)

__all__ = [
    "MEMORY_FEATURE_COUNT",
    "MatchMemory",
    "MatchMemoryObservationWrapper",
    "OWN_DECK_SLOTS",
]

OWN_DECK_SLOTS = 8
#: Per own-deck slot: the card's properties plus an "in hand right now" flag.
_OWN_SLOT_WIDTH = CARD_FEATURE_COUNT + 1
#: Opponent reveal: mean properties, max properties, and how much is revealed.
_ENEMY_WIDTH = CARD_FEATURE_COUNT * 2 + 1
MEMORY_FEATURE_COUNT = OWN_DECK_SLOTS * _OWN_SLOT_WIDTH + _ENEMY_WIDTH


class MatchMemory:
    """Accumulate per-match knowledge and render it as a fixed-length vector."""

    def __init__(
        self,
        table: CardFeatureTable | None = None,
        *,
        agent_team: str = "BLUE",
    ) -> None:
        self.table = table or default_table()
        self.agent_team = agent_team.upper()
        self._unit_to_card = self._build_unit_index(self.table)
        self.reset()

    @staticmethod
    def _build_unit_index(table: CardFeatureTable) -> dict[str, str]:
        """Map a unit name back to the card that summons it.

        The arena reports units, not cards, so a Skeleton has to be traced back
        to the card that put it there before it can count as revealed.
        """
        index: dict[str, str] = {}
        for card_id, card in table.cards.items():
            for key in ("unit", "secondaryUnit"):
                unit = card.get(key)
                if unit and str(unit) not in index:
                    index[str(unit)] = card_id
        return index

    def reset(self, own_deck: Sequence[str] | None = None) -> None:
        """Start a new match, optionally declaring the agent's own deck.

        When the deck is not supplied it is discovered from the hands actually
        held, in first-seen order. The self-play opponent only ever sees a
        mirrored observation and never learns which eight cards it was dealt,
        so both sides have to work from the same incomplete information.
        """
        self._own_deck: list[str] = list(dict.fromkeys(own_deck or ()))[:OWN_DECK_SLOTS]
        self._own_features = [
            self.table.card_features(card_id) for card_id in self._own_deck
        ]
        self._revealed: set[str] = set()

    @property
    def revealed_cards(self) -> tuple[str, ...]:
        return tuple(sorted(self._revealed))

    def observe(self, observation: dict[str, Any]) -> np.ndarray:
        """Fold *observation* into the memory and return the feature vector."""
        if isinstance(observation, dict):
            self._absorb_enemy_units(observation)
            self._absorb_enemy_hand_if_visible(observation)
        return self.features(observation if isinstance(observation, dict) else {})

    def _absorb_enemy_units(self, observation: dict[str, Any]) -> None:
        enemy_team = "RED" if self.agent_team == "BLUE" else "BLUE"
        for entity in observation.get("entities") or []:
            if str(entity.get("team", "")).upper() != enemy_team:
                continue
            card_id = self._unit_to_card.get(str(entity.get("name")))
            if card_id:
                self._revealed.add(card_id)

    def _absorb_enemy_hand_if_visible(self, observation: dict[str, Any]) -> None:
        """Count a spell only once it has been cast.

        Spells leave no unit behind, so they would never be revealed by the
        arena alone. They are picked up from the opponent's own played card
        when the bridge reports one, never from their hidden hand.
        """
        played = observation.get("lastEnemyCardPlayed") or observation.get("redLastCardPlayed")
        if isinstance(played, dict):
            card_id = played.get("id")
            if card_id and str(card_id) in self.table.cards:
                self._revealed.add(str(card_id))
        elif isinstance(played, str) and played in self.table.cards:
            self._revealed.add(played)

    def _discover_own_cards(self, in_hand: set[str]) -> None:
        """Fill deck slots in first-seen order.

        Slot order is arbitrary but fixed for the match, which is what matters:
        the policy reads a slot's properties, never its position.
        """
        for card_id in sorted(in_hand):
            if len(self._own_deck) >= OWN_DECK_SLOTS:
                return
            if card_id not in self._own_deck and card_id in self.table.cards:
                self._own_deck.append(card_id)
                self._own_features.append(self.table.card_features(card_id))

    def _own_hand_ids(self, observation: dict[str, Any]) -> set[str]:
        key = "bluePlayer" if self.agent_team == "BLUE" else "redPlayer"
        player = observation.get(key) or {}
        return {
            str(card.get("id"))
            for card in (player.get("hand") or [])
            if isinstance(card, dict) and card.get("id")
        }

    def features(self, observation: dict[str, Any]) -> np.ndarray:
        vector = np.zeros(MEMORY_FEATURE_COUNT, dtype=np.float32)
        in_hand = self._own_hand_ids(observation)
        self._discover_own_cards(in_hand)

        for slot, card_id in enumerate(self._own_deck[:OWN_DECK_SLOTS]):
            start = slot * _OWN_SLOT_WIDTH
            vector[start : start + CARD_FEATURE_COUNT] = self._own_features[slot]
            # A slot that is not in hand is a card coming back around, which is
            # exactly what deliberate cycling needs to know.
            vector[start + CARD_FEATURE_COUNT] = 1.0 if card_id in in_hand else 0.0

        enemy_start = OWN_DECK_SLOTS * _OWN_SLOT_WIDTH
        if self._revealed:
            revealed = np.stack(
                [self.table.card_features(card_id) for card_id in sorted(self._revealed)]
            )
            vector[enemy_start : enemy_start + CARD_FEATURE_COUNT] = revealed.mean(axis=0)
            vector[
                enemy_start + CARD_FEATURE_COUNT : enemy_start + 2 * CARD_FEATURE_COUNT
            ] = revealed.max(axis=0)
            vector[enemy_start + 2 * CARD_FEATURE_COUNT] = min(
                1.0, len(self._revealed) / float(OWN_DECK_SLOTS)
            )
        return vector

    def summary(self) -> dict[str, Any]:
        """Human-readable memory state, for the replay viewer and debugging."""
        return {
            "own_deck": list(self._own_deck),
            "revealed_enemy_cards": list(self.revealed_cards),
            "revealed_fraction": round(
                min(1.0, len(self._revealed) / float(OWN_DECK_SLOTS)), 3
            ),
        }


def stack_memory_features(memories: Iterable[MatchMemory], observation: dict[str, Any]) -> np.ndarray:
    """Vector-env helper: one memory row per environment."""
    return np.stack([memory.features(observation) for memory in memories])


class MatchMemoryObservationWrapper(gym.ObservationWrapper):
    """Append per-match memory to a flat observation.

    Sits above the static preprocessing wrapper, which requires the exact
    legacy width, and reads the raw dictionary the environment keeps for
    opponents so no extra bridge round trip is needed.
    """

    def __init__(self, env: gym.Env, table: CardFeatureTable | None = None) -> None:
        super().__init__(env)
        self.memory = MatchMemory(table)
        base = env.observation_space
        if not isinstance(base, spaces.Box) or len(base.shape) != 1:
            raise ValueError("MatchMemoryObservationWrapper requires a flat Box observation")
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(base.shape[0]) + MEMORY_FEATURE_COUNT,),
            dtype=np.float32,
        )

    def _raw(self) -> dict[str, Any]:
        raw = getattr(self.env.unwrapped, "_last_obs_raw", None)
        return raw if isinstance(raw, dict) else {}

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        inner = self.env.unwrapped
        # The deck can change every episode, so the memory is re-declared here
        # rather than once at construction.
        self.memory.reset(getattr(inner, "blue_deck", ()) or ())
        return self.observation(observation), info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        features = self.memory.observe(self._raw())
        return np.concatenate(
            [np.asarray(observation, dtype=np.float32).ravel(), features]
        ).astype(np.float32)
