# Modified by NextoCR contributors; see NOTICE for attribution.
"""Sample random decks that are still coherent enough to play.

Eight cards drawn uniformly from the catalogue produce decks no human would
bring: no way to damage a tower, nothing that can hit air, twelve elixir of
tanks. Training against those teaches an agent to beat nonsense.

Roles are derived from the card properties rather than a hand-written list, so
the pool follows the simulator's catalogue instead of drifting from it. A deck
must be able to threaten a tower, answer air, deal with a swarm and cycle at a
playable cost, which is the floor every real archetype clears.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Iterable, Sequence

import gymnasium as gym

from crforge_gym.card_properties import CardFeatureTable, default_table

__all__ = [
    "DECK_SIZE",
    "DeckConstraints",
    "DeckSampler",
    "RandomDeckWrapper",
    "SampledDeck",
    "card_roles",
]

DECK_SIZE = 8

# A win condition must be able to reach a tower. Cards that only target
# buildings walk past defenders, and a Mortar-like building outranges the
# tower it shoots; both threaten a tower without needing a lane cleared.
_TOWER_THREAT_MIN_RANGE = 6.0


@dataclass(frozen=True)
class DeckConstraints:
    """The floor a sampled deck must clear to be worth training against."""

    min_win_conditions: int = 1
    min_spells: int = 1
    min_cheap_spells: int = 1
    min_anti_air: int = 2
    min_defensive: int = 1
    min_average_elixir: float = 3.0
    max_average_elixir: float = 4.4
    max_win_conditions: int = 3
    cheap_spell_max_cost: int = 3

    def __post_init__(self) -> None:
        if self.min_average_elixir > self.max_average_elixir:
            raise ValueError("min_average_elixir cannot exceed max_average_elixir")


@dataclass(frozen=True)
class SampledDeck:
    """One coherent deck plus the reasons it was accepted."""

    card_ids: tuple[str, ...]
    average_elixir: float
    roles: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "card_ids": list(self.card_ids),
            "average_elixir": round(self.average_elixir, 3),
            "roles": {role: list(cards) for role, cards in sorted(self.roles.items())},
        }


def _is_playable(card: dict[str, Any]) -> bool:
    """Keep only cards the simulator can run as an ordinary 8-card slot."""
    card_type = str(card.get("type", "")).upper()
    if card_type not in {"TROOP", "BUILDING", "SPELL"}:
        return False  # HERO forms carry abilities the action schema cannot express.
    if card.get("evolved") or card.get("baseCard") or card.get("heroForm"):
        return False  # Evolutions and hero forms are approximations right now.
    if card.get("mirror") or card.get("variants"):
        return False  # Mirror copies another card; it has no standalone identity.
    if card_type != "SPELL" and not card.get("unit"):
        return False  # A troop with no unit definition cannot be simulated.
    return True


def card_roles(table: CardFeatureTable, card_id: str) -> set[str]:
    """Classify a card from its simulated behaviour, not from a curated list."""
    card = table.cards.get(card_id)
    if card is None:
        return set()
    card_type = str(card.get("type", "")).upper()
    unit = table.units.get(str(card.get("unit"))) if card.get("unit") else None
    roles: set[str] = set()

    cost = float(card.get("cost", 0) or 0)
    hitpoints = float(unit.get("health", 0) or 0) if unit else 0.0
    attack_range = float(unit.get("range", 0) or 0) if unit else 0.0
    targets_air = str(unit.get("targetType", "")).upper() == "ALL" if unit else False
    only_buildings = bool(unit.get("targetOnlyBuildings")) if unit else False
    count = int(card.get("count", 1) or 1) + int(card.get("secondaryCount", 0) or 0)

    if card_type == "SPELL":
        roles.add("spell")
        roles.add("cheap_spell" if cost <= 3 else "big_spell")
        return roles

    if only_buildings:
        roles.add("win_condition")
    if card_type == "BUILDING":
        roles.add("building")
        if attack_range >= _TOWER_THREAT_MIN_RANGE:
            roles.add("win_condition")
        else:
            roles.add("defensive")
    if targets_air:
        roles.add("anti_air")
    if count >= 3:
        roles.add("swarm")
    if hitpoints >= 1400 and card_type == "TROOP":
        roles.add("tank")
        roles.add("defensive")
    if card_type == "TROOP" and not only_buildings and hitpoints >= 600:
        roles.add("defensive")
    return roles


class DeckSampler:
    """Draw seeded, reproducible decks that satisfy :class:`DeckConstraints`."""

    def __init__(
        self,
        table: CardFeatureTable | None = None,
        constraints: DeckConstraints | None = None,
        *,
        allowed_cards: Iterable[str] | None = None,
    ) -> None:
        self.table = table or default_table()
        self.constraints = constraints or DeckConstraints()
        pool = [
            card_id
            for card_id, card in self.table.cards.items()
            if _is_playable(card)
        ]
        if allowed_cards is not None:
            allowed = set(allowed_cards)
            pool = [card_id for card_id in pool if card_id in allowed]
        # Sorted so a seed maps to the same deck across processes and platforms,
        # which dict iteration order alone would not guarantee.
        self.pool: tuple[str, ...] = tuple(sorted(pool))
        self._roles = {card_id: card_roles(self.table, card_id) for card_id in self.pool}
        self._costs = {
            card_id: float(self.table.cards[card_id].get("cost", 0) or 0)
            for card_id in self.pool
        }
        if len(self.pool) < DECK_SIZE:
            raise ValueError(f"card pool too small to build a deck: {len(self.pool)}")
        self._verify_pool_can_satisfy_constraints()

    def _cards_with_role(self, role: str) -> list[str]:
        return [card_id for card_id in self.pool if role in self._roles[card_id]]

    def _verify_pool_can_satisfy_constraints(self) -> None:
        """Fail loudly at construction rather than looping forever at sample time."""
        requirements = {
            "win_condition": self.constraints.min_win_conditions,
            "spell": self.constraints.min_spells,
            "cheap_spell": self.constraints.min_cheap_spells,
            "anti_air": self.constraints.min_anti_air,
            "defensive": self.constraints.min_defensive,
        }
        for role, needed in requirements.items():
            available = len(self._cards_with_role(role))
            if available < needed:
                raise ValueError(
                    f"card pool has {available} cards with role {role!r}, need {needed}"
                )

    def satisfies(self, card_ids: Sequence[str]) -> bool:
        if len(set(card_ids)) != DECK_SIZE:
            return False
        counts: dict[str, int] = {}
        for card_id in card_ids:
            for role in self._roles.get(card_id, ()):  # unknown cards contribute nothing
                counts[role] = counts.get(role, 0) + 1
        constraints = self.constraints
        if counts.get("win_condition", 0) < constraints.min_win_conditions:
            return False
        if counts.get("win_condition", 0) > constraints.max_win_conditions:
            return False
        if counts.get("spell", 0) < constraints.min_spells:
            return False
        if counts.get("cheap_spell", 0) < constraints.min_cheap_spells:
            return False
        if counts.get("anti_air", 0) < constraints.min_anti_air:
            return False
        if counts.get("defensive", 0) < constraints.min_defensive:
            return False
        average = self.average_elixir(card_ids)
        return constraints.min_average_elixir <= average <= constraints.max_average_elixir

    def average_elixir(self, card_ids: Sequence[str]) -> float:
        if not card_ids:
            return 0.0
        return sum(self._costs.get(card_id, 0.0) for card_id in card_ids) / len(card_ids)

    def sample(self, rng: random.Random | int | None = None, *, max_attempts: int = 400) -> SampledDeck:
        """Return one deck satisfying the constraints.

        Roles are seeded first and the remainder filled at random, so the deck
        is varied without ever failing the archetype floor.
        """
        generator = rng if isinstance(rng, random.Random) else random.Random(rng)
        constraints = self.constraints
        for _ in range(max_attempts):
            chosen: list[str] = []
            seeds = (
                ("win_condition", constraints.min_win_conditions),
                ("cheap_spell", constraints.min_cheap_spells),
                ("anti_air", constraints.min_anti_air),
                ("defensive", constraints.min_defensive),
            )
            failed = False
            for role, needed in seeds:
                available = [
                    card_id for card_id in self._cards_with_role(role) if card_id not in chosen
                ]
                # A seeded card often covers several roles, so only top up.
                have = sum(1 for card_id in chosen if role in self._roles[card_id])
                missing = needed - have
                if missing <= 0:
                    continue
                if len(available) < missing:
                    failed = True
                    break
                chosen.extend(generator.sample(available, missing))
            if failed or len(chosen) > DECK_SIZE:
                continue
            remaining = [card_id for card_id in self.pool if card_id not in chosen]
            chosen.extend(generator.sample(remaining, DECK_SIZE - len(chosen)))
            generator.shuffle(chosen)
            if self.satisfies(chosen):
                return self._describe(tuple(chosen))
        raise RuntimeError(
            f"could not sample a deck satisfying the constraints in {max_attempts} attempts"
        )

    def _describe(self, card_ids: tuple[str, ...]) -> SampledDeck:
        roles: dict[str, list[str]] = {}
        for card_id in card_ids:
            for role in sorted(self._roles[card_id]):
                roles.setdefault(role, []).append(card_id)
        return SampledDeck(
            card_ids=card_ids,
            average_elixir=self.average_elixir(card_ids),
            roles={role: tuple(cards) for role, cards in roles.items()},
        )


class RandomDeckWrapper(gym.Wrapper):
    """Draw a fresh coherent deck for both players at every episode.

    Facing one deck for millions of steps produces a specialist: the mortar
    mirror run reached 64% against its own league while still losing to a
    scripted bot. Resampling per episode is what forces a policy to read the
    board instead of replaying a memorised script.

    Must wrap :class:`~crforge_gym.env.CRForgeEnv` closely enough that
    ``set_decks`` is reachable, and sits below any observation wrapper.
    """

    def __init__(
        self,
        env: gym.Env,
        sampler: DeckSampler | None = None,
        *,
        seed: int = 0,
        mirror: bool = False,
    ) -> None:
        super().__init__(env)
        self.sampler = sampler or DeckSampler()
        self.mirror = mirror
        self._rng = random.Random(seed)
        self._episode = 0
        self.current_blue: SampledDeck | None = None
        self.current_red: SampledDeck | None = None

    def reset(self, **kwargs: Any):
        # Derive per-episode streams from the run seed so a resumed run replays
        # the same deck sequence instead of restarting the draw.
        episode_rng = random.Random(self._rng.randrange(1 << 62))
        blue = self.sampler.sample(episode_rng)
        red = blue if self.mirror else self.sampler.sample(episode_rng)
        target = self.env.unwrapped
        target.set_decks(list(blue.card_ids), list(red.card_ids))
        self.current_blue = blue
        self.current_red = red
        self._episode += 1

        observation, info = self.env.reset(**kwargs)
        info = dict(info)
        info["blue_deck"] = list(blue.card_ids)
        info["red_deck"] = list(red.card_ids)
        info["blue_average_elixir"] = blue.average_elixir
        return observation, info
