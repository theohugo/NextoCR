# Modified by NextoCR contributors; see NOTICE for attribution.
"""Describe cards and units by their properties instead of by their identity.

The bridge encodes a card as its cost, its type and a CRC32 identity hashed into
a scalar.  With a single fixed deck a policy simply memorises those eight
scalars, but a hash carries no structure: nothing tells the network that a
Musketeer and an Archer behave alike, or that one of them can hit air.  Training
against varied decks therefore cannot generalise, and a deck a friend brings
that was never seen in training is unplayable.

This module turns an identity into the numbers that actually drive a decision -
hitpoints, damage, range, speed, what it can target - read from the simulator's
own ``cards.json`` and ``units.json`` so the features can never drift from the
engine they describe.

Values are scaled into roughly ``[0, 1]`` by fixed divisors rather than by
statistics of the current card pool, so adding a card later never silently
rescales the features a trained policy already relies on.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

__all__ = [
    "CARD_FEATURE_NAMES",
    "CARD_FEATURE_COUNT",
    "CardFeatureTable",
    "default_table",
    "find_card_data_dir",
]


CARD_FEATURE_NAMES: tuple[str, ...] = (
    "known",
    "cost",
    "is_troop",
    "is_building",
    "is_spell",
    "is_hero",
    "unit_count",
    "hitpoints",
    "total_hitpoints",
    "shield_hitpoints",
    "damage",
    "damage_per_second",
    "area_damage",
    "death_damage",
    "attack_range",
    "sight_range",
    "minimum_range",
    "move_speed",
    "deploy_time",
    "lifetime",
    "targets_air",
    "targets_ground",
    "targets_only_buildings",
    "is_air_unit",
    "spawns_on_death",
    "can_deploy_on_enemy_side",
)
CARD_FEATURE_COUNT = len(CARD_FEATURE_NAMES)

# Fixed scales, chosen so a typical card lands well inside [0, 1] and an
# outlier saturates instead of dominating the input.
_SCALE_COST = 10.0
_SCALE_COUNT = 10.0
_SCALE_HP = 3000.0
_SCALE_TOTAL_HP = 6000.0
_SCALE_DAMAGE = 1000.0
_SCALE_DPS = 500.0
_SCALE_RANGE = 12.0
_SCALE_SPEED = 150.0
_SCALE_TIME = 5.0
_SCALE_LIFETIME = 60.0

_TARGETS_AIR = frozenset({"ALL", "AIR"})
_TARGETS_GROUND = frozenset({"ALL", "GROUND"})

_DATA_DIR_ENVIRONMENT_VARIABLE = "NEXTOCR_CARD_DATA_DIR"
_RELATIVE_DATA_DIR = Path("data") / "src" / "main" / "resources" / "cards"


def find_card_data_dir(start: Path | None = None) -> Path:
    """Locate the simulator's card resources, walking up from this file."""
    override = os.environ.get(_DATA_DIR_ENVIRONMENT_VARIABLE, "").strip()
    if override:
        candidate = Path(override).expanduser()
        if (candidate / "cards.json").is_file():
            return candidate
        raise FileNotFoundError(
            f"{_DATA_DIR_ENVIRONMENT_VARIABLE} does not contain cards.json: {candidate}"
        )
    origin = (start or Path(__file__)).resolve()
    for parent in origin.parents:
        candidate = parent / _RELATIVE_DATA_DIR
        if (candidate / "cards.json").is_file():
            return candidate
    raise FileNotFoundError(
        "could not locate data/src/main/resources/cards; set "
        f"{_DATA_DIR_ENVIRONMENT_VARIABLE} to the card resource directory"
    )


def _number(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _clamp_unit(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


@dataclass(frozen=True)
class CardFeatureTable:
    """Fixed-length property vectors for every card and unit in the catalogue."""

    cards: dict[str, dict[str, Any]]
    units: dict[str, dict[str, Any]]

    @classmethod
    def load(cls, data_dir: Path | None = None) -> "CardFeatureTable":
        directory = data_dir or find_card_data_dir()
        with (directory / "cards.json").open("r", encoding="utf-8") as handle:
            raw_cards = json.load(handle)
        with (directory / "units.json").open("r", encoding="utf-8") as handle:
            raw_units = json.load(handle)
        cards = (
            {str(card["id"]): card for card in raw_cards if isinstance(card, dict) and "id" in card}
            if isinstance(raw_cards, list)
            else {str(key): value for key, value in raw_cards.items()}
        )
        units = (
            {str(unit["name"]): unit for unit in raw_units if isinstance(unit, dict) and "name" in unit}
            if isinstance(raw_units, list)
            else {str(key): value for key, value in raw_units.items()}
        )
        return cls(cards=cards, units=units)

    @property
    def card_ids(self) -> Iterable[str]:
        return self.cards.keys()

    def card_features(self, card_id: str | None) -> np.ndarray:
        """Return the property vector for *card_id*, all zeros when unknown.

        The leading ``known`` flag lets a policy distinguish "no card here" from
        "a card whose statistics happen to be small".
        """
        card = self.cards.get(str(card_id)) if card_id else None
        if card is None:
            return np.zeros(CARD_FEATURE_COUNT, dtype=np.float32)

        card_type = str(card.get("type", "")).upper()
        unit = self.units.get(str(card.get("unit"))) if card.get("unit") else None
        secondary = self.units.get(str(card.get("secondaryUnit"))) if card.get("secondaryUnit") else None

        count = _number(card.get("count"), 1.0 if unit is not None else 0.0)
        count += _number(card.get("secondaryCount"), 0.0)

        hitpoints = _number(unit.get("health") if unit else None)
        if secondary is not None:
            # A mixed group is described by its strongest body, so a Rascals-like
            # card is not flattened to the statistics of its weakest unit.
            hitpoints = max(hitpoints, _number(secondary.get("health")))
        damage = _number(unit.get("damage") if unit else None)
        if secondary is not None:
            damage = max(damage, _number(secondary.get("damage")))
        cooldown = _number(unit.get("attackCooldown") if unit else None)
        dps = damage / cooldown if cooldown > 0 else 0.0

        target_type = str(unit.get("targetType", "")).upper() if unit else ""
        movement = str(unit.get("movementType", "")).upper() if unit else ""

        features = np.zeros(CARD_FEATURE_COUNT, dtype=np.float32)
        values = {
            "known": 1.0,
            "cost": _number(card.get("cost")) / _SCALE_COST,
            "is_troop": 1.0 if card_type == "TROOP" else 0.0,
            "is_building": 1.0 if card_type == "BUILDING" else 0.0,
            "is_spell": 1.0 if card_type == "SPELL" else 0.0,
            "is_hero": 1.0 if card_type == "HERO" else 0.0,
            "unit_count": count / _SCALE_COUNT,
            "hitpoints": hitpoints / _SCALE_HP,
            "total_hitpoints": (hitpoints * max(count, 1.0)) / _SCALE_TOTAL_HP,
            "shield_hitpoints": _number(unit.get("shieldHitpoints") if unit else None) / _SCALE_HP,
            "damage": damage / _SCALE_DAMAGE,
            "damage_per_second": dps / _SCALE_DPS,
            "area_damage": 1.0 if _number(unit.get("areaDamageRadius") if unit else None) > 0 else 0.0,
            "death_damage": _number(unit.get("deathDamage") if unit else None) / _SCALE_DAMAGE,
            "attack_range": _number(unit.get("range") if unit else None) / _SCALE_RANGE,
            "sight_range": _number(unit.get("sightRange") if unit else None) / _SCALE_RANGE,
            "minimum_range": _number(unit.get("minimumRange") if unit else None) / _SCALE_RANGE,
            "move_speed": _number(unit.get("speed") if unit else None) / _SCALE_SPEED,
            "deploy_time": _number(unit.get("deployTime") if unit else None) / _SCALE_TIME,
            "lifetime": _number(unit.get("lifeTime") if unit else None) / _SCALE_LIFETIME,
            "targets_air": 1.0 if target_type in _TARGETS_AIR else 0.0,
            "targets_ground": 1.0 if target_type in _TARGETS_GROUND else 0.0,
            "targets_only_buildings": 1.0 if unit and unit.get("targetOnlyBuildings") else 0.0,
            "is_air_unit": 1.0 if movement == "AIR" else 0.0,
            "spawns_on_death": 1.0 if unit and unit.get("deathSpawn") else 0.0,
            "can_deploy_on_enemy_side": 1.0 if card.get("canDeployOnEnemySide") else 0.0,
        }
        for index, name in enumerate(CARD_FEATURE_NAMES):
            features[index] = _clamp_unit(values[name])
        return features

    def unit_features(self, unit_name: str | None) -> np.ndarray:
        """Return the property vector for a unit already on the arena."""
        unit = self.units.get(str(unit_name)) if unit_name else None
        if unit is None:
            return np.zeros(CARD_FEATURE_COUNT, dtype=np.float32)
        # Reuse the card path so a unit and the card that spawns it are
        # described on exactly the same axes.
        synthetic = {"type": "TROOP", "cost": 0, "unit": unit.get("name")}
        return self.card_features_from(synthetic)

    def card_features_from(self, card: dict[str, Any]) -> np.ndarray:
        """Build features for a card-shaped mapping that is not in the catalogue."""
        card_id = f"__inline__{id(card)}"
        table = CardFeatureTable(cards={card_id: card}, units=self.units)
        return table.card_features(card_id)


@lru_cache(maxsize=1)
def default_table() -> CardFeatureTable:
    """Load the catalogue once per process."""
    return CardFeatureTable.load()
