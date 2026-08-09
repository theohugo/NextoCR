# Modified by NextoCR contributors; see NOTICE for attribution.
"""Tests for property-based card features.

These guard the property that makes varied decks learnable: two cards that play
differently must produce different feature vectors, and the numbers must come
from the simulator's own catalogue rather than a hand-maintained copy.
"""

from __future__ import annotations

import numpy as np
import pytest

from crforge_gym.card_properties import (
    CARD_FEATURE_COUNT,
    CARD_FEATURE_NAMES,
    CardFeatureTable,
    default_table,
)


@pytest.fixture(scope="module")
def table() -> CardFeatureTable:
    return default_table()


def feature(vector: np.ndarray, name: str) -> float:
    return float(vector[CARD_FEATURE_NAMES.index(name)])


def test_catalogue_loads_the_simulator_cards(table: CardFeatureTable) -> None:
    assert len(table.cards) > 100
    assert "mortar" in table.cards
    assert "Minion" in table.units


def test_every_card_produces_a_bounded_vector(table: CardFeatureTable) -> None:
    for card_id in table.card_ids:
        vector = table.card_features(card_id)
        assert vector.shape == (CARD_FEATURE_COUNT,)
        assert vector.dtype == np.float32
        assert np.all(np.isfinite(vector)), card_id
        # Fixed divisors plus clamping keep an outlier from dominating the input.
        assert np.all(vector >= 0.0) and np.all(vector <= 1.0), card_id


def test_unknown_card_is_all_zeros_and_flagged(table: CardFeatureTable) -> None:
    vector = table.card_features("not_a_real_card")

    assert feature(vector, "known") == 0.0
    assert np.count_nonzero(vector) == 0
    # An empty hand slot must be distinguishable from a cheap, weak card.
    assert feature(table.card_features("goblins"), "known") == 1.0


def test_air_targeting_is_exposed(table: CardFeatureTable) -> None:
    """The single most decision-relevant property: can this answer air?"""
    minions = table.card_features("minions")
    mortar = table.card_features("mortar")

    assert feature(minions, "is_air_unit") == 1.0
    assert feature(minions, "targets_ground") == 1.0
    # A Mortar cannot shoot at an air unit, and the vector has to say so.
    assert feature(mortar, "targets_air") == 0.0
    assert feature(mortar, "is_building") == 1.0


def test_spells_and_troops_are_separable(table: CardFeatureTable) -> None:
    fireball = table.card_features("fireball")
    goblins = table.card_features("goblins")

    assert feature(fireball, "is_spell") == 1.0
    assert feature(fireball, "hitpoints") == 0.0
    assert feature(fireball, "can_deploy_on_enemy_side") == 1.0
    assert feature(goblins, "is_troop") == 1.0
    assert feature(goblins, "hitpoints") > 0.0
    # Goblins arrive as a group; the count has to survive into the features.
    assert feature(goblins, "unit_count") > feature(fireball, "unit_count")


def test_similar_cards_are_close_and_different_cards_are_far(table: CardFeatureTable) -> None:
    """Generalising across decks depends on this geometry, not on identity."""
    minions = table.card_features("minions")
    goblins = table.card_features("goblins")
    mortar = table.card_features("mortar")

    swarm_distance = float(np.linalg.norm(minions - goblins))
    building_distance = float(np.linalg.norm(minions - mortar))

    assert swarm_distance < building_distance


def test_multi_unit_card_uses_its_strongest_body(table: CardFeatureTable) -> None:
    rascals = table.card_features("rascals")

    # RascalBoy is the tanky body; averaging it away with the girls would make
    # the card look far flimsier than it plays.
    boy_hp = float(table.units["RascalBoy"]["health"])
    girl_hp = float(table.units["RascalGirl"]["health"])
    assert boy_hp > girl_hp
    assert feature(rascals, "hitpoints") == pytest.approx(boy_hp / 3000.0, rel=1e-6)
    assert feature(rascals, "unit_count") > 0.0


def test_unit_features_match_the_card_that_spawns_them(table: CardFeatureTable) -> None:
    minion_unit = table.unit_features("Minion")
    minions_card = table.card_features("minions")

    for name in ("hitpoints", "damage", "targets_air", "is_air_unit"):
        assert feature(minion_unit, name) == pytest.approx(feature(minions_card, name))


def test_unknown_unit_is_all_zeros(table: CardFeatureTable) -> None:
    assert np.count_nonzero(table.unit_features("NoSuchUnit")) == 0
    assert np.count_nonzero(table.unit_features(None)) == 0
