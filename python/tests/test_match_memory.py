# Modified by NextoCR contributors; see NOTICE for attribution.
"""Tests for within-match memory.

What matters is that the agent can tell which of its own cards are coming back
and what the opponent has already shown; a single frame cannot answer either.
"""

from __future__ import annotations

import numpy as np

from crforge_gym.card_properties import CARD_FEATURE_COUNT, CARD_FEATURE_NAMES
from crforge_gym.match_memory import (
    MEMORY_FEATURE_COUNT,
    OWN_DECK_SLOTS,
    MatchMemory,
)


DECK = [
    "knight", "archer", "fireball", "arrows",
    "giant", "musketeer", "minions", "valkyrie",
]


def frame(hand: list[str], enemy_units: list[str] | None = None) -> dict:
    return {
        "bluePlayer": {"hand": [{"id": card} for card in hand]},
        "redPlayer": {},
        "entities": [
            {"id": index, "name": name, "team": "RED"}
            for index, name in enumerate(enemy_units or [])
        ],
    }


def test_vector_has_a_stable_width() -> None:
    memory = MatchMemory()
    memory.reset(DECK)

    vector = memory.observe(frame(["knight", "archer", "fireball", "arrows"]))

    assert vector.shape == (MEMORY_FEATURE_COUNT,)
    assert vector.dtype == np.float32
    assert np.all(np.isfinite(vector))


def test_cards_not_in_hand_are_flagged_as_coming_back() -> None:
    memory = MatchMemory()
    memory.reset(DECK)

    vector = memory.observe(frame(["knight", "archer", "fireball", "arrows"]))

    slot_width = CARD_FEATURE_COUNT + 1
    in_hand = {
        memory.summary()["own_deck"][slot]: vector[slot * slot_width + CARD_FEATURE_COUNT]
        for slot in range(len(memory.summary()["own_deck"]))
    }
    assert in_hand["knight"] == 1.0
    assert in_hand["archer"] == 1.0
    # The four cards not in hand are exactly the cycle coming back.
    assert in_hand["giant"] == 0.0
    assert in_hand["minions"] == 0.0


def test_enemy_cards_accumulate_and_never_unreveal() -> None:
    memory = MatchMemory()
    memory.reset(DECK)

    memory.observe(frame([], enemy_units=["Giant"]))
    assert "giant" in memory.revealed_cards

    # The Giant dies, but the opponent is still known to hold it.
    memory.observe(frame([], enemy_units=[]))
    assert "giant" in memory.revealed_cards

    memory.observe(frame([], enemy_units=["Minion"]))
    assert set(memory.revealed_cards) >= {"giant", "minions"}


def test_reveal_answers_the_question_that_decides_a_defence() -> None:
    """Once an air answer has been shown, the summary must say so."""
    memory = MatchMemory()
    memory.reset(DECK)
    enemy_start = OWN_DECK_SLOTS * (CARD_FEATURE_COUNT + 1)
    air_index = CARD_FEATURE_NAMES.index("targets_air")

    before = memory.observe(frame([]))
    assert before[enemy_start + CARD_FEATURE_COUNT + air_index] == 0.0

    after = memory.observe(frame([], enemy_units=["Musketeer"]))

    assert after[enemy_start + CARD_FEATURE_COUNT + air_index] == 1.0


def test_reveal_fraction_grows_as_the_match_goes_on() -> None:
    memory = MatchMemory()
    memory.reset(DECK)
    fraction_index = OWN_DECK_SLOTS * (CARD_FEATURE_COUNT + 1) + 2 * CARD_FEATURE_COUNT

    first = memory.observe(frame([], enemy_units=["Giant"]))[fraction_index]
    second = memory.observe(frame([], enemy_units=["Minion", "Knight"]))[fraction_index]

    assert 0.0 < first < second <= 1.0


def test_reset_forgets_the_previous_match() -> None:
    memory = MatchMemory()
    memory.reset(DECK)
    memory.observe(frame([], enemy_units=["Giant"]))
    assert memory.revealed_cards

    memory.reset(DECK)

    assert memory.revealed_cards == ()
    assert not np.any(memory.features({})[OWN_DECK_SLOTS * (CARD_FEATURE_COUNT + 1) :])


def test_unknown_units_do_not_pollute_the_memory() -> None:
    memory = MatchMemory()
    memory.reset(DECK)

    memory.observe(frame([], enemy_units=["NotARealUnit"]))

    assert memory.revealed_cards == ()


def test_slots_stay_put_once_a_card_has_been_seen() -> None:
    """Slot order is arbitrary but must not shift mid-match."""
    memory = MatchMemory()
    memory.reset()

    memory.observe(frame(["knight", "archer", "fireball", "arrows"]))
    first = list(memory.summary()["own_deck"])

    memory.observe(frame(["giant", "musketeer", "minions", "valkyrie"]))
    after = list(memory.summary()["own_deck"])

    assert after[: len(first)] == first
    assert len(after) == OWN_DECK_SLOTS


def test_own_deck_is_discovered_without_being_declared() -> None:
    memory = MatchMemory()
    memory.reset()

    memory.observe(frame(["knight", "archer", "fireball", "arrows"]))
    memory.observe(frame(["giant", "musketeer", "minions", "valkyrie"]))

    assert sorted(memory.summary()["own_deck"]) == sorted(DECK)


def test_summary_reports_what_is_known() -> None:
    memory = MatchMemory()
    memory.reset(DECK)
    memory.observe(frame([], enemy_units=["Giant"]))

    summary = memory.summary()

    assert summary["own_deck"] == DECK
    assert summary["revealed_enemy_cards"] == ["giant"]
    assert 0.0 < summary["revealed_fraction"] <= 1.0
