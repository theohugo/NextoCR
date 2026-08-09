# Modified by NextoCR contributors; see NOTICE for attribution.
"""Tests for coherent random deck sampling.

The point of the sampler is that a random deck is still a *playable* deck: it
can threaten a tower, answer air and cycle at a sane cost. These tests assert
that floor rather than merely that eight cards came back.
"""

from __future__ import annotations

import random

import gymnasium as gym
import pytest

from crforge_gym.card_properties import default_table
from crforge_gym.deck_sampler import (
    DECK_SIZE,
    DeckConstraints,
    DeckSampler,
    card_roles,
)


@pytest.fixture(scope="module")
def sampler() -> DeckSampler:
    return DeckSampler()


def test_pool_excludes_forms_the_simulator_only_approximates(sampler: DeckSampler) -> None:
    table = default_table()
    for card_id in sampler.pool:
        card = table.cards[card_id]
        assert not card.get("evolved"), card_id
        assert not card.get("heroForm"), card_id
        assert not card.get("mirror"), card_id
        assert str(card.get("type")).upper() in {"TROOP", "BUILDING", "SPELL"}, card_id
    assert len(sampler.pool) >= 40


def test_sampled_decks_are_always_coherent(sampler: DeckSampler) -> None:
    for seed in range(60):
        deck = sampler.sample(seed)

        assert len(deck.card_ids) == DECK_SIZE
        assert len(set(deck.card_ids)) == DECK_SIZE
        assert sampler.satisfies(deck.card_ids)
        assert 3.0 <= deck.average_elixir <= 4.4, (seed, deck.card_ids, deck.average_elixir)
        # The three properties that separate a deck from eight random cards.
        assert deck.roles.get("win_condition"), deck.card_ids
        assert len(deck.roles.get("anti_air", ())) >= 2, deck.card_ids
        assert deck.roles.get("spell"), deck.card_ids


def test_sampling_is_reproducible_and_varied(sampler: DeckSampler) -> None:
    assert sampler.sample(7).card_ids == sampler.sample(7).card_ids
    assert sampler.sample(random.Random(11)).card_ids == sampler.sample(random.Random(11)).card_ids

    decks = {sampler.sample(seed).card_ids for seed in range(40)}
    # Reproducible must not mean "always the same deck".
    assert len(decks) >= 30


def test_a_deck_missing_air_defence_is_rejected(sampler: DeckSampler) -> None:
    ground_only = [
        card_id
        for card_id in sampler.pool
        if "anti_air" not in card_roles(default_table(), card_id)
    ][:DECK_SIZE]

    assert len(ground_only) == DECK_SIZE
    assert not sampler.satisfies(ground_only)


def test_duplicate_cards_are_rejected(sampler: DeckSampler) -> None:
    deck = list(sampler.sample(3).card_ids)
    deck[1] = deck[0]

    assert not sampler.satisfies(deck)


def test_mortar_is_recognised_as_a_win_condition() -> None:
    """A Mortar outranges the tower, so it threatens without clearing a lane."""
    roles = card_roles(default_table(), "mortar")

    assert "win_condition" in roles
    assert "building" in roles


def test_impossible_constraints_fail_at_construction_not_at_sample_time() -> None:
    with pytest.raises(ValueError, match="role"):
        DeckSampler(constraints=DeckConstraints(min_anti_air=99))


def test_restricting_the_pool_is_honoured() -> None:
    table = default_table()
    subset = sorted(table.cards)[:80]

    sampler = DeckSampler(allowed_cards=subset)

    assert set(sampler.pool).issubset(set(subset))


def test_manifest_round_trip(sampler: DeckSampler) -> None:
    manifest = sampler.sample(5).to_manifest()

    assert len(manifest["card_ids"]) == DECK_SIZE
    assert isinstance(manifest["average_elixir"], float)
    assert "win_condition" in manifest["roles"]


class _StubEnv(gym.Env):
    """Minimal stand-in for CRForgeEnv: records the decks it was given."""

    def __init__(self) -> None:
        self.decks: list[tuple[list[str], list[str]]] = []
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(1)

    def set_decks(self, blue, red):
        assert len(blue) == DECK_SIZE and len(red) == DECK_SIZE
        self.decks.append((list(blue), list(red)))

    def reset(self, **kwargs):
        return {}, {}

    def close(self):
        pass


def test_random_deck_wrapper_redraws_every_episode() -> None:
    from crforge_gym.deck_sampler import RandomDeckWrapper

    env = _StubEnv()
    wrapper = RandomDeckWrapper(env, seed=1)

    seen = set()
    for _ in range(15):
        _, info = wrapper.reset()
        seen.add(tuple(info["blue_deck"]))
        assert len(info["blue_deck"]) == DECK_SIZE
        assert info["blue_deck"] != info["red_deck"]

    assert len(env.decks) == 15
    # A specialist is exactly what redrawing is meant to prevent.
    assert len(seen) >= 12


def test_random_deck_wrapper_is_reproducible_from_its_seed() -> None:
    from crforge_gym.deck_sampler import RandomDeckWrapper

    first = RandomDeckWrapper(_StubEnv(), seed=99)
    second = RandomDeckWrapper(_StubEnv(), seed=99)

    for _ in range(5):
        assert first.reset()[1]["blue_deck"] == second.reset()[1]["blue_deck"]


def test_mirror_mode_gives_both_players_the_same_deck() -> None:
    from crforge_gym.deck_sampler import RandomDeckWrapper

    wrapper = RandomDeckWrapper(_StubEnv(), seed=4, mirror=True)

    _, info = wrapper.reset()

    assert info["blue_deck"] == info["red_deck"]
