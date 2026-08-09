# Modified by NextoCR contributors; see NOTICE for attribution.
"""Tests for elixir-economy reward shaping.

The dangerous failure is not a wrong sign on one trade: it is shaping that
grows large enough to outrank winning, so the agent learns to farm trades and
avoid closing out matches.
"""

from __future__ import annotations

from crforge_gym.elixir_rewards import (
    ElixirEconomyTracker,
    TradeRewardConfig,
)


def frame(
    *,
    own_elixir: float,
    enemy_elixir: float,
    enemy_units: list[tuple[int, str]] | None = None,
    own_units: list[tuple[int, str]] | None = None,
) -> dict:
    entities = []
    for identifier, name in enemy_units or []:
        entities.append({"id": identifier, "name": name, "team": "RED"})
    for identifier, name in own_units or []:
        entities.append({"id": identifier, "name": name, "team": "BLUE"})
    return {
        "bluePlayer": {"elixir": own_elixir},
        "redPlayer": {"elixir": enemy_elixir},
        "entities": entities,
    }


def test_killing_an_expensive_push_cheaply_is_rewarded() -> None:
    tracker = ElixirEconomyTracker()
    # Opponent commits a Giant; we answer and it dies.
    tracker.update(frame(own_elixir=8, enemy_elixir=3, enemy_units=[(1, "Giant")]))
    reward = tracker.update(
        frame(own_elixir=5, enemy_elixir=4, enemy_units=[], own_units=[(2, "Musketeer")])
    )

    assert reward > 0.0


def test_losing_units_for_nothing_is_penalised() -> None:
    tracker = ElixirEconomyTracker()
    tracker.update(frame(own_elixir=6, enemy_elixir=6, own_units=[(1, "Giant")]))
    reward = tracker.update(frame(own_elixir=6, enemy_elixir=6, own_units=[]))

    assert reward < 0.0


def test_a_swarm_body_is_priced_below_its_whole_card() -> None:
    """One Skeleton dying must not read like trading a whole card."""
    tracker = ElixirEconomyTracker()

    skeleton = tracker.unit_value("Skeleton")
    giant = tracker.unit_value("Giant")

    assert 0.0 < skeleton < 1.5
    assert giant > skeleton


def test_committing_while_the_opponent_is_dry_earns_tempo() -> None:
    tracker = ElixirEconomyTracker()
    tracker.update(frame(own_elixir=10, enemy_elixir=1))
    dry = tracker.update(frame(own_elixir=5, enemy_elixir=1))

    other = ElixirEconomyTracker()
    other.update(frame(own_elixir=10, enemy_elixir=10))
    full = other.update(frame(own_elixir=5, enemy_elixir=10))

    # Same spend, but only one of them is good timing.
    assert dry > full


def test_regenerating_elixir_is_not_treated_as_spending() -> None:
    tracker = ElixirEconomyTracker()
    tracker.update(frame(own_elixir=3, enemy_elixir=3))
    reward = tracker.update(frame(own_elixir=5, enemy_elixir=5))

    assert reward == 0.0


def test_shaping_cannot_outgrow_its_episode_budget() -> None:
    tracker = ElixirEconomyTracker(TradeRewardConfig(episode_shaping_limit=1.0))
    total = 0.0
    for step in range(200):
        tracker.update(frame(own_elixir=10, enemy_elixir=3, enemy_units=[(step, "Giant")]))
        total += tracker.update(frame(own_elixir=10, enemy_elixir=3, enemy_units=[]))

    # A terminal win bonus is far larger than this, which is the whole point.
    assert abs(total) <= 1.0 + 1e-6


def test_budget_is_shared_by_penalties_and_rewards() -> None:
    tracker = ElixirEconomyTracker(TradeRewardConfig(episode_shaping_limit=2.0))
    for step in range(50):
        tracker.update(frame(own_elixir=10, enemy_elixir=5, own_units=[(step, "Giant")]))
        tracker.update(frame(own_elixir=10, enemy_elixir=5, own_units=[]))

    assert abs(tracker.shaping_spent) <= 2.0 + 1e-6


def test_reset_clears_the_budget_between_episodes() -> None:
    tracker = ElixirEconomyTracker(TradeRewardConfig(episode_shaping_limit=1.0))
    tracker.update(frame(own_elixir=10, enemy_elixir=3, enemy_units=[(1, "Giant")]))
    tracker.update(frame(own_elixir=10, enemy_elixir=3, enemy_units=[]))
    assert tracker.shaping_spent != 0.0

    tracker.reset()

    assert tracker.shaping_spent == 0.0


def test_zero_budget_disables_shaping_entirely() -> None:
    tracker = ElixirEconomyTracker(TradeRewardConfig(episode_shaping_limit=0.0))
    tracker.update(frame(own_elixir=10, enemy_elixir=3, enemy_units=[(1, "Giant")]))

    assert tracker.update(frame(own_elixir=10, enemy_elixir=3, enemy_units=[])) == 0.0


def test_non_dict_observation_is_ignored() -> None:
    tracker = ElixirEconomyTracker()

    assert tracker.update([1, 2, 3]) == 0.0  # type: ignore[arg-type]
