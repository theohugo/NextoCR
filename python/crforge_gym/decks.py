# Modified by NextoCR contributors; see NOTICE for attribution.
"""Versioned deck profiles used by reproducible NextoCR experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DeckProfile:
    """An immutable simulator deck plus its intended live-game forms."""

    profile_id: str
    display_name: str
    simulator_card_ids: tuple[str, ...]
    requested_forms: tuple[str, ...]
    elixir_costs: tuple[int, ...]
    known_approximations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.simulator_card_ids) != 8:
            raise ValueError("a Clash Royale deck profile must contain exactly eight cards")
        if len(set(self.simulator_card_ids)) != 8:
            raise ValueError("deck profile card IDs must be unique")
        if len(self.requested_forms) != 8 or len(self.elixir_costs) != 8:
            raise ValueError("requested forms and elixir costs must match the eight-card deck")

    @property
    def average_elixir(self) -> float:
        return sum(self.elixir_costs) / len(self.elixir_costs)

    def to_manifest(self) -> dict:
        """Return a JSON-serializable snapshot for an experiment manifest."""
        payload = asdict(self)
        payload["simulator_card_ids"] = list(self.simulator_card_ids)
        payload["requested_forms"] = list(self.requested_forms)
        payload["elixir_costs"] = list(self.elixir_costs)
        payload["known_approximations"] = list(self.known_approximations)
        payload["average_elixir"] = self.average_elixir
        return payload


MORTAR_SELF_PLAY_V1 = DeckProfile(
    profile_id="mortar_self_play_v1",
    display_name="Evolved Skeleton Barrel / Hero Goblins / Evolved Mortar bait",
    simulator_card_ids=(
        "skeletonballoon",
        "goblins",
        "mortar",
        "movingcannon",
        "fireball",
        "minions",
        "rascals",
        "barblog",
    ),
    requested_forms=(
        "skeletonballoon_ev1",
        "goblins_hero",
        "mortar_ev1",
        "movingcannon",
        "fireball",
        "minions",
        "rascals",
        "barblog",
    ),
    elixir_costs=(3, 2, 4, 5, 4, 3, 5, 2),
    known_approximations=(
        "Skeleton Barrel currently trains with its base form; the inherited evolved unit reference is incomplete.",
        "Hero Goblins currently train as base Goblins; Banner Brigade is not represented in the action schema.",
        "Mortar currently trains with its base form; the evolved Goblin projectile is not implemented yet.",
    ),
)


DECK_PROFILES = {MORTAR_SELF_PLAY_V1.profile_id: MORTAR_SELF_PLAY_V1}


def get_deck_profile(profile_id: str) -> DeckProfile:
    """Resolve a profile ID or raise a CLI-friendly error."""
    try:
        return DECK_PROFILES[profile_id]
    except KeyError as exc:
        choices = ", ".join(sorted(DECK_PROFILES))
        raise ValueError(f"unknown deck profile {profile_id!r}; choose one of: {choices}") from exc
