# Modified by NextoCR contributors; see NOTICE for attribution.
"""Contract tests for versioned experiment decks."""

import pytest

from crforge_gym.decks import MORTAR_SELF_PLAY_V1, get_deck_profile


def test_requested_mortar_deck_is_locked_and_versioned():
    profile = get_deck_profile("mortar_self_play_v1")

    assert profile is MORTAR_SELF_PLAY_V1
    assert profile.simulator_card_ids == (
        "skeletonballoon",
        "goblins",
        "mortar",
        "movingcannon",
        "fireball",
        "minions",
        "rascals",
        "barblog",
    )
    assert profile.average_elixir == 3.5
    assert len(profile.known_approximations) == 3


def test_deck_manifest_is_json_compatible():
    manifest = MORTAR_SELF_PLAY_V1.to_manifest()

    assert manifest["profile_id"] == "mortar_self_play_v1"
    assert isinstance(manifest["simulator_card_ids"], list)
    assert manifest["average_elixir"] == 3.5


def test_unknown_deck_profile_lists_valid_choice():
    with pytest.raises(ValueError, match="mortar_self_play_v1"):
        get_deck_profile("missing")
