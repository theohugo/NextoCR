"""Offline tests for the official Top-1000 curriculum pipeline."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "build_top1000_curriculum.py"
SPEC = importlib.util.spec_from_file_location("build_top1000_curriculum", SCRIPT)
assert SPEC and SPEC.loader
top1000 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = top1000
SPEC.loader.exec_module(top1000)


TRACKER = """
## Troops

| # | CR Name | Internal ID | Status | Notes |
|---|---------|-------------|--------|-------|
| 1 | Knight | knight | `DONE` | |
| 2 | Goblins | goblins | `DONE` | |
| 3 | Archer Queen | archerqueen | `PARTIAL` | ability missing |
| 4 | New Card | newcard | `MISSING` | not implemented |

## Spells

| # | CR Name | Internal ID | Status | Notes |
|---|---------|-------------|--------|-------|
| 1 | Fireball | fireball | `DONE` | |
| 2 | The Log | log | `DONE` | |

## Buildings

| # | CR Name | Internal ID | Status | Notes |
|---|---------|-------------|--------|-------|
| 1 | Mortar | mortar | `DONE` | |
| 2 | Cannon | cannon | `DONE` | |

## Tower Troops (new mechanic -- all Missing)

| # | CR Name | Status | Notes |
|---|---------|--------|-------|
| 1 | Tower Princess | `DONE` | default |
| 2 | Cannoneer | `MISSING` | unsupported |
"""


def card(
    official_id: int,
    name: str,
    *,
    evolution_level: int = 0,
    hero_level: int = 0,
) -> dict[str, object]:
    return {
        "officialCardId": official_id,
        "name": name,
        "evolutionLevel": evolution_level,
        "heroLevel": hero_level,
    }


def exact_cards(offset: int = 0) -> list[dict[str, object]]:
    names = ["Knight", "Goblins", "Fireball", "The Log", "Mortar", "Cannon"]
    # A real deck cannot repeat cards. The fixture uses distinct official IDs while mapping two
    # extra observed aliases to already supported simulator mechanics.
    names.extend(["Knight", "Goblins"])
    return [card(offset + index + 1, name) for index, name in enumerate(names)]


def source_snapshot(records: list[dict[str, object]], *, target: int | None = None):
    actual_target = len(records) if target is None else target
    return {
        "schemaVersion": 1,
        "source": {
            "provider": "Supercell Clash Royale API",
            "season": "2026-08",
            "interpretation": "current decks of ranked players at retrieval time",
        },
        "retrievedAtStarted": "2026-08-09T12:00:00Z",
        "retrievedAtCompleted": "2026-08-09T12:05:00Z",
        "targetRankCount": actual_target,
        "leaderboardPlayersReturned": len(records),
        "profileDecksReturned": len(records),
        "records": records,
        "failures": [],
    }


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def get_json(self, path: str, query=None):
        self.calls.append((path, query))
        if "pathoflegend" in path:
            if query.get("after") == "next-page":
                return {
                    "items": [{"tag": "#P0Y", "rank": 3, "name": "player-three"}],
                    "paging": {},
                }
            return {
                "items": [
                    {"tag": "#P2Y", "rank": 1, "name": "player-one"},
                    {"tag": "#P9Y", "rank": 2, "name": "player-two"},
                ],
                "paging": {"cursors": {"after": "next-page"}},
            }

        rank_by_tag = {"%23P2Y": 1, "%23P9Y": 2, "%23P0Y": 3}
        tag = path.rsplit("/", 1)[-1]
        rank = rank_by_tag[tag]
        return {
            "tag": urllib_tag(tag),
            "name": f"private-player-name-{rank}",
            "currentDeck": [
                {
                    "id": rank * 100 + index,
                    "name": name,
                    "level": 16,
                    "iconUrls": {"medium": "https://assets.invalid/card.png"},
                }
                for index, name in enumerate(
                    ["Knight", "Goblins", "Fireball", "The Log", "Mortar", "Cannon", "A", "B"],
                    start=1,
                )
            ],
            "currentDeckSupportCards": [{"id": 159000000, "name": "Tower Princess"}],
        }


def urllib_tag(encoded_tag: str) -> str:
    return encoded_tag.replace("%23", "#")


class Top1000CurriculumTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = top1000.parse_implementation_tracker(TRACKER)

    def test_tracker_parses_playable_and_tower_statuses(self) -> None:
        self.assertEqual("knight", self.tracker["knight"]["simulatorId"])
        self.assertEqual("partial", self.tracker["archerqueen"]["trackerStatus"])
        self.assertEqual("tower-troop", self.tracker["towerprincess"]["kind"])

    def test_mapping_overrides_are_append_only_and_auditable(self) -> None:
        merged = top1000.apply_mapping_overrides(
            self.tracker,
            {
                "schemaVersion": 1,
                "mappings": [
                    {
                        "officialName": "Cannon Cart",
                        "simulatorId": "movingcannon",
                        "trackerStatus": "done",
                        "kind": "troop",
                        "evidence": "reviewed card configuration",
                    }
                ],
            },
        )
        self.assertEqual("movingcannon", merged["cannoncart"]["simulatorId"])
        self.assertEqual("reviewed card configuration", merged["cannoncart"]["mappingEvidence"])

        with self.assertRaisesRegex(top1000.Top1000Error, "conflicts"):
            top1000.apply_mapping_overrides(
                self.tracker,
                {
                    "schemaVersion": 1,
                    "mappings": [
                        {
                            "officialName": "Knight",
                            "simulatorId": "wrong",
                            "trackerStatus": "done",
                            "kind": "troop",
                            "evidence": "must not replace tracker",
                        }
                    ],
                },
            )

    def test_trainable_mappings_must_exist_in_simulator_catalog(self) -> None:
        simulator_cards = [
            {"id": mapping["simulatorId"]}
            for mapping in self.tracker.values()
            if mapping["simulatorId"] is not None and mapping["trackerStatus"] != "missing"
        ]
        top1000.validate_simulator_ids(self.tracker, simulator_cards)

        incomplete = [card for card in simulator_cards if card["id"] != "knight"]
        with self.assertRaisesRegex(top1000.Top1000Error, "Knight.*absent"):
            top1000.validate_simulator_ids(self.tracker, incomplete)

    def test_fetch_paginates_and_drops_player_identity_and_assets(self) -> None:
        client = FakeClient()
        timestamps = iter(["2026-08-09T12:00:00Z", "2026-08-09T12:05:00Z"])

        snapshot = top1000.fetch_population_snapshot(
            client,
            season="2026-08",
            target=3,
            page_size=2,
            timestamp=lambda: next(timestamps),
        )

        serialized = json.dumps(snapshot)
        self.assertEqual(3, snapshot["leaderboardPlayersReturned"])
        self.assertEqual(3, snapshot["profileDecksReturned"])
        self.assertEqual([1, 2, 3], [record["rank"] for record in snapshot["records"]])
        self.assertNotIn("private-player-name", serialized)
        self.assertNotIn("#P2Y", serialized)
        self.assertNotIn("iconUrls", serialized)
        self.assertNotIn("assets.invalid", serialized)
        self.assertEqual(
            "/v1/locations/global/pathoflegend/2026-08/rankings/players",
            client.calls[0][0],
        )
        self.assertEqual("next-page", client.calls[1][1]["after"])

    def test_profile_requires_eight_unique_cards(self) -> None:
        profile = {
            "currentDeck": [{"id": 1, "name": "Knight"}] * 8,
            "currentDeckSupportCards": [],
        }
        with self.assertRaisesRegex(top1000.Top1000Error, "duplicate"):
            top1000.normalize_profile_deck(profile, rank=1)

    def test_base_fallback_reports_exact_partial_approximated_and_missing(self) -> None:
        records = [
            {
                "rank": 1,
                "cards": exact_cards(0),
                "towerTroops": [{"officialCardId": 1000, "name": "Tower Princess"}],
            },
            {
                "rank": 2,
                "cards": [
                    card(101, "Hero Goblins", hero_level=1),
                    *exact_cards(200)[:7],
                ],
                "towerTroops": [{"officialCardId": 1001, "name": "Cannoneer"}],
            },
            {
                "rank": 3,
                "cards": [card(301, "Archer Queen"), *exact_cards(400)[:7]],
                "towerTroops": [{"officialCardId": 1000, "name": "Tower Princess"}],
            },
            {
                "rank": 4,
                "cards": [card(501, "Never Seen Card"), *exact_cards(600)[:7]],
                "towerTroops": [{"officialCardId": 1000, "name": "Tower Princess"}],
            },
        ]

        document = top1000.compile_curriculum(
            source_snapshot(records),
            self.tracker,
            tracker_sha256="tracker-hash",
            fidelity_policy="base-fallback",
            compiled_at="2026-08-09T12:06:00Z",
        )

        coverage = document["coverage"]
        self.assertEqual(
            {"approximated": 1, "exact-base": 1, "excluded": 1, "partial": 1},
            coverage["deckObservationsByQuality"],
        )
        self.assertEqual(3, coverage["uniqueDecksEligible"])
        self.assertEqual(1, coverage["uniqueDecksExcluded"])
        self.assertAlmostEqual(0.75, coverage["trainableCoverageAgainstTarget"])
        self.assertAlmostEqual(0.25, coverage["exactCoverageAgainstTarget"])
        self.assertFalse(coverage["exactSimulatorCoverage"])
        self.assertEqual(
            "official-ranked-current-decks-partial-coverage", coverage["trainingClaim"]
        )
        self.assertEqual("Never Seen Card", document["unmappedCards"][0]["name"])
        self.assertEqual("Never Seen Card", document["cardAudit"]["unmapped"][0]["name"])
        self.assertEqual(1, len(document["cardAudit"]["partial"]))
        self.assertEqual(1, len(document["deckAudit"]["excluded"]))
        self.assertTrue(any(not deck["eligibleForTraining"] for deck in document["decks"]))

        eligible = [deck for deck in document["decks"] if deck["eligibleForTraining"]]
        self.assertAlmostEqual(
            1.0, sum(deck["sampling"]["combinedWeight"] for deck in eligible)
        )
        self.assertTrue(
            all(
                deck["sampling"]["combinedWeight"] == 0
                for deck in document["decks"]
                if not deck["eligibleForTraining"]
            )
        )

    def test_strict_policy_excludes_forms_partial_cards_and_nondefault_towers(self) -> None:
        records = [
            {
                "rank": 1,
                "cards": [card(1, "Goblins", evolution_level=1), *exact_cards(10)[:7]],
                "towerTroops": [{"officialCardId": 1000, "name": "Tower Princess"}],
            },
            {
                "rank": 2,
                "cards": [card(2, "Archer Queen"), *exact_cards(20)[:7]],
                "towerTroops": [{"officialCardId": 1000, "name": "Tower Princess"}],
            },
            {
                "rank": 3,
                "cards": exact_cards(30),
                "towerTroops": [{"officialCardId": 1001, "name": "Cannoneer"}],
            },
        ]

        document = top1000.compile_curriculum(
            source_snapshot(records),
            self.tracker,
            tracker_sha256="tracker-hash",
            fidelity_policy="strict",
            compiled_at="2026-08-09T12:06:00Z",
        )

        self.assertEqual(0, document["coverage"]["uniqueDecksEligible"])
        self.assertEqual(3, document["coverage"]["deckObservationsByQuality"]["excluded"])
        reasons = {
            reason
            for deck in document["decks"]
            for reason in deck["exclusionReasons"]
        }
        self.assertIn("card:Goblins:unsupported-form", reasons)
        self.assertIn("card:Archer Queen:partial", reasons)
        self.assertIn("tower-troop:unsupported", reasons)

    def test_complete_exact_target_is_the_only_exact_claim(self) -> None:
        record = {
            "rank": 1,
            "cards": exact_cards(),
            "towerTroops": [{"officialCardId": 1000, "name": "Tower Princess"}],
        }
        document = top1000.compile_curriculum(
            source_snapshot([record], target=1),
            self.tracker,
            tracker_sha256="tracker-hash",
            fidelity_policy="strict",
            compiled_at="2026-08-09T12:06:00Z",
        )

        self.assertTrue(document["coverage"]["completeTargetPopulation"])
        self.assertTrue(document["coverage"]["exactSimulatorCoverage"])
        self.assertFalse(document["coverage"]["top1000Exact"])
        self.assertNotEqual("official-top1000-exact", document["coverage"]["trainingClaim"])

    def test_weight_mix_must_sum_to_one(self) -> None:
        with self.assertRaisesRegex(top1000.Top1000Error, "sum to 1"):
            top1000.compile_curriculum(
                source_snapshot([]),
                self.tracker,
                tracker_sha256="tracker-hash",
                weight_mix={"population": 0.5, "rank": 0.5, "uniform": 0.5},
            )


if __name__ == "__main__":
    unittest.main()
