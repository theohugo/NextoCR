#!/usr/bin/env python3
"""Audit NextoCR's machine-readable card catalog without third-party dependencies.

This script is a NextoCR addition. It intentionally reports data coverage separately from
mechanical implementation status: a JSON definition does not prove that a card's special behavior
is simulated correctly. See docs/card_tracker.md for the current mechanic tracker.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CATALOG_FILES = ("cards.json", "units.json", "projectiles.json", "buffs.json")
CARD_TYPES = {"TROOP", "SPELL", "BUILDING", "HERO"}
RARITIES = {"Common", "Rare", "Epic", "Legendary", "Champion"}


@dataclass(frozen=True)
class CatalogSummary:
    card_definitions: int
    base_definitions: int
    deployable_base_definitions: int
    evolution_definitions: int
    alternate_form_definitions: int
    units: int
    projectiles: int
    buffs: int
    base_cards_by_type: dict[str, int]
    base_cards_by_rarity: dict[str, int]
    unit_stat_coverage: dict[str, int]
    errors: list[str]
    warnings: list[str]


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ValueError(f"missing catalog file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def _duplicates(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def audit_catalog(catalog_dir: Path) -> CatalogSummary:
    """Load the catalog, validate its core references, and return a stable summary."""
    loaded = {name: _load_json(catalog_dir / name) for name in CATALOG_FILES}
    cards = loaded["cards.json"]
    units = loaded["units.json"]
    projectiles = loaded["projectiles.json"]
    buffs = loaded["buffs.json"]

    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(cards, list):
        raise ValueError("cards.json must contain a JSON array")
    for name, value in (("units.json", units), ("projectiles.json", projectiles), ("buffs.json", buffs)):
        if not isinstance(value, dict):
            raise ValueError(f"{name} must contain a JSON object")

    card_ids: list[str] = []
    for index, card in enumerate(cards):
        if not isinstance(card, dict):
            errors.append(f"cards[{index}] is not an object")
            continue
        for field in ("id", "name", "type", "cost"):
            if field not in card:
                errors.append(f"cards[{index}] is missing required field '{field}'")
        card_id = card.get("id")
        if isinstance(card_id, str) and card_id:
            card_ids.append(card_id)
        else:
            errors.append(f"cards[{index}].id must be a non-empty string")
        card_type = card.get("type")
        if card_type not in CARD_TYPES:
            errors.append(f"{card_id or f'cards[{index}]'} has unknown type {card_type!r}")
        cost = card.get("cost")
        if not isinstance(cost, int) or isinstance(cost, bool) or cost < 0:
            errors.append(f"{card_id or f'cards[{index}]'} has invalid cost {cost!r}")
        rarity = card.get("rarity", "")
        if cost and rarity not in RARITIES:
            errors.append(f"{card_id or f'cards[{index}]'} has unknown rarity {rarity!r}")

    for duplicate in _duplicates(card_ids):
        errors.append(f"duplicate card id: {duplicate}")

    card_id_set = set(card_ids)
    unit_ids = set(units)
    projectile_ids = set(projectiles)

    for card in cards:
        if not isinstance(card, dict) or not isinstance(card.get("id"), str):
            continue
        card_id = card["id"]
        # Alternate/evolution records are present as forward-looking data even when their mechanics
        # are not loadable yet. Missing references in those records are catalog warnings; the same
        # problem in a base definition is an error because it can break an ordinary deck.
        reference_issues = warnings if "baseCard" in card else errors
        for field in ("unit", "secondaryUnit", "summonCharacter"):
            reference = card.get(field)
            if reference is not None and reference not in unit_ids:
                reference_issues.append(f"{card_id}.{field} references missing unit {reference!r}")
        for field in ("projectile", "spawnProjectile"):
            reference = card.get(field)
            if reference is not None and reference not in projectile_ids:
                reference_issues.append(
                    f"{card_id}.{field} references missing projectile {reference!r}"
                )
        for field in ("baseCard", "evolvedCard"):
            reference = card.get(field)
            if reference is not None and reference not in card_id_set:
                reference_issues.append(f"{card_id}.{field} references missing card {reference!r}")

    base_cards = [card for card in cards if isinstance(card, dict) and "baseCard" not in card]
    deployable_base_cards = [
        card
        for card in base_cards
        if isinstance(card.get("cost"), int) and card["cost"] > 0
    ]
    evolutions = [
        card
        for card in cards
        if isinstance(card, dict) and (card.get("evolved") is True or str(card.get("id", "")).endswith("_ev1"))
    ]
    alternate_forms = [
        card
        for card in cards
        if isinstance(card, dict) and "baseCard" in card and card not in evolutions
    ]

    stat_fields = (
        "health",
        "damage",
        "speed",
        "range",
        "attackCooldown",
        "deployTime",
        "movementType",
        "targetType",
    )
    stat_coverage = {
        field: sum(1 for unit in units.values() if isinstance(unit, dict) and field in unit)
        for field in stat_fields
    }

    return CatalogSummary(
        card_definitions=len(cards),
        base_definitions=len(base_cards),
        deployable_base_definitions=len(deployable_base_cards),
        evolution_definitions=len(evolutions),
        alternate_form_definitions=len(alternate_forms),
        units=len(units),
        projectiles=len(projectiles),
        buffs=len(buffs),
        base_cards_by_type=dict(sorted(Counter(card.get("type", "UNKNOWN") for card in base_cards).items())),
        base_cards_by_rarity=dict(
            sorted(Counter(card.get("rarity") or "UNSPECIFIED" for card in base_cards).items())
        ),
        unit_stat_coverage=stat_coverage,
        errors=sorted(set(errors)),
        warnings=sorted(set(warnings)),
    )


def format_text(summary: CatalogSummary) -> str:
    """Render a concise, deterministic human-readable report."""
    lines = [
        "NextoCR card catalog audit",
        "==========================",
        f"Card definitions:          {summary.card_definitions}",
        f"Base definitions:          {summary.base_definitions}",
        f"Deployable base entries:   {summary.deployable_base_definitions}",
        f"Evolution definitions:     {summary.evolution_definitions}",
        f"Alternate/hero forms:      {summary.alternate_form_definitions}",
        f"Unit definitions:          {summary.units}",
        f"Projectile definitions:    {summary.projectiles}",
        f"Buff definitions:          {summary.buffs}",
        "",
        "Base definitions by type:",
    ]
    lines.extend(f"  {name:<12} {count}" for name, count in summary.base_cards_by_type.items())
    lines.extend(("", "Base definitions by rarity:"))
    lines.extend(f"  {name:<12} {count}" for name, count in summary.base_cards_by_rarity.items())
    lines.extend(("", f"Unit stat field coverage (of {summary.units}):"))
    lines.extend(f"  {name:<16} {count}" for name, count in summary.unit_stat_coverage.items())
    lines.extend(("", f"Validation errors:         {len(summary.errors)}"))
    lines.extend(f"  - {error}" for error in summary.errors)
    lines.extend(("", f"Known incomplete variants: {len(summary.warnings)}"))
    lines.extend(f"  - {warning}" for warning in summary.warnings)
    lines.extend(("", "Note: data presence is not mechanic fidelity; see docs/card_tracker.md."))
    return "\n".join(lines)


def _default_catalog_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "src" / "main" / "resources" / "cards"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-dir",
        type=Path,
        default=_default_catalog_dir(),
        help="directory containing cards.json, units.json, projectiles.json, and buffs.json",
    )
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a non-zero exit code when validation errors are found",
    )
    args = parser.parse_args(argv)

    try:
        summary = audit_catalog(args.catalog_dir)
    except ValueError as exc:
        print(f"catalog audit failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    else:
        print(format_text(summary))

    return 1 if args.strict and summary.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
