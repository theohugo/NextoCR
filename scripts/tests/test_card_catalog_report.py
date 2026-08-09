"""Tests for the dependency-free card catalog audit."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "card_catalog_report.py"
SPEC = importlib.util.spec_from_file_location("card_catalog_report", SCRIPT)
assert SPEC and SPEC.loader
card_catalog_report = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = card_catalog_report
SPEC.loader.exec_module(card_catalog_report)


class CardCatalogReportTest(unittest.TestCase):
    def test_repository_catalog_has_expected_shape_and_no_reference_errors(self) -> None:
        catalog_dir = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "src"
            / "main"
            / "resources"
            / "cards"
        )

        summary = card_catalog_report.audit_catalog(catalog_dir)

        self.assertGreaterEqual(summary.card_definitions, 200)
        self.assertGreaterEqual(summary.deployable_base_definitions, 120)
        self.assertGreaterEqual(summary.evolution_definitions, 20)
        self.assertEqual([], summary.errors)
        self.assertGreater(len(summary.warnings), 0)

    def test_missing_reference_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            catalog_dir = Path(temporary_directory)
            (catalog_dir / "cards.json").write_text(
                json.dumps(
                    [
                        {
                            "id": "test-card",
                            "name": "Test card",
                            "type": "TROOP",
                            "rarity": "Common",
                            "cost": 1,
                            "unit": "MissingUnit",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            for name in ("units.json", "projectiles.json", "buffs.json"):
                (catalog_dir / name).write_text("{}", encoding="utf-8")

            summary = card_catalog_report.audit_catalog(catalog_dir)

            self.assertIn(
                "test-card.unit references missing unit 'MissingUnit'",
                summary.errors,
            )


if __name__ == "__main__":
    unittest.main()
