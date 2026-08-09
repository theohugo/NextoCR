"""Schema-smoke tests for the card-data source ledger."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


MANIFEST = Path(__file__).resolve().parents[2] / "docs" / "data-sources.json"


class DataSourcesManifestTest(unittest.TestCase):
    def test_sources_have_unique_ids_and_required_policy_fields(self) -> None:
        document = json.loads(MANIFEST.read_text(encoding="utf-8"))

        self.assertEqual(1, document["schemaVersion"])
        self.assertRegex(document["auditedAt"], r"^\d{4}-\d{2}-\d{2}$")
        sources = document["sources"]
        ids = [source["id"] for source in sources]
        self.assertEqual(len(ids), len(set(ids)))

        required = {
            "id",
            "url",
            "role",
            "format",
            "automatedAccess",
            "license",
            "redistribution",
            "combatStats",
            "status",
        }
        for source in sources:
            self.assertFalse(required - source.keys(), source["id"])
            self.assertTrue(source["url"].startswith("https://"), source["id"])


if __name__ == "__main__":
    unittest.main()
