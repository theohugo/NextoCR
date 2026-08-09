"""Offline unit tests for the official catalog synchronizer."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "sync_official_catalog.py"
SPEC = importlib.util.spec_from_file_location("sync_official_catalog", SCRIPT)
assert SPEC and SPEC.loader
sync_official_catalog = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sync_official_catalog
SPEC.loader.exec_module(sync_official_catalog)


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *unused: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class OfficialCatalogSyncTest(unittest.TestCase):
    def test_fetch_keeps_support_items_and_excludes_assets_and_unknown_fields(self) -> None:
        source_payload = {
            "items": [
                {
                    "name": "Knight",
                    "id": 26000000,
                    "maxLevel": 16,
                    "maxEvolutionLevel": 3,
                    "elixirCost": 3,
                    "rarity": "common",
                    "iconUrls": {"medium": "https://assets.invalid/knight.png"},
                    "unreviewedField": "must not pass through",
                }
            ],
            "supportItems": [
                {
                    "name": "Tower Princess",
                    "id": 159000000,
                    "maxLevel": 16,
                    "rarity": "common",
                    "iconUrls": {"medium": "https://assets.invalid/tower.png"},
                }
            ],
        }

        def fake_open(request: object, *, timeout: float) -> FakeResponse:
            self.assertEqual(sync_official_catalog.API_URL, request.full_url)
            self.assertEqual("GET", request.get_method())
            self.assertEqual("Bearer test-secret", request.get_header("Authorization"))
            self.assertEqual(sync_official_catalog.DEFAULT_TIMEOUT_SECONDS, timeout)
            return FakeResponse(source_payload)

        document = sync_official_catalog.fetch_official_catalog(
            "test-secret",
            open_url=fake_open,
            retrieved_at="2026-08-09T12:00:00Z",
        )

        self.assertEqual(1, document["schemaVersion"])
        self.assertEqual(sync_official_catalog.API_URL, document["source"])
        self.assertEqual("2026-08-09T12:00:00Z", document["retrievedAt"])
        self.assertEqual("card", document["items"][0]["kind"])
        self.assertEqual("towerTroop", document["supportItems"][0]["kind"])
        self.assertIsNone(document["supportItems"][0]["elixirCost"])
        self.assertIsNone(document["supportItems"][0]["maxEvolutionLevel"])
        self.assertEqual(
            {
                "id",
                "name",
                "rarity",
                "elixirCost",
                "maxLevel",
                "maxEvolutionLevel",
                "kind",
            },
            set(document["items"][0]),
        )
        serialized = json.dumps(document)
        self.assertNotIn("iconUrls", serialized)
        self.assertNotIn("unreviewedField", serialized)
        self.assertNotIn("test-secret", serialized)

    def test_duplicate_id_across_collections_is_rejected(self) -> None:
        duplicate = {"id": 42, "name": "Duplicate", "rarity": "common"}

        with self.assertRaisesRegex(
            sync_official_catalog.CatalogSyncError,
            "duplicate official card id 42",
        ):
            sync_official_catalog.normalize_catalog(
                {"items": [duplicate], "supportItems": [duplicate]},
                retrieved_at="2026-08-09T12:00:00Z",
            )

    def test_redirect_handler_refuses_to_forward_bearer_credentials(self) -> None:
        request = sync_official_catalog.urllib.request.Request(
            sync_official_catalog.API_URL,
            headers={"Authorization": "Bearer must-not-leak"},
        )
        handler = sync_official_catalog._RejectRedirectHandler()

        redirected = handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://evil.invalid/capture",
        )

        self.assertIsNone(redirected)

    def test_missing_token_fails_before_any_network_call(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(sync_official_catalog, "fetch_official_catalog") as fetch,
            mock.patch("sys.stderr", stderr),
        ):
            exit_code = sync_official_catalog.main([])

        self.assertEqual(2, exit_code)
        fetch.assert_not_called()
        self.assertIn(sync_official_catalog.TOKEN_ENVIRONMENT_VARIABLE, stderr.getvalue())

    def test_token_file_and_output_option_do_not_require_network(self) -> None:
        fixture = {
            "schemaVersion": 1,
            "source": sync_official_catalog.API_URL,
            "retrievedAt": "2026-08-09T12:00:00Z",
            "items": [],
            "supportItems": [],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            token_file = directory / "token.txt"
            output_file = directory / "catalog.json"
            token_file.write_text("file-secret\n", encoding="utf-8")

            with mock.patch.object(
                sync_official_catalog,
                "fetch_official_catalog",
                return_value=fixture,
            ) as fetch:
                exit_code = sync_official_catalog.main(
                    ["--token-file", str(token_file), "--output", str(output_file)]
                )

            self.assertEqual(0, exit_code)
            fetch.assert_called_once_with("file-secret")
            self.assertEqual(fixture, json.loads(output_file.read_text(encoding="utf-8")))
            self.assertNotIn("file-secret", output_file.read_text(encoding="utf-8"))

    def test_environment_token_emits_json_to_stdout_by_default(self) -> None:
        fixture = {
            "schemaVersion": 1,
            "source": sync_official_catalog.API_URL,
            "retrievedAt": "2026-08-09T12:00:00Z",
            "items": [],
            "supportItems": [],
        }
        stdout = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {sync_official_catalog.TOKEN_ENVIRONMENT_VARIABLE: "environment-secret"},
                clear=True,
            ),
            mock.patch.object(
                sync_official_catalog,
                "fetch_official_catalog",
                return_value=fixture,
            ) as fetch,
            mock.patch("sys.stdout", stdout),
        ):
            exit_code = sync_official_catalog.main([])

        self.assertEqual(0, exit_code)
        fetch.assert_called_once_with("environment-secret")
        self.assertEqual(fixture, json.loads(stdout.getvalue()))
        self.assertNotIn("environment-secret", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
