#!/usr/bin/env python3
"""Fetch and normalize the official Clash Royale card catalog.

Scope and provenance are intentionally narrow: the official endpoint is authoritative for
catalog identity and classification, but it does not expose combat statistics. This tool keeps
ordinary cards and support items separate, drops asset URLs and every unknown field, and never
stores or prints the API token.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_URL = "https://api.clashroyale.com/v1/cards"
TOKEN_ENVIRONMENT_VARIABLE = "CLASH_ROYALE_API_TOKEN"
SCHEMA_VERSION = 1
USER_AGENT = "NextoCR-official-catalog-sync/1"
DEFAULT_TIMEOUT_SECONDS = 30.0


class CatalogSyncError(ValueError):
    """Raised when credentials, transport, or source data cannot be safely processed."""


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Prevent bearer credentials from being forwarded by urllib redirects."""

    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def _open_without_redirect(request: urllib.request.Request, *, timeout: float):
    opener = urllib.request.build_opener(_RejectRedirectHandler())
    return opener.open(request, timeout=timeout)


def resolve_token(
    token_file: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Load a token without accepting it as a command-line value.

    Passing secrets directly as command-line arguments can expose them in shell history and
    process listings. ``--token-file`` is the explicit alternative to the environment variable.
    """

    environment = os.environ if environ is None else environ
    if token_file is not None:
        try:
            token = token_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise CatalogSyncError(f"could not read token file {token_file}: {exc}") from exc
    else:
        token = environment.get(TOKEN_ENVIRONMENT_VARIABLE, "").strip()

    if not token:
        raise CatalogSyncError(
            f"missing API token: set {TOKEN_ENVIRONMENT_VARIABLE} or use --token-file"
        )
    if any(character.isspace() for character in token):
        raise CatalogSyncError("API token must not contain whitespace")
    return token


def _optional_integer(item: Mapping[str, Any], field: str, *, location: str) -> int | None:
    value = item.get(field)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise CatalogSyncError(f"{location}.{field} must be an integer or null")
    return value


def _normalize_item(item: Any, *, kind: str, location: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise CatalogSyncError(f"{location} must be an object")

    official_id = item.get("id")
    if not isinstance(official_id, int) or isinstance(official_id, bool):
        raise CatalogSyncError(f"{location}.id must be an integer")

    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        raise CatalogSyncError(f"{location}.name must be a non-empty string")

    rarity = item.get("rarity")
    if rarity is not None and (not isinstance(rarity, str) or not rarity.strip()):
        raise CatalogSyncError(f"{location}.rarity must be a non-empty string or null")

    # Deliberately allow-list fields. In particular, iconUrls and future API fields are excluded
    # until their provenance, semantics, and redistribution policy are reviewed.
    return {
        "id": official_id,
        "name": name,
        "rarity": rarity,
        "elixirCost": _optional_integer(item, "elixirCost", location=location),
        "maxLevel": _optional_integer(item, "maxLevel", location=location),
        "maxEvolutionLevel": _optional_integer(
            item, "maxEvolutionLevel", location=location
        ),
        "kind": kind,
    }


def normalize_catalog(
    payload: Any,
    *,
    retrieved_at: str,
) -> dict[str, Any]:
    """Return a deterministic, provenance-bearing subset of an official API response."""

    if not isinstance(payload, dict):
        raise CatalogSyncError("official API response must be a JSON object")

    normalized_collections: dict[str, list[dict[str, Any]]] = {}
    for collection_name, kind in (("items", "card"), ("supportItems", "towerTroop")):
        collection = payload.get(collection_name)
        if not isinstance(collection, list):
            raise CatalogSyncError(
                f"official API response field {collection_name!r} must be an array"
            )
        normalized_collections[collection_name] = sorted(
            (
                _normalize_item(
                    item,
                    kind=kind,
                    location=f"{collection_name}[{index}]",
                )
                for index, item in enumerate(collection)
            ),
            key=lambda item: item["id"],
        )

    seen_ids: dict[int, str] = {}
    for collection_name in ("items", "supportItems"):
        for index, item in enumerate(normalized_collections[collection_name]):
            official_id = item["id"]
            location = f"{collection_name}[{index}]"
            previous = seen_ids.get(official_id)
            if previous is not None:
                raise CatalogSyncError(
                    f"duplicate official card id {official_id} in {previous} and {location}"
                )
            seen_ids[official_id] = location

    return {
        "schemaVersion": SCHEMA_VERSION,
        "source": API_URL,
        "retrievedAt": retrieved_at,
        "items": normalized_collections["items"],
        "supportItems": normalized_collections["supportItems"],
    }


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def fetch_official_catalog(
    token: str,
    *,
    open_url: Callable[..., Any] | None = None,
    retrieved_at: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Fetch the official endpoint and return its normalized catalog.

    ``open_url`` and ``retrieved_at`` are injectable so tests can be fully deterministic and make
    no network requests. The bearer token is used only in the request header.
    """

    request = urllib.request.Request(
        API_URL,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": USER_AGENT,
        },
        method="GET",
    )
    opener = _open_without_redirect if open_url is None else open_url
    try:
        with opener(request, timeout=timeout) as response:
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        if 300 <= exc.code < 400:
            raise CatalogSyncError(
                f"official API redirect refused (HTTP {exc.code})"
            ) from exc
        raise CatalogSyncError(f"official API returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise CatalogSyncError(f"could not reach official API: {exc.reason}") from exc
    except OSError as exc:
        raise CatalogSyncError(f"could not read official API response: {exc}") from exc

    try:
        payload = json.loads(response_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogSyncError("official API returned invalid JSON") from exc

    return normalize_catalog(payload, retrieved_at=retrieved_at or _utc_timestamp())


def _serialize(document: Mapping[str, Any]) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2) + "\n"


def _write_atomically(path: Path, content: str) -> None:
    """Replace an output file only after the complete JSON document has been written."""

    parent = path.parent
    if not parent.exists():
        raise CatalogSyncError(f"output directory does not exist: {parent}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=parent,
            delete=False,
        ) as handle:
            handle.write(content)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    except OSError as exc:
        raise CatalogSyncError(f"could not write output file {path}: {exc}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token-file",
        type=Path,
        help=(
            "read the API token from this UTF-8 file instead of "
            f"{TOKEN_ENVIRONMENT_VARIABLE}; the token itself is never a CLI option"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write normalized JSON atomically to this file instead of stdout",
    )
    args = parser.parse_args(argv)

    try:
        token = resolve_token(args.token_file)
        document = fetch_official_catalog(token)
        serialized = _serialize(document)
        if args.output is None:
            sys.stdout.write(serialized)
        else:
            _write_atomically(args.output, serialized)
    except CatalogSyncError as exc:
        print(f"official catalog sync failed: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
