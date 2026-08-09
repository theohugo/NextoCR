#!/usr/bin/env python3
"""Build a provenance-bearing RL opponent catalog from the official ranked leaderboard.

The pipeline deliberately has two stages:

* ``fetch`` calls only the official Clash Royale API and writes a minimized, private source
  snapshot. Player names and tags are discarded after their profile is queried.
* ``compile`` joins that snapshot to NextoCR's implementation tracker, deduplicates decks,
  computes sampling weights, and reports every unsupported or approximated observation.

The API token is accepted only through ``CLASH_ROYALE_API_TOKEN`` or a token file. It is never
accepted as a command-line value, serialized, or included in an error message.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_BASE_URL = "https://api.clashroyale.com"
DEVELOPER_DOCUMENTATION_URL = "https://developer.clashroyale.com/#/documentation"
FAN_CONTENT_POLICY_URL = "https://supercell.com/en/fan-content-policy/"
TOKEN_ENVIRONMENT_VARIABLE = "CLASH_ROYALE_API_TOKEN"
SOURCE_SCHEMA_VERSION = 1
CURRICULUM_SCHEMA_VERSION = 1
COMPILER_VERSION = "top1000-curriculum-v1"
DEFAULT_TARGET = 1000
DEFAULT_PAGE_SIZE = 50
DEFAULT_REQUESTS_PER_SECOND = 5.0
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 5
RETRIABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})
SEASON_PATTERN = re.compile(r"^[0-9]{4}-(0[1-9]|1[0-2])$")
PLAYER_TAG_PATTERN = re.compile(r"^#[0289PYLQGRJCUV]+$")


class Top1000Error(ValueError):
    """Raised when input, transport, or source data cannot be processed safely."""


class OfficialApiError(Top1000Error):
    """Official API failure with an optional, non-secret HTTP status."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never forward the bearer token to a redirected destination."""

    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _serialize(document: Mapping[str, Any]) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _write_atomically(path: Path, content: str) -> None:
    if not path.parent.exists():
        raise Top1000Error(f"output directory does not exist: {path.parent}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(content)
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    except OSError as exc:
        raise Top1000Error(f"could not write output file {path}: {exc}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def resolve_token(
    token_file: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve the bearer token without exposing it through process arguments."""

    environment = os.environ if environ is None else environ
    if token_file is None:
        token = environment.get(TOKEN_ENVIRONMENT_VARIABLE, "").strip()
    else:
        try:
            token = token_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise Top1000Error(f"could not read token file {token_file}: {exc}") from exc

    if not token:
        raise Top1000Error(
            f"missing API token: set {TOKEN_ENVIRONMENT_VARIABLE} or use --token-file"
        )
    if any(character.isspace() for character in token):
        raise Top1000Error("API token must not contain whitespace")
    return token


def _retry_after_seconds(headers: Mapping[str, str] | None, attempt: int) -> float:
    if headers is not None:
        raw_value = headers.get("Retry-After")
        if raw_value:
            try:
                return min(60.0, max(0.0, float(raw_value)))
            except ValueError:
                pass
    return min(30.0, 0.5 * (2**attempt))


class OfficialApiClient:
    """Small paced client restricted to the official API origin."""

    def __init__(
        self,
        token: str,
        *,
        requests_per_second: float = DEFAULT_REQUESTS_PER_SECOND,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        open_url: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if requests_per_second <= 0:
            raise Top1000Error("requests_per_second must be positive")
        if timeout <= 0:
            raise Top1000Error("timeout must be positive")
        if max_retries < 0:
            raise Top1000Error("max_retries must not be negative")
        self._token = token
        self._minimum_interval = 1.0 / requests_per_second
        self._timeout = timeout
        self._max_retries = max_retries
        self._open_url = open_url or urllib.request.build_opener(
            _RejectRedirectHandler()
        ).open
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_request_at: float | None = None

    def _pace(self) -> None:
        now = self._monotonic()
        if self._last_request_at is not None:
            remaining = self._minimum_interval - (now - self._last_request_at)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request_at = self._monotonic()

    def get_json(
        self,
        path: str,
        query: Mapping[str, str | int] | None = None,
    ) -> Any:
        if not path.startswith("/v1/") or "://" in path or ".." in path:
            raise Top1000Error(f"refusing non-official API path: {path!r}")
        query_string = urllib.parse.urlencode(query or {})
        url = f"{API_BASE_URL}{path}"
        if query_string:
            url = f"{url}?{query_string}"

        for attempt in range(self._max_retries + 1):
            self._pace()
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self._token}",
                    "User-Agent": f"NextoCR/{COMPILER_VERSION}",
                },
                method="GET",
            )
            try:
                with self._open_url(request, timeout=self._timeout) as response:
                    response_body = response.read()
            except urllib.error.HTTPError as exc:
                if exc.code in RETRIABLE_HTTP_STATUSES and attempt < self._max_retries:
                    self._sleep(_retry_after_seconds(exc.headers, attempt))
                    continue
                hint = ""
                if exc.code == 403:
                    hint = "; check the token and its allow-listed egress IP"
                raise OfficialApiError(
                    f"official API returned HTTP {exc.code}{hint}", status=exc.code
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < self._max_retries:
                    self._sleep(_retry_after_seconds(None, attempt))
                    continue
                raise OfficialApiError(
                    f"could not reach official API: {exc.reason}"
                ) from exc
            except OSError as exc:
                raise OfficialApiError("could not read official API response") from exc

            try:
                return json.loads(response_body)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise OfficialApiError("official API returned invalid JSON") from exc

        raise AssertionError("retry loop exhausted without returning or raising")


def _require_integer(value: Any, location: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise Top1000Error(f"{location} must be an integer")
    if positive and value <= 0:
        raise Top1000Error(f"{location} must be positive")
    return value


def _optional_nonnegative_integer(item: Mapping[str, Any], field: str, location: str) -> int:
    value = item.get(field, 0)
    if value is None:
        return 0
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Top1000Error(f"{location}.{field} must be a non-negative integer or null")
    return value


def _normalize_card(item: Any, *, location: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise Top1000Error(f"{location} must be an object")
    official_id = _require_integer(item.get("id"), f"{location}.id", positive=True)
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        raise Top1000Error(f"{location}.name must be a non-empty string")

    # Allow-list only identity and equipped-form hints. Levels, icon URLs, and unknown fields are
    # intentionally omitted because the curriculum uses tournament-standard simulation levels.
    return {
        "officialCardId": official_id,
        "name": name.strip(),
        "evolutionLevel": _optional_nonnegative_integer(item, "evolutionLevel", location),
        "heroLevel": _optional_nonnegative_integer(item, "heroLevel", location),
    }


def normalize_profile_deck(profile: Any, *, rank: int) -> dict[str, Any]:
    """Return a minimized profile record without a player name or tag."""

    if not isinstance(profile, dict):
        raise Top1000Error("player profile must be an object")
    current_deck = profile.get("currentDeck")
    if not isinstance(current_deck, list) or len(current_deck) != 8:
        raise Top1000Error("player profile currentDeck must contain exactly 8 cards")
    cards = [
        _normalize_card(card, location=f"currentDeck[{index}]")
        for index, card in enumerate(current_deck)
    ]
    official_ids = [card["officialCardId"] for card in cards]
    if len(set(official_ids)) != 8:
        raise Top1000Error("player profile currentDeck contains duplicate official card IDs")

    support_items = profile.get("currentDeckSupportCards", [])
    if support_items is None:
        support_items = []
    if not isinstance(support_items, list):
        raise Top1000Error("player profile currentDeckSupportCards must be an array or null")

    return {
        "rank": rank,
        "cards": cards,
        "towerTroops": [
            {
                "officialCardId": normalized["officialCardId"],
                "name": normalized["name"],
            }
            for normalized in (
                _normalize_card(item, location=f"currentDeckSupportCards[{index}]")
                for index, item in enumerate(support_items)
            )
        ],
    }


def fetch_ranked_players(
    client: Any,
    *,
    season: str,
    target: int,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> list[dict[str, Any]]:
    if not SEASON_PATTERN.fullmatch(season):
        raise Top1000Error("season must use YYYY-MM, for example 2026-08")
    if target <= 0 or target > 1000:
        raise Top1000Error("target must be between 1 and 1000")
    if page_size <= 0 or page_size > 200:
        raise Top1000Error("page_size must be between 1 and 200")

    path = f"/v1/locations/global/pathoflegend/{season}/rankings/players"
    players: list[dict[str, Any]] = []
    seen_tags: set[str] = set()
    seen_ranks: set[int] = set()
    seen_cursors: set[str] = set()
    after: str | None = None

    while len(players) < target:
        query: dict[str, str | int] = {"limit": min(page_size, target - len(players))}
        if after is not None:
            query["after"] = after
        payload = client.get_json(path, query)
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise Top1000Error("leaderboard response must contain an items array")

        items = payload["items"]
        if not items:
            break
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise Top1000Error(f"leaderboard.items[{index}] must be an object")
            tag = item.get("tag")
            if not isinstance(tag, str) or not PLAYER_TAG_PATTERN.fullmatch(tag):
                raise Top1000Error(f"leaderboard.items[{index}].tag is invalid")
            rank = _require_integer(
                item.get("rank"), f"leaderboard.items[{index}].rank", positive=True
            )
            if tag in seen_tags or rank in seen_ranks:
                raise Top1000Error("leaderboard contains a duplicate player tag or rank")
            seen_tags.add(tag)
            seen_ranks.add(rank)
            players.append({"tag": tag, "rank": rank})
            if len(players) == target:
                break

        if len(players) == target:
            break
        paging = payload.get("paging")
        cursors = paging.get("cursors") if isinstance(paging, dict) else None
        next_cursor = cursors.get("after") if isinstance(cursors, dict) else None
        if not isinstance(next_cursor, str) or not next_cursor:
            break
        if next_cursor in seen_cursors:
            raise Top1000Error("leaderboard pagination repeated an after cursor")
        seen_cursors.add(next_cursor)
        after = next_cursor

    return sorted(players, key=lambda player: player["rank"])


def _failure_record(rank: int, exc: Exception) -> dict[str, Any]:
    if isinstance(exc, OfficialApiError):
        return {
            "rank": rank,
            "category": "official-api-error",
            "httpStatus": exc.status,
        }
    return {"rank": rank, "category": "invalid-profile", "httpStatus": None}


def fetch_population_snapshot(
    client: Any,
    *,
    season: str,
    target: int = DEFAULT_TARGET,
    page_size: int = DEFAULT_PAGE_SIZE,
    timestamp: Callable[[], str] = _utc_timestamp,
) -> dict[str, Any]:
    """Fetch rankers and profiles while discarding player identity from persisted data."""

    started_at = timestamp()
    players = fetch_ranked_players(
        client, season=season, target=target, page_size=page_size
    )
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for player in players:
        rank = player["rank"]
        encoded_tag = urllib.parse.quote(player["tag"], safe="")
        try:
            profile = client.get_json(f"/v1/players/{encoded_tag}")
            records.append(normalize_profile_deck(profile, rank=rank))
        except (OfficialApiError, Top1000Error) as exc:
            failures.append(_failure_record(rank, exc))

    completed_at = timestamp()
    return {
        "schemaVersion": SOURCE_SCHEMA_VERSION,
        "snapshotKind": "official-ranked-current-decks",
        "source": {
            "provider": "Supercell Clash Royale API",
            "documentation": DEVELOPER_DOCUMENTATION_URL,
            "leaderboardEndpoint": (
                "/v1/locations/global/pathoflegend/{season}/rankings/players"
            ),
            "profileEndpoint": "/v1/players/{playerTag}",
            "season": season,
            "interpretation": "current decks of ranked players at retrieval time",
        },
        "retrievedAtStarted": started_at,
        "retrievedAtCompleted": completed_at,
        "targetRankCount": target,
        "leaderboardPlayersReturned": len(players),
        "profileDecksReturned": len(records),
        "records": records,
        "failures": failures,
        "privacy": {
            "playerNamesPersisted": False,
            "playerTagsPersisted": False,
        },
        "redistribution": {
            "policy": "local-private-snapshot",
            "note": "Official API terms apply; no open-data redistribution grant is asserted.",
        },
    }


def normalize_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def parse_implementation_tracker(markdown: str) -> dict[str, dict[str, Any]]:
    """Parse only playable-card and tower-troop tables from ``docs/card_tracker.md``."""

    section: str | None = None
    entries: dict[str, dict[str, Any]] = {}
    ordinary_sections = {"Troops", "Spells", "Buildings"}

    for line_number, raw_line in enumerate(markdown.splitlines(), start=1):
        if raw_line.startswith("## "):
            section = raw_line[3:].strip()
            continue
        if section not in ordinary_sections | {"Tower Troops (new mechanic -- all Missing)"}:
            continue
        if not raw_line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in raw_line.strip().strip("|").split("|")]
        if not cells or not cells[0].isdigit():
            continue

        if section in ordinary_sections:
            if len(cells) < 4:
                raise Top1000Error(f"invalid tracker table row at line {line_number}")
            cr_name, simulator_id, raw_status = cells[1], cells[2], cells[3]
            kind = section[:-1].casefold() if section.endswith("s") else section.casefold()
        else:
            if len(cells) < 3:
                raise Top1000Error(f"invalid tower tracker row at line {line_number}")
            cr_name, simulator_id, raw_status = cells[1], None, cells[2]
            kind = "tower-troop"

        status = raw_status.strip("`").upper()
        if status not in {"DONE", "PARTIAL", "MISSING"}:
            raise Top1000Error(
                f"unknown tracker status {raw_status!r} at line {line_number}"
            )
        key = normalize_name(cr_name)
        if not key:
            raise Top1000Error(f"empty tracker card name at line {line_number}")
        if key in entries:
            raise Top1000Error(f"duplicate tracker card name {cr_name!r}")
        entries[key] = {
            "crName": cr_name,
            "simulatorId": simulator_id,
            "trackerStatus": status.casefold(),
            "kind": kind,
        }

    if not entries:
        raise Top1000Error("implementation tracker contained no supported tables")
    return entries


def apply_mapping_overrides(
    tracker: Mapping[str, Mapping[str, Any]],
    payload: Any,
) -> dict[str, dict[str, Any]]:
    """Merge reviewed mappings that are absent from the Markdown tracker.

    Overrides are intentionally append-only: replacing an existing tracker row would hide a
    conflict instead of making it visible during review.
    """

    if not isinstance(payload, dict) or payload.get("schemaVersion") != 1:
        raise Top1000Error("mapping overrides must use schemaVersion 1")
    mappings = payload.get("mappings")
    if not isinstance(mappings, list):
        raise Top1000Error("mapping overrides must contain a mappings array")

    merged = {key: dict(value) for key, value in tracker.items()}
    for index, mapping in enumerate(mappings):
        if not isinstance(mapping, dict):
            raise Top1000Error(f"mapping overrides[{index}] must be an object")
        official_name = mapping.get("officialName")
        simulator_id = mapping.get("simulatorId")
        status = mapping.get("trackerStatus")
        kind = mapping.get("kind")
        evidence = mapping.get("evidence")
        if not isinstance(official_name, str) or not official_name.strip():
            raise Top1000Error(f"mapping overrides[{index}].officialName is invalid")
        if not isinstance(simulator_id, str) or not simulator_id.strip():
            raise Top1000Error(f"mapping overrides[{index}].simulatorId is invalid")
        if status not in {"done", "partial", "missing"}:
            raise Top1000Error(f"mapping overrides[{index}].trackerStatus is invalid")
        if not isinstance(kind, str) or not kind.strip():
            raise Top1000Error(f"mapping overrides[{index}].kind is invalid")
        if not isinstance(evidence, str) or not evidence.strip():
            raise Top1000Error(f"mapping overrides[{index}].evidence is required")
        key = normalize_name(official_name)
        if key in merged:
            raise Top1000Error(
                f"mapping override {official_name!r} conflicts with an existing tracker row"
            )
        merged[key] = {
            "crName": official_name.strip(),
            "simulatorId": simulator_id.strip(),
            "trackerStatus": status,
            "kind": kind.strip(),
            "mappingEvidence": evidence.strip(),
        }
    return merged


def validate_simulator_ids(
    tracker: Mapping[str, Mapping[str, Any]],
    cards_payload: Any,
) -> None:
    """Ensure every trainable tracker mapping names a card present in cards.json."""

    if not isinstance(cards_payload, list):
        raise Top1000Error("simulator cards catalog must be an array")
    simulator_ids: set[str] = set()
    for index, card_item in enumerate(cards_payload):
        if not isinstance(card_item, dict):
            raise Top1000Error(f"simulator cards[{index}] must be an object")
        simulator_id = card_item.get("id")
        if not isinstance(simulator_id, str) or not simulator_id:
            raise Top1000Error(f"simulator cards[{index}].id is invalid")
        if simulator_id in simulator_ids:
            raise Top1000Error(f"duplicate simulator card ID {simulator_id!r}")
        simulator_ids.add(simulator_id)

    for mapping in tracker.values():
        simulator_id = mapping.get("simulatorId")
        if mapping["kind"] == "tower-troop" or mapping["trackerStatus"] == "missing":
            continue
        if simulator_id not in simulator_ids:
            raise Top1000Error(
                f"trainable mapping {mapping['crName']!r} references absent simulator ID "
                f"{simulator_id!r}"
            )


def _form_and_base_name(card: Mapping[str, Any]) -> tuple[str, str]:
    name = str(card["name"]).strip()
    normalized = normalize_name(name)
    hero_level = int(card.get("heroLevel", 0))
    evolution_level = int(card.get("evolutionLevel", 0))

    if hero_level > 0 or normalized.startswith("hero"):
        if name.casefold().startswith("hero "):
            name = name[5:].strip()
        return "hero", name
    if evolution_level > 0:
        return "evolution", name
    if name.casefold().endswith(" evolution"):
        return "evolution", name[: -len(" evolution")].strip()
    return "base", name


def classify_card(
    card: Mapping[str, Any],
    tracker: Mapping[str, Mapping[str, Any]],
    *,
    fidelity_policy: str,
) -> dict[str, Any]:
    form, base_name = _form_and_base_name(card)
    implementation = tracker.get(normalize_name(base_name))
    result: dict[str, Any] = {
        "officialCardId": card["officialCardId"],
        "name": card["name"],
        "observedForm": form,
        "evolutionLevel": card.get("evolutionLevel", 0),
        "heroLevel": card.get("heroLevel", 0),
        "mappingStatus": "unmapped" if implementation is None else "mapped",
        "simulatorId": None if implementation is None else implementation["simulatorId"],
        "trackerStatus": "missing" if implementation is None else implementation["trackerStatus"],
    }

    if implementation is None or implementation["trackerStatus"] == "missing":
        result.update(
            supportStatus="missing",
            trainable=False,
            approximation=None,
        )
    elif implementation["trackerStatus"] == "partial":
        if fidelity_policy == "base-fallback":
            result.update(
                supportStatus="partial",
                trainable=True,
                approximation=(
                    "partial-base-and-form-fallback" if form != "base" else "partial-base"
                ),
            )
        else:
            result.update(
                supportStatus="partial",
                trainable=False,
                approximation=None,
            )
    elif form != "base":
        if fidelity_policy == "base-fallback":
            result.update(
                supportStatus="base-fallback",
                trainable=True,
                approximation=f"{form}-mechanic-replaced-by-base-card",
            )
        else:
            result.update(
                supportStatus="unsupported-form",
                trainable=False,
                approximation=None,
            )
    else:
        result.update(
            supportStatus="exact-base",
            trainable=True,
            approximation=None,
        )
    return result


def classify_tower_troops(
    items: Sequence[Mapping[str, Any]],
    tracker: Mapping[str, Mapping[str, Any]],
    *,
    fidelity_policy: str,
) -> tuple[list[dict[str, Any]], bool, bool]:
    if not items:
        if fidelity_policy == "base-fallback":
            return (
                [
                    {
                        "officialCardId": None,
                        "name": None,
                        "mappingStatus": "unknown",
                        "supportStatus": "default-tower-fallback",
                        "approximation": "missing API field treated as Tower Princess",
                    }
                ],
                True,
                True,
            )
        return (
            [
                {
                    "officialCardId": None,
                    "name": None,
                    "mappingStatus": "unknown",
                    "supportStatus": "missing",
                    "approximation": None,
                }
            ],
            False,
            False,
        )

    classified: list[dict[str, Any]] = []
    trainable = True
    approximated = False
    for item in items:
        implementation = tracker.get(normalize_name(str(item["name"])))
        is_default = normalize_name(str(item["name"])) == normalize_name("Tower Princess")
        exact = (
            implementation is not None
            and implementation["trackerStatus"] == "done"
            and is_default
        )
        if exact:
            classified.append(
                {
                    **item,
                    "mappingStatus": "mapped",
                    "supportStatus": "exact-default",
                    "approximation": None,
                }
            )
            continue

        if fidelity_policy == "base-fallback":
            classified.append(
                {
                    **item,
                    "mappingStatus": "unmapped" if implementation is None else "mapped",
                    "supportStatus": "default-tower-fallback",
                    "approximation": f"{item['name']} replaced by Tower Princess",
                }
            )
            approximated = True
        else:
            classified.append(
                {
                    **item,
                    "mappingStatus": "unmapped" if implementation is None else "mapped",
                    "supportStatus": "missing",
                    "approximation": None,
                }
            )
            trainable = False
    return classified, trainable, approximated


def _rank_score(rank: int) -> float:
    return 1.0 / math.sqrt(rank)


def _deck_signature(cards: Sequence[Mapping[str, Any]], towers: Sequence[Mapping[str, Any]]) -> str:
    identity = {
        "cards": sorted(
            (card["officialCardId"], card["observedForm"]) for card in cards
        ),
        "towerTroops": sorted(
            (tower.get("officialCardId"), tower.get("name") or "unknown") for tower in towers
        ),
    }
    return hashlib.sha256(_canonical_json(identity)).hexdigest()[:24]


def _deck_quality(
    cards: Sequence[Mapping[str, Any]],
    *,
    tower_trainable: bool,
    tower_approximated: bool,
) -> tuple[str, bool, list[str]]:
    reasons: set[str] = set()
    for card in cards:
        if not card["trainable"]:
            reasons.add(f"card:{card['name']}:{card['supportStatus']}")
        elif card["supportStatus"] == "partial":
            reasons.add(f"card:{card['name']}:partial")
        elif card["supportStatus"] == "base-fallback":
            reasons.add(f"card:{card['name']}:{card['observedForm']}-fallback")
    if not tower_trainable:
        reasons.add("tower-troop:unsupported")
    elif tower_approximated:
        reasons.add("tower-troop:default-fallback")

    eligible = all(card["trainable"] for card in cards) and tower_trainable
    if not eligible:
        quality = "excluded"
    elif any(card["supportStatus"] == "partial" for card in cards):
        quality = "partial"
    elif tower_approximated or any(
        card["supportStatus"] == "base-fallback" for card in cards
    ):
        quality = "approximated"
    else:
        quality = "exact-base"
    return quality, eligible, sorted(reasons)


def _validate_weight_mix(weight_mix: Mapping[str, float]) -> None:
    required = {"population", "rank", "uniform"}
    if set(weight_mix) != required:
        raise Top1000Error(f"weight mix must contain exactly {sorted(required)}")
    if any(not math.isfinite(value) or value < 0 for value in weight_mix.values()):
        raise Top1000Error("weight mix values must be finite and non-negative")
    if not math.isclose(sum(weight_mix.values()), 1.0, abs_tol=1e-9):
        raise Top1000Error("weight mix values must sum to 1")


def compile_curriculum(
    source_snapshot: Mapping[str, Any],
    tracker: Mapping[str, Mapping[str, Any]],
    *,
    tracker_sha256: str,
    fidelity_policy: str = "base-fallback",
    weight_mix: Mapping[str, float] | None = None,
    compiled_at: str | None = None,
) -> dict[str, Any]:
    if source_snapshot.get("schemaVersion") != SOURCE_SCHEMA_VERSION:
        raise Top1000Error("unsupported source snapshot schemaVersion")
    if fidelity_policy not in {"strict", "base-fallback"}:
        raise Top1000Error("fidelity_policy must be strict or base-fallback")
    mix = dict(weight_mix or {"population": 0.70, "rank": 0.20, "uniform": 0.10})
    _validate_weight_mix(mix)

    records = source_snapshot.get("records")
    failures = source_snapshot.get("failures")
    if not isinstance(records, list) or not isinstance(failures, list):
        raise Top1000Error("source snapshot records and failures must be arrays")

    deck_groups: dict[str, dict[str, Any]] = {}
    card_groups: dict[tuple[int, str], dict[str, Any]] = {}
    quality_observations: Counter[str] = Counter()
    support_observations: Counter[str] = Counter()
    total_rank_mass = 0.0
    eligible_rank_mass = 0.0
    exact_rank_mass = 0.0

    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise Top1000Error(f"records[{record_index}] must be an object")
        rank = _require_integer(record.get("rank"), f"records[{record_index}].rank", positive=True)
        raw_cards = record.get("cards")
        raw_towers = record.get("towerTroops")
        if not isinstance(raw_cards, list) or len(raw_cards) != 8:
            raise Top1000Error(f"records[{record_index}].cards must contain 8 cards")
        if not isinstance(raw_towers, list):
            raise Top1000Error(f"records[{record_index}].towerTroops must be an array")

        cards = [
            classify_card(card, tracker, fidelity_policy=fidelity_policy)
            for card in raw_cards
        ]
        towers, tower_trainable, tower_approximated = classify_tower_troops(
            raw_towers, tracker, fidelity_policy=fidelity_policy
        )
        quality, eligible, exclusion_reasons = _deck_quality(
            cards,
            tower_trainable=tower_trainable,
            tower_approximated=tower_approximated,
        )
        rank_mass = _rank_score(rank)
        total_rank_mass += rank_mass
        if eligible:
            eligible_rank_mass += rank_mass
        if quality == "exact-base":
            exact_rank_mass += rank_mass
        quality_observations[quality] += 1

        for card in cards:
            support_observations[card["supportStatus"]] += 1
            key = (card["officialCardId"], card["observedForm"])
            aggregate = card_groups.get(key)
            if aggregate is None:
                aggregate = {
                    **card,
                    "deckObservations": 0,
                    "rankWeightRaw": 0.0,
                }
                card_groups[key] = aggregate
            else:
                stable_fields = (
                    "name",
                    "mappingStatus",
                    "simulatorId",
                    "trackerStatus",
                    "supportStatus",
                )
                if any(aggregate[field] != card[field] for field in stable_fields):
                    raise Top1000Error(
                        "official card identity or mapping changed within one source snapshot: "
                        f"{key}"
                    )
            aggregate["deckObservations"] += 1
            aggregate["rankWeightRaw"] += rank_mass

        deck_id = _deck_signature(cards, towers)
        aggregate_deck = deck_groups.get(deck_id)
        if aggregate_deck is None:
            aggregate_deck = {
                "deckId": deck_id,
                "cards": sorted(cards, key=lambda item: item["officialCardId"]),
                "towerTroops": towers,
                "quality": quality,
                "eligibleForTraining": eligible,
                "exclusionReasons": exclusion_reasons,
                "simulatorDeck": (
                    sorted((card["simulatorId"] for card in cards)) if eligible else None
                ),
                "observations": 0,
                "rankMin": rank,
                "rankMax": rank,
                "rankWeightRaw": 0.0,
                "sampling": {
                    "populationWeight": 0.0,
                    "rankWeight": 0.0,
                    "uniformWeight": 0.0,
                    "combinedWeight": 0.0,
                },
            }
            deck_groups[deck_id] = aggregate_deck
        aggregate_deck["observations"] += 1
        aggregate_deck["rankMin"] = min(aggregate_deck["rankMin"], rank)
        aggregate_deck["rankMax"] = max(aggregate_deck["rankMax"], rank)
        aggregate_deck["rankWeightRaw"] += rank_mass

    decks = sorted(deck_groups.values(), key=lambda item: item["deckId"])
    eligible_decks = [deck for deck in decks if deck["eligibleForTraining"]]
    eligible_observations = sum(deck["observations"] for deck in eligible_decks)
    eligible_deck_rank_mass = sum(deck["rankWeightRaw"] for deck in eligible_decks)
    for deck in eligible_decks:
        population_weight = deck["observations"] / eligible_observations
        rank_weight = deck["rankWeightRaw"] / eligible_deck_rank_mass
        uniform_weight = 1.0 / len(eligible_decks)
        deck["sampling"] = {
            "populationWeight": population_weight,
            "rankWeight": rank_weight,
            "uniformWeight": uniform_weight,
            "combinedWeight": (
                mix["population"] * population_weight
                + mix["rank"] * rank_weight
                + mix["uniform"] * uniform_weight
            ),
        }

    target = _require_integer(source_snapshot.get("targetRankCount"), "targetRankCount", positive=True)
    leaderboard_count = _require_integer(
        source_snapshot.get("leaderboardPlayersReturned"),
        "leaderboardPlayersReturned",
    )
    observed_count = len(records)
    complete_population = leaderboard_count == target and observed_count == target and not failures
    exact_observations = quality_observations["exact-base"]
    exact_coverage = complete_population and exact_observations == target
    top1000_exact = target == 1000 and exact_coverage
    source_sha256 = _sha256(source_snapshot)
    snapshot_basis = {
        "sourceSnapshotSha256": source_sha256,
        "supportMapSha256": tracker_sha256,
        "compilerVersion": COMPILER_VERSION,
        "fidelityPolicy": fidelity_policy,
        "weightMix": mix,
    }

    def fraction(numerator: float, denominator: float) -> float:
        return 0.0 if denominator == 0 else numerator / denominator

    cards = sorted(card_groups.values(), key=lambda item: (item["officialCardId"], item["observedForm"]))
    unmapped_cards = [
        {
            "officialCardId": card["officialCardId"],
            "name": card["name"],
            "observedForm": card["observedForm"],
            "deckObservations": card["deckObservations"],
        }
        for card in cards
        if card["mappingStatus"] == "unmapped"
    ]
    excluded_decks = [deck for deck in decks if not deck["eligibleForTraining"]]

    def card_reference(card: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "officialCardId": card["officialCardId"],
            "name": card["name"],
            "observedForm": card["observedForm"],
            "simulatorId": card["simulatorId"],
            "deckObservations": card["deckObservations"],
        }

    card_audit = {
        "mapped": [card_reference(card) for card in cards if card["mappingStatus"] == "mapped"],
        "unmapped": [
            card_reference(card) for card in cards if card["mappingStatus"] == "unmapped"
        ],
        "supportedExactBase": [
            card_reference(card) for card in cards if card["supportStatus"] == "exact-base"
        ],
        "approximatedByBase": [
            card_reference(card) for card in cards if card["supportStatus"] == "base-fallback"
        ],
        "partial": [
            card_reference(card) for card in cards if card["supportStatus"] == "partial"
        ],
        "missingOrUnsupported": [
            card_reference(card)
            for card in cards
            if card["supportStatus"] in {"missing", "unsupported-form"}
        ],
    }
    deck_audit = {
        quality: [deck["deckId"] for deck in decks if deck["quality"] == quality]
        for quality in ("exact-base", "approximated", "partial", "excluded")
    }

    return {
        "schemaVersion": CURRICULUM_SCHEMA_VERSION,
        "compilerVersion": COMPILER_VERSION,
        "snapshotId": f"ranked-{hashlib.sha256(_canonical_json(snapshot_basis)).hexdigest()[:20]}",
        "compiledAt": compiled_at or _utc_timestamp(),
        "source": {
            **source_snapshot["source"],
            "sourceSnapshotSha256": source_sha256,
            "retrievedAtStarted": source_snapshot.get("retrievedAtStarted"),
            "retrievedAtCompleted": source_snapshot.get("retrievedAtCompleted"),
            "license": "Official API terms apply; no open-data license is asserted.",
            "publishPolicy": "keep generated snapshots local unless redistribution is reviewed",
        },
        "mapping": {
            "supportMapSha256": tracker_sha256,
            "fidelityPolicy": fidelity_policy,
            "meaning": {
                "exact-base": "mapped DONE base card with default Tower Princess",
                "approximated": "at least one Evolution, Hero, or Tower Troop uses a base fallback",
                "partial": "at least one mapped simulator card is marked PARTIAL",
                "excluded": "at least one card/form has no trainable simulator mapping",
            },
        },
        "samplingPolicy": {
            "weightMix": mix,
            "rankScore": "1/sqrt(rank)",
            "onlyEligibleDecksReceiveNonzeroWeight": True,
        },
        "coverage": {
            "targetRankCount": target,
            "leaderboardPlayersReturned": leaderboard_count,
            "profileDecksReturned": observed_count,
            "profileFetchFailures": len(failures),
            "uniqueDecksObserved": len(decks),
            "uniqueDecksEligible": len(eligible_decks),
            "uniqueDecksExcluded": len(excluded_decks),
            "deckObservationsByQuality": dict(sorted(quality_observations.items())),
            "cardSlotsBySupportStatus": dict(sorted(support_observations.items())),
            "profileCoverageAgainstTarget": fraction(observed_count, target),
            "trainableCoverageAgainstTarget": fraction(eligible_observations, target),
            "exactCoverageAgainstTarget": fraction(exact_observations, target),
            "rankWeightedTrainableCoverageAmongObserved": fraction(
                eligible_rank_mass, total_rank_mass
            ),
            "rankWeightedExactCoverageAmongObserved": fraction(
                exact_rank_mass, total_rank_mass
            ),
            "completeTargetPopulation": complete_population,
            "exactSimulatorCoverage": exact_coverage,
            "top1000Exact": top1000_exact,
            "trainingClaim": (
                "official-top1000-exact"
                if top1000_exact
                else "official-ranked-current-decks-partial-coverage"
            ),
        },
        "fetchFailures": failures,
        "unmappedCards": unmapped_cards,
        "cardAudit": card_audit,
        "deckAudit": deck_audit,
        "cardCoverage": cards,
        "decks": decks,
        "disclaimer": (
            "This material is unofficial and is not endorsed by Supercell. For more "
            f"information see Supercell's Fan Content Policy: {FAN_CONTENT_POLICY_URL}"
        ),
    }


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Top1000Error(f"could not read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise Top1000Error(f"invalid JSON in {path}: {exc}") from exc


def _load_tracker(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise Top1000Error(f"could not read implementation tracker {path}: {exc}") from exc
    return parse_implementation_tracker(content), hashlib.sha256(content.encode("utf-8")).hexdigest()


def _load_support_map(
    tracker_path: Path,
    overrides_path: Path,
    simulator_cards_path: Path,
) -> tuple[dict[str, dict[str, Any]], str, dict[str, str]]:
    tracker, tracker_sha256 = _load_tracker(tracker_path)
    overrides = _load_json(overrides_path)
    merged = apply_mapping_overrides(tracker, overrides)
    simulator_cards = _load_json(simulator_cards_path)
    validate_simulator_ids(merged, simulator_cards)
    overrides_sha256 = _sha256(overrides)
    sources = {
        "trackerSha256": tracker_sha256,
        "overridesSha256": overrides_sha256,
        "simulatorCardsSha256": _sha256(simulator_cards),
    }
    return merged, _sha256(sources), sources


def _add_output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True, help="write JSON atomically here")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="fetch a minimized private API snapshot")
    fetch_parser.add_argument("--season", required=True, help="ranked season ID in YYYY-MM form")
    fetch_parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    fetch_parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    fetch_parser.add_argument("--requests-per-second", type=float, default=DEFAULT_REQUESTS_PER_SECOND)
    fetch_parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    fetch_parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    fetch_parser.add_argument(
        "--token-file",
        type=Path,
        help=f"read the API token here instead of {TOKEN_ENVIRONMENT_VARIABLE}",
    )
    _add_output_argument(fetch_parser)

    compile_parser = subparsers.add_parser(
        "compile", help="compile a source snapshot into a weighted curriculum"
    )
    compile_parser.add_argument("--source", type=Path, required=True)
    compile_parser.add_argument(
        "--tracker", type=Path, default=Path("docs/card_tracker.md")
    )
    compile_parser.add_argument(
        "--mapping-overrides",
        type=Path,
        default=Path("scripts/top1000_mapping_overrides.json"),
        help="reviewed append-only mappings missing from the Markdown tracker",
    )
    compile_parser.add_argument(
        "--simulator-cards",
        type=Path,
        default=Path("data/src/main/resources/cards/cards.json"),
        help="validate trainable simulator IDs against this card catalog",
    )
    compile_parser.add_argument(
        "--fidelity-policy",
        choices=("strict", "base-fallback"),
        default="base-fallback",
    )
    compile_parser.add_argument("--population-weight", type=float, default=0.70)
    compile_parser.add_argument("--rank-weight", type=float, default=0.20)
    compile_parser.add_argument("--uniform-weight", type=float, default=0.10)
    compile_parser.add_argument(
        "--require-target-complete",
        action="store_true",
        help="return exit code 3 unless every requested rank yielded a profile deck",
    )
    compile_parser.add_argument(
        "--require-exact-coverage",
        action="store_true",
        help="return exit code 3 unless all requested decks are exact in the simulator",
    )
    _add_output_argument(compile_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "fetch":
            token = resolve_token(args.token_file)
            client = OfficialApiClient(
                token,
                requests_per_second=args.requests_per_second,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            document = fetch_population_snapshot(
                client,
                season=args.season,
                target=args.target,
                page_size=args.page_size,
            )
        else:
            source = _load_json(args.source)
            tracker, tracker_sha256, mapping_sources = _load_support_map(
                args.tracker, args.mapping_overrides, args.simulator_cards
            )
            document = compile_curriculum(
                source,
                tracker,
                tracker_sha256=tracker_sha256,
                fidelity_policy=args.fidelity_policy,
                weight_mix={
                    "population": args.population_weight,
                    "rank": args.rank_weight,
                    "uniform": args.uniform_weight,
                },
            )
            document["mapping"]["sourceHashes"] = mapping_sources

        _write_atomically(args.output, _serialize(document))
        if args.command == "compile":
            coverage = document["coverage"]
            if args.require_target_complete and not coverage["completeTargetPopulation"]:
                return 3
            if args.require_exact_coverage and not coverage["exactSimulatorCoverage"]:
                return 3
    except Top1000Error as exc:
        print(f"top-1000 curriculum failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
