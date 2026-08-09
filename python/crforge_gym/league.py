# SPDX-License-Identifier: Apache-2.0
"""Bounded, reproducible checkpoint league for NextoCR self-play."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import random
from typing import Any, Callable

from stable_baselines3.common.callbacks import BaseCallback

from nextocr_fsio import replace_with_retry, unlink_with_retry


LEAGUE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LeagueEntry:
    """One immutable policy snapshot participating in opponent sampling."""

    opponent_id: str
    path: str
    checkpoint_step: int
    category: str
    created_index: int


class CheckpointLeague:
    """Maintain initial, recent and reservoir-sampled historical opponents.

    Snapshot files are immutable while present.  The active pool is bounded to
    ``1 + max_recent + max_historical`` files; evicted generated snapshots are
    deleted after they leave the pool.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        seed: int,
        model_loader: Callable[[Path], Any],
        max_recent: int = 4,
        max_historical: int = 8,
        max_loaded_models: int = 2,
        initial_weight: float = 0.10,
        recent_weight: float = 0.60,
        historical_weight: float = 0.30,
    ):
        if max_recent < 1 or max_historical < 0 or max_loaded_models < 1:
            raise ValueError("league capacities must be max_recent>=1, max_historical>=0, cache>=1")
        weights = (initial_weight, recent_weight, historical_weight)
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("league sampling weights must be non-negative with a positive sum")

        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "league.json"
        self.seed = int(seed)
        self.model_loader = model_loader
        self.max_recent = int(max_recent)
        self.max_historical = int(max_historical)
        self.max_loaded_models = int(max_loaded_models)
        self.weights = {
            "initial": float(initial_weight),
            "recent": float(recent_weight),
            "historical": float(historical_weight),
        }
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._state = self._load_or_create()
        self._validate_state()

    @property
    def size(self) -> int:
        return (
            (1 if self._state.get("initial") else 0)
            + len(self._state["recent"])
            + len(self._state["historical"])
        )

    @property
    def sample_count(self) -> int:
        return int(self._state["sampleCount"])

    def ensure_initial(self, model: Any, *, checkpoint_step: int) -> LeagueEntry:
        """Persist the initial policy exactly once and return it."""
        initial = self._state.get("initial")
        if initial:
            return LeagueEntry(**initial)
        entry = self._save_snapshot(model, checkpoint_step=checkpoint_step, category="initial")
        self._state["initial"] = asdict(entry)
        self._flush()
        return entry

    def add_snapshot(self, model: Any, *, checkpoint_step: int) -> LeagueEntry:
        """Add an immutable recent snapshot and reservoir-sample evictions."""
        for raw in self._state["recent"]:
            if int(raw["checkpoint_step"]) == int(checkpoint_step):
                return LeagueEntry(**raw)

        entry = self._save_snapshot(model, checkpoint_step=checkpoint_step, category="recent")
        self._state["recent"].append(asdict(entry))
        if len(self._state["recent"]) > self.max_recent:
            evicted = LeagueEntry(**self._state["recent"].pop(0))
            self._consider_historical(replace(evicted, category="historical"))
        self._flush()
        return entry

    def sample(self) -> tuple[LeagueEntry, Any]:
        """Sample one opponent deterministically from the current categories."""
        categories: list[tuple[str, list[dict[str, Any]], float]] = []
        initial = self._state.get("initial")
        if initial:
            categories.append(("initial", [initial], self.weights["initial"]))
        if self._state["recent"]:
            categories.append(("recent", self._state["recent"], self.weights["recent"]))
        if self._state["historical"]:
            categories.append(
                ("historical", self._state["historical"], self.weights["historical"])
            )
        if not categories:
            raise RuntimeError("the self-play league is empty; call ensure_initial() first")

        sample_index = int(self._state["sampleCount"])
        rng = random.Random(self.seed + sample_index * 1_000_003)
        positive = [category for category in categories if category[2] > 0]
        available = positive or categories
        total_weight = sum(category[2] for category in available)
        if total_weight <= 0:
            selected_category = available[rng.randrange(len(available))]
        else:
            threshold = rng.random() * total_weight
            selected_category = available[-1]
            cumulative = 0.0
            for category in available:
                cumulative += category[2]
                if threshold <= cumulative:
                    selected_category = category
                    break
        raw_entry = selected_category[1][rng.randrange(len(selected_category[1]))]
        entry = LeagueEntry(**raw_entry)
        self._state["sampleCount"] = sample_index + 1
        self._state["lastSelection"] = {
            "sampleIndex": sample_index,
            "opponentId": entry.opponent_id,
            "category": entry.category,
            "checkpointStep": entry.checkpoint_step,
        }
        self._flush()
        return entry, self._load_model(entry)

    def summary(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "initial": 1 if self._state.get("initial") else 0,
            "recent": len(self._state["recent"]),
            "historical": len(self._state["historical"]),
            "sampleCount": self.sample_count,
            "maxLoadedModels": self.max_loaded_models,
        }

    def _save_snapshot(
        self, model: Any, *, checkpoint_step: int, category: str
    ) -> LeagueEntry:
        created_index = int(self._state["nextSnapshotIndex"])
        opponent_id = f"policy_{created_index:06d}_step_{int(checkpoint_step):012d}"
        destination = self.root / f"{opponent_id}.zip"
        temporary_base = self.root / f".{opponent_id}.tmp"
        model.save(str(temporary_base))
        temporary = Path(f"{temporary_base}.zip")
        if not temporary.is_file() and temporary_base.is_file():
            temporary = temporary_base
        if not temporary.is_file():
            raise FileNotFoundError(f"model.save() did not create a league snapshot for {opponent_id}")
        replace_with_retry(temporary, destination)
        self._state["nextSnapshotIndex"] = created_index + 1
        return LeagueEntry(
            opponent_id=opponent_id,
            path=destination.name,
            checkpoint_step=int(checkpoint_step),
            category=category,
            created_index=created_index,
        )

    def _consider_historical(self, candidate: LeagueEntry) -> None:
        self._state["historicalCandidates"] += 1
        seen = int(self._state["historicalCandidates"])
        historical = self._state["historical"]
        if self.max_historical == 0:
            self._discard(candidate)
            return
        if len(historical) < self.max_historical:
            historical.append(asdict(candidate))
            return

        rng = random.Random(self.seed + seen * 2_000_003)
        slot = rng.randrange(seen)
        if slot < self.max_historical:
            replaced = LeagueEntry(**historical[slot])
            historical[slot] = asdict(candidate)
            self._discard(replaced)
        else:
            self._discard(candidate)

    def _discard(self, entry: LeagueEntry) -> None:
        self._cache.pop(entry.opponent_id, None)
        path = self.root / entry.path
        if path.is_file():
            # The replay viewer may still have this policy open; the entry is
            # already dropped from the league state, so an orphaned file on
            # disk is a better outcome than aborting training.
            unlink_with_retry(path)

    def _load_model(self, entry: LeagueEntry) -> Any:
        cached = self._cache.pop(entry.opponent_id, None)
        if cached is not None:
            self._cache[entry.opponent_id] = cached
            return cached
        model = self.model_loader(self.root / entry.path)
        self._cache[entry.opponent_id] = model
        while len(self._cache) > self.max_loaded_models:
            self._cache.popitem(last=False)
        return model

    def _load_or_create(self) -> dict[str, Any]:
        if self.manifest_path.is_file():
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
            configured = state.get("configuration", {})
            expected = {
                "seed": self.seed,
                "maxRecent": self.max_recent,
                "maxHistorical": self.max_historical,
                "maxLoadedModels": self.max_loaded_models,
                "weights": self.weights,
            }
            if configured != expected:
                raise ValueError(
                    "Cannot resume league with different capacity, seed, cache, or weights"
                )
            return state
        return {
            "schemaVersion": LEAGUE_SCHEMA_VERSION,
            "configuration": {
                "seed": self.seed,
                "maxRecent": self.max_recent,
                "maxHistorical": self.max_historical,
                "maxLoadedModels": self.max_loaded_models,
                "weights": self.weights,
            },
            "initial": None,
            "recent": [],
            "historical": [],
            "nextSnapshotIndex": 0,
            "historicalCandidates": 0,
            "sampleCount": 0,
            "lastSelection": None,
        }

    def _validate_state(self) -> None:
        entries: list[dict[str, Any]] = []
        if self._state.get("initial"):
            entries.append(self._state["initial"])
        entries.extend(self._state["recent"])
        entries.extend(self._state["historical"])
        if len(self._state["recent"]) > self.max_recent:
            raise ValueError("league manifest exceeds maxRecent")
        if len(self._state["historical"]) > self.max_historical:
            raise ValueError("league manifest exceeds maxHistorical")
        for raw in entries:
            path = self.root / raw["path"]
            if not path.is_file():
                raise FileNotFoundError(f"league checkpoint is missing: {path}")

    def _flush(self) -> None:
        temporary = self.root / f".league.json.{os.getpid()}.tmp"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(self._state, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if not replace_with_retry(temporary, self.manifest_path, tolerate_failure=True):
            temporary.unlink(missing_ok=True)


class LeagueSelfPlayOpponent:
    """Delegate red actions to one league policy sampled at each episode boundary."""

    def __init__(
        self,
        league: CheckpointLeague,
        *,
        on_selection: Callable[[LeagueEntry, int], None] | None = None,
    ):
        from crforge_gym.opponents import SelfPlayOpponent

        self.league = league
        self._delegate = SelfPlayOpponent(model=None)
        self._on_selection = on_selection
        self._last_frame: int | None = None
        self._episode = 0
        self.selected_entry: LeagueEntry | None = None

    def act(
        self,
        obs_raw: dict | None,
        player: str = "red",
        obs_flat: Any | None = None,
    ) -> dict | None:
        frame = int(obs_raw.get("frame", 0)) if obs_raw is not None else 0
        if self.selected_entry is None or (
            self._last_frame is not None and frame <= self._last_frame
        ):
            self._episode += 1
            self.selected_entry, self._delegate.model = self.league.sample()
            if self._on_selection is not None:
                self._on_selection(self.selected_entry, self._episode)
        self._last_frame = frame
        return self._delegate.act(obs_raw, player=player, obs_flat=obs_flat)


class LeagueSnapshotCallback(BaseCallback):
    """Periodically add immutable current-policy snapshots to the league."""

    def __init__(self, league: CheckpointLeague, frequency: int, verbose: int = 1):
        super().__init__(verbose)
        self.league = league
        self.frequency = int(frequency)
        self._next_timestep = 0

    def _on_training_start(self) -> None:
        current = int(self.model.num_timesteps)
        self._next_timestep = ((current // self.frequency) + 1) * self.frequency

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_timestep:
            return True
        entry = self.league.add_snapshot(self.model, checkpoint_step=self.num_timesteps)
        while self._next_timestep <= self.num_timesteps:
            self._next_timestep += self.frequency
        self.logger.record("league/pool_size", self.league.size)
        if self.verbose:
            print(
                f"League snapshot {entry.opponent_id} added at {self.num_timesteps}; "
                f"pool={self.league.size}"
            )
        return True
