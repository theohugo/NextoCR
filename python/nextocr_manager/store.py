# SPDX-License-Identifier: Apache-2.0
"""Validated, durable storage helpers for the local NextoCR manager."""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any

from nextocr_fsio import atomic_write_json as _fsio_write_json


SCHEMA_VERSION = 1
SAFE_REPLAY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,95}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def atomic_write_json(path: Path, value: Any) -> None:
    # Several HTTP worker threads read these files while one writes, which is
    # enough to trigger the Windows sharing rules described in nextocr_fsio.
    _fsio_write_json(path, value, fsync=True)


class RunStore:
    """Read and create runs while keeping every resolved path under ``runs/``."""

    def __init__(self, runs_root: Path):
        self.root = runs_root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def validate_run_id(self, run_id: str, *, must_exist: bool = True) -> Path:
        if not isinstance(run_id, str) or not run_id or "\\" in run_id or "\x00" in run_id:
            raise ValueError("invalid runId")
        pure = PurePosixPath(run_id)
        if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
            raise ValueError("invalid runId")
        candidate = (self.root / Path(*pure.parts)).resolve()
        self._require_under_root(candidate)
        if must_exist and not candidate.is_dir():
            raise FileNotFoundError(f"unknown run: {run_id}")
        return candidate

    def run_id(self, run_dir: Path) -> str:
        resolved = run_dir.resolve()
        self._require_under_root(resolved)
        return resolved.relative_to(self.root).as_posix()

    def discover(self) -> list[str]:
        found: set[str] = set()
        for marker_name in ("manifest.json", "manager.json"):
            for marker in self.root.rglob(marker_name):
                try:
                    run_dir = marker.parent.resolve()
                    self._require_under_root(run_dir)
                    if run_dir == self.root:
                        continue
                    found.add(self.run_id(run_dir))
                except (OSError, ValueError):
                    continue
        return sorted(found)

    def summary(self, run_id: str, *, live: bool = False) -> dict[str, Any]:
        run_dir = self.validate_run_id(run_id)
        manifest = read_json(run_dir / "manifest.json", {}) or {}
        config = read_json(run_dir / "config.json", {}) or {}
        metrics = read_json(run_dir / "metrics.json", {}) or {}
        manager = read_json(run_dir / "manager.json", {}) or {}
        training = metrics.get("training", {}) if isinstance(metrics, dict) else {}
        checkpoints = metrics.get("checkpoints", []) if isinstance(metrics, dict) else []

        checkpoint_steps = [
            int(entry.get("timesteps", 0))
            for entry in checkpoints
            if isinstance(entry, dict)
        ]
        attempts = manifest.get("attempts", []) if isinstance(manifest, dict) else []
        initial_start = 0
        if attempts and isinstance(attempts[0], dict):
            initial_start = int(attempts[0].get("startTimesteps", 0) or 0)
        observed = int(training.get("lastTimesteps", 0) or 0)
        durable = max(
            [
                int(manifest.get("finalTimesteps", 0) or 0),
                int(manager.get("sourceCheckpointTimesteps", 0) or 0),
                *checkpoint_steps,
            ]
        )
        current = max(durable, observed) if live else durable
        configured_attempt = int(config.get("timesteps_this_attempt", 0) or 0)
        target = int(
            manager.get("targetTotalTimesteps", 0)
            or (initial_start + configured_attempt)
            or current
        )
        target = max(target, current)
        episodes = int(training.get("episodes", 0) or 0)
        wins = int(training.get("wins", 0) or 0)
        losses = int(training.get("losses", 0) or 0)
        draws = int(training.get("draws", 0) or 0)
        known = wins + losses + draws
        manifest_status = str(manifest.get("status") or manager.get("status") or "created")
        status = "running" if live else manifest_status
        latest = self.latest_checkpoint(run_id, required=False)
        updated = (
            manifest.get("updatedAtUtc")
            or metrics.get("updatedAtUtc")
            or manager.get("updatedAtUtc")
            or manager.get("createdAtUtc")
        )
        return {
            "runId": run_id,
            "label": manager.get("label") or run_dir.name,
            "version": manager.get("version"),
            "parentRunId": manager.get("parentRunId"),
            "sourceCheckpoint": manager.get("sourceCheckpoint"),
            "status": status,
            "manifestStatus": manifest_status,
            "live": bool(live),
            "currentTimesteps": current,
            "durableTimesteps": durable,
            "observedTimesteps": observed,
            "targetTotalTimesteps": target,
            "remainingTimesteps": max(0, target - current),
            "progress": (min(1.0, current / target) if target > 0 else 0.0),
            "episodes": episodes,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "unknownOutcomes": int(training.get("unknownOutcomes", 0) or 0),
            "winRate": (wins / known if known else 0.0),
            "meanReward": float(training.get("meanReward", 0.0) or 0.0),
            "meanLength": float(training.get("meanLength", 0.0) or 0.0),
            "latestCheckpoint": str(latest) if latest else None,
            "checkpointCount": len(checkpoints),
            "updatedAtUtc": updated,
            "deckProfile": config.get("deck_profile", "mortar_self_play_v1"),
            "opponent": config.get("opponent", "self_play"),
            "evaluationOpponent": config.get("evaluation_opponent", "rule_based"),
            "seed": int(config.get("seed", 42) or 42),
        }

    def detail(self, run_id: str, *, live: bool = False) -> dict[str, Any]:
        run_dir = self.validate_run_id(run_id)
        result = self.summary(run_id, live=live)
        metrics = read_json(run_dir / "metrics.json", {}) or {}
        league = read_json(run_dir / "league" / "league.json", {}) or {}
        result.update(
            {
                "manifest": read_json(run_dir / "manifest.json", {}) or {},
                "config": read_json(run_dir / "config.json", {}) or {},
                "manager": read_json(run_dir / "manager.json", {}) or {},
                "checkpoints": metrics.get("checkpoints", []),
                "evaluations": metrics.get("evaluations", []),
                "league": league,
            }
        )
        return result

    def metrics(self, run_id: str) -> dict[str, Any]:
        run_dir = self.validate_run_id(run_id)
        result = read_json(run_dir / "metrics.json", {}) or {}
        episodes = _recent_episode_rows(run_dir / "metrics.csv", limit=100)
        result["recentEpisodes"] = episodes
        result["episodes"] = episodes
        result["latest"] = episodes[-1] if episodes else None
        return result

    def tail_logs(self, run_id: str, lines: int = 200) -> dict[str, Any]:
        run_dir = self.validate_run_id(run_id)
        count = max(1, min(int(lines), 2_000))
        result: dict[str, list[str]] = {}
        for filename in ("trainer.out.log", "trainer.err.log", "manager.log"):
            result[filename] = _tail_file(run_dir / filename, count)
        combined = []
        for filename, content in result.items():
            combined.extend(f"[{filename}] {line}" for line in content)
        return {"runId": run_id, "tail": count, "logs": result, "lines": combined[-count:]}

    def latest_checkpoint(self, run_id: str, *, required: bool = True) -> Path | None:
        run_dir = self.validate_run_id(run_id)
        manifest = read_json(run_dir / "manifest.json", {}) or {}
        candidates: list[Path] = []
        latest = manifest.get("latestCheckpoint")
        if isinstance(latest, str) and latest:
            path = Path(latest)
            candidates.append(path if path.is_absolute() else run_dir / path)
        candidates.extend(
            [run_dir / "checkpoints" / "latest.zip", run_dir / "final_model.zip"]
        )
        candidates.extend(
            reversed(sorted((run_dir / "checkpoints").glob("step_*.zip")))
        )
        for candidate in candidates:
            try:
                resolved = candidate.expanduser().resolve()
                resolved.relative_to(run_dir)
            except (OSError, ValueError):
                continue
            if resolved.is_file() and resolved.suffix.lower() == ".zip":
                return resolved
        source = read_json(run_dir / "manager.json", {}) or {}
        source_checkpoint = source.get("sourceCheckpoint")
        if isinstance(source_checkpoint, str) and source_checkpoint:
            try:
                resolved = Path(source_checkpoint).expanduser().resolve()
                resolved.relative_to(self.root)
            except (OSError, ValueError):
                resolved = Path()
            if resolved.is_file() and resolved.suffix.lower() == ".zip":
                return resolved
        if required:
            raise FileNotFoundError(f"run has no checkpoint: {run_id}")
        return None

    def checkpoint_from_value(self, run_id: str, value: str | None) -> Path:
        if not value or value == "latest":
            checkpoint = self.latest_checkpoint(run_id)
            assert checkpoint is not None
            return checkpoint
        run_dir = self.validate_run_id(run_id)
        candidate = Path(value)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            pure = PurePosixPath(value)
            if any(part in ("", ".", "..") for part in pure.parts):
                raise ValueError("invalid checkpoint path")
            resolved = (run_dir / Path(*pure.parts)).resolve()
        try:
            resolved.relative_to(run_dir)
        except ValueError as exc:
            raise ValueError("checkpoint must stay inside its run") from exc
        if not resolved.is_file() or resolved.suffix.lower() != ".zip":
            raise FileNotFoundError("checkpoint does not exist")
        return resolved

    def create_run(
        self,
        *,
        label: str,
        target_total_timesteps: int,
        parent_run_id: str | None = None,
        source_checkpoint: Path | None = None,
        source_checkpoint_timesteps: int = 0,
    ) -> tuple[str, Path]:
        clean_label = _clean_label(label)
        version = self._next_version()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        slug = _slugify(clean_label)
        run_id = f"nextocr/mortar-v{version:03d}-{slug}-{stamp}"
        run_dir = self.validate_run_id(run_id, must_exist=False)
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError("generated run directory already exists") from exc
        metadata = {
            "schemaVersion": SCHEMA_VERSION,
            "runId": run_id,
            "label": clean_label,
            "version": version,
            "parentRunId": parent_run_id,
            "sourceCheckpoint": str(source_checkpoint) if source_checkpoint else None,
            "sourceCheckpointTimesteps": int(source_checkpoint_timesteps),
            "targetTotalTimesteps": int(target_total_timesteps),
            "status": "created",
            "createdAtUtc": utc_now(),
            "updatedAtUtc": utc_now(),
        }
        atomic_write_json(run_dir / "manager.json", metadata)
        return run_id, run_dir

    def update_manager(self, run_id: str, **updates: Any) -> dict[str, Any]:
        run_dir = self.validate_run_id(run_id)
        metadata = read_json(run_dir / "manager.json", {}) or {
            "schemaVersion": SCHEMA_VERSION,
            "runId": run_id,
            "label": run_dir.name,
            "targetTotalTimesteps": self.summary(run_id)["targetTotalTimesteps"],
            "createdAtUtc": utc_now(),
        }
        metadata.update(updates)
        metadata["updatedAtUtc"] = utc_now()
        atomic_write_json(run_dir / "manager.json", metadata)
        return metadata

    def _next_version(self) -> int:
        versions = []
        for run_id in self.discover():
            data = read_json(self.validate_run_id(run_id) / "manager.json", {}) or {}
            try:
                versions.append(int(data.get("version", 0)))
            except (TypeError, ValueError):
                continue
        return max(versions, default=0) + 1

    def _require_under_root(self, candidate: Path) -> None:
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("path escapes runs root") from exc


def _clean_label(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("label must be a string")
    cleaned = " ".join(value.strip().split())
    if not cleaned or len(cleaned) > 80 or any(ord(ch) < 32 for ch in cleaned):
        raise ValueError("label must contain 1-80 printable characters")
    return cleaned


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (slug or "training")[:32]


def _tail_file(path: Path, line_count: int) -> list[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            chunks: list[bytes] = []
            newline_count = 0
            while position > 0 and newline_count <= line_count:
                size = min(8192, position)
                position -= size
                handle.seek(position)
                chunk = handle.read(size)
                chunks.append(chunk)
                newline_count += chunk.count(b"\n")
            text = b"".join(reversed(chunks)).decode("utf-8", errors="replace")
            return text.splitlines()[-line_count:]
    except OSError:
        return []


def _recent_episode_rows(path: Path, *, limit: int) -> list[dict[str, Any]]:
    """Parse a bounded tail of metrics.csv without loading an unbounded run."""
    max_bytes = 1024 * 1024
    try:
        with path.open("rb") as handle:
            header = handle.readline().decode("utf-8", errors="replace").rstrip("\r\n")
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            start = max(0, end - max_bytes)
            handle.seek(start)
            tail = handle.read().decode("utf-8", errors="replace")
        lines = tail.splitlines()
        if start > 0 and lines:
            lines = lines[1:]
        if not header or not lines:
            return []
        reader = csv.DictReader([header, *lines])
        rows: list[dict[str, Any]] = []
        for row in reader:
            if row.get("event") != "train_episode":
                continue
            try:
                rows.append(
                    {
                        "timestampUtc": row.get("timestamp_utc") or None,
                        "timesteps": int(row.get("timesteps") or 0),
                        "episode": int(row.get("episode") or 0),
                        "reward": float(row.get("reward") or 0.0),
                        "length": int(row.get("length") or 0),
                        "outcome": row.get("outcome") or "unknown",
                    }
                )
            except (TypeError, ValueError):
                continue
        return rows[-max(1, min(int(limit), 500)):]
    except OSError:
        return []
