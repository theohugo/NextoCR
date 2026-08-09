"""Tests for the bounded immutable self-play checkpoint league."""

from __future__ import annotations

from pathlib import Path

from crforge_gym.league import (
    CheckpointLeague,
    LeagueEntry,
    LeagueSelfPlayOpponent,
)


class _SavedModel:
    def __init__(self, payload: bytes):
        self.payload = payload

    def save(self, path: str) -> None:
        Path(f"{path}.zip").write_bytes(self.payload)


def _make_league(path: Path) -> CheckpointLeague:
    return CheckpointLeague(
        path,
        seed=91,
        model_loader=lambda checkpoint: checkpoint.name,
        max_recent=2,
        max_historical=2,
        max_loaded_models=1,
    )


def test_pool_is_bounded_and_snapshots_are_immutable(tmp_path: Path) -> None:
    league = _make_league(tmp_path / "league")
    initial = league.ensure_initial(_SavedModel(b"initial"), checkpoint_step=0)
    initial_path = league.root / initial.path
    for step in range(1, 8):
        league.add_snapshot(_SavedModel(f"step-{step}".encode()), checkpoint_step=step)

    assert league.size <= 1 + 2 + 2
    assert len(list(league.root.glob("*.zip"))) == league.size
    assert initial_path.read_bytes() == b"initial"
    assert league.summary()["recent"] == 2
    assert league.summary()["historical"] <= 2


def test_sampling_is_seeded_and_model_cache_is_bounded(tmp_path: Path) -> None:
    first = _make_league(tmp_path / "first")
    second = _make_league(tmp_path / "second")
    for league in (first, second):
        league.ensure_initial(_SavedModel(b"initial"), checkpoint_step=0)
        for step in range(1, 6):
            league.add_snapshot(_SavedModel(str(step).encode()), checkpoint_step=step)

    first_ids = [first.sample()[0].opponent_id for _ in range(20)]
    second_ids = [second.sample()[0].opponent_id for _ in range(20)]

    assert first_ids == second_ids
    assert len(first._cache) <= 1
    resumed = _make_league(tmp_path / "first")
    assert resumed.sample_count == 20


class _FakeLeague:
    def refresh(self) -> bool:
        # Parallel environments re-read the manifest at every episode
        # boundary; a fake league has nothing to reload.
        return False

    def __init__(self):
        self.calls = 0

    def sample(self):
        self.calls += 1
        entry = LeagueEntry(
            opponent_id=f"opponent-{self.calls}",
            path="unused.zip",
            checkpoint_step=self.calls,
            category="recent",
            created_index=self.calls,
        )
        return entry, f"model-{self.calls}"


class _Delegate:
    model = None

    def act(self, obs_raw, player="red", obs_flat=None):
        return {"model": self.model, "frame": obs_raw["frame"]}


def test_opponent_is_sampled_once_per_episode_boundary() -> None:
    league = _FakeLeague()
    selections = []
    opponent = LeagueSelfPlayOpponent(
        league, on_selection=lambda entry, episode: selections.append((entry.opponent_id, episode))
    )
    opponent._delegate = _Delegate()

    actions = [opponent.act({"frame": frame}) for frame in (0, 15, 30, 0, 15)]

    assert league.calls == 2
    assert selections == [("opponent-1", 1), ("opponent-2", 2)]
    assert [action["model"] for action in actions] == [
        "model-1", "model-1", "model-1", "model-2", "model-2"
    ]
