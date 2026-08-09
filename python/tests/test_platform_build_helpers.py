"""Platform-contract tests for the Python bridge build helpers."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch

import pytest


PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from crforge_gym import jpype_bridge  # noqa: E402


def _load_train_module():
    path = PYTHON_ROOT / "examples" / "train_ppo.py"
    spec = importlib.util.spec_from_file_location("nextocr_train_ppo_contracts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_ppo = _load_train_module()


@pytest.mark.parametrize(
    ("os_name", "wrapper", "launcher"),
    [
        ("nt", "gradlew.bat", "gym-bridge.bat"),
        ("posix", "gradlew", "gym-bridge"),
    ],
)
def test_native_paths_match_platform(
    tmp_path: Path, os_name: str, wrapper: str, launcher: str
) -> None:
    root = str(tmp_path)

    assert os.path.basename(jpype_bridge._gradle_wrapper_path(root, os_name)) == wrapper
    assert os.path.basename(train_ppo._gradle_wrapper_path(root, os_name)) == wrapper
    assert os.path.basename(train_ppo._bridge_launcher_path(root, os_name)) == launcher


@pytest.mark.parametrize(("os_name", "wrapper"), [("nt", "gradlew.bat"), ("posix", "gradlew")])
def test_project_root_uses_native_wrapper(
    tmp_path: Path, os_name: str, wrapper: str
) -> None:
    nested = tmp_path / "python" / "crforge_gym"
    nested.mkdir(parents=True)
    (tmp_path / wrapper).touch()

    assert jpype_bridge._find_project_root(str(nested), os_name) == str(tmp_path)
    assert train_ppo._find_project_root(str(nested), os_name) == str(tmp_path)


@pytest.mark.parametrize(("os_name", "wrapper"), [("nt", "gradlew.bat"), ("posix", "gradlew")])
def test_jpype_build_is_refreshed_even_when_distribution_exists(
    tmp_path: Path, os_name: str, wrapper: str
) -> None:
    lib_dir = tmp_path / "gym-bridge" / "build" / "install" / "gym-bridge" / "lib"
    lib_dir.mkdir(parents=True)
    (lib_dir / "bridge.jar").touch()
    (tmp_path / wrapper).touch()
    completed = Mock(returncode=0, stderr="")

    with patch.dict(os.environ, {"JAVA_HOME": "test-java-home"}), patch.object(
        subprocess, "run", return_value=completed
    ) as run:
        first = jpype_bridge._ensure_jars_built(str(tmp_path), os_name)
        second = jpype_bridge._ensure_jars_built(str(tmp_path), os_name)

    assert first == second == str(lib_dir)
    assert run.call_count == 2
    assert run.call_args.args[0] == [
        str(tmp_path / wrapper),
        ":gym-bridge:installDist",
        "-q",
    ]


@pytest.mark.parametrize(
    ("os_name", "wrapper", "launcher"),
    [
        ("nt", "gradlew.bat", "gym-bridge.bat"),
        ("posix", "gradlew", "gym-bridge"),
    ],
)
def test_server_build_is_refreshed_and_returns_native_launcher(
    tmp_path: Path, os_name: str, wrapper: str, launcher: str
) -> None:
    script = (
        tmp_path
        / "gym-bridge"
        / "build"
        / "install"
        / "gym-bridge"
        / "bin"
        / launcher
    )
    script.parent.mkdir(parents=True)
    script.touch()
    (tmp_path / wrapper).touch()
    completed = Mock(returncode=0, stderr="")

    with patch.object(train_ppo, "_get_java_home", return_value="test-java-home"), patch.object(
        train_ppo.subprocess, "run", return_value=completed
    ) as run:
        first = train_ppo._build_bridge_dist(str(tmp_path), os_name)
        second = train_ppo._build_bridge_dist(str(tmp_path), os_name)

    assert first == second == str(script)
    assert run.call_count == 2
    assert run.call_args.args[0] == [
        str(tmp_path / wrapper),
        ":gym-bridge:installDist",
        "-q",
    ]


def test_save_parent_is_created_before_training(tmp_path: Path) -> None:
    save_path = tmp_path / "nested" / "models" / "nextocr_agent"

    parent = train_ppo._ensure_save_parent(str(save_path))

    assert parent == str(save_path.parent)
    assert save_path.parent.is_dir()
