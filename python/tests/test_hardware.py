# Modified by NextoCR contributors; see NOTICE for attribution.
"""Tests for sizing a run to the machine.

The failure that matters is not "too slow": it is a plan that exhausts RAM and
makes the desktop unusable while a multi-day run is in flight.
"""

from __future__ import annotations

from crforge_gym.hardware import (
    BRIDGE_MEMORY_BYTES,
    HardwarePlan,
    detect_available_memory_bytes,
    plan_training_resources,
)


GIB = 1024 ** 3


def test_bigger_machines_get_more_environments() -> None:
    small = plan_training_resources(cpu_count=4, available_memory_bytes=16 * GIB)
    large = plan_training_resources(cpu_count=32, available_memory_bytes=64 * GIB)

    assert small.num_envs < large.num_envs
    assert small.num_envs >= 1


def test_cores_are_always_left_for_the_desktop() -> None:
    plan = plan_training_resources(cpu_count=12, available_memory_bytes=32 * GIB)

    # Environments plus the update pool must not claim every core, or the
    # dashboard and the desktop stall while training runs.
    assert plan.num_envs < 12
    assert plan.num_envs + plan.torch_threads <= 12 + 1


def test_a_tiny_machine_still_produces_a_runnable_plan() -> None:
    plan = plan_training_resources(cpu_count=1, available_memory_bytes=2 * GIB)

    assert plan.num_envs == 1
    assert plan.torch_threads >= 1


def test_low_memory_caps_environments_before_cores_do() -> None:
    """A 32-core box with 6 GiB free must not launch 20 JVMs."""
    plan = plan_training_resources(cpu_count=32, available_memory_bytes=6 * GIB)

    assert plan.limited_by == "memory"
    assert plan.num_envs * BRIDGE_MEMORY_BYTES < 6 * GIB


def test_explicit_request_is_still_capped_by_memory() -> None:
    plan = plan_training_resources(
        cpu_count=32, available_memory_bytes=5 * GIB, requested_envs=16
    )

    assert plan.num_envs < 16
    assert plan.limited_by == "memory"


def test_explicit_request_is_honoured_when_it_fits() -> None:
    plan = plan_training_resources(
        cpu_count=32, available_memory_bytes=64 * GIB, requested_envs=6
    )

    assert plan.num_envs == 6
    assert plan.limited_by == "request"


def test_unknown_memory_falls_back_to_cores_without_crashing() -> None:
    plan = plan_training_resources(cpu_count=8, available_memory_bytes=0)

    assert plan.num_envs >= 1
    assert plan.limited_by in {"cpu", "limit"}


def test_environment_count_is_bounded_even_on_huge_machines() -> None:
    plan = plan_training_resources(cpu_count=256, available_memory_bytes=512 * GIB)

    assert plan.num_envs <= 16
    assert plan.limited_by == "limit"


def test_detection_works_on_this_machine() -> None:
    memory = detect_available_memory_bytes()
    plan = plan_training_resources()

    assert memory > 0, "memory detection should work on a supported platform"
    assert isinstance(plan, HardwarePlan)
    assert plan.num_envs >= 1
    assert plan.detected_cores >= 1
    assert "core" in plan.describe()
