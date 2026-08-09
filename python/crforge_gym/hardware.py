# Modified by NextoCR contributors; see NOTICE for attribution.
"""Size a training run to the machine it runs on, without taking the machine down.

Each parallel environment costs a JVM bridge process, so oversubscribing does
not merely slow training: it can exhaust RAM and make the desktop unusable
while a multi-day run is in flight. The plan therefore leaves cores and memory
free for the operating system, the dashboard and the person using the PC.

Throughput has two halves. Environments are separate processes and scale with
cores; the PPO update is one process using torch's thread pool. A measured run
on a 12-core desktop spent roughly half its wall clock in the update, so both
halves get a share rather than handing every core to environments.

Detection is dependency-free: ``os.cpu_count`` everywhere, plus the native
memory call for the platform.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
import sys

__all__ = [
    "HardwarePlan",
    "detect_available_memory_bytes",
    "detect_total_memory_bytes",
    "plan_training_resources",
]

# A gym-bridge JVM settles around 350-450 MB; round up so a plan that fits on
# paper still fits once the JVM has warmed up.
BRIDGE_MEMORY_BYTES = 600 * 1024 * 1024
# The trainer holds the policy, the rollout buffer and the league model cache.
TRAINER_MEMORY_BYTES = 2 * 1024 * 1024 * 1024
# Never hand the whole machine over: the desktop, the dashboard and the
# evaluation bridge all need room.
RESERVED_CORES = 2
RESERVED_MEMORY_BYTES = 2 * 1024 * 1024 * 1024
MAX_ENVS = 16


@dataclass(frozen=True)
class HardwarePlan:
    """How many environments and torch threads this machine should use."""

    num_envs: int
    torch_threads: int
    detected_cores: int
    available_memory_bytes: int
    limited_by: str

    def to_manifest(self) -> dict[str, object]:
        return {
            "num_envs": self.num_envs,
            "torch_threads": self.torch_threads,
            "detected_cores": self.detected_cores,
            "available_memory_gib": round(self.available_memory_bytes / (1024 ** 3), 2),
            "limited_by": self.limited_by,
        }

    def describe(self) -> str:
        gib = self.available_memory_bytes / (1024 ** 3)
        return (
            f"{self.detected_cores} cores, {gib:.1f} GiB free -> "
            f"{self.num_envs} env(s), {self.torch_threads} torch thread(s) "
            f"(limited by {self.limited_by})"
        )


def detect_total_memory_bytes() -> int:
    """Total physical RAM, or 0 when the platform cannot be queried."""
    if sys.platform == "win32":
        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        return 0
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return 0


def detect_available_memory_bytes() -> int:
    """Free RAM right now, falling back to total when unavailable.

    Free matters more than total: a run started next to a browser and a game
    client has far less to work with than the spec sheet suggests.
    """
    if sys.platform == "win32":
        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
        return detect_total_memory_bytes()
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return detect_total_memory_bytes()


def plan_training_resources(
    *,
    cpu_count: int | None = None,
    available_memory_bytes: int | None = None,
    requested_envs: int | None = None,
    reserved_cores: int = RESERVED_CORES,
    reserved_memory_bytes: int = RESERVED_MEMORY_BYTES,
    max_envs: int = MAX_ENVS,
) -> HardwarePlan:
    """Choose environment and thread counts for this machine.

    ``requested_envs`` still passes through the memory ceiling, so an explicit
    request cannot push the machine into swap.
    """
    cores = int(cpu_count or os.cpu_count() or 1)
    memory = int(
        available_memory_bytes
        if available_memory_bytes is not None
        else detect_available_memory_bytes()
    )

    usable_cores = max(1, cores - max(0, reserved_cores))
    # Environments and the update phase share the machine; a single core cannot
    # be split, so the smaller side keeps at least one.
    cpu_envs = max(1, (usable_cores * 2) // 3)

    spare_memory = memory - reserved_memory_bytes - TRAINER_MEMORY_BYTES
    if memory <= 0:
        memory_envs = cpu_envs  # Unknown memory: trust the core count alone.
        memory_known = False
    else:
        memory_envs = max(1, int(spare_memory // BRIDGE_MEMORY_BYTES))
        memory_known = True

    ceiling = max(1, int(max_envs))
    num_envs = min(cpu_envs, memory_envs, ceiling)
    if requested_envs is not None:
        num_envs = max(1, min(int(requested_envs), memory_envs, ceiling))
        limited_by = "request"
        if num_envs < int(requested_envs):
            limited_by = "memory" if memory_known else "limit"
    elif num_envs == ceiling:
        limited_by = "limit"
    elif memory_known and memory_envs <= cpu_envs:
        limited_by = "memory"
    else:
        limited_by = "cpu"

    # Whatever the environments do not need drives the PPO update, which is
    # about half the wall clock of a self-play run.
    torch_threads = max(1, min(usable_cores, usable_cores - num_envs + 1))

    return HardwarePlan(
        num_envs=num_envs,
        torch_threads=torch_threads,
        detected_cores=cores,
        available_memory_bytes=max(0, memory),
        limited_by=limited_by,
    )
