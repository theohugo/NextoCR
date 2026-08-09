# Local throughput baseline

This document records a small, reproducible smoke benchmark of the current JPype execution paths.
It is a development baseline, not a general performance claim.

## Run metadata

| Item | Value |
|---|---|
| Date | 2026-08-09 (Europe/Paris) |
| Base commit | `c41c16a85e6474e8a59b088ace8358a76b9a9d1a` (`build(deps): bump gdx from 1.14.0 to 1.14.1 (#23)`) |
| Working tree | Dirty; includes the in-progress NextoCR phase-one fixes, including world-local IDs. This is not a pristine upstream-commit measurement. |
| Branch | `feat/nextocr-foundation` |
| CPU | Intel Core i5-11400F at 2.60 GHz, 6 physical cores / 12 logical processors |
| Memory | 34,187,804,672 bytes installed (approximately 31.8 GiB) |
| OS | Windows 11 Pro 64-bit, version `10.0.26200`, build `26200` |
| JVM | Microsoft OpenJDK 17.0.19+10-LTS, 64-bit HotSpot |
| Gradle | 9.4.1 |
| Python | CPython 3.12.10 |
| Python packages | `nextocr-gym 0.1.0`, `JPype1 1.7.1`, `NumPy 2.5.2`, `Gymnasium 1.3.0`, `stable-baselines3 2.9.0`, `torch 2.13.0+cpu` |

The Python dependencies were installed only in the repository-local, Git-ignored `.venv`.

## Commands

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".\python[jpype]" stable-baselines3
.\gradlew.bat :gym-bridge:installDist

.\.venv\Scripts\python.exe python\examples\bench_step.py --jpype-only --steps 5000
.\.venv\Scripts\python.exe python\examples\bench_step.py --jpype-only --threaded-envs 2 --steps 5000
.\.venv\Scripts\python.exe python\examples\bench_step.py --jpype-only --threaded-envs 4 --steps 5000
.\.venv\Scripts\python.exe python\examples\bench_step.py --jpype-only --threaded-envs 8 --steps 5000
```

Each command started a fresh Python process and embedded JVM. The existing benchmark performs 200
single-environment warm-up steps. Its threaded section additionally performs 50 vector calls as a
warm-up. `--steps 5000` means 5,000 measured steps **per environment**. The environment default is
six engine ticks per Gymnasium step.

## Results

The script prints integer-rounded rates. Engine ticks per second below are derived as total Gym
steps per second multiplied by the default six ticks per step.

| Mode | Environments | Measured Gym steps | Single baseline in same process (steps/s) | Total steps/s | Per-env steps/s | Approx. engine ticks/s | Script-reported speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| JPype single | 1 | 5,000 | 12,641 | 12,641 | 12,641 | 75,846 | 1.0x |
| Threaded JPype | 2 | 10,000 | 12,647 | 7,177 | 3,589 | 43,062 | 0.6x |
| Threaded JPype | 4 | 20,000 | 11,601 | 6,633 | 1,658 | 39,798 | 0.6x |
| Threaded JPype | 8 | 40,000 | 8,065 | 6,584 | 823 | 39,504 | 0.8x |

The threaded commands run the single-JPype section first, which is why each row has its own
same-process baseline. Including these repeated single measurements, all commands completed 90,000
timed Gym steps, or approximately 540,000 engine ticks, plus warm-up work.

## Interpretation and stability

All four commands exited successfully. The 2-, 4-, and 8-environment runs produced no Java
exception, JVM crash, or observable ID collision after the world-local ID changes.

On this run, single JPype reached 12,641 steps/s, while threaded total throughput plateaued between
6,584 and 7,177 steps/s. Threading therefore did not improve total throughput on this machine under
the benchmark's current workload. A likely explanation is the cost of one executor task per world,
Python buffer copies, and vector-wrapper/action-mask work relative to the small Java step, but this
is an inference and needs profiling before changing the implementation. A batched Java API or
persistent worker loop should be evaluated against this baseline.

## Methodological limits

- There was one timed sample per configuration, so no median, dispersion, or confidence interval is
  available. The single measurements varied from 8,065 to 12,647 steps/s across fresh processes.
- The benchmark uses only a short warm-up. JVM JIT compilation may still be changing during the
  measured interval.
- The blue action is a no-op, but `CRForgeEnv` keeps its default random red opponent and the script
  does not set a seed. Entity counts and reset timing can therefore vary between runs.
- The threaded path adds `EpisodeStatsWrapper` and `ActionMaskedWrapper`; the single path does not.
  The comparison includes their Python-side overhead and is not a pure engine-scaling measurement.
- Other development agents and build processes were active on the machine. CPU affinity, Windows
  power mode, background load, temperature, and clock frequency were not controlled.
- The working tree was dirty and changed from the recorded base commit. Re-run after the phase-one
  work is committed and record the resulting commit hash for a release-quality result.
- A "step" here is one Gymnasium call covering six engine ticks. It is not a match, policy sample,
  or training update, and the result excludes neural-network inference and learning.
- Successful execution is only a smoke signal for concurrency safety. Determinism and ID isolation
  remain covered by dedicated Java tests rather than proven by throughput measurements.
