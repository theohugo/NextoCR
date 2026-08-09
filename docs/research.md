# Foundation research

Last reviewed: 2026-08-09.

NextoCR needs the same separation of concerns that made the Rocket League training ecosystem
effective: a fast simulator, a standard environment API, scalable rollouts, and a self-play/evaluation
system. A perception bot attached to a real-time client cannot generate enough experience for that
kind of work.

## Compared projects

| Project | What is reusable | Main limitation for NextoCR | License seen during review |
| --- | --- | --- | --- |
| [voonhous/crforge](https://github.com/voonhous/crforge) | Deterministic Java engine, JSON cards, combat systems, tests, Gymnasium via ZMQ/JPype, debug viewer, vector prototype | Fidelity gaps, incomplete Tower Troops/champion/evolution mechanics, concurrency and determinism work still needed | Apache-2.0 |
| [nguiaSoren/clash-royale-suite](https://github.com/nguiaSoren/clash-royale-suite) (`cr-rudy-sim`) | Deterministic Rust simulator, integer-oriented mechanics, PyO3 bindings, useful differential-test candidate | Training layer is still a TODO and the project documents remaining simulator mismatches; data provenance also needs auditing | MIT |
| [wty-yy/KataCR](https://github.com/wty-yy/KataCR) | Strong visual perception pipeline, datasets, offline imitation/RL research, real-screen state extraction | No fast headless self-play simulator; training remains tied to recorded or real-time visual input | MIT |
| [MSU-AI/clash-royale-gym](https://github.com/MSU-AI/clash-royale-gym) | Clear Gymnasium framing and a small API example | README describes mock API calls in v0.0.1; it is an environment shell rather than a battle engine | MIT |
| [Jason-XII/clash-royale-simulator](https://github.com/Jason-XII/clash-royale-simulator) | Recent playable engine, 47 cards, Gym/SB3 integration, useful design and performance comparison | Partial card set/fidelity and no declared repository license, so code/data cannot safely be copied | No license detected |
| [Aryan Singh self-play PPO project](https://aryansingh.org/portfolio/) | Public description is very close to the intended PPO+self-play direction | Source, simulator, experiments, and license are not public/verifiable | Private/unavailable |

## Decision

NextoCR uses `crforge` as its starting point because it is the only reviewed project that combines:

1. an explicit permissive license;
2. a genuinely headless tick-based engine;
3. broad data-driven card coverage;
4. detailed mechanic and integration tests;
5. Python Gymnasium access without requiring the renderer;
6. both process-isolated and in-process rollout paths.

The GitHub fork relationship and upstream commit history are intentionally preserved. Java package
names remain `org.crforge` and the original Python import remains `crforge_gym` so that upstream
changes can still be merged with manageable conflicts.

## What is borrowed versus referenced

- **Inherited under Apache-2.0:** `crforge` source, tests, documentation, and its existing community
  data snapshot, subject to [LICENSE](../LICENSE) and [NOTICE](../NOTICE).
- **Referenced, not copied:** KataCR architecture/paper, MSU Gym framing, the Jason-XII simulator,
  Nexto/Necto/RLGym training patterns, and public project descriptions.
- **Not used:** code/data from repositories with no compatible explicit license, official client
  binaries/assets, private server implementations, or unpublished project code.

## Immediate technical audit of the selected base

The inherited test suite and catalog are substantial, but a passing build is not the same as a safe
parallel RL engine. The initial NextoCR audit identified three phase-0 blockers:

1. Entity/projectile IDs used process-global mutable counters. Resetting one world could affect a
   second world in threaded JPype rollouts.
2. Targeting had an internal fixed random generator that was not part of the episode seed/reset
   lifecycle.
3. The bridge inferred whether an action failed from an elixir comparison before the queued action
   was committed on a simulation tick.

These issues are now tracked as explicit baseline work in [ROADMAP.md](../ROADMAP.md), with regression
tests required before training claims.

## Why the other projects still matter

### KataCR

KataCR and its paper,
[Playing Non-Embedded Card-Based Games with Reinforcement Learning](https://arxiv.org/abs/2504.04783),
are useful for offline visual validation. They may eventually help compare simulator trajectories
with recorded gameplay, but NextoCR does not include live client automation.

### Clash Royale Gym

Its value is conceptual: expose observations, actions, rewards, and termination through a familiar
RL API. NextoCR keeps that standard but gets the state from an actual simulation rather than mock
requests or rendered pixels.

### cr-rudy-sim

Its Rust/PyO3 architecture is attractive for a future differential harness and performance
comparison. It was not selected as the primary base because its published training module remains
unfinished and its own documentation still records mechanic mismatches. NextoCR may compare focused
interaction traces with it, but it must not treat either simulator as ground truth.

### Jason-XII simulator

The project reports 47 implemented cards, an 83-microsecond tick, and an SB3-compatible PPO example.
Those are useful external benchmarks. Because the repository did not declare a license during this
review, NextoCR does not incorporate its source or data.

## Legal and research boundary

An open-source software license only covers rights granted by that software's contributors. It does
not grant rights over a third party's game, trademarks, assets, or service.

Supercell's current
[Fan Content Policy](https://supercell.com/en/fan-content-policy/) restricts new products based on
Supercell assets and content associated with bots/automation. Its
[Terms of Service](https://supercell.com/en/terms-of-service/) also prohibit software designed to
modify or interfere with the service or game experience. NextoCR therefore:

- stays offline and research-oriented;
- does not control or connect to the official client;
- does not include official art, audio, binaries, or credentials;
- uses an explicit unofficial-project disclaimer;
- treats any future sim-to-real bridge as a separate legal/policy review, not a promised feature.

Written permission from Supercell would be the strongest basis for expanding beyond this boundary.
This document records project decisions and is not legal advice.

## Primary references

- [crforge repository](https://github.com/voonhous/crforge)
- [crforge architecture](https://github.com/voonhous/crforge/blob/main/docs/architecture.md)
- [KataCR repository](https://github.com/wty-yy/KataCR)
- [KataCR paper](https://arxiv.org/abs/2504.04783)
- [clash-royale-suite / cr-rudy-sim](https://github.com/nguiaSoren/clash-royale-suite)
- [MSU-AI Clash Royale Gym](https://github.com/MSU-AI/clash-royale-gym)
- [Jason-XII Clash Royale Simulator](https://github.com/Jason-XII/clash-royale-simulator)
- [Official Clash Royale developer API](https://developer.clashroyale.com/)
- [Supercell Fan Content Policy](https://supercell.com/en/fan-content-policy/)
- [Supercell Terms of Service](https://supercell.com/en/terms-of-service/)
