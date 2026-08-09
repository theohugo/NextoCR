# NextoCR Gymnasium environment

> **Modified-file notice:** this document derives from the Apache-2.0 crforge Python README and was
> updated for NextoCR. The compatibility implementation still lives in `crforge_gym`.

Python RL interface for the offline NextoCR research simulator.

```
Python (Gymnasium)  --ZMQ or JPype-->  Java (crforge engine)
   env.step()       binary/JSON          GameEngine.tick()
   env.reset()      process/in-process   20 TPS deterministic
```

## Prerequisites

- Python 3.10+
- Java 17 (for the bridge server)
- The NextoCR project built: `./gradlew build`

## Installation

```bash
# Basic install (env + bridge client)
python -m pip install -e python/

# With training dependencies (SB3 + TensorBoard)
python -m pip install -e "python[train]"

# For contributors and integration tests (training stack + pytest)
python -m pip install -e "python[dev]"
```

## Quick Start

### 1. Start the Java bridge server

```bash
./gradlew :gym-bridge:run
```

The server listens on `tcp://localhost:9876` by default.
`--allow-remote` exposes the unauthenticated bridge protocol and is discouraged outside a trusted
network.

### 2. Run random episodes (smoke test)

```bash
python python/examples/run_episodes.py
```

This runs 3 episodes with random actions and prints per-episode stats.

### 3. Train a PPO agent

```bash
python python/examples/train_ppo.py --timesteps 50000
```

The current self-play prototype requires one ZMQ environment in JSON mode so the opponent receives
its real hand. Binary/JPype self-play is rejected until the observation schema encodes both hands:

```bash
python python/examples/train_ppo.py --opponent self_play --num-envs 1 --timesteps 100000
```

Monitor training:

```bash
tensorboard --logdir logs/ppo_crforge
```

### 4. Evaluate a trained model

```bash
python python/examples/evaluate.py --model models/ppo_crforge --episodes 50
```

### 5. Watch a trained model play (AI Visualizer)

The desktop visualizer can run in AI mode, where Python controls the game via the same ZMQ protocol
as the headless bridge. This lets you watch the trained model deploy cards with full rendering.

```bash
# Terminal 1: Start the desktop visualizer in AI mode (with Java 17 configured)
./gradlew :desktop:run --args="--ai-port 9876"

# Terminal 2: Run the trained model (same command as headless evaluation)
python python/examples/evaluate.py --model models/ppo_crforge
```

The visualizer renders each step in real-time: when the model sends a `step` message, the engine
ticks are spread across render frames so you can see entities move, projectiles fly, and cards
deploy visually.

**Controls during AI playback:**

| Key   | Action                         |
|-------|--------------------------------|
| SPACE | Pause/resume (Python blocks)   |
| +/-   | Speed up/slow down (0.25x-8x)  |
| P     | Toggle path visualization      |
| O     | Toggle attack range circles    |
| D     | Toggle floating damage numbers |
| A     | Toggle AOE damage indicators   |
| H     | Toggle HP numbers              |

Any script that connects to the bridge server works -- `run_episodes.py`, `evaluate.py`, or your own
custom loop. The Python side requires no changes; it cannot tell whether the server is headless or
rendering.

## Environment Details

### Action Space

`MultiDiscrete([2, 4, 10])`

| Index | Meaning     | Values                                      |
|-------|-------------|---------------------------------------------|
| 0     | action_type | 0=no-op, 1=play card                        |
| 1     | hand_index  | 0-3 (which card slot)                       |
| 2     | zone        | 0-6 own-half zones, 7-9 enemy spell zones  |

The ten-zone abstraction is a training baseline. Exact 18 x 32 placement remains roadmap work for a
versioned action schema.

The current factorized action mask can still admit an incompatible card/zone combination when
another affordable card makes that zone legal. Engine-generated exact masks are roadmap work.

### Observation Space

Binary mode is the default and returns the V2 flat `float32` vector with shape `(1153,)`. JSON mode
returns the structured dictionary below. Spatial coordinates and stable identities are normalized
to `[0, 1]` for neural-network inputs.

| Key                            | Shape    | Description |
|--------------------------------|----------|-------------|
| frame                          | (1,)     | Current simulation frame |
| game_time                      | (1,)     | Seconds; space allows 0-600, Standard1v1 currently <=300 |
| is_overtime                    | (1,)     | 1.0 if overtime |
| observation_schema_version     | (1,)     | Observation contract version (`2`) |
| elixir                         | (2,)     | [blue, red] elixir (0-10) |
| crowns                         | (2,)     | [blue, red] crown count |
| hand_costs                     | (4,)     | Card costs / 10 |
| hand_types                     | (4,)     | 0=troop, 1=spell, 2=building |
| hand_card_ids                  | (4,)     | Legacy registry indices; not stable across catalog reorderings |
| hand_card_identities           | (4,)     | Stable card identities divided by `2^24` |
| hand_allows_enemy_placement    | (4,)     | Exact per-slot enemy-half permission (0/1) |
| next_card_cost                 | (1,)     | Next card cost / 10 |
| next_card_type                 | (1,)     | Next card type |
| next_card_id                   | (1,)     | Legacy next-card registry index |
| next_card_identity             | (1,)     | Stable next-card identity divided by `2^24` |
| towers                         | (6, 4)   | [hp_frac, x_norm, y_norm, alive] |
| entities                       | (64, 16) | Team/generic type, position, HP/shield, combat state, effects, lifetime |
| entity_identities              | (64,)    | Stable runtime archetype identities divided by `2^24`; zero-padded |
| num_entities                   | (1,)     | Encoded entity count, capped at 64 |
| lane_summary                   | (8,)     | Aggregated friendly/enemy lane pressure |

#### Observation schema V2

V2 is append-only. Binary offsets `0..1078` are the byte-for-byte V1 prefix; the extension is:

| Offset | Count | Field |
|--------|-------|-------|
| 1079   | 1     | Schema version (`2`) |
| 1080   | 4     | Stable hand identities / `2^24` |
| 1084   | 1     | Stable next-card identity / `2^24` |
| 1085   | 64    | Stable entity identities / `2^24` |
| 1149   | 4     | `hand_allows_enemy_placement` flags |

The raw JSON bridge retains the exact integer `identityId` on each hand card and entity. The Gym
parser and binary encoder divide it by `16_777_216`, avoiding very large ordinal inputs to an MLP.
IDs use the deterministic cross-language contract
`1 + (CRC32(UTF-8(namespace + ":" + canonical_key)) & 0xFFFFFF)`, with namespace `card` and the
card's registry ID, or namespace `entity` and the runtime entity name. Zero means padding, unknown,
or a detected catalog collision. The current catalog is collision-tested during the Java test
suite.

`allowsEnemyPlacement` follows the engine rule
`(type == SPELL && !spellAsDeploy) || canDeployOnEnemySide`. Thus Fireball may target the enemy
half, Barbarian Barrel may not, and Miner/Goblin Drill may cross the normal troop boundary.

A legacy 1079-value binary response is padded to the V2 Gym shape without changing its prefix; its
schema value is `1` and all appended values are zero. Stable-Baselines3 checkpoints persist their
observation space: a V2 flattened/binary checkpoint therefore records `Box(..., (1153,), float32)`,
while an unflattened JSON policy records the V2 Dict keys above. A V1 `(1079,)` checkpoint is not
shape-compatible and must be migrated or retrained. When exporting outside SB3, include
`observation_schema_version: 2` in the checkpoint manifest explicitly.

### Reward Structure

| Source         | Magnitude   | Purpose                                |
|----------------|-------------|----------------------------------------|
| Tower HP delta | +/-0.005/HP | Symmetric tower-damage shaping         |
| Crown delta    | +/-10.0     | Symmetric tower-destruction milestone  |
| Win / loss     | +30 / -30   | Terminal outcome                       |
| Draw           | -10 each    | Discourage passive draws                |
| Elixir waste   | -0.02/step  | Penalize remaining capped at 10        |
| Time penalty   | -0.001/step | Discourage passive play                |
| Invalid action | -0.01/step  | Python-side penalty for rejected plays |

Terminal rewards use the engine's canonical `outcome` (`ONGOING`, `BLUE_WIN`, `RED_WIN`, or
`DRAW`) and are emitted exactly once. JSON step results carry that enum as a string. Binary step
results preserve the 12-byte header and observation offset, then append a little-endian `int32`
outcome code after the observation (`0..3` in the order above). Gym exposes the blue-player view in
`info["game_outcome"]` as `ongoing`, `win`, `loss`, or `draw`; terminal results are never inferred
from reward magnitude or crown count.

### Configuration

```python
from nextocr_gym import NextoCREnv

env = NextoCREnv(
    endpoint="tcp://localhost:9876",  # Bridge server address
    blue_deck=[
        "knight", "archer", "fireball", "arrows",
        "giant", "musketeer", "minions", "valkyrie",
    ],
    red_deck=[
        "knight", "archer", "fireball", "arrows",
        "giant", "musketeer", "minions", "valkyrie",
    ],
    level=11,                        # Card/tower level (1-15)
    ticks_per_step=15,               # ~240 regulation steps; ~400 with overtime
    opponent="rule_based",           # "noop", "random", "rule_based", or callable
    invalid_action_penalty=-0.01,    # Penalty for failed actions
)
```

### Deterministic Seeding

Pass `seed` to `env.reset()` for reproducible episodes:

```python
obs, info = env.reset(seed=42)  # Same seed -> same deck shuffle
```

The explicit seed is used for that episode without consuming the episode-sequence RNG. Later
`env.reset()` calls derive reproducible child seeds from Gymnasium's `np_random`; the effective Java
seed is returned as `info["episode_seed"]`. Two environments first reset with the same seed therefore
produce the same complete sequence of episode seeds, including vector-environment auto-resets.

### FlattenedObsWrapper

For SB3's MlpPolicy, wrap the env to flatten Dict obs into a single vector:

```python
from nextocr_gym import FlattenedObsWrapper, NextoCREnv
env = FlattenedObsWrapper(NextoCREnv(binary_obs=False, ...))
```

## Integration Tests

Install the development extra, then run these tests with the Java server in another terminal.

macOS/Linux:

```bash
./gradlew :gym-bridge:run

CRFORGE_INTEGRATION=1 python -m pytest python/tests/ -v
```

Windows PowerShell:

```powershell
.\gradlew.bat :gym-bridge:run

$env:CRFORGE_INTEGRATION = "1"
python -m pytest python/tests/ -v
```

## Architecture

- **Java bridge** (`gym-bridge/`): ZMQ PAIR server, wraps GameEngine in a step/reset API
- **Python bridge** (`crforge_gym/bridge.py`): inherited ZMQ/binary compatibility implementation
- **NextoCR API** (`nextocr_gym/`): public named entry point over the compatible engine API
- **Gymnasium env** (`crforge_gym/env.py`): wraps bridge clients in the standard Gym interface
- **Wrappers** (`crforge_gym/wrappers.py`): FlattenedObsWrapper for SB3 compatibility
