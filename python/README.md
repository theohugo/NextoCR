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

Binary mode is the default and returns a flat float32 vector with shape `(1079,)`. JSON mode returns
the structured dictionary below. Spatial coordinates are normalized to `[0, 1]`.

| Key            | Shape   | Description                          |
|----------------|---------|--------------------------------------|
| frame          | (1,)    | Current simulation frame             |
| game_time      | (1,)    | Seconds; space allows 0-600, Standard1v1 currently <=300 |
| is_overtime    | (1,)    | 1.0 if overtime                      |
| elixir         | (2,)    | [blue, red] elixir (0-10)            |
| crowns         | (2,)    | [blue, red] crown count              |
| hand_costs     | (4,)    | Card costs / 10 (normalized)         |
| hand_types     | (4,)    | 0=troop, 1=spell, 2=building         |
| hand_card_ids  | (4,)    | Card-vocabulary indices              |
| next_card_cost | (1,)    | Next card cost / 10                  |
| next_card_type | (1,)    | Next card type                       |
| next_card_id   | (1,)    | Next card-vocabulary index           |
| towers         | (6, 4)  | [hp_frac, x_norm, y_norm, alive]     |
| entities       | (64, 16) | Team/generic type, position, HP/shield, combat state, effects, lifetime |
| num_entities   | (1,)    | Encoded entity count, capped at 64   |
| lane_summary   | (8,)    | Aggregated friendly/enemy lane pressure |

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
