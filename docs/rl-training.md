# Reproducible RL training

NextoCR's PPO runner is an experiment harness, not only an example loop. It records the inputs
needed to understand a run, uses exact action masks during training and evaluation, checkpoints at
absolute timestep intervals, and can continue the policy and optimizer state after interruption.

The current curriculum is deliberately marked `mirror`: both sides use the same versioned
`mortar_self_play_v1` simulator deck. An opponent-catalog snapshot is nullable in the manifest;
no Top-1000 or live-game deck data is invented by this runner.

## Install

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e "python[dev,jpype]"
$env:PYTHONHASHSEED = "42"
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e "python[dev,jpype]"
export PYTHONHASHSEED=42
```

The script seeds Python, NumPy, Torch, policy spaces, each simulator environment, and every episode.
Setting `PYTHONHASHSEED` before starting Python also makes the interpreter's hash seed effective from
process startup; changing it inside an already-running interpreter cannot retroactively re-seed
hashing. The manifest records whether it was effective at startup.

## Action and deck contracts

- `action_schema=exact_discrete_v1`: `Discrete(41)`, with no-op plus every valid hand/zone pair.
- `observation_preprocessing=nextocr_static_v1`: a fixed, non-running transform scales frame,
  time, elixir, crowns, types, entity counts and categorical entity features to bounded inputs.
  Legacy insertion-ordered `cardIndex` fields are zeroed; V2 stable card/entity identities remain.
- Evaluation uses `sb3_contrib.common.maskable.evaluation.evaluate_policy`, so the same exact masks
  are applied to deterministic evaluation actions.
- Legacy `MultiDiscrete([2, 4, 10])` checkpoints are rejected clearly instead of being attached to
  an incompatible environment.
- Checkpoints without the matching preprocessing tag are also rejected even though their tensor
  dimensions happen to match. Blue observations and mirrored red-policy observations use the same
  transform; no online `VecNormalize` state is involved.
- Both players receive the eight simulator IDs from `mortar_self_play_v1`.

The requested live-game idea includes evolved Skeleton Barrel, Hero Goblins, and evolved Mortar.
Those variants are not yet fidelity-certified: training currently uses the documented base-form
approximations from the deck profile. `config.json` stores both the simulator IDs and these caveats.

## Long baseline run

For a fast fixed-opponent baseline, JPype supports several environments inside one JVM:

```powershell
.\.venv\Scripts\python.exe python\examples\train_ppo.py `
  --jpype --num-envs 8 --opponent rule_based `
  --run-dir runs\ppo-rule-seed42 --seed 42 `
  --timesteps 10000000 --checkpoint-freq 100000 `
  --eval-freq 250000 --eval-episodes 20 --device cpu
```

The current league self-play path needs JSON observations, one ZMQ training environment, and a
separate evaluation server. Build the native launchers once (the Python helper refreshes this task
incrementally as well):

```powershell
.\gradlew.bat :gym-bridge:installDist
```

Start the training bridge in one PowerShell terminal:

```powershell
.\gym-bridge\build\install\gym-bridge\bin\gym-bridge.bat 9876
```

Start the evaluation bridge in a second terminal:

```powershell
.\gym-bridge\build\install\gym-bridge\bin\gym-bridge.bat 9877
```

Then start a mirror-league run in a third terminal:

```powershell
.\.venv\Scripts\python.exe python\examples\train_ppo.py `
  --endpoint tcp://localhost:9876 --eval-endpoint tcp://localhost:9877 `
  --opponent self_play --eval-opponent rule_based `
  --run-dir runs\mirror-league-seed42 --seed 42 `
  --timesteps 10000000 --checkpoint-freq 100000 `
  --self-play-interval 50000 --eval-freq 250000 --eval-episodes 20
```

The corresponding native launcher on macOS/Linux is
`./gym-bridge/build/install/gym-bridge/bin/gym-bridge`.

Keeping `--eval-opponent rule_based` makes periodic scores comparable while the training league
changes. Setting it to `self_play` is supported for exploratory measurements, but those scores then
depend on the league state at that timestep.

## League behavior

At the start of self-play, the runner saves an immutable initial policy. It then maintains three
opponent groups:

- the initial policy, which is never replaced;
- a bounded queue of recent policy snapshots;
- a bounded reservoir sample of older historical snapshots.

One checkpoint is sampled with a seeded RNG at each episode boundary. The default pool contains at
most 1 initial + 4 recent + 8 historical policies, and the in-memory model cache contains at most
two loaded opponents. Snapshot files are never modified in place; files pruned from the bounded pool
are removed. Every selected opponent ID, category, source step, episode, and phase is appended to
the run metrics.

Use `--league-max-recent`, `--league-max-historical`, `--league-model-cache`, and the three
`--league-*-weight` flags to change the population. These values are reproducibility-critical and
cannot change when continuing the same run directory.

## Run artifacts

Each run directory is self-contained:

```text
run/
  config.json                 immutable original configuration and deck snapshot
  manifest.json               status, runtime versions, Git commit, attempts, latest model
  metrics.csv                 append-only episode/evaluation/checkpoint/league events
  metrics.json                compact aggregate metrics and evaluation history
  attempts/attempt_001.json   exact command configuration for each continuation
  checkpoints/latest.zip     copy of the latest periodic/failure/interruption checkpoint
  checkpoints/step_*.zip     immutable numbered checkpoints
  league/league.json          bounded population state and deterministic sample counter
  league/policy_*.zip         immutable active opponent snapshots
  tensorboard/                SB3 logs
  final_model.zip             most recent completed policy
```

`Ctrl+C`, `SIGINT`, and `SIGTERM` request a stop at the next callback boundary. The runner then saves
an `interrupted` checkpoint and marks the attempt accordingly. Unexpected failures also trigger a
best-effort `failed` checkpoint without hiding the original exception.

## Resume semantics

`--timesteps` always means additional timesteps. A resume loads the model and optimizer, retains the
existing absolute timestep counter, and calls SB3 with `reset_num_timesteps=False`:

```powershell
.\.venv\Scripts\python.exe python\examples\train_ppo.py `
  --jpype --num-envs 8 --opponent rule_based `
  --resume runs\ppo-rule-seed42 --run-dir runs\ppo-rule-seed42 `
  --seed 42 --timesteps 2000000 --checkpoint-freq 100000 `
  --eval-freq 250000 --eval-episodes 20 --device cpu
```

Repeat the original environment and PPO hyperparameters. The runner refuses changes to critical
fields in the same run directory. PPO completes whole rollout buffers, so actual progress may exceed
the requested additional count by up to one rollout. Resume starts a new simulator episode and does
not restore a partially collected rollout or mid-match engine state. To avoid replaying episode seeds
such as 42 and 43, the new attempt derives each environment's next seed namespace from the durable
checkpoint timestep and records that offset in its attempt config. Resume is therefore a durable,
non-replaying continuation, not a bit-for-bit continuation from the exact interrupted instruction.

## Quick CPU smoke

This command validates build discovery, exact masks, checkpointing, manifests, and a short PPO
update without waiting for an episode:

```powershell
.\.venv\Scripts\python.exe python\examples\train_ppo.py `
  --jpype --opponent rule_based --run-dir .local\ppo-smoke `
  --timesteps 32 --n-steps 32 --batch-size 16 --n-epochs 1 --net-arch 32 `
  --checkpoint-freq 16 --eval-episodes 0 --device cpu
```

Run `python python/examples/train_ppo.py --help` for all options.
