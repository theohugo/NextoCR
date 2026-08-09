# Local training manager

The NextoCR manager is a localhost-only control plane for the offline simulator and RL trainer. On
Windows, start it by double-clicking `NextoCR.bat`. The browser UI is served from
`http://127.0.0.1:8765/`; no cloud service or Node.js runtime is needed to use the built dashboard.

## Daily workflow

1. Start `NextoCR.bat` and open the selected run.
2. Press **Resume** (or **Start** for the first run).
3. Inspect progress, fixed-opponent evaluation, league composition, logs, and checkpoint history.
4. Optionally press **Checkpoint** to save the policy and optimizer without stopping.
5. Before shutting down, press **Pause + checkpoint** and wait for the paused confirmation.
6. The next day, launch the same file and press **Resume**. The trainer loads the latest durable
   checkpoint and trains only the remaining steps toward the run's total target.

`--timesteps` in the underlying trainer means additional steps. The manager deliberately computes
`targetTotalTimesteps - checkpointTimesteps` on every continuation, so repeated daily resumes do
not accidentally add another 10 million steps.

## Runs, checkpoints, and versions

Every run remains self-contained under `runs/` with immutable configuration, attempts, metrics,
league state, TensorBoard logs, and SB3 checkpoints. A manual or pause checkpoint is acknowledged
only after the trainer writes it atomically at a safe callback boundary.

Creating a version makes a new run with its own identifier and records the parent run and source
checkpoint. It never edits or replaces the parent checkpoint. Only one training run is active at a
time to avoid bridge-port and CPU contention.

## Watching a match

**See a match** starts a separate loopback bridge, loads the selected immutable checkpoint, and
runs one seeded evaluation episode. The generated replay stores abstract tower and entity
positions, hit points, actions, outcome, reward, seed, opponent, checkpoint step, and the lag
between that checkpoint and the live run. The web arena interpolates those frames; it does not use
or redistribute game art.

Replay generation never connects to the training or periodic-evaluation bridge and cleans up its
temporary Java process afterward.

## Failure and recovery behavior

- Closing only the browser tab leaves the manager and training active.
- `Ctrl+C` on the manager requests a cooperative pause, waits for a checkpoint, then cleans up its
  trainer and Java process trees.
- If the manager itself restarts while a managed trainer is still alive, persisted PIDs and the
  file-backed command channel allow it to recover control after validating command lines and run
  paths.
- If Windows terminates processes before a final checkpoint can complete, the latest previously
  acknowledged checkpoint remains valid and can be resumed. A partially collected PPO rollout is
  not restored.

Runtime state and API tokens stay under `.local/`; runs and replay artifacts stay under `runs/`.
Both directories are ignored by Git.

## Local security boundary

The HTTP server binds exactly to `127.0.0.1`, validates the `Host` and `Origin`, disables CORS, and
requires a random per-process token for every mutation. Run IDs, checkpoint paths, static paths,
and replay IDs are constrained to their workspace roots. The API does not accept arbitrary shell
commands.

This manager operates only the offline NextoCR simulator. It does not click, automate, modify, or
connect to the official Clash Royale client or service.
