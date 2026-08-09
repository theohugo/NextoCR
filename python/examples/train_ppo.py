#!/usr/bin/env python3
# Modified by NextoCR contributors; see NOTICE for attribution.
"""
Train a PPO agent on CRForge using Stable Baselines 3.

Prerequisites:
  1. Start the Java bridge server: ./gradlew :gym-bridge:run
     (or use --num-envs N to auto-launch N servers)
  2. Install dependencies: pip install -e "python[train]"

Usage:
  python python/examples/train_ppo.py
  python python/examples/train_ppo.py --timesteps 1000000 --num-envs 4
  python python/examples/train_ppo.py --resume models/ppo_crforge --timesteps 50000
  python python/examples/train_ppo.py --opponent self_play --timesteps 100000
"""

import argparse
import atexit
import json
import os
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# Server management (auto-launch for multi-env)
# ---------------------------------------------------------------------------

def _is_windows(os_name: str | None = None) -> bool:
    """Return whether *os_name* (or the current platform) is Windows."""
    return (os_name or os.name) == "nt"


def _gradle_wrapper_path(project_root: str, os_name: str | None = None) -> str:
    """Return the native Gradle wrapper path for the requested platform."""
    wrapper = "gradlew.bat" if _is_windows(os_name) else "gradlew"
    return os.path.join(project_root, wrapper)


def _bridge_launcher_path(project_root: str, os_name: str | None = None) -> str:
    """Return the native installDist launcher path for the requested platform."""
    launcher = "gym-bridge.bat" if _is_windows(os_name) else "gym-bridge"
    return os.path.join(
        project_root,
        "gym-bridge",
        "build",
        "install",
        "gym-bridge",
        "bin",
        launcher,
    )


def _find_project_root(
    start_path: str | None = None, os_name: str | None = None
) -> str | None:
    """Walk upwards until the native Gradle wrapper is found."""
    path = start_path or os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(path)
    for _ in range(10):
        if os.path.isfile(_gradle_wrapper_path(path, os_name)):
            return path
        path = os.path.dirname(path)
    return None


def _get_java_home() -> str:
    """Resolve JAVA_HOME for Java 17 on macOS."""
    try:
        return subprocess.check_output(
            ["/usr/libexec/java_home", "-v", "17"], text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return os.environ.get("JAVA_HOME", "")


def _build_bridge_dist(project_root: str, os_name: str | None = None) -> str:
    """Incrementally refresh installDist and return its native launcher."""
    script = _bridge_launcher_path(project_root, os_name)

    print("Refreshing gym-bridge distribution...")
    env = {**os.environ}
    java_home = _get_java_home()
    if java_home:
        env["JAVA_HOME"] = java_home
    result = subprocess.run(
        [
            _gradle_wrapper_path(project_root, os_name),
            ":gym-bridge:installDist",
            "-q",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        sys.exit(1)
    if not os.path.isfile(script):
        print(f"Build failed: native bridge launcher was not created at {script}")
        sys.exit(1)
    print("Distribution is up to date.")
    return script


def _ensure_save_parent(save_path: str) -> str:
    """Create and return the parent directory used by model.save()."""
    parent = os.path.dirname(os.path.abspath(os.path.expanduser(save_path)))
    os.makedirs(parent, exist_ok=True)
    return parent


def _wait_for_server(endpoint: str, timeout: int = 30) -> bool:
    """Wait for a Java bridge server to respond on the given endpoint."""
    import zmq

    dummy_deck = [
        "knight", "archer", "fireball", "arrows",
        "giant", "musketeer", "minions", "valkyrie",
    ]
    deadline = time.time() + timeout

    while time.time() < deadline:
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PAIR)
        sock.setsockopt(zmq.RCVTIMEO, 2000)
        sock.setsockopt(zmq.SNDTIMEO, 2000)
        try:
            sock.connect(endpoint)
            sock.send_string(json.dumps({
                "type": "init",
                "data": {
                    "blueDeck": dummy_deck,
                    "redDeck": dummy_deck,
                    "level": 11,
                    "ticksPerStep": 6,
                },
            }))
            resp = sock.recv_string()
            if "init_ok" in resp:
                # Cleanly end the health-check session
                sock.send_string(json.dumps({"type": "close"}))
                try:
                    sock.recv_string()
                except zmq.error.Again:
                    pass
                return True
        except zmq.error.Again:
            pass
        finally:
            sock.close()
            ctx.term()
        time.sleep(1)
    return False


def launch_servers(base_port: int, num_envs: int) -> list:
    """Auto-launch N Java bridge server processes on sequential ports.

    Returns the list of subprocess.Popen objects. Registers an atexit
    handler to terminate them on exit.
    """
    project_root = _find_project_root()
    if project_root is None:
        print("Error: Cannot find project root (gradlew not found).")
        sys.exit(1)

    script = _build_bridge_dist(project_root)
    java_home = _get_java_home()
    env = {**os.environ}
    if java_home:
        env["JAVA_HOME"] = java_home

    processes = []
    for i in range(num_envs):
        port = base_port + i
        proc = subprocess.Popen(
            [script, str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
        )
        processes.append(proc)

    def cleanup():
        for p in processes:
            try:
                p.terminate()
            except OSError:
                pass
        for p in processes:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
    atexit.register(cleanup)

    print(f"Waiting for {num_envs} server(s) on ports {base_port}-{base_port + num_envs - 1}...")
    for i in range(num_envs):
        endpoint = f"tcp://localhost:{base_port + i}"
        if not _wait_for_server(endpoint):
            print(f"Error: Server on port {base_port + i} failed to start within timeout.")
            cleanup()
            sys.exit(1)
    print(f"All {num_envs} servers ready.")
    return processes


def check_server(endpoint: str) -> bool:
    """Check if the Java bridge server is reachable (single-env mode)."""
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PAIR)
    sock.setsockopt(zmq.RCVTIMEO, 3000)
    sock.setsockopt(zmq.SNDTIMEO, 3000)

    try:
        sock.connect(endpoint)
        sock.send_string(json.dumps({
            "type": "init",
            "data": {
                "blueDeck": ["knight", "archer", "fireball", "arrows",
                             "giant", "musketeer", "minions", "valkyrie"],
                "redDeck": ["knight", "archer", "fireball", "arrows",
                            "giant", "musketeer", "minions", "valkyrie"],
                "level": 11,
                "ticksPerStep": 6,
            }
        }))
        response = sock.recv_string()
        return "init_ok" in response
    except zmq.error.Again:
        return False
    finally:
        sock.close()
        ctx.term()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PPO on CRForge")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Total training timesteps (default: 50000)")
    parser.add_argument("--save-path", type=str, default="models/ppo_crforge",
                        help="Path to save the trained model")
    parser.add_argument("--endpoint", default="tcp://localhost:9876",
                        help="Bridge server endpoint (single-env mode)")
    parser.add_argument("--ticks-per-step", type=int, default=15,
                        help="Simulation ticks per step (default: 15, ~240 regulation steps, "
                             "~400 with overtime)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a saved model .zip to resume training from")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Number of evaluation episodes after training")
    parser.add_argument("--log-dir", type=str, default="logs/ppo_crforge",
                        help="TensorBoard log directory")
    parser.add_argument("--opponent", type=str, default="rule_based",
                        choices=["noop", "random", "rule_based", "self_play"],
                        help="Opponent type (default: rule_based)")
    parser.add_argument("--self-play-interval", type=int, default=10000,
                        help="Timesteps between opponent model updates in self-play (default: 10000)")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of parallel environments (default: 1). "
                             "When >1, auto-launches Java bridge servers.")
    parser.add_argument("--base-port", type=int, default=9876,
                        help="Base port for auto-launched servers (default: 9876)")
    parser.add_argument("--jpype", action="store_true",
                        help="Use in-process JPype backend (no Java server needed)")
    args = parser.parse_args()

    # Check dependencies
    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError:
        print("Error: sb3-contrib or stable-baselines3 not installed.")
        print("Install with: pip install -e \"python[train]\"")
        sys.exit(1)

    from collections import deque

    from crforge_gym import CRForgeEnv
    from crforge_gym.opponents import SelfPlayOpponent
    from crforge_gym.wrappers import ActionMaskedWrapper, EpisodeStatsWrapper, FlattenedObsWrapper

    # Validate flags
    if args.opponent == "self_play" and args.jpype:
        print("Error: self-play requires JSON observations with the red player's hand.")
        print("The JPype backend currently exposes only the binary blue-hand schema.")
        print("Use the ZMQ backend with --num-envs 1.")
        sys.exit(1)
    if args.opponent == "self_play" and args.num_envs > 1:
        print("Error: self-play is not supported with --num-envs > 1 (subprocess mode).")
        print("The prototype requires shared opponent state and JSON observations.")
        print("Use the ZMQ backend with --num-envs 1.")
        sys.exit(1)

    binary_observations = args.opponent != "self_play"

    # -- Server setup --

    server_processes = None
    if args.jpype:
        # JPype mode: ensure JARs are built, no server processes needed
        project_root = _find_project_root()
        if project_root is None:
            print("Error: Cannot find project root (gradlew not found).")
            sys.exit(1)
        _build_bridge_dist(project_root)
        print("JPype mode: using in-process JVM (no server processes)")
    elif args.num_envs > 1:
        server_processes = launch_servers(args.base_port, args.num_envs)
    else:
        # Single-env: check manually started server
        print(f"Checking Java bridge server at {args.endpoint}...")
        if not check_server(args.endpoint):
            print("Error: Cannot connect to the Java bridge server.")
            print("Start it with: ./gradlew :gym-bridge:run")
            print("Or use --num-envs N to auto-launch servers.")
            sys.exit(1)
        print("Server is running.")

    # -- Self-play callback --

    class SelfPlayCallback(BaseCallback):
        """Periodically snapshot the training model into the SelfPlayOpponent.

        Every `update_interval` timesteps, save the current model to a
        checkpoint file and reload it into the opponent. The first iteration
        starts with the initial (random) policy.
        """

        def __init__(self, opponent: SelfPlayOpponent, update_interval: int,
                     checkpoint_dir: str = "models", verbose: int = 0):
            super().__init__(verbose)
            self.opponent = opponent
            self.update_interval = update_interval
            self.checkpoint_dir = checkpoint_dir
            self._last_update_step = 0

        def _on_training_start(self) -> None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self._snapshot_model(tag="init")

        def _on_step(self) -> bool:
            elapsed = self.num_timesteps - self._last_update_step
            if elapsed >= self.update_interval:
                self._snapshot_model(tag=f"step_{self.num_timesteps}")
                self._last_update_step = self.num_timesteps
            return True

        def _snapshot_model(self, tag: str) -> None:
            path = os.path.join(self.checkpoint_dir, f"self_play_{tag}")
            self.model.save(path)
            loaded = MaskablePPO.load(path)
            self.opponent.model = loaded
            if self.verbose > 0:
                print(f"[SelfPlayCallback] Updated opponent at step {self.num_timesteps} ({tag})")

    # -- Episode logging callback --

    class EpisodeLogCallback(BaseCallback):
        """Tracks and logs game-level statistics: win rate, episode reward, episode length.

        Collects game outcomes from info["game_outcome"] (set by EpisodeStatsWrapper)
        and logs rolling averages to TensorBoard and stdout.
        """

        def __init__(self, total_timesteps: int, window_size: int = 100,
                     log_interval: int = 50, verbose: int = 1):
            super().__init__(verbose)
            self.total_timesteps = total_timesteps
            self.window_size = window_size
            self.log_interval = log_interval
            self._outcomes = deque(maxlen=window_size)
            self._ep_rewards = deque(maxlen=window_size)
            self._ep_lengths = deque(maxlen=window_size)
            self._total_episodes = 0
            self._last_log_episode = 0

        def _on_step(self) -> bool:
            # With VecEnv, infos is a list of dicts (one per env)
            infos = self.locals.get("infos", [])
            for info in infos:
                if "episode" in info:
                    self._total_episodes += 1
                    self._ep_rewards.append(info["episode"]["r"])
                    self._ep_lengths.append(info["episode"]["l"])
                    outcome = info.get("game_outcome", "unknown")
                    self._outcomes.append(outcome)

                    if self._total_episodes - self._last_log_episode >= self.log_interval:
                        self._log_stats()
                        self._last_log_episode = self._total_episodes

            return True

        def _on_training_end(self) -> None:
            if self._total_episodes > 0:
                self._log_stats()

        def _log_stats(self):
            n = len(self._outcomes)
            if n == 0:
                return

            wins = sum(1 for o in self._outcomes if o == "win")
            losses = sum(1 for o in self._outcomes if o == "loss")
            draws = sum(1 for o in self._outcomes if o == "draw")
            win_rate = wins / n
            loss_rate = losses / n
            draw_rate = draws / n
            avg_reward = sum(self._ep_rewards) / len(self._ep_rewards)
            avg_length = sum(self._ep_lengths) / len(self._ep_lengths)

            # Log to TensorBoard
            self.logger.record("game/win_rate", win_rate)
            self.logger.record("game/loss_rate", loss_rate)
            self.logger.record("game/draw_rate", draw_rate)
            self.logger.record("game/ep_reward_mean", avg_reward)
            self.logger.record("game/ep_length_mean", avg_length)
            self.logger.record("game/total_episodes", self._total_episodes)

            if self.verbose > 0:
                pct = self.num_timesteps / self.total_timesteps * 100 if self.total_timesteps > 0 else 0
                print(
                    f"[{self.num_timesteps}/{self.total_timesteps} steps ({pct:.0f}%) | "
                    f"ep {self._total_episodes}] "
                    f"win={win_rate:.1%} loss={loss_rate:.1%} draw={draw_rate:.1%} | "
                    f"reward={avg_reward:.1f} len={avg_length:.0f} "
                    f"(last {n} games)"
                )

    # -- Opponent setup --

    if args.opponent == "self_play":
        self_play_opponent = SelfPlayOpponent(model=None)
        env_opponent = self_play_opponent
    else:
        self_play_opponent = None
        env_opponent = args.opponent

    # -- Environment creation --

    def make_env(endpoint: str | None = None, backend: str = "zmq"):
        """Factory that returns a no-arg callable for SubprocVecEnv."""
        def _init():
            env = CRForgeEnv(
                endpoint=endpoint or "tcp://localhost:9876",
                ticks_per_step=args.ticks_per_step,
                opponent=env_opponent,
                binary_obs=binary_observations,
                backend=backend,
            )
            env = EpisodeStatsWrapper(env)
            if not binary_observations:
                env = FlattenedObsWrapper(env)
            # Binary observations are already flat; JSON self-play is flattened above.
            env = ActionMaskedWrapper(env)
            return env
        return _init

    if args.jpype:
        if args.num_envs > 1:
            from crforge_gym import ThreadedJPypeVecEnv
            env_fns = [make_env(backend="jpype") for _ in range(args.num_envs)]
            env = ThreadedJPypeVecEnv(env_fns)
            eval_endpoint = None
        else:
            env = make_env(backend="jpype")()
            eval_endpoint = None
    elif args.num_envs > 1:
        env_fns = [
            make_env(f"tcp://localhost:{args.base_port + i}")
            for i in range(args.num_envs)
        ]
        env = SubprocVecEnv(env_fns)
        # Create a separate single env for evaluation (reuses first server
        # after training ends, since SubprocVecEnv will have closed its conn)
        eval_endpoint = f"tcp://localhost:{args.base_port}"
    else:
        env = make_env(args.endpoint)()
        eval_endpoint = None

    # -- Model creation --

    if args.resume:
        print(f"Resuming training from {args.resume}...")
        model = MaskablePPO.load(args.resume, env=env, tensorboard_log=args.log_dir, verbose=1)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            policy_kwargs={"net_arch": [512, 256]},
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=args.seed,
            verbose=1,
            tensorboard_log=args.log_dir,
        )

    # -- Callbacks --

    callbacks = [EpisodeLogCallback(total_timesteps=args.timesteps, window_size=100, log_interval=50, verbose=1)]
    if args.opponent == "self_play":
        self_play_opponent.model = model
        callbacks.append(SelfPlayCallback(
            opponent=self_play_opponent,
            update_interval=args.self_play_interval,
            checkpoint_dir=os.path.dirname(args.save_path) or "models",
            verbose=1,
        ))

    # -- Training --

    _ensure_save_parent(args.save_path)

    print(f"\nStarting training for {args.timesteps} timesteps...")
    print(f"Backend: {'jpype (in-process JVM)' if args.jpype else 'zmq'}")
    print(f"Parallel envs: {args.num_envs}")
    print(f"Opponent: {args.opponent}")
    if args.opponent == "self_play":
        print(f"Self-play update interval: {args.self_play_interval} steps")
    effective_rollout = 2048 * args.num_envs
    print(f"Effective rollout size: {effective_rollout} ({args.num_envs} x 2048)")
    print(f"TensorBoard logs: {args.log_dir}")
    print(f"Monitor with: tensorboard --logdir {args.log_dir}\n")

    start_time = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks if callbacks else None,
    )
    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time:.1f}s")
    print(f"Throughput: {args.timesteps / training_time:.0f} steps/sec")

    # -- Save --

    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")

    # -- Evaluate --

    if args.num_envs > 1:
        # Close the vec env and create a single env for evaluation
        env.close()
        if args.jpype:
            eval_env = make_env(backend="jpype")()
        else:
            eval_env = make_env(eval_endpoint)()
    else:
        eval_env = env

    print(f"\nEvaluating over {args.eval_episodes} episodes...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes)
    print(f"Mean reward: {mean_reward:.3f} +/- {std_reward:.3f}")

    eval_env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
