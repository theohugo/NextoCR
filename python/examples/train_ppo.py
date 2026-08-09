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
  python python/examples/train_ppo.py --resume runs/my-run --run-dir runs/my-run --timesteps 50000
  python python/examples/train_ppo.py --opponent self_play --eval-episodes 0 --timesteps 100000
"""

import argparse
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
            creationflags=(
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if _is_windows()
                else 0
            ),
        )
        processes.append(proc)

    def cleanup():
        _terminate_processes(processes)

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
# Experiment CLI
# ---------------------------------------------------------------------------


def _parse_net_arch(value: str) -> list[int]:
    try:
        widths = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("net architecture must be comma-separated integers") from exc
    if not widths or any(width <= 0 for width in widths):
        raise argparse.ArgumentTypeError("net architecture widths must be positive")
    return widths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a reproducible MaskablePPO policy on NextoCR"
    )
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Additional timesteps to learn (also when resuming)")
    parser.add_argument("--run-dir", default=None,
                        help="Artifact directory; generated automatically when omitted")
    parser.add_argument("--save-path", default=None,
                        help="Final model path (default: RUN_DIR/final_model)")
    parser.add_argument("--log-dir", default=None,
                        help="TensorBoard directory (default: RUN_DIR/tensorboard)")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint .zip, extensionless path, or prior run directory")
    parser.add_argument("--endpoint", default="tcp://localhost:9876",
                        help="Training bridge endpoint in single-env ZMQ mode")
    parser.add_argument("--eval-endpoint", default=None,
                        help="Separate ZMQ endpoint required for periodic single-env evaluation")
    parser.add_argument("--base-port", type=int, default=9876,
                        help="First auto-launched ZMQ port when --num-envs > 1")
    parser.add_argument("--jpype", action="store_true",
                        help="Use the in-process JPype backend")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Parallel training environments")
    parser.add_argument("--ticks-per-step", type=int, default=15,
                        help="15 gives about 240 regulation decisions, 400 with overtime")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--opponent", choices=["noop", "random", "rule_based", "self_play"],
                        default="rule_based")
    parser.add_argument(
        "--eval-opponent",
        choices=["noop", "random", "rule_based", "self_play"],
        default=None,
        help="Fixed evaluation opponent (default: rule_based for self-play, otherwise training opponent)",
    )
    parser.add_argument("--deck-profile", default="mortar_self_play_v1")
    parser.add_argument(
        "--deck-mode",
        choices=["profile", "random", "random_mirror"],
        default="profile",
        help="profile keeps one fixed deck; random draws a fresh coherent deck "
             "for both players every episode (random_mirror gives both the same one)",
    )
    parser.add_argument(
        "--match-memory",
        action="store_true",
        help="Append own-cycle and revealed-opponent-card memory to the observation",
    )
    parser.add_argument(
        "--elixir-shaping",
        type=float,
        default=0.0,
        help="Per-episode budget for elixir-trade reward shaping; 0 disables it",
    )
    parser.add_argument(
        "--auto-scale",
        action="store_true",
        help="Size --num-envs and torch threads to this machine, leaving it usable",
    )
    parser.add_argument("--checkpoint-freq", type=int, default=100000,
                        help="Periodic checkpoint interval; 0 disables")
    parser.add_argument("--eval-freq", type=int, default=0,
                        help="Periodic evaluation interval; 0 disables")
    parser.add_argument("--eval-episodes", type=int, default=10,
                        help="Episodes per periodic/final evaluation; 0 disables final evaluation")
    parser.add_argument("--episode-log-interval", type=int, default=50)
    parser.add_argument("--self-play-interval", type=int, default=10000,
                        help="Interval between immutable league snapshots")
    parser.add_argument("--league-max-recent", type=int, default=4)
    parser.add_argument("--league-max-historical", type=int, default=8)
    parser.add_argument("--league-model-cache", type=int, default=2)
    parser.add_argument("--league-initial-weight", type=float, default=0.10)
    parser.add_argument("--league-recent-weight", type=float, default=0.60)
    parser.add_argument("--league-historical-weight", type=float, default=0.30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--net-arch", type=_parse_net_arch, default=[512, 256])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument(
        "--deterministic-torch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request deterministic Torch algorithms (default: enabled)",
    )
    parser.add_argument("--progress-bar", action="store_true")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = {
        "timesteps": args.timesteps,
        "num-envs": args.num_envs,
        "ticks-per-step": args.ticks_per_step,
        "n-steps": args.n_steps,
        "batch-size": args.batch_size,
        "n-epochs": args.n_epochs,
        "episode-log-interval": args.episode_log_interval,
        "self-play-interval": args.self_play_interval,
        "league-max-recent": args.league_max_recent,
        "league-model-cache": args.league_model_cache,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"--{name} must be positive")
    for name, value in {
        "checkpoint-freq": args.checkpoint_freq,
        "eval-freq": args.eval_freq,
        "eval-episodes": args.eval_episodes,
        "league-max-historical": args.league_max_historical,
    }.items():
        if value < 0:
            parser.error(f"--{name} must be non-negative")
    rollout_size = args.n_steps * args.num_envs
    if rollout_size <= 1:
        parser.error("--n-steps * --num-envs must exceed one")
    if args.batch_size > rollout_size:
        parser.error("--batch-size cannot exceed --n-steps * --num-envs")
    if args.eval_freq > 0 and args.eval_episodes == 0:
        parser.error("--eval-freq requires --eval-episodes > 0")
    if args.opponent == "self_play" and (args.jpype or args.num_envs != 1):
        parser.error("self_play currently requires one ZMQ environment with JSON observations")
    if args.eval_opponent == "self_play" and args.opponent != "self_play":
        parser.error("--eval-opponent self_play requires a self-play training league")
    resolved_eval_opponent = args.eval_opponent or (
        "rule_based" if args.opponent == "self_play" else args.opponent
    )
    weights = (
        args.league_initial_weight,
        args.league_recent_weight,
        args.league_historical_weight,
    )
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        parser.error("league weights must be non-negative and sum to more than zero")
    if args.eval_freq > 0 and not args.jpype and args.num_envs == 1 and not args.eval_endpoint:
        parser.error("periodic single-env ZMQ evaluation requires --eval-endpoint")
    if (
        args.eval_episodes > 0
        and not args.jpype
        and args.num_envs == 1
        and resolved_eval_opponent != args.opponent
        and not args.eval_endpoint
    ):
        parser.error("a distinct single-env ZMQ evaluation opponent requires --eval-endpoint")
    if args.eval_endpoint and args.eval_endpoint == args.endpoint:
        parser.error("--eval-endpoint must differ from --endpoint")


def _terminate_processes(processes: list[subprocess.Popen] | None) -> None:
    if not processes:
        return
    for process in processes:
        if process.poll() is not None:
            continue
        if _is_windows():
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
        else:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _validate_resume_hyperparameters(model, args: argparse.Namespace) -> None:
    expected = {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
    }
    mismatches = []
    for name, wanted in expected.items():
        actual = getattr(model, name, None)
        if name == "clip_range" and callable(actual):
            actual = actual(1.0)
        if actual is None or abs(float(actual) - float(wanted)) > 1e-12:
            mismatches.append(f"{name}={actual!r} (CLI {wanted!r})")
    if mismatches:
        raise ValueError(
            "Resume checkpoint hyperparameters differ from this invocation: "
            + ", ".join(mismatches)
        )
    actual_arch = list(getattr(model.policy, "net_arch", []))
    if actual_arch != args.net_arch:
        raise ValueError(
            "Resume checkpoint network architecture differs from this invocation: "
            f"net_arch={actual_arch!r} (CLI {args.net_arch!r})"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError:
        print("Error: training dependencies are not installed.")
        print("Install with: python -m pip install -e \"python[train,jpype]\"")
        return 1

    from crforge_gym import (
        CRForgeEnv,
        EpisodeStatsWrapper,
        ExactDiscreteActionWrapper,
        FlattenedObsWrapper,
    )
    from crforge_gym.decks import get_deck_profile
    from crforge_gym.league import (
        CheckpointLeague,
        LeagueSelfPlayOpponent,
        LeagueSnapshotCallback,
    )
    from crforge_gym.observation_preprocessing import (
        OBSERVATION_PREPROCESSING_SCHEMA,
        PreprocessedPolicyAdapter,
        StaticObservationPreprocessingWrapper,
        tag_model_preprocessing,
        validate_model_preprocessing,
    )
    from crforge_gym.training_runtime import (
        ACTION_SCHEMA,
        DeterministicEpisodeSeedWrapper,
        EpisodeMetricsCallback,
        PeriodicCheckpointCallback,
        PeriodicEvaluationCallback,
        RunArtifacts,
        SignalController,
        StopRequestedCallback,
        TrainingCommandChannel,
        TrainingControlCallback,
        default_run_dir,
        derive_episode_seed,
        evaluate_maskable_policy,
        infer_run_dir_from_checkpoint,
        resolve_resume_checkpoint,
        seed_everything,
        validate_exact_action_space,
    )

    try:
        deck_profile = get_deck_profile(args.deck_profile)
    except ValueError as exc:
        parser.error(str(exc))

    resume_path = resolve_resume_checkpoint(args.resume) if args.resume else None
    inferred_run_dir = infer_run_dir_from_checkpoint(resume_path) if resume_path else None
    run_dir = os.path.abspath(
        os.path.expanduser(args.run_dir)
        if args.run_dir
        else str(inferred_run_dir or default_run_dir(args.seed))
    )
    save_path = os.path.abspath(
        os.path.expanduser(args.save_path or os.path.join(run_dir, "final_model"))
    )
    log_dir = os.path.abspath(
        os.path.expanduser(args.log_dir or os.path.join(run_dir, "tensorboard"))
    )
    eval_seed = args.eval_seed if args.eval_seed is not None else args.seed + 10_000_000
    eval_opponent_name = args.eval_opponent or (
        "rule_based" if args.opponent == "self_play" else args.opponent
    )
    backend = "jpype" if args.jpype else "zmq"
    hardware_plan = None
    if args.auto_scale:
        import torch

        from crforge_gym.hardware import plan_training_resources

        hardware_plan = plan_training_resources(
            requested_envs=args.num_envs if args.num_envs > 1 else None
        )
        # Self-play still drives one environment; the thread budget is the part
        # that helps there, since about half the wall clock is the PPO update.
        if args.opponent != "self_play" and not args.jpype:
            args.num_envs = hardware_plan.num_envs
        torch.set_num_threads(hardware_plan.torch_threads)
        print(f"Auto-scale: {hardware_plan.describe()}")
    hash_seed_at_start = os.environ.get("PYTHONHASHSEED") == str(args.seed)
    seed_metadata = seed_everything(args.seed, args.deterministic_torch)

    config = {
        "schema_version": 1,
        "action_schema": ACTION_SCHEMA,
        "observation_preprocessing": OBSERVATION_PREPROCESSING_SCHEMA,
        "deck_profile": deck_profile.profile_id,
        "deck": deck_profile.to_manifest(),
        "deck_mode": args.deck_mode,
        "match_memory": bool(args.match_memory),
        "elixir_shaping_budget": float(args.elixir_shaping),
        "hardware_plan": hardware_plan.to_manifest() if hardware_plan else None,
        "curriculum": {
            "stage": "mirror" if args.deck_mode != "random" else "random_decks",
            "opponent_catalog_snapshot": None,
        },
        "seed": args.seed,
        "eval_seed": eval_seed,
        "python_hash_seed_effective_at_process_start": hash_seed_at_start,
        "seed_runtime": seed_metadata,
        "backend": backend,
        "num_envs": args.num_envs,
        "opponent": args.opponent,
        "evaluation_opponent": eval_opponent_name,
        "ticks_per_step": args.ticks_per_step,
        "timesteps_this_attempt": args.timesteps,
        "checkpoint_freq": args.checkpoint_freq,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "self_play_interval": args.self_play_interval,
        "league_max_recent": args.league_max_recent,
        "league_max_historical": args.league_max_historical,
        "league_model_cache": args.league_model_cache,
        "league_initial_weight": args.league_initial_weight,
        "league_recent_weight": args.league_recent_weight,
        "league_historical_weight": args.league_historical_weight,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "net_arch": args.net_arch,
        "device": args.device,
        "deterministic_torch": args.deterministic_torch,
        "run_dir": run_dir,
        "save_path": save_path,
        "log_dir": log_dir,
        "resume_from": str(resume_path) if resume_path else None,
        "endpoint": args.endpoint,
        "eval_endpoint": args.eval_endpoint,
    }

    artifacts = RunArtifacts(run_dir)
    server_processes: list[subprocess.Popen] | None = None
    train_env = None
    eval_env = None
    model = None
    initialized_artifacts = False
    signal_controller = SignalController()
    training_started = 0.0
    start_timesteps = 0
    binary_observations = args.opponent != "self_play"
    evaluation_requested = args.eval_episodes > 0

    try:
        if args.jpype:
            project_root = _find_project_root()
            if project_root is None:
                raise FileNotFoundError("Cannot find the native Gradle wrapper")
            _build_bridge_dist(project_root)
            print("JPype mode: using an in-process JVM")
        elif args.num_envs > 1:
            extra_eval_server = 1 if evaluation_requested else 0
            server_processes = launch_servers(
                args.base_port, args.num_envs + extra_eval_server
            )
        else:
            print(f"Checking training bridge at {args.endpoint}...")
            if not check_server(args.endpoint):
                raise ConnectionError(
                    "Cannot reach the training bridge. Start it with the native Gradle wrapper."
                )
            if evaluation_requested and args.eval_endpoint and not check_server(args.eval_endpoint):
                raise ConnectionError(f"Cannot reach evaluation bridge at {args.eval_endpoint}")

        if resume_path:
            print(f"Loading checkpoint: {resume_path}")
            model = MaskablePPO.load(
                str(resume_path),
                device=args.device,
                tensorboard_log=log_dir,
                verbose=1,
            )
            validate_exact_action_space(model.action_space)
            validate_model_preprocessing(model)
            _validate_resume_hyperparameters(model, args)
            start_timesteps = int(model.num_timesteps)
        config["episode_seed_strategy"] = (
            "base_plus_rank_stride_plus_checkpoint_timesteps_v1"
        )
        config["episode_seed_offset"] = start_timesteps

        model_holder: list[Any] = [None]
        league = None

        def load_league_model(path):
            loaded = MaskablePPO.load(str(path), device=args.device)
            validate_exact_action_space(loaded.action_space)
            validate_model_preprocessing(loaded)
            return PreprocessedPolicyAdapter(loaded)

        def selection_logger(phase: str):
            def record(entry, episode):
                current_model = model_holder[0]
                artifacts.record_league_selection(
                    timesteps=int(current_model.num_timesteps) if current_model else 0,
                    episode=episode,
                    opponent_id=entry.opponent_id,
                    category=entry.category,
                    checkpoint_step=entry.checkpoint_step,
                    phase=phase,
                )
            return record

        if args.opponent == "self_play":
            league = CheckpointLeague(
                os.path.join(run_dir, "league"),
                seed=args.seed + 20_000_000,
                model_loader=load_league_model,
                max_recent=args.league_max_recent,
                max_historical=args.league_max_historical,
                max_loaded_models=args.league_model_cache,
                initial_weight=args.league_initial_weight,
                recent_weight=args.league_recent_weight,
                historical_weight=args.league_historical_weight,
            )
            train_opponent = LeagueSelfPlayOpponent(
                league, on_selection=selection_logger("train")
            )
            eval_opponent = (
                LeagueSelfPlayOpponent(
                    league, on_selection=selection_logger("evaluation")
                )
                if eval_opponent_name == "self_play"
                else eval_opponent_name
            )
        else:
            train_opponent = args.opponent
            eval_opponent = eval_opponent_name

        def make_env(
            *,
            rank: int,
            endpoint: str | None = None,
            environment_backend: str = "zmq",
            evaluation: bool = False,
        ):
            def initialize():
                opponent = eval_opponent if evaluation else train_opponent
                env = CRForgeEnv(
                    endpoint=endpoint or "tcp://localhost:9876",
                    blue_deck=list(deck_profile.simulator_card_ids),
                    red_deck=list(deck_profile.simulator_card_ids),
                    ticks_per_step=args.ticks_per_step,
                    opponent=opponent,
                    binary_obs=binary_observations,
                    backend=environment_backend,
                )
                episode_seed = derive_episode_seed(
                    eval_seed if evaluation else args.seed,
                    rank=rank,
                    checkpoint_timesteps=start_timesteps,
                )
                if args.deck_mode != "profile":
                    from crforge_gym.deck_sampler import RandomDeckWrapper

                    # Offset by rank so parallel environments explore different
                    # decks instead of replaying one stream in lockstep.
                    env = RandomDeckWrapper(
                        env,
                        seed=(eval_seed if evaluation else args.seed) + rank * 7919,
                        mirror=args.deck_mode == "random_mirror",
                    )
                if args.elixir_shaping > 0:
                    from crforge_gym.elixir_rewards import (
                        ElixirTradeRewardWrapper,
                        TradeRewardConfig,
                    )

                    # Shaping belongs below the episode statistics so the logged
                    # episode reward is the one PPO actually optimises.
                    env = ElixirTradeRewardWrapper(
                        env,
                        TradeRewardConfig(episode_shaping_limit=args.elixir_shaping),
                    )
                env = DeterministicEpisodeSeedWrapper(env, episode_seed)
                env = EpisodeStatsWrapper(env)
                if evaluation:
                    env = Monitor(env)
                if not binary_observations:
                    env = FlattenedObsWrapper(env)
                env = StaticObservationPreprocessingWrapper(env)
                if args.match_memory:
                    from crforge_gym.match_memory import MatchMemoryObservationWrapper

                    # Above the static preprocessing, which requires the exact
                    # legacy width, and below the action wrapper.
                    env = MatchMemoryObservationWrapper(env)
                env = ExactDiscreteActionWrapper(env)
                env.action_space.seed(episode_seed)
                env.observation_space.seed(episode_seed)
                return env
            return initialize

        if args.jpype:
            if args.num_envs > 1:
                from crforge_gym import ThreadedJPypeVecEnv
                train_env = ThreadedJPypeVecEnv(
                    [
                        make_env(rank=rank, environment_backend="jpype")
                        for rank in range(args.num_envs)
                    ]
                )
            else:
                train_env = make_env(rank=0, environment_backend="jpype")()
            if evaluation_requested:
                eval_env = make_env(
                    rank=10_000, environment_backend="jpype", evaluation=True
                )()
        elif args.num_envs > 1:
            train_env = SubprocVecEnv(
                [
                    make_env(
                        rank=rank,
                        endpoint=f"tcp://localhost:{args.base_port + rank}",
                    )
                    for rank in range(args.num_envs)
                ]
            )
            if evaluation_requested:
                eval_env = make_env(
                    rank=10_000,
                    endpoint=f"tcp://localhost:{args.base_port + args.num_envs}",
                    evaluation=True,
                )()
        else:
            train_env = make_env(rank=0, endpoint=args.endpoint)()
            if evaluation_requested and args.eval_endpoint:
                eval_env = make_env(
                    rank=10_000, endpoint=args.eval_endpoint, evaluation=True
                )()
            elif evaluation_requested:
                eval_env = train_env

        if model is not None:
            model.set_env(train_env)
        else:
            model = MaskablePPO(
                "MlpPolicy",
                train_env,
                policy_kwargs={"net_arch": args.net_arch},
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                ent_coef=args.ent_coef,
                vf_coef=args.vf_coef,
                max_grad_norm=args.max_grad_norm,
                seed=args.seed,
                verbose=1,
                tensorboard_log=log_dir,
                device=args.device,
            )

        validate_exact_action_space(model.action_space)
        if not resume_path:
            tag_model_preprocessing(model)
        validate_model_preprocessing(model)
        model_holder[0] = model
        start_timesteps = int(model.num_timesteps)
        artifacts.initialize(
            config,
            resume_from=resume_path,
            start_timesteps=start_timesteps,
        )
        initialized_artifacts = True
        if league is not None:
            league.ensure_initial(model, checkpoint_step=start_timesteps)

        target_timesteps = start_timesteps + args.timesteps
        callbacks = [
            EpisodeMetricsCallback(
                artifacts,
                target_timesteps=target_timesteps,
                log_interval_episodes=args.episode_log_interval,
            ),
            StopRequestedCallback(lambda: signal_controller.requested),
        ]
        control_callback = TrainingControlCallback(
            artifacts,
            TrainingCommandChannel(run_dir),
        )
        callbacks.append(control_callback)
        if args.checkpoint_freq > 0:
            callbacks.append(
                PeriodicCheckpointCallback(artifacts, args.checkpoint_freq)
            )
        if args.eval_freq > 0:
            callbacks.append(
                PeriodicEvaluationCallback(
                    artifacts,
                    eval_env,
                    frequency=args.eval_freq,
                    episodes=args.eval_episodes,
                    base_seed=eval_seed + start_timesteps,
                )
            )
        if league is not None:
            callbacks.append(
                LeagueSnapshotCallback(league, args.self_play_interval)
            )

        _ensure_save_parent(save_path)
        os.makedirs(log_dir, exist_ok=True)
        print(f"\nRun directory: {run_dir}")
        print(f"Action schema: {ACTION_SCHEMA}")
        print(f"Observation preprocessing: {OBSERVATION_PREPROCESSING_SCHEMA}")
        print(f"Deck profile: {deck_profile.profile_id} ({deck_profile.average_elixir:.3g} elixir)")
        if deck_profile.known_approximations:
            print("Deck fidelity note: requested evolution/hero forms use documented base approximations.")
        print(f"Backend/envs: {backend}/{args.num_envs}")
        print(f"Opponent: {args.opponent}")
        print(f"Evaluation opponent: {eval_opponent_name}")
        print(f"Learning {args.timesteps} additional steps ({start_timesteps} -> >= {target_timesteps})")
        print(f"Rollout: {args.num_envs} x {args.n_steps} = {args.num_envs * args.n_steps}")
        print(f"TensorBoard: tensorboard --logdir {log_dir}\n")

        signal_controller.install()
        training_started = time.time()
        interrupted = False
        try:
            model.learn(
                total_timesteps=args.timesteps,
                callback=callbacks,
                reset_num_timesteps=not bool(resume_path),
                tb_log_name=os.path.basename(run_dir),
                progress_bar=args.progress_bar,
            )
            interrupted = signal_controller.requested or control_callback.stop_requested
        except KeyboardInterrupt:
            interrupted = True
        finally:
            signal_controller.restore()

        training_seconds = time.time() - training_started
        completed_steps = int(model.num_timesteps) - start_timesteps
        if control_callback.stop_requested:
            artifacts.finish(
                "paused",
                timesteps=int(model.num_timesteps),
                training_seconds=training_seconds,
                attempt_steps=completed_steps,
            )
            print(
                "Training paused cleanly; resumable checkpoint: "
                f"{control_callback.pause_checkpoint}"
            )
            return 0
        if interrupted:
            checkpoint = artifacts.save_checkpoint(
                model, timesteps=int(model.num_timesteps), reason="interrupted"
            )
            artifacts.finish(
                "interrupted",
                timesteps=int(model.num_timesteps),
                training_seconds=training_seconds,
                attempt_steps=completed_steps,
            )
            print(f"Training interrupted cleanly; resumable checkpoint: {checkpoint}")
            return 130

        saved_model = artifacts.save_final_model(model, save_path)
        print(f"Final model: {saved_model}")

        if args.eval_episodes > 0:
            print(f"Evaluating {args.eval_episodes} deterministic masked episodes...")
            result = evaluate_maskable_policy(
                model,
                eval_env,
                n_episodes=args.eval_episodes,
                base_seed=eval_seed + 1_000_000_000 + start_timesteps,
            )
            artifacts.record_evaluation(timesteps=int(model.num_timesteps), result=result)
            print(
                f"Evaluation reward: {result.mean_reward:.3f} +/- {result.std_reward:.3f}; "
                f"win={result.win_rate:.1%}"
            )

        artifacts.finish(
            "completed",
            timesteps=int(model.num_timesteps),
            training_seconds=training_seconds,
            attempt_steps=completed_steps,
        )
        throughput = completed_steps / training_seconds if training_seconds > 0 else 0.0
        print(f"Training completed in {training_seconds:.1f}s ({throughput:.0f} steps/s)")
        return 0
    except Exception as exc:
        if initialized_artifacts:
            elapsed = time.time() - training_started if training_started else 0.0
            current_steps = int(model.num_timesteps) if model is not None else start_timesteps
            checkpoint_error = None
            if model is not None:
                try:
                    artifacts.save_checkpoint(
                        model, timesteps=current_steps, reason="failed"
                    )
                except Exception as save_exc:  # Preserve the original training failure.
                    checkpoint_error = f"; failed checkpoint: {type(save_exc).__name__}: {save_exc}"
            artifacts.finish(
                "failed",
                timesteps=current_steps,
                training_seconds=elapsed,
                attempt_steps=max(0, current_steps - start_timesteps),
                error=f"{type(exc).__name__}: {exc}{checkpoint_error or ''}",
            )
        raise
    finally:
        signal_controller.restore()
        if eval_env is not None and eval_env is not train_env:
            eval_env.close()
        if train_env is not None:
            train_env.close()
        _terminate_processes(server_processes)


if __name__ == "__main__":
    raise SystemExit(main())
