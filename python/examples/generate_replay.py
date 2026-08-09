#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate one deterministic browser replay from a NextoCR checkpoint."""

from __future__ import annotations

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a NextoCR checkpoint and save an abstract JSON replay"
    )
    parser.add_argument("--run-dir", required=True, help="Existing NextoCR run directory")
    parser.add_argument(
        "--endpoint",
        required=True,
        help="Dedicated loopback bridge, e.g. tcp://127.0.0.1:9888",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint inside RUN_DIR (default: latest durable checkpoint)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Evaluation seed")
    parser.add_argument(
        "--opponent",
        choices=["rule_based", "league"],
        default="rule_based",
    )
    parser.add_argument(
        "--opponent-id",
        default=None,
        help="Active league opponent ID (default: newest league member)",
    )
    parser.add_argument(
        "--replay-id",
        default=None,
        help="Manager-provided opaque replay ID",
    )
    parser.add_argument("--max-steps", type=int, default=2_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from crforge_gym.replay import ReplayRequest, generate_replay

        summary = generate_replay(
            ReplayRequest(
                run_dir=args.run_dir,
                endpoint=args.endpoint,
                checkpoint=args.checkpoint,
                seed=args.seed,
                opponent=args.opponent,
                opponent_id=args.opponent_id,
                replay_id=args.replay_id,
                max_steps=args.max_steps,
            )
        )
    except Exception as exc:
        print(f"Replay generation failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary.to_dict(), ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
