# SPDX-License-Identifier: Apache-2.0
"""Command-line entrypoint for the localhost NextoCR Manager."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import webbrowser

from .http_server import APP_SIGNATURE, NextoCRHTTPServer
from .service import ManagerConfig, ManagerService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local NextoCR training dashboard")
    parser.add_argument("--port", type=int, default=8765, help="localhost port (default: 8765)")
    parser.add_argument(
        "--open-browser",
        dest="open_browser",
        action="store_true",
        default=True,
        help="open the dashboard in the default browser (default)",
    )
    parser.add_argument(
        "--no-browser",
        dest="open_browser",
        action="store_false",
        help="do not open the dashboard in the default browser",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = ManagerConfig.discover(port=args.port).validated()
    except ValueError as exc:
        print(f"NextoCR Manager: {exc}", file=sys.stderr)
        return 2

    url = f"http://127.0.0.1:{config.port}/"
    probe = _probe_port(config.port)
    if probe == "nextocr":
        print(f"NextoCR Manager est déjà lancé : {url}")
        if args.open_browser:
            webbrowser.open(url, new=2)
        return 0
    if probe == "occupied":
        print(
            f"NextoCR Manager: le port 127.0.0.1:{config.port} est utilisé par une autre application.",
            file=sys.stderr,
        )
        return 1

    service = ManagerService(config)
    try:
        server = NextoCRHTTPServer(service)
    except OSError as exc:
        service.close()
        print(f"NextoCR Manager: impossible d'ouvrir {url}: {exc}", file=sys.stderr)
        return 1

    stopping = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        if stopping.is_set():
            return
        stopping.set()
        # shutdown() must not run in the serve_forever() thread.
        threading.Thread(target=server.shutdown, name="NextoCR-http-shutdown", daemon=True).start()

    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGBREAK"):
        handled_signals.append(signal.SIGBREAK)
    previous_handlers: dict[signal.Signals, object] = {}
    for handled in handled_signals:
        previous_handlers[handled] = signal.getsignal(handled)
        signal.signal(handled, request_stop)

    print(f"NextoCR Manager prêt sur {url}")
    if args.open_browser:
        timer = threading.Timer(0.15, webbrowser.open, args=(url,), kwargs={"new": 2})
        timer.daemon = True
        timer.start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        stopping.set()
        server.server_close()
        service.close()
        for handled, previous in previous_handlers.items():
            signal.signal(handled, previous)
    return 0


def _probe_port(port: int) -> str:
    request = Request(
        f"http://127.0.0.1:{port}/api/session",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=0.5) as response:
            payload = json.loads(response.read(64 * 1024).decode("utf-8"))
    except (URLError, HTTPError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError, OSError):
        return "free" if _can_bind(port) else "occupied"
    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    if payload.get("ok") is True and data.get("signature") == APP_SIGNATURE:
        return "nextocr"
    return "occupied"


def _can_bind(port: int) -> bool:
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
