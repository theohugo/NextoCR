"""
ZeroMQ bridge client for communicating with the CRForge Java server.

Supports both JSON and binary observation modes. Binary mode sends
observations as raw float32 arrays for minimal serialization overhead.
"""

import json
import struct
from typing import Any

import numpy as np
import zmq


_BINARY_OUTCOMES = {
    0: "ONGOING",
    1: "BLUE_WIN",
    2: "RED_WIN",
    3: "DRAW",
}


def _decode_binary_step(raw: bytes):
    """Decode the append-only binary step protocol, including its outcome trailer."""
    from crforge_gym.env import OBS_SIZE, OBS_V1_SIZE

    if len(raw) < 12:
        raise RuntimeError(f"Binary step response is too short: {len(raw)} bytes")

    blue_reward, red_reward = struct.unpack_from("<ff", raw, 0)
    terminated = raw[8] != 0
    truncated = raw[9] != 0
    blue_action_failed = raw[10] != 0
    red_action_failed = raw[11] != 0

    observation_floats = None
    outcome = None
    for candidate in (OBS_SIZE, OBS_V1_SIZE):
        observation_end = 12 + candidate * 4
        if len(raw) == observation_end + 4:
            outcome_code = struct.unpack_from("<i", raw, observation_end)[0]
            if outcome_code not in _BINARY_OUTCOMES:
                raise RuntimeError(f"Unknown binary game outcome code: {outcome_code}")
            observation_floats = candidate
            outcome = _BINARY_OUTCOMES[outcome_code]
            break
        if len(raw) == observation_end:
            # Compatibility with a bridge predating the canonical outcome trailer.
            observation_floats = candidate
            outcome = "UNKNOWN" if terminated or truncated else "ONGOING"
            break

    if observation_floats is None:
        raise RuntimeError(f"Unsupported binary step response size: {len(raw)} bytes")

    obs = np.frombuffer(
        raw,
        dtype=np.float32,
        count=observation_floats,
        offset=12,
    ).copy()
    return (
        obs,
        blue_reward,
        red_reward,
        terminated,
        truncated,
        blue_action_failed,
        red_action_failed,
        outcome,
    )


class BridgeClient:
    """ZMQ PAIR socket client that connects to the CRForge bridge server.

    When binary_obs=True, observations are received as raw byte arrays
    containing float32 values, bypassing JSON serialization entirely.
    """

    def __init__(self, endpoint: str = "tcp://localhost:9876", binary_obs: bool = True):
        self.endpoint = endpoint
        self.binary_obs = binary_obs
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self._connected = False

    def connect(self) -> None:
        """Connect to the bridge server."""
        self.socket.connect(self.endpoint)
        self._connected = True

    def send(self, msg_type: str, data: dict | None = None) -> None:
        """Send a JSON message to the server."""
        message = {"type": msg_type}
        if data is not None:
            message["data"] = data
        self.socket.send_json(message)

    def receive(self) -> dict[str, Any]:
        """Receive a JSON message from the server."""
        return self.socket.recv_json()

    def init(
        self,
        blue_deck: list[str],
        red_deck: list[str],
        level: int = 11,
        ticks_per_step: int = 1,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Initialize a match on the server."""
        data = {
            "blueDeck": blue_deck,
            "redDeck": red_deck,
            "level": level,
            "ticksPerStep": ticks_per_step,
            "binaryObs": self.binary_obs,
        }
        if seed is not None:
            data["seed"] = seed
        self.send("init", data)
        response = self.receive()
        if response.get("type") == "error":
            raise RuntimeError(f"Init failed: {response.get('message')}")
        return response

    def reset(self, seed: int | None = None) -> dict[str, Any] | np.ndarray:
        """Reset the match and get initial observation.

        Args:
            seed: optional seed override for deterministic deck shuffling

        Returns:
            In binary mode: flat float32 numpy array sized by the bridge observation schema.
            In JSON mode: raw observation dict from server.
        """
        data = {}
        if seed is not None:
            data["seed"] = seed
        self.send("reset", data if data else None)

        if self.binary_obs:
            raw = self.socket.recv()
            return np.frombuffer(raw, dtype=np.float32).copy()

        response = self.receive()
        if response.get("type") == "error":
            raise RuntimeError(f"Reset failed: {response.get('message')}")
        return response.get("data", {})

    def step(
        self,
        blue_action: dict | None = None,
        red_action: dict | None = None,
    ) -> dict[str, Any] | tuple[np.ndarray, float, float, bool, bool, bool, bool, str]:
        """Submit actions for both players and advance the simulation.

        Returns:
            In binary mode: tuple of (obs_array, blue_reward, red_reward,
                terminated, truncated, blue_action_failed, red_action_failed, outcome).
            In JSON mode: raw step result dict from server.
        """
        self.send(
            "step",
            {
                "blueAction": blue_action,
                "redAction": red_action,
            },
        )

        if self.binary_obs:
            raw = self.socket.recv()
            return _decode_binary_step(raw)

        response = self.receive()
        if response.get("type") == "error":
            raise RuntimeError(f"Step failed: {response.get('message')}")
        return response.get("data", {})

    def close(self) -> None:
        """Send close message and disconnect."""
        if self._connected:
            try:
                self.send("close")
                self.receive()  # wait for close_ok
            except Exception:
                pass
            self.socket.close()
            self.context.term()
            self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
