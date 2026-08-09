# Modified by NextoCR contributors; see NOTICE for attribution.
"""
In-process JVM bridge using JPype for zero-IPC CRForge RL training.

Drop-in replacement for BridgeClient with identical method signatures.
Calls Java objects directly via JPype, eliminating all serialization
and socket overhead. Target: ~25k+ steps/sec per env.

Requires: pip install jpype1>=1.5.0
"""

import os
import threading
from typing import Any

import numpy as np

# Serializes JVM startup across threads (ThreadedJPypeVecEnv calls
# connect() from N threads concurrently on first reset).
_jvm_lock = threading.Lock()


def _is_windows(os_name: str | None = None) -> bool:
    """Return whether *os_name* (or the current platform) is Windows."""
    return (os_name or os.name) == "nt"


def _gradle_wrapper_path(project_root: str, os_name: str | None = None) -> str:
    """Return the native Gradle wrapper path for the requested platform."""
    wrapper = "gradlew.bat" if _is_windows(os_name) else "gradlew"
    return os.path.join(project_root, wrapper)


def _find_project_root(
    start_path: str | None = None, os_name: str | None = None
) -> str:
    """Walk upwards until the native Gradle wrapper is found."""
    path = start_path or os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(path)
    for _ in range(10):
        if os.path.isfile(_gradle_wrapper_path(path, os_name)):
            return path
        path = os.path.dirname(path)
    raise FileNotFoundError(
        "Cannot find project root (native Gradle wrapper not found). "
        "Run from the crforge project directory."
    )


def _ensure_jars_built(project_root: str, os_name: str | None = None) -> str:
    """Incrementally refresh installDist and return its library directory."""
    import subprocess

    lib_dir = os.path.join(
        project_root, "gym-bridge", "build", "install", "gym-bridge", "lib"
    )

    print("Refreshing gym-bridge distribution for JPype...")
    java_home = os.environ.get("JAVA_HOME", "")
    if not java_home:
        try:
            java_home = subprocess.check_output(
                ["/usr/libexec/java_home", "-v", "17"], text=True
            ).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

    env = {**os.environ}
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
        raise RuntimeError(f"gradle installDist failed:\n{result.stderr}")
    if not os.path.isdir(lib_dir) or not any(
        name.endswith(".jar") for name in os.listdir(lib_dir)
    ):
        raise RuntimeError(
            "gradle installDist completed, but no JAR was found in " f"{lib_dir}"
        )
    print("Distribution is up to date.")
    return lib_dir


class InProcessBridge:
    """In-process JVM bridge via JPype. Same interface as BridgeClient.

    Each instance creates Java objects that run inside the Python process's
    embedded JVM. No sockets, no serialization, no IPC overhead.

    The JVM is started lazily on first connect() call. In SubprocVecEnv
    this means each subprocess gets its own JVM after fork, which is safe.
    """

    def __init__(self):
        self._session = None
        self._encoder = None
        self._connected = False
        # Java class references (set after JVM starts)
        self._StepAction = None
        self._InitConfig = None
        self._ArrayList = None

    def connect(self) -> None:
        """Start the JVM (if not already running) and import Java classes."""
        import jpype
        import jpype.imports

        with _jvm_lock:
            if not jpype.isJVMStarted():
                project_root = _find_project_root()
                lib_dir = _ensure_jars_built(project_root)

                # Build classpath from all JARs in the lib directory
                jars = [
                    os.path.join(lib_dir, f)
                    for f in os.listdir(lib_dir)
                    if f.endswith(".jar")
                ]
                classpath = os.pathsep.join(jars)

                jpype.startJVM(
                    "-Dorg.slf4j.simpleLogger.defaultLogLevel=warn",
                    classpath=[classpath],
                    convertStrings=True,
                )

        # Import Java classes
        from org.crforge.bridge import GameSession
        from org.crforge.bridge.dto import InitConfig, StepAction
        from org.crforge.bridge.observation import BinaryObservationEncoder
        from org.crforge.data.card import CardRegistry

        # Ensure CardRegistry is loaded (loads all card JSON data)
        CardRegistry.getAll()

        self._session = GameSession()
        self._encoder = BinaryObservationEncoder()
        self._StepAction = StepAction
        self._InitConfig = InitConfig
        self._ArrayList = jpype.JClass("java.util.ArrayList")
        self._connected = True

    def init(
        self,
        blue_deck: list[str],
        red_deck: list[str],
        level: int = 11,
        ticks_per_step: int = 1,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Initialize a match session."""
        blue_list = self._to_java_list(blue_deck)
        red_list = self._to_java_list(red_deck)

        java_seed = None
        if seed is not None:
            import jpype
            java_seed = jpype.JLong(seed)

        config = self._InitConfig(blue_list, red_list, level, ticks_per_step, java_seed)
        self._session.init(config)
        return {"type": "init_ok"}

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset match and return initial observation as flat float32 array."""
        if seed is not None:
            import jpype
            self._session.reset(jpype.JLong(seed))
        else:
            self._session.reset()

        java_obs = self._encoder.fillAndGetObsBuffer(
            self._session.getEngine(),
            self._session.getBluePlayer(),
            self._session.getRedPlayer(),
        )
        return np.array(java_obs, dtype=np.float32)

    def step(
        self,
        blue_action: dict | None = None,
        red_action: dict | None = None,
    ) -> tuple[np.ndarray, float, float, bool, bool, bool, bool, str]:
        """Execute one step and return (obs, blue_reward, red_reward,
        terminated, truncated, blue_action_failed, red_action_failed, outcome).

        Same return signature as BridgeClient.step() in binary mode.
        """
        java_blue = self._to_step_action(blue_action)
        java_red = self._to_step_action(red_action)

        result = self._session.stepBinary(java_blue, java_red)

        # Read observation directly from encoder
        java_obs = self._encoder.fillAndGetObsBuffer(
            self._session.getEngine(),
            self._session.getBluePlayer(),
            self._session.getRedPlayer(),
        )
        obs = np.array(java_obs, dtype=np.float32)

        reward = result.reward()
        return (
            obs,
            float(reward.blue()),
            float(reward.red()),
            bool(result.terminated()),
            bool(result.truncated()),
            bool(result.blueActionFailed()),
            bool(result.redActionFailed()),
            str(result.outcome().name()),
        )

    def close(self) -> None:
        """Release Java references. Never shuts down the JVM."""
        self._session = None
        self._encoder = None
        self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _to_step_action(self, action: dict | None):
        """Convert a Python action dict to a Java StepAction, or None for no-op."""
        if action is None:
            return None
        return self._StepAction(
            int(action["handIndex"]),
            float(action["x"]),
            float(action["y"]),
        )

    def _to_java_list(self, items: list[str]):
        """Convert a Python list of strings to a java.util.ArrayList."""
        jlist = self._ArrayList()
        for item in items:
            jlist.add(item)
        return jlist
