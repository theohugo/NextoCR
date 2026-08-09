"""Public NextoCR Python API with compatibility exports from the crforge engine.

The inherited implementation remains in :mod:`crforge_gym` so upstream updates can be merged. New
NextoCR code should import :class:`NextoCREnv` from this package.
"""

from crforge_gym import (
    ActionMaskedWrapper,
    BridgeClient,
    CRForgeEnv,
    EpisodeStatsWrapper,
    FlattenedObsWrapper,
    RuleBasedOpponent,
    SelfPlayOpponent,
    parse_observation,
)


class NextoCREnv(CRForgeEnv):
    """NextoCR-named entry point for the inherited Gymnasium environment."""


NextoCRBridgeClient = BridgeClient

__all__ = [
    "NextoCREnv",
    "NextoCRBridgeClient",
    "ActionMaskedWrapper",
    "EpisodeStatsWrapper",
    "FlattenedObsWrapper",
    "RuleBasedOpponent",
    "SelfPlayOpponent",
    "parse_observation",
]

try:
    from crforge_gym import InProcessBridge

    __all__.append("InProcessBridge")
except ImportError:
    pass

try:
    from crforge_gym import ThreadedJPypeVecEnv

    __all__.append("ThreadedJPypeVecEnv")
except ImportError:
    pass
