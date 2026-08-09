"""Tests for the public NextoCR compatibility package."""

from crforge_gym import BridgeClient, CRForgeEnv
from nextocr_gym import NextoCRBridgeClient, NextoCREnv


def test_nextocr_environment_extends_inherited_environment():
    assert issubclass(NextoCREnv, CRForgeEnv)


def test_nextocr_bridge_alias_preserves_protocol_compatibility():
    assert NextoCRBridgeClient is BridgeClient
