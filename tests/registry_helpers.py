"""Test helper: re-create a BLOCKED route for a registry participant so fail-closed behaviour stays covered."""

from msu_llm_lab.models import registry
from msu_llm_lab.models.registry import RouteStatus


def blocked_copy(key: str, reason: str = "route not approved"):
    rec = registry.REGISTRY[key]
    return rec.model_copy(update={"route": rec.route.model_copy(update={"status": RouteStatus.BLOCKED, "blocked_reasons": [reason]})})


def block_route(monkeypatch, key: str) -> None:
    monkeypatch.setitem(registry.REGISTRY, key, blocked_copy(key))
