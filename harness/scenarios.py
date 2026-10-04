"""Scenario definitions live in scenarios/*.yaml; this module loads and validates them."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCENARIO_DIR = Path(__file__).resolve().parent.parent / "scenarios"
SEVERITY_WEIGHT = {"critical": 3, "major": 2, "minor": 1}


@dataclass
class Scenario:
    id: str
    split: str  # "train" (failures feed the improver) or "holdout" (only measured)
    title: str
    persona: str
    opening: str
    checks: list[dict[str, Any]]
    now: str = "2026-10-05T09:30"
    setup: dict[str, Any] = field(default_factory=dict)
    faults: dict[str, list[str]] = field(default_factory=dict)
    max_turns: int = 14
    tags: list[str] = field(default_factory=list)
    scenario_terms: list[str] = field(default_factory=list)  # literals a learned rule must not mention


def load_scenarios(ids: list[str] | None = None, split: str | None = None) -> list[Scenario]:
    out = []
    for path in sorted(SCENARIO_DIR.glob("*.yaml")):
        d = yaml.safe_load(path.read_text())
        sc = Scenario(**d)
        for c in sc.checks:
            assert c.get("id") and c.get("kind") in ("state", "trace", "transcript", "judge"), f"{sc.id}: bad check {c}"
            assert c.get("severity", "major") in SEVERITY_WEIGHT, f"{sc.id}/{c['id']}: bad severity"
        out.append(sc)
    if ids:
        out = [s for s in out if s.id in ids]
    if split:
        out = [s for s in out if s.split == split]
    return out
