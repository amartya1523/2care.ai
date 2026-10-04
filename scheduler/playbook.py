"""The playbook: the only part of the agent the improvement loop is allowed to change.

The core system prompt is human-owned. Learned behaviour lives here as small, structured
rules, each with provenance (which scenario/check produced it). That keeps machine-made
changes auditable, diffable and individually revertible, and bounds their blast radius.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_RULES = 12
MAX_FIELD_CHARS = 450
RULE_FIELDS = ("title", "when", "do", "avoid", "why")

_DATE_RE = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b|\b\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b", re.I)
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
_ID_RE = re.compile(r"\b(P|A|S)-\d{3,}|\bS-[A-Z]+-\d{12}")


@dataclass
class Playbook:
    version: int = 0
    rules: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | Path) -> "Playbook":
        d = json.loads(Path(path).read_text())
        return cls(version=d.get("version", 0), rules=d.get("rules", []), history=d.get("history", []))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"version": self.version, "rules": self.rules, "history": self.history}, indent=2) + "\n")

    def copy(self) -> "Playbook":
        return copy.deepcopy(self)

    # ------------------------------------------------------------------ render
    def render(self) -> str:
        if not self.rules:
            return ""
        lines = [
            f"## Learned playbook (v{self.version})",
            "These rules were learned from evaluated call failures. Follow them as strictly as the core rules; "
            "if one ever seems to conflict with a core rule, the core rule wins.",
        ]
        for r in self.rules:
            lines.append(f"\n{r['id']}. {r['title']}")
            lines.append(f"   When: {r['when']}")
            lines.append(f"   Do: {r['do']}")
            if r.get("avoid"):
                lines.append(f"   Avoid: {r['avoid']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ change
    def apply(self, change: dict[str, Any], provenance: dict[str, Any]) -> "Playbook":
        """Return a new playbook with `change` applied. Raises ValueError if the change is invalid."""
        new = self.copy()
        op = change.get("op")
        rule = {k: str(change.get("rule", {}).get(k, "")).strip() for k in RULE_FIELDS}
        if op == "add":
            if len(new.rules) >= MAX_RULES:
                raise ValueError(f"Playbook is full ({MAX_RULES} rules). Modify or merge an existing rule instead of adding.")
            next_n = max([int(r["id"][1:]) for r in new.rules] + [0]) + 1
            rule["id"] = f"R{next_n}"
            new.rules.append({**rule, "source": provenance, "added_in": new.version + 1})
        elif op == "modify":
            rid = change.get("rule_id")
            idx = next((i for i, r in enumerate(new.rules) if r["id"] == rid), None)
            if idx is None:
                raise ValueError(f"modify: no rule with id {rid!r}")
            old = new.rules[idx]
            new.rules[idx] = {**old, **{k: v for k, v in rule.items() if v}, "source": provenance, "modified_in": new.version + 1}
        else:
            raise ValueError(f"op must be 'add' or 'modify', got {op!r}")
        new.version += 1
        new.history.append({"version": new.version, "op": op, "rule_id": change.get("rule_id") or new.rules[-1]["id"], **provenance})
        return new


def validate_rule(rule: dict[str, Any], forbidden_literals: list[str]) -> list[str]:
    """Reject rules that are malformed or overfit to a specific test case.

    A rule that mentions a scenario's patient name, a concrete date/time or a record id
    would "fix" that scenario by memorising it, which is exactly what we don't want.
    """
    problems = []
    for k in ("title", "when", "do"):
        if not str(rule.get(k, "")).strip():
            problems.append(f"missing field {k!r}")
    for k in RULE_FIELDS:
        v = str(rule.get(k, ""))
        if len(v) > MAX_FIELD_CHARS:
            problems.append(f"{k!r} is longer than {MAX_FIELD_CHARS} chars; rules must be short and general")
        if _DATE_RE.search(v) or _TIME_RE.search(v):
            problems.append(f"{k!r} contains a concrete date/time; rules must generalise across calls")
        if _ID_RE.search(v):
            problems.append(f"{k!r} contains a record id")
        for lit in forbidden_literals:
            if lit and re.search(rf"\b{re.escape(lit)}\b", v, re.I):
                problems.append(f"{k!r} mentions scenario-specific term {lit!r}")
    return problems
