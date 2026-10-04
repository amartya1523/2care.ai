"""Turn evaluated failures into one concrete, structured playbook change."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from scheduler.clinic import DATA_PATH
from scheduler.llm import LLM
from scheduler.playbook import Playbook, validate_rule
from scheduler.prompts import CORE_PROMPT
from scheduler.tools import TOOL_SPECS

from .runner import SuiteResult
from .scenarios import SEVERITY_WEIGHT, Scenario

REFLECT_SYSTEM = """You improve a voice AI receptionist for a medical clinic by editing its *playbook*: a short list of structured rules appended to its system prompt.

You are given one cluster of evaluated failures (the same check failing in one or more simulated calls), the agent's current core prompt and playbook, and its tools.

Your job:
1. Diagnose the root cause from the evidence: what did the agent do, what should it have done, and why did the current instructions allow the mistake? Distinguish a missing rule from an ambiguous or conflicting one.
2. Propose exactly ONE change: add a new rule, or modify an existing rule if a rule already covers this area (prefer modifying over piling on near-duplicates).

Constraints on the rule:
- It must generalise to any caller: no patient names, concrete dates or times, record ids, or details specific to these transcripts.
- It must be concrete and operational ("before calling X, do Y"), not a vague value ("be careful").
- Never weaken safety, privacy, verification or confirmation behaviour to make a check pass.
- Keep each field under 300 characters. "when" is the trigger condition, "do" the required behaviour, "avoid" the specific mistake, "why" the reason.
- Consider what else the rule could change. A rule that makes the agent stricter everywhere can break calls that currently pass.

Return JSON:
{"diagnosis": "...", "root_cause": "missing_rule|ambiguous_rule|conflicting_rule|ignored_rule|tool_misuse",
 "change": {"op": "add" | "modify", "rule_id": null | "R<n>", "rule": {"title": "...", "when": "...", "do": "...", "avoid": "...", "why": "..."}},
 "expected_effect": "...", "regression_risk": "..."}"""


@dataclass
class FailureCluster:
    check_id: str
    severity: str
    description: str
    examples: list[dict[str, Any]] = field(default_factory=list)  # {scenario_id, scenario_title, detail, evidence, transcript}

    @property
    def weight(self) -> int:
        return SEVERITY_WEIGHT[self.severity] * len(self.examples)

    @property
    def scenario_ids(self) -> list[str]:
        return sorted({e["scenario_id"] for e in self.examples})


def cluster_failures(result: SuiteResult, scenarios: list[Scenario], split: str = "train") -> list[FailureCluster]:
    titles = {s.id: s.title for s in scenarios}
    clusters: dict[str, FailureCluster] = {}
    for r in result.runs:
        if r.error or result.splits.get(r.scenario_id) != split:
            continue
        for c in r.checks:
            if c["passed"]:
                continue
            cl = clusters.setdefault(c["id"], FailureCluster(c["id"], c["severity"], c["description"]))
            cl.examples.append(
                {
                    "scenario_id": r.scenario_id,
                    "scenario_title": titles.get(r.scenario_id, r.scenario_id),
                    "detail": c["detail"],
                    "evidence": c.get("evidence", ""),
                    "transcript": r.render_transcript(with_tools=True),
                }
            )
    return sorted(clusters.values(), key=lambda c: (-c.weight, c.check_id))


def forbidden_terms(scenarios: list[Scenario]) -> list[str]:
    data = json.loads(DATA_PATH.read_text())
    terms = set()
    for p in data["patients"]:
        terms.update(p["full_name"].split())
        terms.add(p["date_of_birth"])
    for s in scenarios:
        terms.update(s.scenario_terms)
        for p in s.setup.get("extra_patients", []):
            terms.update(p["full_name"].split())
    return sorted(t for t in terms if len(t) > 2)


def _prompt(cluster: FailureCluster, playbook: Playbook, rejected: list[dict[str, Any]]) -> str:
    tools = "\n".join(f"- {t['name']}: {t['description']}" for t in TOOL_SPECS)
    rules = json.dumps([{k: r.get(k) for k in ("id", "title", "when", "do", "avoid")} for r in playbook.rules], indent=2) if playbook.rules else "(empty)"
    ex_parts = []
    seen = set()
    for e in cluster.examples:
        if e["scenario_id"] in seen or len(ex_parts) >= 3:
            continue
        seen.add(e["scenario_id"])
        tr = e["transcript"]
        if len(tr) > 7000:
            tr = tr[:2500] + "\n...\n" + tr[-4000:]
        ex_parts.append(
            f"### Example from scenario: {e['scenario_title']}\nCheck result: {e['detail']}\n"
            + (f"Judge evidence: {e['evidence']!r}\n" if e["evidence"] else "")
            + f"Transcript (with tool calls):\n{tr}"
        )
    rej = ""
    if rejected:
        rej = "\n\n# Previously rejected attempts for this failure (do not repeat them)\n" + "\n".join(
            f"- rule {json.dumps(r['change'].get('rule', {}))} -> rejected because: {r['reason']}" for r in rejected
        )
    return (
        f"# Core prompt (human-owned, you cannot change it)\n{CORE_PROMPT}\n\n# Tools\n{tools}\n\n"
        f"# Current playbook rules\n{rules}\n\n"
        f"# Failing check: {cluster.check_id} (severity {cluster.severity})\nWhat it verifies: {cluster.description}\n"
        f"Failed in {len(cluster.examples)} run(s) across scenarios: {', '.join(cluster.scenario_ids)}\n\n"
        + "\n\n".join(ex_parts)
        + rej
    )


def propose_improvement(
    cluster: FailureCluster,
    playbook: Playbook,
    scenarios: list[Scenario],
    rejected: list[dict[str, Any]] | None = None,
    llm: LLM | None = None,
) -> dict[str, Any]:
    """Returns {"diagnosis", "root_cause", "change", "expected_effect", "regression_risk"}; raises if no valid change."""
    llm = llm or LLM("reflector", temperature=0.4)
    forbidden = forbidden_terms(scenarios)
    feedback = ""
    for _ in range(3):
        out = llm.json(REFLECT_SYSTEM, _prompt(cluster, playbook, rejected or []) + feedback)
        change = out.get("change") or {}
        problems = validate_rule(change.get("rule") or {}, forbidden)
        if change.get("op") not in ("add", "modify"):
            problems.append("change.op must be 'add' or 'modify'")
        if change.get("op") == "modify" and change.get("rule_id") not in {r["id"] for r in playbook.rules}:
            problems.append(f"rule_id {change.get('rule_id')!r} does not exist")
        if not problems:
            return out
        feedback = "\n\n# Your previous proposal was rejected by the validator\n" + "\n".join(f"- {p}" for p in problems) + "\nFix these and answer again."
    raise ValueError(f"reflector could not produce a valid rule: {problems}")
