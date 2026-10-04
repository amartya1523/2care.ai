"""Run scenarios against the agent and score them."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean
from pathlib import Path
from typing import Any, Callable

from scheduler.agent import Agent
from scheduler.clinic import Clinic
from scheduler.llm import model_for, provider_name
from scheduler.playbook import RULE_FIELDS, Playbook
from scheduler.prompts import CORE_PROMPT
from scheduler.tools import TOOL_SPECS, Session

from .checks import UNIVERSAL_CHECKS, run_check
from .judge import judge_run
from .record import RunRecord
from .scenarios import SEVERITY_WEIGHT, Scenario
from .simulator import SIM_SYSTEM, CallerSimulator

CACHE_DIR = Path(__file__).resolve().parent.parent / "runs" / "cache"


def _cache_key(sc: Scenario, playbook: Playbook, trial: int) -> str:
    """Everything that can change a simulated call. Any prompt, tool, model or scenario edit misses the cache."""
    material = {
        "rules": [{k: r.get(k) for k in RULE_FIELDS} for r in playbook.rules],
        "scenario": sc.__dict__,
        "trial": trial,
        "models": [provider_name(), model_for("agent"), model_for("simulator")],
        "prompts": [CORE_PROMPT, SIM_SYSTEM, TOOL_SPECS],
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()[:24]


def simulate_cached(sc: Scenario, playbook: Playbook, trial: int = 0) -> RunRecord:
    """Reuse a finished, error-free simulation with identical inputs (set RUN_CACHE=0 to disable).
    LLM calls are the scarce resource (free-tier quotas); this makes an interrupted loop resumable."""
    if os.getenv("RUN_CACHE", "1") == "0":
        return simulate(sc, playbook, trial)
    path = CACHE_DIR / f"{_cache_key(sc, playbook, trial)}.json"
    if path.exists():
        d = json.loads(path.read_text())
        return RunRecord(**{k: d[k] for k in ("scenario_id", "trial", "transcript", "trace", "initial_appointments",
                                              "final_appointments", "escalation", "ended_by", "error")})
    rec = simulate(sc, playbook, trial)
    if rec.error is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec.to_dict(), default=str))
    return rec


def simulate(sc: Scenario, playbook: Playbook, trial: int = 0) -> RunRecord:
    clinic = Clinic.load(datetime.strptime(sc.now, "%Y-%m-%dT%H:%M"), sc.setup)
    session = Session(clinic=clinic, faults=copy.deepcopy(sc.faults))
    initial = copy.deepcopy(clinic.data["appointments"])
    agent = Agent(session, playbook)
    sim = CallerSimulator(sc.persona, sc.opening, now=datetime.strptime(sc.now, "%Y-%m-%dT%H:%M"))
    transcript: list[dict[str, Any]] = [{"turn": 0, "speaker": "agent", "text": agent.greeting()}]
    ended_by, error = "max_turns", None
    try:
        for turn in range(1, sc.max_turns + 1):
            utterance, caller_done = sim.next(transcript)
            if not utterance:
                ended_by = "caller_hangup"
                break
            transcript.append({"turn": turn, "speaker": "caller", "text": utterance})
            reply = agent.respond(utterance)
            transcript.append({"turn": turn, "speaker": "agent", "text": reply})
            if session.escalation:
                ended_by = "transferred"
                break
            if caller_done:
                ended_by = "caller_hangup"
                break
    except Exception as e:  # noqa: BLE001 — infra errors are recorded, not scored as agent failures
        ended_by, error = "error", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"
    return RunRecord(
        scenario_id=sc.id,
        trial=trial,
        transcript=transcript,
        trace=session.trace,
        initial_appointments=initial,
        final_appointments=clinic.data["appointments"],
        escalation=session.escalation,
        ended_by=ended_by,
        error=error,
    )


def score(rec: RunRecord, sc: Scenario, use_judge: bool = True) -> RunRecord:
    checks = UNIVERSAL_CHECKS + sc.checks
    judge_checks = [c for c in checks if c["kind"] == "judge"]
    judged = judge_run(rec, judge_checks) if (use_judge and judge_checks and rec.error is None) else {}
    results = []
    for c in checks:
        base = {"id": c["id"], "kind": c["kind"], "severity": c.get("severity", "major"), "description": c.get("criterion") or c.get("description") or c.get("type")}
        if c["kind"] == "judge":
            if not use_judge:
                continue
            j = judged.get(c["id"], {"passed": False, "detail": "not judged"})
            results.append({**base, "passed": j["passed"], "detail": j["detail"], "evidence": j.get("evidence", ""), "evidence_verified": j.get("evidence_verified", True)})
        else:
            passed, detail = run_check(c, rec)
            results.append({**base, "passed": passed, "detail": detail})
    total = sum(SEVERITY_WEIGHT[r["severity"]] for r in results)
    got = sum(SEVERITY_WEIGHT[r["severity"]] for r in results if r["passed"])
    rec.checks = results
    rec.score = round(got / total, 4) if total else 0.0
    rec.passed = rec.error is None and not any(not r["passed"] and r["severity"] in ("critical", "major") for r in results)
    return rec


@dataclass
class SuiteResult:
    label: str
    playbook_version: int
    runs: list[RunRecord] = field(default_factory=list)
    splits: dict[str, str] = field(default_factory=dict)  # scenario_id -> split

    def scenario_ids(self) -> list[str]:
        return sorted({r.scenario_id for r in self.runs})

    def runs_for(self, sid: str) -> list[RunRecord]:
        return [r for r in self.runs if r.scenario_id == sid and r.error is None]

    def pass_rate(self, sid: str) -> float:
        rs = self.runs_for(sid)
        return mean(r.passed for r in rs) if rs else 0.0

    def mean_score(self, sid: str) -> float:
        rs = self.runs_for(sid)
        return mean(r.score for r in rs) if rs else 0.0

    def check_rate(self, sid: str, check_id: str) -> float | None:
        vals = [c["passed"] for r in self.runs_for(sid) for c in r.checks if c["id"] == check_id]
        return mean(vals) if vals else None

    def metrics(self, split: str | None = None) -> dict[str, Any]:
        ids = [s for s in self.scenario_ids() if split in (None, self.splits.get(s))]
        runs = [r for s in ids for r in self.runs_for(s)]
        return {
            "scenarios": len(ids),
            "runs": len(runs),
            # None, not 0, when nothing was measured: "no data" must never read as "everything failed".
            "pass_rate": round(mean(r.passed for r in runs), 4) if runs else None,
            "mean_score": round(mean(r.score for r in runs), 4) if runs else None,
            "critical_failures": sum(1 for r in runs for c in r.checks if not c["passed"] and c["severity"] == "critical"),
            "errors": sum(1 for s in ids for r in self.runs if r.scenario_id == s and r.error),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "playbook_version": self.playbook_version,
            "metrics": {"all": self.metrics(), "train": self.metrics("train"), "holdout": self.metrics("holdout")},
            "scenarios": {
                s: {"split": self.splits.get(s), "pass_rate": self.pass_rate(s), "mean_score": round(self.mean_score(s), 4)} for s in self.scenario_ids()
            },
            "runs": [r.to_dict() for r in self.runs],
        }


def run_suite(
    scenarios: list[Scenario],
    playbook: Playbook,
    trials: int = 1,
    workers: int = 4,
    label: str = "",
    use_judge: bool = True,
    on_done: Callable[[RunRecord], None] | None = None,
) -> SuiteResult:
    jobs = [(sc, t) for sc in scenarios for t in range(trials)]

    def one(job: tuple[Scenario, int]) -> RunRecord:
        sc, t = job
        rec = simulate_cached(sc, playbook, t)
        if rec.error:  # one retry for infrastructure flakiness
            rec = simulate_cached(sc, playbook, t)
        rec = score(rec, sc, use_judge=use_judge)
        if on_done:
            on_done(rec)
        return rec

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        runs = list(ex.map(one, jobs))
    return SuiteResult(label=label, playbook_version=playbook.version, runs=runs, splits={sc.id: sc.split for sc in scenarios})
