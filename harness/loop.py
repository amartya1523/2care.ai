"""The self-improvement loop: evaluate -> cluster failures -> propose one rule -> re-run everything -> gate.

One change per iteration, so every score movement is attributable to a single rule.
A candidate playbook is accepted only if
  1. the failure it targeted actually improved,
  2. no scenario (train *or* held-out) regresses — suspected regressions are re-run
     against both playbooks before being believed, since LLM calls are noisy, and
  3. the overall train score does not drop.
Rejected candidates are logged with the reason and fed back to the reflector.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from scheduler.playbook import Playbook

from .reflect import FailureCluster, cluster_failures, propose_improvement
from .runner import SuiteResult, run_suite, score, simulate_cached
from .scenarios import Scenario

SCORE_TOLERANCE = 0.02
CONFIRM_TRIALS = 2


@dataclass
class GateDecision:
    accepted: bool
    reasons: list[str]
    target_before: float
    target_after: float
    regressions: list[str] = field(default_factory=list)


def target_rate(result: SuiteResult, cluster: FailureCluster) -> float:
    rates = [result.check_rate(sid, cluster.check_id) for sid in cluster.scenario_ids]
    rates = [r for r in rates if r is not None]
    return mean(rates) if rates else 0.0


def _rerun(result: SuiteResult, sc: Scenario, playbook: Playbook, n: int) -> None:
    for t in range(n):
        rec = simulate_cached(sc, playbook, trial=100 + t)
        if rec.error is None:
            result.runs.append(score(rec, sc))


def gate(
    current: SuiteResult,
    current_pb: Playbook,
    candidate: SuiteResult,
    candidate_pb: Playbook,
    cluster: FailureCluster,
    scenarios: list[Scenario],
    log: Callable[[str], None] = print,
) -> GateDecision:
    reasons: list[str] = []
    before, after = target_rate(current, cluster), target_rate(candidate, cluster)
    by_id = {s.id: s for s in scenarios}

    def crit(res: SuiteResult, sid: str) -> float:
        rs = res.runs_for(sid)
        return mean(sum(1 for c in r.checks if not c["passed"] and c["severity"] == "critical") for r in rs) if rs else 0.0

    suspects = [
        sid for sid in current.scenario_ids()
        if candidate.pass_rate(sid) < current.pass_rate(sid) or crit(candidate, sid) > crit(current, sid)
    ]
    regressions = []
    for sid in suspects:
        log(f"    suspected regression in {sid}: re-running {CONFIRM_TRIALS}x on both playbooks to rule out noise")
        _rerun(current, by_id[sid], current_pb, CONFIRM_TRIALS)
        _rerun(candidate, by_id[sid], candidate_pb, CONFIRM_TRIALS)
        if candidate.pass_rate(sid) < current.pass_rate(sid) or crit(candidate, sid) > crit(current, sid):
            regressions.append(
                f"{sid} ({current.splits.get(sid)}): pass rate {current.pass_rate(sid):.2f} -> {candidate.pass_rate(sid):.2f}, "
                f"critical failures/run {crit(current, sid):.2f} -> {crit(candidate, sid):.2f}"
            )
    after = target_rate(candidate, cluster)
    before = target_rate(current, cluster)
    if after <= before:
        reasons.append(f"targeted check {cluster.check_id} did not improve ({before:.2f} -> {after:.2f})")
    if regressions:
        reasons.append("confirmed regressions: " + "; ".join(regressions))
    s_before, s_after = current.metrics("train")["mean_score"] or 0.0, candidate.metrics("train")["mean_score"] or 0.0
    if s_after < s_before - SCORE_TOLERANCE:
        reasons.append(f"train mean score dropped {s_before:.3f} -> {s_after:.3f}")
    accepted = not reasons
    if accepted:
        reasons.append(f"{cluster.check_id} {before:.2f} -> {after:.2f}; train score {s_before:.3f} -> {s_after:.3f}; no regressions")
    return GateDecision(accepted, reasons, before, after, regressions)


def improvement_loop(
    scenarios: list[Scenario],
    playbook: Playbook,
    out_dir: Path,
    iterations: int = 3,
    trials: int = 1,
    workers: int = 4,
    max_attempts_per_cluster: int = 2,
    log: Callable[[str], None] = print,
    on_run: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    save = lambda name, obj: (out_dir / name).write_text(json.dumps(obj, indent=2, default=str))  # noqa: E731

    log(f"[baseline] playbook v{playbook.version}: running {len(scenarios)} scenarios x {trials} trial(s)")
    baseline = run_suite(scenarios, playbook, trials, workers, label=f"v{playbook.version} (baseline)", on_done=on_run)
    current, current_pb = baseline, playbook
    baseline_snapshot = baseline.to_dict()  # frozen before confirm re-runs add data
    save("iter0_baseline.json", baseline_snapshot)
    playbook.save(out_dir / f"playbook_v{playbook.version}.json")
    log(f"[baseline] {_fmt(baseline)}")
    incomplete = None
    if _errors(baseline):
        # Never learn from (or report on) a partial evaluation: missing runs would look like passes or failures.
        incomplete = (f"Baseline incomplete: {len(_errors(baseline))} scenario(s) errored, so the loop stopped before proposing anything. "
                      "Finished runs are cached; re-run the same command later to resume. Errors: " + " || ".join(_errors(baseline)))
        log(f"[baseline] {incomplete}")

    tried: dict[str, list[dict[str, Any]]] = defaultdict(list)
    iterations_log: list[dict[str, Any]] = []
    for it in range(1, (0 if incomplete else iterations) + 1):
        clusters = [c for c in cluster_failures(current, scenarios, "train") if len(tried[c.check_id]) < max_attempts_per_cluster]
        # Fewest failed attempts first: a failure a prompt rule could not fix (e.g. a model that invents tool
        # arguments) should not consume every iteration while other, fixable failures wait.
        clusters.sort(key=lambda c: len(tried[c.check_id]))
        if not clusters:
            log(f"[iter {it}] no remaining train failures to learn from — stopping")
            break
        cluster = clusters[0]
        log(f"[iter {it}] targeting {cluster.check_id} ({cluster.severity}) failing in {cluster.scenario_ids}")
        try:
            proposal = propose_improvement(cluster, current_pb, scenarios, tried[cluster.check_id])
        except ValueError as e:
            log(f"[iter {it}] reflector failed: {e}")
            tried[cluster.check_id].append({"change": {}, "reason": str(e)})
            iterations_log.append({"iteration": it, "cluster": _cluster_info(cluster), "error": str(e)})
            continue
        provenance = {"iteration": it, "check": cluster.check_id, "scenarios": cluster.scenario_ids, "diagnosis": proposal.get("diagnosis", "")}
        candidate_pb = current_pb.apply(proposal["change"], provenance)
        rule = proposal["change"]["rule"]
        log(f"[iter {it}] diagnosis: {proposal.get('diagnosis', '')}")
        log(f"[iter {it}] proposed {proposal['change']['op']} rule: {rule['title']} — when {rule['when']} → {rule['do']}")
        save(f"iter{it}_proposal.json", proposal)

        log(f"[iter {it}] re-running all scenarios with candidate playbook v{candidate_pb.version}")
        candidate = run_suite(scenarios, candidate_pb, trials, workers, label=f"v{candidate_pb.version} (candidate)", on_done=on_run)
        if _errors(candidate):
            incomplete = (f"Candidate v{candidate_pb.version} evaluation incomplete ({len(_errors(candidate))} errored); it was neither accepted nor "
                          "rejected and the loop stopped. Re-run later to resume from cache. Errors: " + " || ".join(_errors(candidate)))
            log(f"[iter {it}] {incomplete}")
            iterations_log.append({"iteration": it, "cluster": _cluster_info(cluster), "error": incomplete})
            break
        decision = gate(current, current_pb, candidate, candidate_pb, cluster, scenarios, log)
        save(f"iter{it}_candidate.json", candidate.to_dict())
        log(f"[iter {it}] candidate {_fmt(candidate)}")
        log(f"[iter {it}] {'ACCEPTED' if decision.accepted else 'REJECTED'}: {' | '.join(decision.reasons)}")

        entry = {
            "iteration": it,
            "cluster": _cluster_info(cluster),
            "proposal": proposal,
            "candidate_version": candidate_pb.version,
            "decision": decision.__dict__,
            "before": current.to_dict()["metrics"],
            "after": candidate.to_dict()["metrics"],
            "before_scenarios": current.to_dict()["scenarios"],
            "after_scenarios": candidate.to_dict()["scenarios"],
            "example_before": _example(current, cluster),
            "example_after": _example(candidate, cluster),
        }
        iterations_log.append(entry)
        if decision.accepted:
            current, current_pb = candidate, candidate_pb
            current_pb.save(out_dir / f"playbook_v{current_pb.version}.json")
        else:
            tried[cluster.check_id].append({"change": proposal["change"], "reason": "; ".join(decision.reasons)})

    summary = {
        "started": out_dir.name,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "trials": trials,
        "baseline": baseline_snapshot,
        "final": current.to_dict(),
        "final_playbook": {"version": current_pb.version, "rules": current_pb.rules, "history": current_pb.history},
        "iterations": iterations_log,
        "incomplete": incomplete,
    }
    save("summary.json", {k: v for k, v in summary.items() if k not in ("baseline", "final")} | {
        "baseline_metrics": baseline_snapshot["metrics"], "final_metrics": current.to_dict()["metrics"]})
    return {"summary": summary, "final_playbook": current_pb, "baseline": baseline, "final": current}


def _fmt(r: SuiteResult) -> str:
    t, h = r.metrics("train"), r.metrics("holdout")
    pct = lambda x: "n/a" if x is None else f"{x:.0%}"  # noqa: E731
    num = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
    errs = t["errors"] + h["errors"]
    return (f"train pass {pct(t['pass_rate'])} score {num(t['mean_score'])} | holdout pass {pct(h['pass_rate'])} score {num(h['mean_score'])} | "
            f"critical failures {t['critical_failures'] + h['critical_failures']}" + (f" | {errs} run(s) ERRORED" if errs else ""))


def _errors(r: SuiteResult) -> list[str]:
    return sorted({f"{x.scenario_id}: {(x.error or '').splitlines()[0][:160]}" for x in r.runs if x.error})


def _cluster_info(c: FailureCluster) -> dict[str, Any]:
    return {"check_id": c.check_id, "severity": c.severity, "description": c.description, "scenarios": c.scenario_ids,
            "details": [e["detail"] for e in c.examples][:5]}


def _example(result: SuiteResult, cluster: FailureCluster) -> dict[str, Any] | None:
    """One transcript of the targeted scenario, for the before/after section of the report."""
    for sid in cluster.scenario_ids:
        for r in result.runs_for(sid):
            chk = next((c for c in r.checks if c["id"] == cluster.check_id), None)
            if chk:
                return {"scenario_id": sid, "passed": chk["passed"], "detail": chk["detail"], "transcript": r.render_transcript()}
    return None
