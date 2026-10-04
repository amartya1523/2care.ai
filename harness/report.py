"""Render a loop run as a human-readable markdown report."""

from __future__ import annotations

from typing import Any

from scheduler.llm import USAGE, model_for, provider_name


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _metrics_row(label: str, m: dict[str, Any], decision: str = "") -> str:
    t, h = m["train"], m["holdout"]
    return (f"| {label} | {_pct(t['pass_rate'])} | {_num(t['mean_score'])} | {_pct(h['pass_rate'])} | {_num(h['mean_score'])} | "
            f"{t['critical_failures'] + h['critical_failures']} | {decision} |")


def render_report(summary: dict[str, Any]) -> str:
    base, final = summary["baseline"], summary["final"]
    out = [
        "# Self-improvement loop report",
        "",
        f"Run `{summary['started']}` · provider `{provider_name()}` · agent `{model_for('agent')}` · simulator `{model_for('simulator')}` · "
        f"judge `{model_for('judge')}` · reflector `{model_for('reflector')}` · {summary['trials']} trial(s) per scenario",
        "",
        *([f"> ⚠️ **Incomplete evaluation — no conclusions drawn.** {summary['incomplete']}", ""] if summary.get("incomplete") else []),
        "## Before → after",
        "",
        "| Playbook | Train pass | Train score | Held-out pass | Held-out score | Critical failures | Gate |",
        "|---|---|---|---|---|---|---|",
        _metrics_row(f"v{base['playbook_version']} baseline", base["metrics"]),
    ]
    for it in summary["iterations"]:
        if "decision" not in it:
            out.append(f"| iter {it['iteration']} | — | — | — | — | — | reflector error |")
            continue
        verdict = "✅ accepted" if it["decision"]["accepted"] else "❌ rejected"
        out.append(_metrics_row(f"v{it['candidate_version']} candidate (iter {it['iteration']})", it["after"], verdict))
    out += [_metrics_row(f"**v{final['playbook_version']} final**", final["metrics"], "—"), ""]
    out.append("Pass = no critical or major check failed in that run. Score = severity-weighted share of checks passed "
               "(critical 3, major 2, minor 1). Held-out scenarios never feed the improver; they only guard against overfitting.")

    out += ["", "## Per scenario (pass rate · mean score)", "", "| Scenario | Split | Before | After |", "|---|---|---|---|"]
    for sid, b in base["scenarios"].items():
        a = final["scenarios"].get(sid, {})
        arrow = " ⬆" if a.get("mean_score", 0) > b["mean_score"] + 1e-9 else (" ⬇" if a.get("mean_score", 0) < b["mean_score"] - 1e-9 else "")
        out.append(f"| {sid} | {b['split']} | {_pct(b['pass_rate'])} · {b['mean_score']:.2f} | {_pct(a.get('pass_rate', 0))} · {a.get('mean_score', 0):.2f}{arrow} |")

    for it in summary["iterations"]:
        out += ["", f"## Iteration {it['iteration']}: `{it['cluster']['check_id']}`", ""]
        out.append(f"**Flagged failure** ({it['cluster']['severity']}, scenarios: {', '.join(it['cluster']['scenarios'])}): {it['cluster']['description']}")
        for d in it["cluster"]["details"][:3]:
            out.append(f"- {d}")
        if "error" in it:
            out += ["", f"Reflector error: {it['error']}"]
            continue
        p = it["proposal"]
        r = p["change"]["rule"]
        out += [
            "",
            f"**Diagnosis** ({p.get('root_cause', '?')}): {p.get('diagnosis', '')}",
            "",
            f"**Change applied** — `{p['change']['op']}` rule *{r['title']}*",
            f"- When: {r['when']}",
            f"- Do: {r['do']}",
            f"- Avoid: {r.get('avoid', '')}",
            f"- Why: {r.get('why', '')}",
            f"- Expected effect: {p.get('expected_effect', '')}",
            f"- Regression risk (reflector's own estimate): {p.get('regression_risk', '')}",
            "",
            f"**Gate: {'ACCEPTED' if it['decision']['accepted'] else 'REJECTED'}** — " + " | ".join(it["decision"]["reasons"]),
        ]
        for label, key in (("Before", "example_before"), ("After", "example_after")):
            ex = it.get(key)
            if ex:
                out += ["", f"<details><summary>{label}: {ex['scenario_id']} — {'pass' if ex['passed'] else 'FAIL'}: {ex['detail']}</summary>",
                        "", "```", ex["transcript"], "```", "", "</details>"]

    pb = summary["final_playbook"]
    out += ["", f"## Final playbook (v{pb['version']})", ""]
    if not pb["rules"]:
        out.append("(no rules accepted)")
    for r in pb["rules"]:
        out.append(f"- **{r['id']}. {r['title']}** — when {r['when']} → {r['do']}" + (f" (avoid: {r['avoid']})" if r.get("avoid") else "")
                   + f" _[from {r['source'].get('check')} in iter {r['source'].get('iteration')}]_")

    unverified = sum(1 for run in final["runs"] for c in run["checks"] if c["kind"] == "judge" and not c["passed"] and not c.get("evidence_verified", True))
    out += ["", "## Evaluator health", "",
            f"- Judge 'fail' verdicts whose quoted evidence was not found in the transcript (treat with suspicion): {unverified}",
            f"- Runs that errored (infra, excluded from scores): baseline {base['metrics']['all']['errors']}, final {final['metrics']['all']['errors']}"]
    if USAGE:
        out += ["", "| Role | Calls | Input tokens | Output tokens |", "|---|---|---|---|"]
        for role, u in USAGE.items():
            out.append(f"| {role} | {u.calls} | {u.input_tokens:,} | {u.output_tokens:,} |")
    return "\n".join(out) + "\n"
