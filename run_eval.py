#!/usr/bin/env python3
"""Evaluate one (or two, for an A/B) playbooks against the scenario suite, without changing anything.

    python run_eval.py                                            # latest playbook, all scenarios
    python run_eval.py --playbook playbooks/v0.json --compare playbooks/latest.json --trials 3
    python run_eval.py --scenarios emergency_midcall --show-transcripts
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from harness.runner import SuiteResult, run_suite
from harness.scenarios import load_scenarios
from scheduler.llm import USAGE
from scheduler.playbook import Playbook

ROOT = Path(__file__).resolve().parent
console = Console()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    latest = ROOT / "playbooks" / "latest.json"
    ap.add_argument("--playbook", type=Path, default=latest if latest.exists() else ROOT / "playbooks" / "v0.json")
    ap.add_argument("--compare", type=Path, default=None, help="second playbook for an A/B on the same scenarios")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scenarios", nargs="*")
    ap.add_argument("--show-transcripts", action="store_true")
    args = ap.parse_args()

    scenarios = load_scenarios(args.scenarios)
    out_dir = ROOT / "runs" / f"eval-{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[SuiteResult] = []
    for path in [args.playbook] + ([args.compare] if args.compare else []):
        pb = Playbook.load(path)
        console.print(f"[bold]Evaluating {path.name}[/bold] (v{pb.version}, {len(pb.rules)} rules)")
        res = run_suite(scenarios, pb, args.trials, args.workers, label=path.name)
        results.append(res)
        (out_dir / f"{path.stem}.json").write_text(json.dumps(res.to_dict(), indent=2, default=str))
        if args.show_transcripts:
            for r in res.runs:
                console.rule(f"{r.scenario_id} trial {r.trial} — {'PASS' if r.passed else 'FAIL'} ({r.score:.2f})")
                console.print(r.render_transcript(), markup=False)
                for c in r.checks:
                    console.print(f"  {'✓' if c['passed'] else '✗'} [{c['severity']}] {c['id']}: {c['detail']}", markup=False)

    table = Table(title="Scenario results (pass rate · mean score)")
    table.add_column("Scenario")
    table.add_column("Split")
    for r in results:
        table.add_column(r.label)
    for sid in results[0].scenario_ids():
        row = [sid, results[0].splits[sid]]
        for r in results:
            fails = sorted({c["id"] for run in r.runs_for(sid) for c in run.checks if not c["passed"]})
            row.append(f"{r.pass_rate(sid):.0%} · {r.mean_score(sid):.2f}" + (f"\n[red]{', '.join(fails)}[/red]" if fails else ""))
        table.add_row(*row)
    for split in ("train", "holdout", None):
        row = [f"[bold]{split or 'ALL'}[/bold]", ""]
        for r in results:
            m = r.metrics(split)
            if m["runs"]:
                row.append(f"[bold]{m['pass_rate']:.0%} · {m['mean_score']:.3f}[/bold] (crit {m['critical_failures']})"
                           + (f" [yellow]{m['errors']} errored[/yellow]" if m["errors"] else ""))
            else:
                row.append("n/a" + (f" [yellow]{m['errors']} errored[/yellow]" if m["errors"] else ""))
        table.add_row(*row)
    console.print(table)
    for role, u in USAGE.items():
        console.print(f"[dim]{role}: {u.calls} calls, {u.input_tokens:,} in / {u.output_tokens:,} out tokens[/dim]")
    console.print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
