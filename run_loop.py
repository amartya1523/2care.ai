#!/usr/bin/env python3
"""Run the self-improvement loop: baseline eval -> flag failures -> propose a playbook rule ->
re-run every scenario -> accept only if the target improves with no regressions -> repeat.

    python run_loop.py                     # 3 iterations from playbooks/v0.json
    python run_loop.py --iterations 4 --trials 2
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markup import escape

from harness.loop import improvement_loop
from harness.report import render_report
from harness.scenarios import load_scenarios
from scheduler.llm import model_for, provider_name
from scheduler.playbook import Playbook

ROOT = Path(__file__).resolve().parent
console = Console()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--playbook", type=Path, default=ROOT / "playbooks" / "v0.json", help="starting playbook")
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--trials", type=int, default=1, help="runs per scenario per evaluation (more = less noise, more cost)")
    ap.add_argument("--workers", type=int, default=4, help="scenarios simulated in parallel")
    ap.add_argument("--scenarios", nargs="*", help="limit to these scenario ids")
    ap.add_argument("--no-promote", action="store_true", help="don't write the final playbook to playbooks/latest.json")
    args = ap.parse_args()

    scenarios = load_scenarios(args.scenarios)
    out_dir = ROOT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    console.print(f"[bold]Self-improvement loop[/bold] · {provider_name()} · agent {model_for('agent')} · judge {model_for('judge')} · "
                  f"{len(scenarios)} scenarios ({sum(s.split == 'train' for s in scenarios)} train / {sum(s.split == 'holdout' for s in scenarios)} held-out)")

    def on_run(rec) -> None:
        mark = "[green]✓[/green]" if rec.passed else ("[yellow]![/yellow]" if rec.error else "[red]✗[/red]")
        fails = [c["id"] for c in rec.checks if not c["passed"]]
        console.print(f"   {mark} {rec.scenario_id:<28} score {rec.score:.2f}  {'' if not fails else 'failed: ' + escape(', '.join(fails))}")

    result = improvement_loop(
        scenarios, Playbook.load(args.playbook), out_dir, iterations=args.iterations, trials=args.trials, workers=args.workers,
        log=lambda m: console.print(f"[cyan]{escape(m)}[/cyan]"), on_run=on_run,
    )
    report = render_report(result["summary"])
    (out_dir / "report.md").write_text(report)
    (ROOT / "reports").mkdir(exist_ok=True)
    shutil.copy(out_dir / "report.md", ROOT / "reports" / "latest_loop_report.md")
    final_pb: Playbook = result["final_playbook"]
    if final_pb.version > 0 and not args.no_promote:
        final_pb.save(ROOT / "playbooks" / "latest.json")
        console.print(f"[green]Promoted playbook v{final_pb.version} to playbooks/latest.json[/green]")

    b, f = result["summary"]["baseline"]["metrics"], result["summary"]["final"]["metrics"]
    console.print("\n[bold]Before → after[/bold]")
    for split in ("train", "holdout"):
        mb, mf = b[split], f[split]
        pct = lambda x: "n/a" if x is None else f"{x:.0%}"  # noqa: E731
        num = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
        console.print(f"  {split:<8} pass {pct(mb['pass_rate'])} → {pct(mf['pass_rate'])}   score {num(mb['mean_score'])} → {num(mf['mean_score'])}   "
                      f"critical {mb['critical_failures']} → {mf['critical_failures']}")
    if result["summary"].get("incomplete"):
        console.print(f"[yellow]INCOMPLETE: {escape(result['summary']['incomplete'])}[/yellow]")
    console.print(f"\nReport: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
