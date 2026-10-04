#!/usr/bin/env python3
"""Talk to the scheduling agent in your terminal (you play the caller).

    python run_agent.py                       # uses the latest accepted playbook
    python run_agent.py --playbook playbooks/v0.json
    python run_agent.py --scenario reschedule_existing   # load a scenario's clinic setup + clock
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from rich.console import Console

from harness.scenarios import load_scenarios
from scheduler.agent import Agent
from scheduler.clinic import Clinic
from scheduler.playbook import Playbook
from scheduler.tools import Session

ROOT = Path(__file__).resolve().parent
DEFAULT_NOW = "2026-10-05T09:30"
console = Console()


def default_playbook() -> Path:
    latest = ROOT / "playbooks" / "latest.json"
    return latest if latest.exists() else ROOT / "playbooks" / "v0.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--playbook", type=Path, default=None)
    ap.add_argument("--now", default=DEFAULT_NOW, help="clinic clock, YYYY-MM-DDTHH:MM (fixed so runs are reproducible)")
    ap.add_argument("--scenario", default=None, help="load the clinic setup (extra records, injected faults) of a scenario id")
    ap.add_argument("--hide-tools", action="store_true", help="don't print tool calls")
    args = ap.parse_args()

    setup, faults, now = {}, {}, args.now
    if args.scenario:
        sc = {s.id: s for s in load_scenarios()}[args.scenario]
        setup, faults, now = sc.setup, sc.faults, sc.now
    playbook_path = args.playbook or default_playbook()
    playbook = Playbook.load(playbook_path)
    session = Session(clinic=Clinic.load(datetime.strptime(now, "%Y-%m-%dT%H:%M"), setup), faults={k: list(v) for k, v in faults.items()})
    agent = Agent(session, playbook)

    console.print(f"[dim]playbook {playbook_path.name} (v{playbook.version}, {len(playbook.rules)} learned rules) | clinic clock {now} | "
                  f"agent model {agent.llm.provider}:{agent.llm.model} | type 'quit' to hang up[/dim]\n")
    console.print(f"[bold cyan]Maya:[/bold cyan] {agent.greeting()}")
    while True:
        try:
            text = console.input("[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit", "bye!"}:
            break
        if not text:
            continue
        seen = len(session.trace)
        reply = agent.respond(text)
        if not args.hide_tools:
            for t in session.trace[seen:]:
                status = "ok" if t["result"].get("ok") else "ERROR"
                console.print(f"  [dim]→ {t['tool']}({json.dumps(t['args'])}) [{status}] {json.dumps(t['result'])[:220]}[/dim]")
        console.print(f"[bold cyan]Maya:[/bold cyan] {reply}")
        if session.escalation:
            console.print(f"[yellow]Call transferred to a human ({session.escalation['urgency']}).[/yellow]")
            break
    console.print("[dim]Call ended.[/dim]")


if __name__ == "__main__":
    main()
