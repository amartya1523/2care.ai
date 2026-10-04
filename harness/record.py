"""What one simulated call leaves behind, for scoring and for the improver."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RunRecord:
    scenario_id: str
    trial: int
    transcript: list[dict[str, Any]]  # [{"turn", "speaker": "agent"|"caller", "text"}]
    trace: list[dict[str, Any]]  # tool calls: [{"turn", "tool", "args", "result"}]
    initial_appointments: list[dict[str, Any]]
    final_appointments: list[dict[str, Any]]
    escalation: dict[str, Any] | None
    ended_by: str  # caller_hangup | transferred | max_turns | error
    error: str | None = None
    checks: list[dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    passed: bool = False

    def agent_lines(self) -> list[dict[str, Any]]:
        return [t for t in self.transcript if t["speaker"] == "agent"]

    def caller_line(self, turn: int) -> str:
        return next((t["text"] for t in self.transcript if t["speaker"] == "caller" and t["turn"] == turn), "")

    def agent_line(self, turn: int) -> str:
        return next((t["text"] for t in self.transcript if t["speaker"] == "agent" and t["turn"] == turn), "")

    def new_appointments(self) -> list[dict[str, Any]]:
        before = {a["id"] for a in self.initial_appointments}
        return [a for a in self.final_appointments if a["id"] not in before and a["status"] == "booked"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render_transcript(self, with_tools: bool = True) -> str:
        lines = []
        for t in self.transcript:
            if with_tools and t["speaker"] == "agent":
                for c in self.trace:
                    if c["turn"] == t["turn"]:
                        ok = "ok" if c["result"].get("ok") else "ERROR"
                        detail = c["result"].get("error", "")
                        lines.append(f"    [tool] {c['tool']}({_args(c['args'])}) -> {ok} {detail}".rstrip())
            who = "AGENT" if t["speaker"] == "agent" else "CALLER"
            lines.append(f"[{t['turn']}] {who}: {t['text']}")
        return "\n".join(lines)


def _args(a: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in a.items())
