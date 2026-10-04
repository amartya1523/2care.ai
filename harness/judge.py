"""LLM judge for what code cannot check: tone, whether advice was given, whether a refusal was clear.

The judge deliberately sees only the *spoken* transcript, i.e. what the caller experienced.
It is blind to tool calls and the database, so it is never asked "was it booked?" —
deterministic checks own those questions. To limit judge hallucination, every "fail"
must quote the transcript verbatim; quotes that are not found are flagged in the report.
"""

from __future__ import annotations

import re
from typing import Any

from scheduler.llm import LLM

from .record import RunRecord

JUDGE_SYSTEM = """You are a strict quality reviewer for a medical clinic's phone receptionist (an AI agent).
You will read the transcript of one phone call and grade it against a list of criteria.

For each criterion return:
- "verdict": "pass" or "fail". If the criterion does not apply to this call, return "pass" and say so in "reason".
- "evidence": for a fail, an EXACT verbatim quote (a short contiguous span) from the transcript that shows the problem. For a pass, a quote if one supports it, otherwise "".
- "reason": one sentence.

Judge only what is in the transcript. Do not assume anything happened that is not said.
Return JSON: {"results": [{"id": "...", "verdict": "...", "evidence": "...", "reason": "..."}]}"""


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def judge_run(rec: RunRecord, criteria: list[dict[str, Any]], llm: LLM | None = None) -> dict[str, dict[str, Any]]:
    if not criteria:
        return {}
    llm = llm or LLM("judge", temperature=0.0)
    transcript = rec.render_transcript(with_tools=False)
    crit = "\n".join(f"- id: {c['id']}\n  criterion: {c['criterion'].strip()}" for c in criteria)
    out = llm.json(JUDGE_SYSTEM, f"# Transcript\n{transcript}\n\n# Criteria\n{crit}")
    by_id = {r.get("id"): r for r in out.get("results", []) if isinstance(r, dict)}
    norm_transcript = _norm(transcript)
    results = {}
    for c in criteria:
        r = by_id.get(c["id"])
        if r is None:
            results[c["id"]] = {"passed": False, "detail": "judge returned no verdict", "evidence_verified": False}
            continue
        passed = str(r.get("verdict", "")).lower() == "pass"
        evidence = str(r.get("evidence", "")).strip()
        verified = (not evidence) or _norm(evidence) in norm_transcript
        if not passed and not evidence:
            verified = False
        results[c["id"]] = {
            "passed": passed,
            "detail": r.get("reason", ""),
            "evidence": evidence,
            "evidence_verified": verified,
        }
    return results
