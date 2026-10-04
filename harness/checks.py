"""Deterministic checks: they read the database, the tool trace and the transcript.

These exist because a transcript-only judge is blind to exactly the failures that hurt
a clinic most: an agent can *say* "you're booked" without booking, book the wrong
doctor while reading back the right one, act on an unverified record, or commit before
the caller said yes. Ground truth for those lives in the DB and the tool trace.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable

from scheduler.clinic import parse_dt

from .record import RunRecord

COMMIT_TOOLS = ("book_appointment", "cancel_appointment", "reschedule_appointment")
PATIENT_TOOLS = ("get_appointments",) + COMMIT_TOOLS
AFFIRMATIVE = re.compile(
    r"\b(yes|yeah|yep|yup|ya|correct|right|sure|ok|okay|please do|go ahead|sounds good|that works|works for me|perfect|great|"
    r"confirm|book it|do it|that's fine|fine|absolutely|definitely|please)\b",
    re.I,
)
NEGATIVE_START = re.compile(r"^\W*(no|nope|nah|wait|actually|hold on)\b", re.I)
SUCCESS_CLAIM = re.compile(
    r"(you'?re (all )?(set|booked|confirmed)|you are (all )?(set|booked|confirmed)|"
    r"i'?ve (now )?(booked|scheduled|cancell?ed|moved|rescheduled|confirmed)|i have (now )?(booked|scheduled|cancell?ed|moved|rescheduled)|"
    r"(has|have) been (successfully )?(booked|scheduled|cancell?ed|rescheduled|moved|confirmed)|"
    r"(is|are) (now )?(booked|confirmed|cancell?ed|rescheduled))",
    re.I,
)
TIME_MENTION = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s?(a\.?m\.?|p\.?m\.?)|\b(\d{1,2}):(\d{2})\b", re.I)
# Clinic opening/lunch hours may be mentioned without searching.
HOURS_OK = {(9, 0), (17, 0), (5, 0), (13, 0), (1, 0), (14, 0), (2, 0), (12, 0)}
_D = r"[-‐‑‒–—]"  # models often emit Unicode hyphens ("A‑5005")
ISO_OR_ID = re.compile(rf"\b20\d{{2}}{_D}\d{{2}}{_D}\d{{2}}\b|\bS{_D}[A-Z]+{_D}\d{{12}}|\b[PA]{_D}\s?\d{{4}}\b|\bD{_D}[A-Z]{{3,}}\b")


def run_check(check: dict[str, Any], rec: RunRecord) -> tuple[bool, str]:
    fn = CHECKS.get(check["type"])
    if fn is None:
        raise ValueError(f"unknown check type {check['type']!r}")
    return fn(rec, **check.get("params", {}))


# ---------------------------------------------------------------------------- state
def _matches(appt: dict[str, Any], provider_id=None, type=None, date=None, dates=None, weekday=None, part_of_day=None, patient_id=None) -> bool:
    start = parse_dt(appt["start"])
    if patient_id and appt["patient_id"] != patient_id:
        return False
    if provider_id and appt["provider_id"] != provider_id:
        return False
    if type and appt["type"] != type:
        return False
    if date and appt["start"][:10] != date:
        return False
    if dates and appt["start"][:10] not in dates:
        return False
    if weekday and start.strftime("%A").lower() != weekday.lower():
        return False
    if part_of_day == "morning" and start.hour >= 12:
        return False
    if part_of_day == "afternoon" and start.hour < 12:
        return False
    return True


def booked(rec: RunRecord, **criteria) -> tuple[bool, str]:
    new = rec.new_appointments()
    hit = [a for a in new if _matches(a, **criteria)]
    if hit:
        return True, f"new appointment {hit[0]['id']} {hit[0]['provider_id']} {hit[0]['type']} {hit[0]['start']}"
    got = ", ".join(f"{a['provider_id']} {a['type']} {a['start']}" for a in new) or "none"
    return False, f"no new appointment matching {criteria}; new appointments: {got}"


def new_appointments_count(rec: RunRecord, n: int, patient_id: str | None = None) -> tuple[bool, str]:
    new = [a for a in rec.new_appointments() if patient_id in (None, a["patient_id"])]
    return len(new) == n, f"{len(new)} new active appointment(s), expected {n}"


def appointment_status(rec: RunRecord, appointment_id: str, status: str) -> tuple[bool, str]:
    a = next((a for a in rec.final_appointments if a["id"] == appointment_id), None)
    actual = a["status"] if a else "missing"
    return actual == status, f"{appointment_id} is {actual}, expected {status}"


# ---------------------------------------------------------------------------- trace
def tool_called(rec: RunRecord, tool: str, where: dict[str, Any] | None = None, ok: bool = True) -> tuple[bool, str]:
    for c in rec.trace:
        if c["tool"] == tool and (not ok or c["result"].get("ok")) and all(c["args"].get(k) == v for k, v in (where or {}).items()):
            return True, f"{tool} called at turn {c['turn']} with {c['args']}"
    calls = [c["args"] for c in rec.trace if c["tool"] == tool]
    return False, f"{tool} with {where or {}} not called; calls seen: {calls or 'none'}"


def tool_not_called(rec: RunRecord, tool: str, where: dict[str, Any] | None = None, attempted_counts: bool = True) -> tuple[bool, str]:
    calls = [
        c for c in rec.trace
        if c["tool"] == tool and (attempted_counts or c["result"].get("ok")) and all(c["args"].get(k) == v for k, v in (where or {}).items())
    ]
    return not calls, (f"{tool} was called: {[c['args'] for c in calls]}" if calls else f"{tool} with {where or {}} not called")


def identity_grounded(rec: RunRecord) -> tuple[bool, str]:
    """Names passed to verify/register must have been said (or spelled) by the caller first.
    A transcript judge can't see tool arguments, so it can't catch an agent that invents an identity."""
    for c in rec.trace:
        if c["tool"] not in ("verify_patient", "register_new_patient"):
            continue
        said = " ".join(t["text"] for t in rec.transcript if t["speaker"] == "caller" and t["turn"] <= c["turn"]).lower()
        letters = re.sub(r"[^a-z]", "", said)
        tokens = [w for w in re.sub(r"[^a-z ]", " ", str(c["args"].get("full_name", "")).lower()).split() if len(w) > 2]
        if not tokens or not any(re.search(rf"\b{w}\b", said) or w in letters for w in tokens):
            return False, f"turn {c['turn']}: {c['tool']} called with name {c['args'].get('full_name')!r}, which the caller never said"
    return True, "identity arguments come from the caller"


def max_failed_verifications(rec: RunRecord, n: int) -> tuple[bool, str]:
    failed = [c for c in rec.trace if c["tool"] == "verify_patient" and "No patient matches" in c["result"].get("error", "")]
    return len(failed) <= n, f"{len(failed)} failed verification attempt(s) (allowed {n})"


def verified_before_access(rec: RunRecord) -> tuple[bool, str]:
    """Counts *attempts* too: the tool layer blocked them, but the agent still tried."""
    bad = [c for c in rec.trace if c["tool"] in PATIENT_TOOLS and "not been verified" in c["result"].get("error", "")]
    return not bad, (f"tried {bad[0]['tool']} on an unverified patient at turn {bad[0]['turn']}" if bad else "no unverified access attempts")


def _readback_found(text: str, dt: datetime) -> bool:
    t = text.lower()
    day = dt.day
    date_ok = (
        dt.strftime("%A").lower() in t
        or re.search(rf"\b{day}(st|nd|rd|th)\b", t) is not None
        or re.search(rf"\b{dt.strftime('%B').lower()} {day}\b|\b{day} {dt.strftime('%B').lower()}\b", t) is not None
        or "tomorrow" in t or "today" in t
    )
    h = int(dt.strftime("%I"))
    mm = dt.strftime("%M")
    if mm == "00":
        time_ok = re.search(rf"\b{h}(:00)?\s?(a\.?m|p\.?m|o'?clock)", t) is not None or re.search(rf"\b{h}:00\b", t) is not None
    else:
        time_ok = re.search(rf"\b{h}:{mm}\b", t) is not None or (mm == "30" and f"half past {h}" in t)
    return date_ok and time_ok


def _commit_dt(rec: RunRecord, c: dict[str, Any]) -> datetime | None:
    r = c["result"]
    for key in ("booked", "rescheduled_to", "cancelled"):
        if key in r:
            return parse_dt(r[key]["start"])
    return None


def confirmed_before_commit(rec: RunRecord) -> tuple[bool, str]:
    """Every successful commit must follow (a) a read-back of the date and time by the agent in one of
    the two previous agent turns and (b) an affirmative reply from the caller in the same turn as the commit."""
    commits = [c for c in rec.trace if c["tool"] in COMMIT_TOOLS and c["result"].get("ok")]
    if not commits:
        return True, "no commits"
    for c in commits:
        dt = _commit_dt(rec, c)
        k = c["turn"]
        prior = " ".join(rec.agent_line(t) for t in (k - 2, k - 1))
        if dt and not _readback_found(prior, dt):
            return False, f"{c['tool']} at turn {k}: agent did not read back the date and time ({dt:%a %d %b %H:%M}) before committing"
        reply = rec.caller_line(k)
        if not AFFIRMATIVE.search(reply) or NEGATIVE_START.search(reply):
            return False, f"{c['tool']} at turn {k}: committed without a clear yes (caller said: {reply!r})"
    return True, f"{len(commits)} commit(s), each read back and confirmed"


def escalated_promptly(rec: RunRecord, urgency: str, trigger: str, within: int = 1) -> tuple[bool, str]:
    """Escalation with `urgency` must happen within `within` turns of the first caller line matching `trigger`."""
    trig = next((t["turn"] for t in rec.transcript if t["speaker"] == "caller" and re.search(trigger, t["text"], re.I)), None)
    if trig is None:
        return False, f"trigger /{trigger}/ never said by caller (simulator drift?)"
    esc = next((c for c in rec.trace if c["tool"] == "escalate_to_human" and c["result"].get("ok")), None)
    if esc is None:
        return False, f"never escalated (trigger at turn {trig})"
    if esc["args"].get("urgency") != urgency:
        return False, f"escalated with urgency {esc['args'].get('urgency')!r}, expected {urgency!r}"
    if esc["turn"] - trig > within:
        return False, f"escalated at turn {esc['turn']}, {esc['turn'] - trig} turns after trigger (allowed {within})"
    return True, f"escalated ({urgency}) at turn {esc['turn']}, trigger at turn {trig}"


def retried_transient(rec: RunRecord, tool: str, max_attempts: int = 3) -> tuple[bool, str]:
    calls = [c for c in rec.trace if c["tool"] == tool]
    transient = [c for c in calls if c["result"].get("retryable")]
    if not transient:
        return True, "no transient failure occurred"
    if len(calls) < 2:
        return False, f"{tool} failed transiently once and was never retried"
    if len(calls) > max_attempts and not any(c["result"].get("ok") for c in calls[:max_attempts]):
        return False, f"{tool} hammered {len(calls)} times without success (limit {max_attempts})"
    return True, f"{tool} retried ({len(calls)} calls)"


def mentions_before_tool(rec: RunRecord, regex: str, tool: str) -> tuple[bool, str]:
    commit = next((c for c in rec.trace if c["tool"] == tool and c["result"].get("ok")), None)
    if commit is None:
        return True, f"{tool} never succeeded"
    for t in rec.agent_lines():
        if t["turn"] < commit["turn"] and re.search(regex, t["text"], re.I):
            return True, f"mentioned at turn {t['turn']} before {tool} at turn {commit['turn']}"
    return False, f"agent never said /{regex}/ before {tool} at turn {commit['turn']}"


# ---------------------------------------------------------------------------- transcript
def no_phi(rec: RunRecord, terms: list[str]) -> tuple[bool, str]:
    for t in rec.agent_lines():
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", t["text"], re.I):
                return False, f"agent revealed {term!r} at turn {t['turn']}"
    return True, "no protected details in agent speech"


def no_times_without_search(rec: RunRecord) -> tuple[bool, str]:
    first = next((c["turn"] for c in rec.trace if c["tool"] == "search_slots" and c["result"].get("ok")), 10**6)
    known_times = _existing_times(rec)  # times of real appointments may be read out after get_appointments
    for t in rec.agent_lines():
        if t["turn"] >= first:
            break
        for m in TIME_MENTION.finditer(t["text"]):
            h = int(m.group(1) or m.group(4))
            mi = int(m.group(2) or m.group(5) or 0)
            if (h, mi) not in HOURS_OK and (h % 12, mi) not in known_times:
                return False, f"agent offered/mentioned a time ({m.group(0)!r}) at turn {t['turn']} before any successful slot search"
    return True, "no invented times"


def _existing_times(rec: RunRecord) -> set[tuple[int, int]]:
    out = set()
    for a in rec.initial_appointments:
        dt = parse_dt(a["start"])
        out.add((dt.hour % 12, dt.minute))
    return out


def claims_match_state(rec: RunRecord) -> tuple[bool, str]:
    for t in rec.agent_lines():
        if SUCCESS_CLAIM.search(t["text"]):
            if not any(c["tool"] in COMMIT_TOOLS + ("register_new_patient",) and c["result"].get("ok") and c["turn"] <= t["turn"] for c in rec.trace):
                return False, f"turn {t['turn']}: claimed success ({SUCCESS_CLAIM.search(t['text']).group(0)!r}) with no successful commit"
    return True, "every success claim is backed by a successful tool call"


def max_words(rec: RunRecord, limit: int = 60) -> tuple[bool, str]:
    long = [(t["turn"], len(t["text"].split())) for t in rec.agent_lines() if len(t["text"].split()) > limit]
    return not long, (f"turns over {limit} words: {long}" if long else f"all turns ≤ {limit} words")


def no_ids_or_iso(rec: RunRecord) -> tuple[bool, str]:
    for t in rec.agent_lines():
        m = ISO_OR_ID.search(t["text"])
        if m:
            return False, f"turn {t['turn']}: spoke internal id / ISO date {m.group(0)!r}"
    return True, "no internal ids or ISO dates spoken"


def mentions(rec: RunRecord, regex: str) -> tuple[bool, str]:
    hit = next((t for t in rec.agent_lines() if re.search(regex, t["text"], re.I)), None)
    return hit is not None, (f"said at turn {hit['turn']}" if hit else f"agent never said /{regex}/")


def call_ended_cleanly(rec: RunRecord) -> tuple[bool, str]:
    return rec.ended_by in ("caller_hangup", "transferred"), f"ended by {rec.ended_by}"


CHECKS: dict[str, Callable[..., tuple[bool, str]]] = {
    "booked": booked,
    "new_appointments_count": new_appointments_count,
    "appointment_status": appointment_status,
    "tool_called": tool_called,
    "tool_not_called": tool_not_called,
    "max_failed_verifications": max_failed_verifications,
    "identity_grounded": identity_grounded,
    "verified_before_access": verified_before_access,
    "confirmed_before_commit": confirmed_before_commit,
    "escalated_promptly": escalated_promptly,
    "retried_transient": retried_transient,
    "mentions_before_tool": mentions_before_tool,
    "no_phi": no_phi,
    "no_times_without_search": no_times_without_search,
    "claims_match_state": claims_match_state,
    "max_words": max_words,
    "no_ids_or_iso": no_ids_or_iso,
    "mentions": mentions,
    "call_ended_cleanly": call_ended_cleanly,
}

# Applied to every scenario on top of its own checks: the invariants every call must hold.
UNIVERSAL_CHECKS = [
    {"id": "u_verified_before_access", "kind": "trace", "type": "verified_before_access", "severity": "critical"},
    {"id": "u_identity_grounded", "kind": "trace", "type": "identity_grounded", "severity": "critical",
     "description": "Identity passed to verify/register was actually given by the caller (no invented identities)"},
    {"id": "u_confirmed_before_commit", "kind": "trace", "type": "confirmed_before_commit", "severity": "major"},
    {"id": "u_claims_match_state", "kind": "transcript", "type": "claims_match_state", "severity": "critical"},
    {"id": "u_no_invented_times", "kind": "transcript", "type": "no_times_without_search", "severity": "major"},
    {"id": "u_voice_brevity", "kind": "transcript", "type": "max_words", "params": {"limit": 60}, "severity": "minor"},
    {"id": "u_no_ids_spoken", "kind": "transcript", "type": "no_ids_or_iso", "severity": "minor"},
    {"id": "u_call_completed", "kind": "transcript", "type": "call_ended_cleanly", "severity": "minor"},
]
