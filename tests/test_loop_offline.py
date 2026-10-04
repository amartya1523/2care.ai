"""End-to-end test of the improvement loop with scripted fake LLMs (no API key needed).

The fake agent only reads back and waits for a yes if its system prompt contains a
learned rule about read-back. The fake reflector first proposes an irrelevant rule
(which the gate must reject) and then the right one (which the gate must accept).
This checks the loop's mechanics, not model quality.
"""

import json

from harness.loop import improvement_loop
from harness.report import render_report
from harness.scenarios import Scenario
from scheduler import llm as llm_mod
from scheduler.llm import Reply, ToolCall
from scheduler.playbook import Playbook

SCENARIO = Scenario(
    id="fake_booking",
    split="train",
    title="fake booking",
    persona="Priya Nair, 22 July 1988",
    opening="I'd like a follow-up with Dr. Rao on Wednesday morning.",
    checks=[{"id": "booked", "kind": "state", "type": "booked", "severity": "critical",
             "params": {"patient_id": "P-1001", "provider_id": "D-RAO", "date": "2026-10-07"}}],
)
HOLDOUT = Scenario(**{**SCENARIO.__dict__, "id": "fake_booking_holdout", "split": "holdout"})

SIM_LINES = ["Priya Nair, born 22 July 1988.", "Yes, that works.", "Thanks, bye! <END_CALL>"]
REFLECTIONS = [
    {"title": "Be warm", "when": "Always", "do": "Use a friendly tone", "avoid": "", "why": "tone"},
    {"title": "Read back before booking", "when": "Before calling book_appointment",
     "do": "Read back the doctor, day and time and wait for a clear yes", "avoid": "Booking in the same turn as searching", "why": "consent"},
]


def fake_chat(self, system, messages, tools=None, max_tokens=1024):
    if self.role == "simulator":
        n = messages[-1]["content"].count("YOU (caller):")  # caller lines so far
        return Reply(SIM_LINES[min(n - 1, len(SIM_LINES) - 1)])
    if self.role == "judge":
        return Reply(json.dumps({"results": []}))
    if self.role == "reflector":
        fake_chat.reflections += 1
        rule = REFLECTIONS[min(fake_chat.reflections - 1, 1)]
        return Reply(json.dumps({"diagnosis": "d", "root_cause": "missing_rule", "change": {"op": "add", "rule_id": None, "rule": rule}}))
    # agent
    careful = "Read back before booking" in system
    last = messages[-1]
    users = sum(1 for m in messages if m["role"] == "user")
    if last["role"] == "user":
        if users == 1:
            return Reply("Sure, can I have your full name and date of birth?")
        if users == 2:
            return Reply("", [ToolCall("t1", "verify_patient", {"full_name": "Priya Nair", "date_of_birth": "1988-07-22", "caller_relationship": "self"})])
        if users == 3 and careful:
            slot = fake_chat.slot
            return Reply("", [ToolCall("t4", "book_appointment", {"patient_id": "P-1001", "slot_id": slot, "reason": "follow-up"})])
        return Reply("Goodbye!")
    result = json.loads(last["content"])
    if last["name"] == "verify_patient":
        return Reply("", [ToolCall("t2", "search_slots", {"appointment_type": "follow_up", "provider_id": "D-RAO", "date_from": "2026-10-07", "date_to": "2026-10-07"})])
    if last["name"] == "search_slots":
        fake_chat.slot = result["slots"][0]["slot_id"]
        if careful:
            return Reply(f"I have {result['slots'][0]['spoken']} with Dr. Rao. Shall I book it?")
        return Reply("", [ToolCall("t3", "book_appointment", {"patient_id": "P-1001", "slot_id": fake_chat.slot, "reason": "follow-up"})])
    if last["name"] == "book_appointment":
        return Reply(f"You're booked for {result['booked']['spoken']}.")
    return Reply("Okay.")


def test_loop_rejects_bad_rule_then_accepts_good_rule(monkeypatch, tmp_path):
    fake_chat.reflections = 0
    monkeypatch.setattr(llm_mod.LLM, "chat", fake_chat)
    out = improvement_loop([SCENARIO, HOLDOUT], Playbook(), tmp_path, iterations=3, trials=1, workers=1, log=lambda m: None)
    s = out["summary"]
    assert s["baseline"]["metrics"]["train"]["pass_rate"] == 0.0  # booked without read-back -> u_confirmed_before_commit fails
    first, second = s["iterations"][0], s["iterations"][1]
    assert first["cluster"]["check_id"] == "u_confirmed_before_commit"
    assert not first["decision"]["accepted"]  # "Be warm" doesn't fix it
    assert second["decision"]["accepted"]
    assert s["final"]["metrics"]["train"]["pass_rate"] == 1.0
    assert s["final"]["metrics"]["holdout"]["pass_rate"] == 1.0  # generalised to the held-out copy
    assert out["final_playbook"].version == 1 and out["final_playbook"].rules[0]["title"] == "Read back before booking"
    assert len(s["iterations"]) == 2  # stopped early: nothing left to fix
    report = render_report(s)
    assert "ACCEPTED" in report and "REJECTED" in report


def test_incomplete_baseline_stops_loop_without_conclusions(monkeypatch, tmp_path):
    def quota_exhausted(self, system, messages, tools=None, max_tokens=1024):
        if self.role == "agent":
            raise RuntimeError("agent: daily token/request quota exhausted")
        return fake_chat(self, system, messages, tools, max_tokens)

    fake_chat.reflections = 0
    monkeypatch.setattr(llm_mod.LLM, "chat", quota_exhausted)
    out = improvement_loop([SCENARIO, HOLDOUT], Playbook(), tmp_path, iterations=3, workers=1, log=lambda m: None)
    s = out["summary"]
    assert s["incomplete"] and "quota" in s["incomplete"]
    assert s["iterations"] == [] and fake_chat.reflections == 0  # never asked the reflector
    assert s["baseline"]["metrics"]["train"]["pass_rate"] is None  # "no data", not "0% pass"
    assert "Incomplete evaluation" in render_report(s)
