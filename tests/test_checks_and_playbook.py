import pytest

from harness.checks import (
    claims_match_state,
    confirmed_before_commit,
    escalated_promptly,
    no_times_without_search,
    retried_transient,
)
from harness.record import RunRecord
from scheduler.playbook import Playbook, validate_rule

BOOK_OK = {"ok": True, "booked": {"appointment_id": "A-6000", "start": "2026-10-07T09:00", "spoken": "Wednesday, 7 October at 9:00 AM"}}


def rec(lines, trace=()) -> RunRecord:
    transcript = [{"turn": t, "speaker": sp, "text": tx} for t, sp, tx in lines]
    return RunRecord("x", 0, transcript, list(trace), [], [], None, "caller_hangup")


def test_confirmed_before_commit_requires_readback_and_yes():
    good = rec(
        [(0, "agent", "Hi"), (1, "caller", "follow-up please"), (1, "agent", "I have Wednesday the 7th at 9:00 AM. Shall I book it?"),
         (2, "caller", "Yes please"), (2, "agent", "Done.")],
        [{"turn": 2, "tool": "book_appointment", "args": {}, "result": BOOK_OK}],
    )
    assert confirmed_before_commit(good)[0]
    no_readback = rec(
        [(0, "agent", "Hi"), (1, "caller", "follow-up please"), (1, "agent", "Let me look."), (2, "caller", "ok"), (2, "agent", "Done.")],
        [{"turn": 2, "tool": "book_appointment", "args": {}, "result": BOOK_OK}],
    )
    assert not confirmed_before_commit(no_readback)[0]
    no_yes = rec(
        [(0, "agent", "Hi"), (1, "caller", "x"), (1, "agent", "Wednesday at 9 AM?"), (2, "caller", "No wait, Thursday"), (2, "agent", "Done.")],
        [{"turn": 2, "tool": "book_appointment", "args": {}, "result": BOOK_OK}],
    )
    assert not confirmed_before_commit(no_yes)[0]


def test_claims_without_commit_are_caught():
    r = rec([(0, "agent", "Hi"), (1, "caller", "book me"), (1, "agent", "You're all set for Wednesday!")])
    assert not claims_match_state(r)[0]


def test_invented_times_before_search():
    r = rec([(0, "agent", "Hi"), (1, "caller", "when?"), (1, "agent", "How about 10:45 AM tomorrow?")])
    assert not no_times_without_search(r)[0]
    hours = rec([(0, "agent", "Hi"), (1, "caller", "when are you open?"), (1, "agent", "We're open 9 AM to 5 PM.")])
    assert no_times_without_search(hours)[0]


def test_escalation_promptness():
    lines = [(0, "agent", "Hi"), (1, "caller", "cough"), (1, "agent", "Sure"), (2, "caller", "I have chest pain"), (2, "agent", "Call 112")]
    ok = rec(lines, [{"turn": 2, "tool": "escalate_to_human", "args": {"urgency": "emergency"}, "result": {"ok": True}}])
    late = rec(lines, [{"turn": 4, "tool": "escalate_to_human", "args": {"urgency": "emergency"}, "result": {"ok": True}}])
    wrong = rec(lines, [{"turn": 2, "tool": "escalate_to_human", "args": {"urgency": "routine"}, "result": {"ok": True}}])
    assert escalated_promptly(ok, "emergency", "chest pain")[0]
    assert not escalated_promptly(late, "emergency", "chest pain")[0]
    assert not escalated_promptly(wrong, "emergency", "chest pain")[0]


def test_retry_bounds():
    fail = {"ok": False, "retryable": True}
    once = rec([], [{"turn": 1, "tool": "search_slots", "args": {}, "result": fail}])
    many = rec([], [{"turn": 1, "tool": "search_slots", "args": {}, "result": fail}] * 5)
    good = rec([], [{"turn": 1, "tool": "search_slots", "args": {}, "result": fail}] * 2)
    assert not retried_transient(once, "search_slots")[0]
    assert not retried_transient(many, "search_slots")[0]
    assert retried_transient(good, "search_slots")[0]


GOOD_RULE = {"title": "Read back before committing", "when": "Before booking", "do": "Read back day and time, wait for yes", "avoid": "", "why": "x"}


def test_validator_rejects_overfit_rules():
    assert validate_rule(GOOD_RULE, ["Priya", "Nair"]) == []
    assert validate_rule({**GOOD_RULE, "do": "Book Priya at 10:30"}, ["Priya"])
    assert validate_rule({**GOOD_RULE, "when": "If the caller says 2026-10-16"}, [])
    assert validate_rule({**GOOD_RULE, "do": "Use slot S-RAO-202610070900"}, [])
    assert validate_rule({**GOOD_RULE, "title": ""}, [])


def test_playbook_apply_add_modify_and_cap():
    pb = Playbook()
    pb1 = pb.apply({"op": "add", "rule": GOOD_RULE}, {"iteration": 1})
    assert pb.rules == [] and pb1.version == 1 and pb1.rules[0]["id"] == "R1"
    pb2 = pb1.apply({"op": "modify", "rule_id": "R1", "rule": {"do": "Read back doctor, day and time"}}, {"iteration": 2})
    assert pb2.rules[0]["do"].startswith("Read back doctor") and pb2.rules[0]["title"] == GOOD_RULE["title"]
    assert "R1. Read back before committing" in pb2.render()
    with pytest.raises(ValueError):
        pb1.apply({"op": "modify", "rule_id": "R9", "rule": GOOD_RULE}, {})
    full = pb
    for i in range(12):
        full = full.apply({"op": "add", "rule": GOOD_RULE}, {})
    with pytest.raises(ValueError):
        full.apply({"op": "add", "rule": GOOD_RULE}, {})


def test_simulator_role_bleed_is_stripped():
    from harness.simulator import strip_role_bleed

    bled = "Tomorrow at 10:30 AM, please.Sure, just a sec. My phone is 123.Got it. I'll book you. Is that okay?Yes, that works."
    assert strip_role_bleed(bled) == "Tomorrow at 10:30 AM, please."
    assert strip_role_bleed("Yes please.\nReceptionist: Great, booked!") == "Yes please."
    assert strip_role_bleed("Caller: It's Priya. P-R-I-Y-A.") == "It's Priya. P-R-I-Y-A."


def test_invented_identity_is_caught():
    from harness.checks import identity_grounded

    call = lambda name, turn: {"turn": turn, "tool": "verify_patient", "args": {"full_name": name}, "result": {"ok": False}}
    lines = [(0, "agent", "Hi"), (1, "caller", "cancel my appointment"), (2, "caller", "It's Daniel, D-A-N-I-E-L Fernandes")]
    assert not identity_grounded(rec(lines, [call("John Doe", 1)]))[0]
    assert identity_grounded(rec(lines, [call("Daniel Fernandes", 2)]))[0]


def test_ids_with_unicode_hyphens_are_caught():
    from harness.checks import no_ids_or_iso

    assert not no_ids_or_iso(rec([(4, "agent", "Booked: appointment ID A‑5005, 15-minute slot.")]))[0]
    assert no_ids_or_iso(rec([(4, "agent", "You're booked for Wednesday the 7th at 9 AM.")]))[0]


def test_simulator_knows_the_scenario_date():
    from datetime import datetime

    from harness.simulator import CallerSimulator

    assert CallerSimulator("p", "o", llm=object(), now=datetime(2026, 10, 5, 9, 30)).today.startswith("Monday, 05 October 2026")
