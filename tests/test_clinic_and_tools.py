import json
from datetime import datetime

import pytest

from scheduler.clinic import Clinic, ClinicError
from scheduler.tools import Session

NOW = datetime(2026, 10, 5, 9, 30)


def session(**kw) -> Session:
    return Session(clinic=Clinic.load(NOW), **kw)


def call(s: Session, name: str, **args) -> dict:
    return json.loads(s.call(name, args))


def verified(s: Session, name="Priya Nair", dob="1988-07-22", rel="self") -> str:
    r = call(s, "verify_patient", full_name=name, date_of_birth=dob, caller_relationship=rel)
    assert r["ok"], r
    return r["patient_id"]


def first_slot(s: Session, **kw) -> str:
    args = {"appointment_type": "follow_up", "provider_id": "D-RAO", "date_from": "2026-10-07", "date_to": "2026-10-08"} | kw
    r = call(s, "search_slots", **args)
    assert r["ok"] and r["slots"], r
    return r["slots"][0]["slot_id"]


def test_cannot_touch_records_before_verification():
    s = session()
    r = call(s, "get_appointments", patient_id="P-1001")
    assert not r["ok"] and "not been verified" in r["error"]


def test_cannot_book_invented_slot():
    s = session()
    pid = verified(s)
    r = call(s, "book_appointment", patient_id=pid, slot_id="S-RAO-202610070900-follow_up", reason="x")
    assert not r["ok"] and "not returned by search_slots" in r["error"]


def test_book_happy_path_and_no_same_day_double_booking():
    s = session()
    pid = verified(s)
    slot = first_slot(s)
    assert call(s, "book_appointment", patient_id=pid, slot_id=slot, reason="results")["ok"]
    slot2 = first_slot(s, date_from="2026-10-07", date_to="2026-10-07")
    r = call(s, "book_appointment", patient_id=pid, slot_id=slot2, reason="again")
    assert not r["ok"] and "already has an appointment" in r["error"]


def test_name_normalisation_and_lockout():
    s = session()
    assert verified(s, name="  priya   NAIR ") == "P-1001"
    s2 = session()
    for _ in range(3):
        assert not call(s2, "verify_patient", full_name="Priya Nair", date_of_birth="1990-01-01", caller_relationship="self")["ok"]
    r = call(s2, "verify_patient", full_name="Priya Nair", date_of_birth="1988-07-22", caller_relationship="self")
    assert not r["ok"] and "locked" in r["error"]


def test_privacy_third_party_and_adult_guardian_refused():
    s = session()
    r = call(s, "verify_patient", full_name="Meera Kapoor", date_of_birth="1992-11-02", caller_relationship="other")
    assert not r["ok"] and "PRIVACY" in r["error"]
    r = call(s, "verify_patient", full_name="Meera Kapoor", date_of_birth="1992-11-02", caller_relationship="parent_or_guardian")
    assert not r["ok"] and "PRIVACY" in r["error"]
    assert verified(s, "Aarav Shah", "2018-05-09", "parent_or_guardian") == "P-1004"


def test_age_rules_route_children_to_pediatrics():
    s = session()
    pid = verified(s, "Aarav Shah", "2018-05-09", "parent_or_guardian")
    slot = first_slot(s)
    r = call(s, "book_appointment", patient_id=pid, slot_id=slot, reason="cough")
    assert not r["ok"] and "pediatrics" in r["error"]


def test_closed_days_lunch_and_hours():
    c = Clinic.load(NOW)
    assert "closed" in c.slot_problem("D-RAO", datetime(2026, 10, 11, 10, 0), "follow_up")
    assert "lunch" in c.slot_problem("D-RAO", datetime(2026, 10, 7, 13, 0), "follow_up")
    assert "hours" in c.slot_problem("D-RAO", datetime(2026, 10, 10, 13, 0), "follow_up")
    assert "work" in c.slot_problem("D-ORTIZ", datetime(2026, 10, 7, 10, 0), "follow_up")


def test_late_cancellation_requires_acknowledged_fee():
    s = session()
    pid = verified(s, "Daniel Fernandes", "1965-09-12")
    r = call(s, "cancel_appointment", patient_id=pid, appointment_id="A-5002")
    assert not r["ok"] and "LATE_CANCELLATION" in r["error"]
    assert call(s, "cancel_appointment", patient_id=pid, appointment_id="A-5002", late_fee_acknowledged=True)["ok"]


def test_cannot_cancel_someone_elses_appointment():
    s = session()
    pid = verified(s)
    r = call(s, "cancel_appointment", patient_id=pid, appointment_id="A-5001")
    assert not r["ok"] and "Unknown appointment_id" in r["error"]


def test_reschedule_is_atomic_and_rolls_back():
    s = session()
    pid = verified(s, "Rahul Verma", "1979-03-14")
    slot = first_slot(s, date_from="2026-10-09", date_to="2026-10-09", not_before="14:00")
    r = call(s, "reschedule_appointment", patient_id=pid, appointment_id="A-5003", new_slot_id=slot)
    assert r["ok"]
    assert s.clinic.appointment("A-5003")["status"] == "rescheduled"
    # Rollback: a failing new slot leaves the original booked.
    s2 = session()
    pid = verified(s2, "Rahul Verma", "1979-03-14")
    s2.offered_slots["S-RAO-202610111000-follow_up"] = {}  # Sunday: will fail at booking time
    r = call(s2, "reschedule_appointment", patient_id=pid, appointment_id="A-5003", new_slot_id="S-RAO-202610111000-follow_up")
    assert not r["ok"] and s2.clinic.appointment("A-5003")["status"] == "booked"


def test_fault_injection_is_retryable_then_recovers():
    s = session(faults={"search_slots": ["timeout"]})
    r = call(s, "search_slots", appointment_type="follow_up", provider_id="D-RAO", date_from="2026-10-07", date_to="2026-10-07")
    assert not r["ok"] and r["retryable"]
    assert call(s, "search_slots", appointment_type="follow_up", provider_id="D-RAO", date_from="2026-10-07", date_to="2026-10-07")["ok"]


def test_time_window_search():
    s = session()
    r = call(s, "search_slots", appointment_type="follow_up", provider_id="D-RAO", date_from="2026-10-09", date_to="2026-10-09", not_before="14:00")
    assert all(x["start"][11:] >= "14:00" for x in r["slots"])


def test_no_actions_after_transfer():
    s = session()
    assert call(s, "escalate_to_human", urgency="emergency", reason="chest pain", summary="x")["ok"]
    assert not call(s, "list_providers")["ok"]


def test_bad_dates_are_reported_not_raised():
    s = session()
    r = call(s, "verify_patient", full_name="Priya Nair", date_of_birth="22/07/1988", caller_relationship="self")
    assert not r["ok"] and "YYYY-MM-DD" in r["error"]
    with pytest.raises(ClinicError):
        Clinic.parse_slot_id("garbage")


def test_provider_resolves_by_name():
    c = Clinic.load(NOW)
    assert c.provider("Dr. Asha Rao")["id"] == "D-RAO" and c.provider("rao")["id"] == "D-RAO"
    with pytest.raises(ClinicError):
        c.provider("Dr. Nobody")
