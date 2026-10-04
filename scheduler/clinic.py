"""In-memory clinic system of record: patients, providers, appointments, availability.

Everything that must never go wrong is enforced *here*, in code, not in the prompt:
age/specialty rules, opening hours, overlaps, the late-cancellation window. The
prompt can make the agent behave well; this module makes sure that even a badly
behaved agent cannot corrupt the schedule.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "clinic.json"
DAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
SLOT_STEP_MIN = 15


class ClinicError(Exception):
    """A business-rule violation. The message is shown to the agent verbatim."""


def parse_dt(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M")


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        raise ClinicError(f"Dates must be in YYYY-MM-DD format, got {value!r}.")


def normalize_name(name: str) -> str:
    name = re.sub(r"\b(mr|mrs|ms|miss|dr|master)\.?\s+", "", name.lower())
    name = re.sub(r"[^a-z\s]", "", name)
    return " ".join(name.split())


def spoken_dt(dt: datetime) -> str:
    """'Tuesday, 6 October at 10:30 AM' — the form a voice agent should read back."""
    return f"{dt.strftime('%A')}, {dt.day} {dt.strftime('%B')} at {dt.strftime('%I:%M %p').lstrip('0')}"


def _parse_hhmm(value: str):
    try:
        return datetime.strptime(value.strip(), "%H:%M").time()
    except (ValueError, AttributeError):
        raise ClinicError(f"Times must be HH:MM (24-hour), got {value!r}.")


def age_on(dob: date, on: date) -> int:
    return on.year - dob.year - ((on.month, on.day) < (dob.month, dob.day))


@dataclass
class Clinic:
    data: dict[str, Any]
    now: datetime
    events: list[dict[str, Any]] = field(default_factory=list)  # audit log of mutations

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, now: datetime, setup: dict[str, Any] | None = None, path: Path = DATA_PATH) -> "Clinic":
        data = json.loads(Path(path).read_text())
        setup = setup or {}
        data["patients"] += copy.deepcopy(setup.get("extra_patients", []))
        data["appointments"] += copy.deepcopy(setup.get("extra_appointments", []))
        data["busy_blocks"] += copy.deepcopy(setup.get("extra_busy_blocks", []))
        return cls(data=data, now=now)

    @property
    def info(self) -> dict[str, Any]:
        return self.data["clinic"]

    @property
    def appointment_types(self) -> dict[str, Any]:
        return self.data["appointment_types"]

    # ------------------------------------------------------------------ lookups
    def provider(self, provider_id: str) -> dict[str, Any]:
        for p in self.data["providers"]:
            if p["id"] == provider_id:
                return p
        # Be forgiving about names ("Dr. Asha Rao", "rao") — a common model mistake with no safety impact.
        key = normalize_name(provider_id or "")
        hits = [p for p in self.data["providers"] if key and (key in normalize_name(p["name"]) or normalize_name(p["name"]).endswith(key))]
        if len(hits) == 1:
            return hits[0]
        ids = ", ".join(f"{p['id']} ({p['name']})" for p in self.data["providers"])
        raise ClinicError(f"Unknown provider_id {provider_id!r}. Valid ids: {ids}.")

    def patient(self, patient_id: str) -> dict[str, Any]:
        for p in self.data["patients"]:
            if p["id"] == patient_id:
                return p
        raise ClinicError(f"Unknown patient_id {patient_id!r}.")

    def appointment(self, appointment_id: str) -> dict[str, Any]:
        for a in self.data["appointments"]:
            if a["id"] == appointment_id:
                return a
        raise ClinicError(f"Unknown appointment_id {appointment_id!r}.")

    def patient_age(self, patient_id: str) -> int:
        return age_on(parse_date(self.patient(patient_id)["date_of_birth"]), self.now.date())

    def find_patient(self, full_name: str, dob: date) -> dict[str, Any] | None:
        target = normalize_name(full_name)
        for p in self.data["patients"]:
            if normalize_name(p["full_name"]) == target and p["date_of_birth"] == dob.isoformat():
                return p
        return None

    def upcoming_appointments(self, patient_id: str) -> list[dict[str, Any]]:
        out = [
            a for a in self.data["appointments"]
            if a["patient_id"] == patient_id and a["status"] == "booked" and parse_dt(a["start"]) >= self.now
        ]
        return sorted(out, key=lambda a: a["start"])

    def describe_appointment(self, a: dict[str, Any]) -> dict[str, Any]:
        start = parse_dt(a["start"])
        return {
            "appointment_id": a["id"],
            "provider": self.provider(a["provider_id"])["name"],
            "type": self.appointment_types[a["type"]]["label"],
            "start": a["start"],
            "spoken": spoken_dt(start),
            "minutes": self.appointment_types[a["type"]]["minutes"],
        }

    # ------------------------------------------------------------------ availability
    def _busy_intervals(self, provider_id: str) -> list[tuple[datetime, datetime]]:
        out = [(parse_dt(b["start"]), parse_dt(b["end"])) for b in self.data["busy_blocks"] if b["provider_id"] == provider_id]
        for a in self.data["appointments"]:
            if a["provider_id"] == provider_id and a["status"] == "booked":
                s = parse_dt(a["start"])
                out.append((s, s + timedelta(minutes=self.appointment_types[a["type"]]["minutes"])))
        return out

    def slot_problem(self, provider_id: str, start: datetime, appt_type: str) -> str | None:
        """Return why a slot is not bookable, or None if it is."""
        prov = self.provider(provider_id)
        if appt_type not in self.appointment_types:
            return f"Unknown appointment type {appt_type!r}."
        if appt_type not in prov["appointment_types"]:
            return f"{prov['name']} does not offer {appt_type} appointments."
        if start <= self.now:
            return "That time is in the past."
        if start.date() > (self.now + timedelta(days=self.info["booking_horizon_days"])).date():
            return "That date is beyond the booking horizon."
        day = DAY_KEYS[start.weekday()]
        hours = self.info["hours"].get(day)
        if hours is None:
            return "The clinic is closed on that day."
        if day not in prov["days"]:
            return f"{prov['name']} does not work on {start.strftime('%A')}s."
        end = start + timedelta(minutes=self.appointment_types[appt_type]["minutes"])
        open_t = datetime.combine(start.date(), datetime.strptime(hours[0], "%H:%M").time())
        close_t = datetime.combine(start.date(), datetime.strptime(hours[1], "%H:%M").time())
        if start < open_t or end > close_t:
            return "That time is outside clinic hours."
        lb = self.info.get("lunch_break")
        if lb and day != "sat":
            ls = datetime.combine(start.date(), datetime.strptime(lb[0], "%H:%M").time())
            le = datetime.combine(start.date(), datetime.strptime(lb[1], "%H:%M").time())
            if start < le and end > ls:
                return "That time overlaps the lunch break."
        for bs, be in self._busy_intervals(provider_id):
            if start < be and end > bs:
                return "That slot is already taken."
        return None

    def available_slots(
        self,
        provider_ids: list[str],
        appt_type: str,
        date_from: date,
        date_to: date,
        part_of_day: str = "any",
        limit: int = 6,
        per_day: int = 3,
        not_before: str | None = None,
        not_after: str | None = None,
    ) -> list[dict[str, Any]]:
        """Earliest open slots, at most `per_day` per doctor per day. not_before/not_after are HH:MM start-time bounds."""
        lo = _parse_hhmm(not_before) if not_before else None
        hi = _parse_hhmm(not_after) if not_after else None
        if date_to < date_from:
            raise ClinicError("date_to is before date_from.")
        if (date_to - date_from).days > 31:
            raise ClinicError("Search at most 31 days at a time.")
        found: list[dict[str, Any]] = []
        day = date_from
        while day <= date_to and len(found) < limit:
            for pid in provider_ids:
                count = 0
                t = datetime.combine(day, datetime.min.time()).replace(hour=8)
                while t.hour < 18 and count < per_day and len(found) < limit:
                    in_part = part_of_day == "any" or (part_of_day == "morning" and t.hour < 12) or (
                        part_of_day == "afternoon" and t.hour >= 12
                    )
                    in_window = (lo is None or t.time() >= lo) and (hi is None or t.time() <= hi)
                    if in_part and in_window and self.slot_problem(pid, t, appt_type) is None:
                        found.append(self._slot(pid, t, appt_type))
                        count += 1
                    t += timedelta(minutes=SLOT_STEP_MIN)
            day += timedelta(days=1)
        return found

    def _slot(self, provider_id: str, start: datetime, appt_type: str) -> dict[str, Any]:
        prov = self.provider(provider_id)
        return {
            "slot_id": f"S-{provider_id[2:]}-{start.strftime('%Y%m%d%H%M')}-{appt_type}",
            "provider_id": provider_id,
            "provider": prov["name"],
            "type": appt_type,
            "start": start.strftime("%Y-%m-%dT%H:%M"),
            "spoken": spoken_dt(start),
        }

    @staticmethod
    def parse_slot_id(slot_id: str) -> tuple[str, datetime, str]:
        m = re.fullmatch(r"S-([A-Z]+)-(\d{12})-([a-z_]+)", slot_id or "")
        if not m:
            raise ClinicError(f"Malformed slot_id {slot_id!r}. Only use slot_ids returned by search_slots.")
        return f"D-{m.group(1)}", datetime.strptime(m.group(2), "%Y%m%d%H%M"), m.group(3)

    # ------------------------------------------------------------------ eligibility
    def check_eligibility(self, patient_id: str, provider_id: str) -> None:
        prov = self.provider(provider_id)
        age = self.patient_age(patient_id)
        if prov["min_age"] is not None and age < prov["min_age"]:
            raise ClinicError(f"{prov['name']} only sees patients aged {prov['min_age']}+; this patient is {age}. Use pediatrics.")
        if prov["max_age"] is not None and age > prov["max_age"]:
            raise ClinicError(f"{prov['name']} only sees patients up to age {prov['max_age']}; this patient is {age}.")

    # ------------------------------------------------------------------ mutations
    def _next_appt_id(self) -> str:
        nums = [int(a["id"].split("-")[1]) for a in self.data["appointments"]]
        return f"A-{max(nums, default=5000) + 1}"

    def book(self, patient_id: str, slot_id: str, reason: str) -> dict[str, Any]:
        provider_id, start, appt_type = self.parse_slot_id(slot_id)
        self.check_eligibility(patient_id, provider_id)
        problem = self.slot_problem(provider_id, start, appt_type)
        if problem:
            raise ClinicError(f"Cannot book: {problem}")
        for a in self.upcoming_appointments(patient_id):
            if a["provider_id"] == provider_id and a["start"][:10] == start.strftime("%Y-%m-%d"):
                raise ClinicError(
                    f"Patient already has an appointment with this provider that day ({a['id']} at {a['start'][11:]}). "
                    "Reschedule that one instead of double-booking."
                )
        appt = {
            "id": self._next_appt_id(),
            "patient_id": patient_id,
            "provider_id": provider_id,
            "start": start.strftime("%Y-%m-%dT%H:%M"),
            "type": appt_type,
            "reason": reason,
            "status": "booked",
        }
        self.data["appointments"].append(appt)
        self.events.append({"op": "book", "appointment_id": appt["id"], "patient_id": patient_id, "slot_id": slot_id})
        return appt

    def is_late_cancellation(self, appt: dict[str, Any]) -> bool:
        window = timedelta(hours=self.info["late_cancellation"]["window_hours"])
        return parse_dt(appt["start"]) - self.now < window

    def cancel(self, patient_id: str, appointment_id: str, late_fee_acknowledged: bool) -> dict[str, Any]:
        appt = self.appointment(appointment_id)
        if appt["patient_id"] != patient_id:
            # Deliberately indistinguishable from "does not exist": don't confirm other patients' records.
            raise ClinicError(f"Unknown appointment_id {appointment_id!r}.")
        if appt["status"] != "booked":
            raise ClinicError("That appointment is not active.")
        late = self.is_late_cancellation(appt)
        if late and not late_fee_acknowledged:
            fee = self.info["late_cancellation"]["fee_inr"]
            raise ClinicError(
                f"LATE_CANCELLATION: this appointment is less than {self.info['late_cancellation']['window_hours']} hours away, "
                f"so a INR {fee} late-cancellation fee applies. Tell the patient about the fee and only retry with "
                "late_fee_acknowledged=true if they explicitly agree."
            )
        appt["status"] = "cancelled"
        self.events.append({"op": "cancel", "appointment_id": appointment_id, "patient_id": patient_id, "late_fee": late})
        return appt

    def reschedule(self, patient_id: str, appointment_id: str, new_slot_id: str) -> dict[str, Any]:
        """Atomic: the new slot is secured before the old one is released."""
        old = self.appointment(appointment_id)
        if old["patient_id"] != patient_id or old["status"] != "booked":
            raise ClinicError(f"Unknown or inactive appointment_id {appointment_id!r}.")
        old["status"] = "rescheduling"  # free the patient's same-day rule + overlap for the swap
        try:
            new = self.book(patient_id, new_slot_id, old.get("reason", ""))
        except ClinicError:
            old["status"] = "booked"
            raise
        old["status"] = "rescheduled"
        old["replaced_by"] = new["id"]
        self.events[-1]["op"] = "reschedule_book"
        self.events.append({"op": "reschedule", "appointment_id": appointment_id, "new_appointment_id": new["id"], "patient_id": patient_id})
        return new

    def register(self, full_name: str, dob: date, phone: str) -> dict[str, Any]:
        if self.find_patient(full_name, dob):
            raise ClinicError("A patient with this name and date of birth already exists. Verify them instead.")
        if dob > self.now.date():
            raise ClinicError("Date of birth is in the future.")
        digits = re.sub(r"\D", "", phone or "")
        if len(digits) < 10:
            raise ClinicError("Phone number must have at least 10 digits.")
        nums = [int(p["id"].split("-")[1]) for p in self.data["patients"]]
        p = {"id": f"P-{max(nums) + 1}", "full_name": " ".join(full_name.split()).title(), "date_of_birth": dob.isoformat(), "phone": digits[-10:]}
        self.data["patients"].append(p)
        self.events.append({"op": "register", "patient_id": p["id"]})
        return p
