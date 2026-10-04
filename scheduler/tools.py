"""Tools exposed to the agent, scoped to a single call (session).

Scoping decisions:
- Every patient-specific tool takes a patient_id, and that id must have been verified
  *in this call*. The model cannot act on a record it has not verified, even if it
  guesses or is told an id.
- book/reschedule only accept slot_ids that search_slots returned in this call, so the
  agent cannot invent availability.
- Errors come back as data ({"ok": false, "error": ..., "retryable": ...}) so the model
  can recover, and every call is written to `trace` for the evaluation harness.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .clinic import Clinic, ClinicError, parse_date

MAX_VERIFY_ATTEMPTS = 3
STATE_CHANGING = {"book_appointment", "cancel_appointment", "reschedule_appointment", "register_new_patient"}

PRIVACY_REFUSAL = (
    "PRIVACY: appointments can only be discussed or changed with the patient themself, or with a parent/guardian "
    "for a patient under 18. Do not confirm or deny whether this person is a patient here."
)

_patient_id = {"type": "string", "description": "A patient_id returned by verify_patient or register_new_patient in this call."}

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "verify_patient",
        "description": "Verify the identity of the patient the call is about, using full name and date of birth. Required before reading or changing any patient record. Never reveals which field mismatched.",
        "parameters": {
            "type": "object",
            "properties": {
                "full_name": {"type": "string", "description": "Patient's full name as spelled by the caller."},
                "date_of_birth": {"type": "string", "description": "YYYY-MM-DD"},
                "caller_relationship": {"type": "string", "enum": ["self", "parent_or_guardian", "other"], "description": "Who is calling relative to the patient."},
            },
            "required": ["full_name", "date_of_birth", "caller_relationship"],
        },
    },
    {
        "name": "register_new_patient",
        "description": "Create a record for a patient who has never visited the clinic. Only after verify_patient found no match and the caller confirms they are new.",
        "parameters": {
            "type": "object",
            "properties": {
                "full_name": {"type": "string"},
                "date_of_birth": {"type": "string", "description": "YYYY-MM-DD"},
                "phone": {"type": "string", "description": "10-digit mobile number"},
                "caller_relationship": {"type": "string", "enum": ["self", "parent_or_guardian"]},
            },
            "required": ["full_name", "date_of_birth", "phone", "caller_relationship"],
        },
    },
    {
        "name": "list_providers",
        "description": "List the clinic's doctors, their specialty, working days and the appointment types they offer.",
        "parameters": {"type": "object", "properties": {"specialty": {"type": "string", "enum": ["general_practice", "pediatrics", "dermatology"]}}},
    },
    {
        "name": "search_slots",
        "description": "Find open appointment slots, earliest first. Give either specialty or provider_id. Returns at most 6 slots (max 3 per day per doctor), each with a slot_id and a spoken label. Narrow the date range or time window to see later slots.",
        "parameters": {
            "type": "object",
            "properties": {
                "appointment_type": {"type": "string", "enum": ["new_patient", "follow_up", "annual_physical", "vaccination"]},
                "specialty": {"type": "string", "enum": ["general_practice", "pediatrics", "dermatology"]},
                "provider_id": {"type": "string"},
                "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                "date_to": {"type": "string", "description": "YYYY-MM-DD, at most 31 days after date_from"},
                "part_of_day": {"type": "string", "enum": ["any", "morning", "afternoon"], "description": "morning = before 12:00, afternoon = 12:00 or later"},
                "not_before": {"type": "string", "description": "Optional earliest start time, HH:MM 24-hour (e.g. '14:00' for 'after 2 PM')"},
                "not_after": {"type": "string", "description": "Optional latest start time, HH:MM 24-hour"},
            },
            "required": ["appointment_type", "date_from", "date_to"],
        },
    },
    {
        "name": "get_appointments",
        "description": "List a verified patient's upcoming appointments.",
        "parameters": {"type": "object", "properties": {"patient_id": _patient_id}, "required": ["patient_id"]},
    },
    {
        "name": "book_appointment",
        "description": "Book a slot previously returned by search_slots for a verified patient.",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": _patient_id,
                "slot_id": {"type": "string"},
                "reason": {"type": "string", "description": "Short non-diagnostic reason for visit, in the patient's words."},
            },
            "required": ["patient_id", "slot_id", "reason"],
        },
    },
    {
        "name": "cancel_appointment",
        "description": "Cancel a verified patient's appointment.",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": _patient_id,
                "appointment_id": {"type": "string"},
                "late_fee_acknowledged": {"type": "boolean", "description": "Set true only after the patient explicitly agreed to the late-cancellation fee."},
            },
            "required": ["patient_id", "appointment_id"],
        },
    },
    {
        "name": "reschedule_appointment",
        "description": "Atomically move a verified patient's appointment to a new slot from search_slots (the old slot is only released once the new one is secured).",
        "parameters": {
            "type": "object",
            "properties": {"patient_id": _patient_id, "appointment_id": {"type": "string"}, "new_slot_id": {"type": "string"}},
            "required": ["patient_id", "appointment_id", "new_slot_id"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Transfer the call to front-desk staff (routine/urgent) or the on-call nurse (emergency). Ends your part of the call.",
        "parameters": {
            "type": "object",
            "properties": {
                "urgency": {"type": "string", "enum": ["emergency", "urgent", "routine"]},
                "reason": {"type": "string"},
                "summary": {"type": "string", "description": "One-paragraph handoff summary for the human."},
            },
            "required": ["urgency", "reason", "summary"],
        },
    },
]


@dataclass
class Session:
    """Authoritative per-call state. The model sees a rendered snapshot of it every turn."""

    clinic: Clinic
    faults: dict[str, list[str]] = field(default_factory=dict)
    verified: dict[str, str] = field(default_factory=dict)  # patient_id -> caller_relationship
    failed_verifications: int = 0
    offered_slots: dict[str, dict[str, Any]] = field(default_factory=dict)
    escalation: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    turn: int = 0
    caller_said: list[str] = field(default_factory=list)  # everything the caller said, appended by the agent loop

    @property
    def locked(self) -> bool:
        return self.failed_verifications >= MAX_VERIFY_ATTEMPTS

    # ------------------------------------------------------------------ dispatch
    def call(self, name: str, args: dict[str, Any]) -> str:
        handler: Callable[..., dict[str, Any]] | None = getattr(self, f"_t_{name}", None)
        if handler is None:
            result = {"ok": False, "error": f"Unknown tool {name!r}.", "retryable": False}
        elif self.escalation is not None and name != "escalate_to_human":
            result = {"ok": False, "error": "The call has been transferred to a human; take no further actions.", "retryable": False}
        elif self.faults.get(name):
            kind = self.faults[name].pop(0)
            result = {"ok": False, "error": f"Scheduling system {kind}: the upstream EHR did not respond.", "retryable": True}
        else:
            try:
                result = {"ok": True, **handler(**args)}
            except ClinicError as e:
                result = {"ok": False, "error": str(e), "retryable": False}
            except TypeError as e:  # bad/missing arguments from the model
                result = {"ok": False, "error": f"Bad arguments: {e}", "retryable": False}
        self.trace.append({"turn": self.turn, "tool": name, "args": args, "result": result})
        return json.dumps(result)

    def _require_verified(self, patient_id: str) -> None:
        if patient_id not in self.verified:
            raise ClinicError("That patient_id has not been verified in this call. Call verify_patient first.")

    # ------------------------------------------------------------------ tools
    def _require_spoken(self, full_name: str) -> None:
        """A model can invent tool arguments ("John Doe") before the caller has said anything. If an invented
        identity ever matched a real patient it would expose their record, so names must come from the caller."""
        said = " ".join(self.caller_said).lower()
        letters = re.sub(r"[^a-z]", "", said)  # catches spelled-out names: "D-A-N-I-E-L"
        tokens = [w for w in re.sub(r"[^a-z ]", " ", (full_name or "").lower()).split() if len(w) > 2]
        if not tokens or not any(re.search(rf"\b{re.escape(w)}\b", said) or w in letters for w in tokens):
            raise ClinicError("The caller has not said this name. Ask the caller for their full name and date of birth; never guess or invent them.")

    def _t_verify_patient(self, full_name: str, date_of_birth: str, caller_relationship: str) -> dict[str, Any]:
        self._require_spoken(full_name)
        if self.locked:
            raise ClinicError("Verification is locked after 3 failed attempts. Do not ask for more guesses; offer a transfer to front-desk staff.")
        if caller_relationship not in ("self", "parent_or_guardian"):
            raise ClinicError(PRIVACY_REFUSAL)
        patient = self.clinic.find_patient(full_name, parse_date(date_of_birth))
        if patient is None:
            self.failed_verifications += 1
            raise ClinicError(
                f"No patient matches that name and date of birth (attempt {self.failed_verifications} of {MAX_VERIFY_ATTEMPTS}). "
                "Ask the caller to spell their name and repeat the date of birth, or ask if they are a new patient."
            )
        age = self.clinic.patient_age(patient["id"])
        if caller_relationship == "parent_or_guardian" and age >= 18:
            raise ClinicError(PRIVACY_REFUSAL)
        self.verified[patient["id"]] = caller_relationship
        return {
            "patient_id": patient["id"],
            "full_name": patient["full_name"],
            "age": age,
            "upcoming_appointments": len(self.clinic.upcoming_appointments(patient["id"])),
        }

    def _t_register_new_patient(self, full_name: str, date_of_birth: str, phone: str, caller_relationship: str = "self") -> dict[str, Any]:
        self._require_spoken(full_name)
        dob = parse_date(date_of_birth)
        p = self.clinic.register(full_name, dob, phone)
        self.verified[p["id"]] = caller_relationship
        return {"patient_id": p["id"], "full_name": p["full_name"], "age": self.clinic.patient_age(p["id"])}

    def _t_list_providers(self, specialty: str | None = None) -> dict[str, Any]:
        provs = [p for p in self.clinic.data["providers"] if specialty in (None, p["specialty"])]
        return {
            "providers": [
                {k: p[k] for k in ("id", "name", "specialty", "days", "appointment_types", "min_age", "max_age")} for p in provs
            ]
        }

    def _t_search_slots(
        self,
        appointment_type: str,
        date_from: str,
        date_to: str,
        specialty: str | None = None,
        provider_id: str | None = None,
        part_of_day: str = "any",
        not_before: str | None = None,
        not_after: str | None = None,
    ) -> dict[str, Any]:
        if provider_id and provider_id.strip():
            provider_ids = [self.clinic.provider(provider_id)["id"]]
        elif specialty:
            provider_ids = [p["id"] for p in self.clinic.data["providers"] if p["specialty"] == specialty]
        else:
            raise ClinicError("Give either specialty or provider_id.")
        provider_ids = [pid for pid in provider_ids if appointment_type in self.clinic.provider(pid)["appointment_types"]]
        if not provider_ids:
            raise ClinicError(f"No matching doctor offers {appointment_type} appointments.")
        slots = self.clinic.available_slots(
            provider_ids, appointment_type, parse_date(date_from), parse_date(date_to), part_of_day, not_before=not_before, not_after=not_after
        )
        for s in slots:
            self.offered_slots[s["slot_id"]] = s
        out: dict[str, Any] = {"slots": slots}
        if not slots:
            out["note"] = "No open slots in that range. Offer a different date range, part of day, or doctor."
        return out

    def _t_get_appointments(self, patient_id: str) -> dict[str, Any]:
        self._require_verified(patient_id)
        return {"appointments": [self.clinic.describe_appointment(a) for a in self.clinic.upcoming_appointments(patient_id)]}

    def _t_book_appointment(self, patient_id: str, slot_id: str, reason: str) -> dict[str, Any]:
        self._require_verified(patient_id)
        if slot_id not in self.offered_slots:
            raise ClinicError("That slot_id was not returned by search_slots in this call. Search first; never invent slots.")
        appt = self.clinic.book(patient_id, slot_id, reason)
        return {"booked": self.clinic.describe_appointment(appt)}

    def _t_cancel_appointment(self, patient_id: str, appointment_id: str, late_fee_acknowledged: bool = False) -> dict[str, Any]:
        self._require_verified(patient_id)
        appt = self.clinic.cancel(patient_id, appointment_id, late_fee_acknowledged)
        return {"cancelled": self.clinic.describe_appointment(appt)}

    def _t_reschedule_appointment(self, patient_id: str, appointment_id: str, new_slot_id: str) -> dict[str, Any]:
        self._require_verified(patient_id)
        if new_slot_id not in self.offered_slots:
            raise ClinicError("That slot_id was not returned by search_slots in this call. Search first; never invent slots.")
        new = self.clinic.reschedule(patient_id, appointment_id, new_slot_id)
        return {"rescheduled_to": self.clinic.describe_appointment(new), "old_appointment_id": appointment_id}

    def _t_escalate_to_human(self, urgency: str, reason: str, summary: str) -> dict[str, Any]:
        if urgency not in ("emergency", "urgent", "routine"):
            raise ClinicError("urgency must be emergency, urgent or routine.")
        self.escalation = {"urgency": urgency, "reason": reason, "summary": summary}
        if urgency == "emergency":
            msg = "Connecting the on-call nurse. Tell the caller to hang up and dial 112 now if they are in danger; keep it to one or two sentences."
        else:
            msg = "Transferring to front-desk staff. Tell the caller what happens next in one sentence."
        return {"transferred": True, "instruction": msg}
