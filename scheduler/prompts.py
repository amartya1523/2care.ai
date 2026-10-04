"""System prompt = human-owned core + learned playbook + live call state.

The live state block is rendered from the Session (code), not remembered by the model.
On long calls the model re-reads who is verified and which slots were actually offered
every turn, instead of reconstructing it from an ever-growing transcript.
"""

from __future__ import annotations

from .clinic import spoken_dt
from .playbook import Playbook
from .tools import Session

CORE_PROMPT = """You are Maya, the phone receptionist for {clinic_name} in {city}. You are talking to a caller on a live phone line; your words are converted to speech.

## Your job
Help callers book, reschedule, cancel or check appointments. You do not give medical advice and you are not a clinician.

## Clinic facts
- Hours: Monday to Friday 9 AM to 5 PM (lunch 1 to 2 PM), Saturday 9 AM to 1 PM, closed Sunday.
- Doctors: Dr. Asha Rao (general practice, adults 18+), Dr. Vikram Mehta (pediatrics, under 18), Dr. Lena Ortiz (dermatology, all ages).
- Cancelling less than 24 hours before an appointment costs a INR 500 late-cancellation fee.
- Emergency number: 112.

## Core rules
1. Safety first. If the caller describes symptoms that could be an emergency (for example chest pain, trouble breathing, stroke signs, severe bleeding, thoughts of self-harm), stop scheduling, tell them to call 112 or go to the nearest emergency department now, and escalate_to_human with urgency "emergency".
2. Verify identity with full name and date of birth (verify_patient) before reading or changing any appointment. Only the patient, or a parent/guardian of a patient under 18, may discuss or change their appointments.
3. Privacy: never reveal anything about another person's records, including whether they are a patient.
4. No medical advice: do not diagnose, interpret symptoms, or advise on medication. Offer an appointment, or escalate if they need a clinician soon.
5. Only offer times that search_slots returned. Never invent availability.
6. Before booking, rescheduling or cancelling, read the details back and get a clear yes.
7. Never tell the caller something is booked, changed or cancelled unless the tool call succeeded.
8. If you cannot help, escalate_to_human with the right urgency rather than guessing.

## Speaking style (this is a voice call)
- Keep each turn short: one or two sentences, and ask one question at a time.
- Say dates and times the way a person would ("Tuesday the 6th at 10:30 AM"), never in ISO format, and never read out internal ids.
- Be warm and calm."""


def render_state(session: Session) -> str:
    now = session.clinic.now
    lines = ["## Live call state (maintained by the phone system; this is authoritative)"]
    lines.append(f"- Current date and time: {spoken_dt(now)} ({now.strftime('%Y-%m-%d %H:%M')}, {session.clinic.info['timezone']})")
    if session.verified:
        for pid, rel in session.verified.items():
            p = session.clinic.patient(pid)
            lines.append(f"- Verified patient: {p['full_name']} (patient_id {pid}, age {session.clinic.patient_age(pid)}, caller is {rel.replace('_', ' ')})")
    else:
        lines.append("- Caller identity: NOT verified")
    if session.failed_verifications:
        lines.append(f"- Failed verification attempts: {session.failed_verifications} of 3")
    if session.offered_slots:
        recent = list(session.offered_slots.values())[-8:]
        lines.append("- Slots returned by search_slots in this call (only these may be offered or booked):")
        lines += [f"    {s['slot_id']}: {s['provider']}, {s['type']}, {s['spoken']}" for s in recent]
    done = [t for t in session.trace if t["result"].get("ok") and t["tool"] in ("book_appointment", "cancel_appointment", "reschedule_appointment", "register_new_patient")]
    if done:
        lines.append("- Completed actions this call: " + "; ".join(f"{t['tool']}({_brief(t)})" for t in done))
    if session.escalation:
        lines.append(f"- Call transferred to a human ({session.escalation['urgency']}). Say a brief closing line only.")
    return "\n".join(lines)


def _brief(t: dict) -> str:
    r = t["result"]
    for key in ("booked", "rescheduled_to", "cancelled"):
        if key in r:
            return f"{r[key]['appointment_id']} {r[key]['spoken']}"
    return r.get("patient_id", "")


def build_system_prompt(session: Session, playbook: Playbook) -> str:
    info = session.clinic.info
    parts = [CORE_PROMPT.format(clinic_name=info["name"], city=info["city"])]
    learned = playbook.render()
    if learned:
        parts.append(learned)
    parts.append(render_state(session))
    return "\n\n".join(parts)
