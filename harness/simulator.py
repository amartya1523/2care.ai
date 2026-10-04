"""LLM-driven caller that plays a scenario persona against the agent."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from scheduler.llm import LLM

END = "<END_CALL>"

SIM_SYSTEM = """You are role-playing a CALLER on the phone with a medical clinic's receptionist. You are helping test the receptionist, so behave like the real person described below — not like a helpful assistant.

# Today
It is {today} (local time). "Today", "tomorrow", "this Friday" etc. are relative to this date.

# Who you are and what you want
{persona}

# How to play
- Talk like a real person on the phone: one or two short, casual sentences per turn.
- Give personal details (date of birth, phone number, etc.) only when asked, and exactly as written above. Never invent different details. If asked something your persona doesn't cover, give a plausible short answer that doesn't change the facts above.
- Follow any scripted moments in your persona exactly when they say to.
- If the receptionist gets your details or request wrong, correct them.
- You don't know anything about the clinic's internal systems, ids or tools.
- When the call is over (your goal is done, you were told to hang up or call emergency services, you were transferred, or you've said goodbye), say your last line and then append {end}. If you have nothing more to say, reply with just {end}.
- Never break character and never mention being an AI or a test.
- Write ONLY your own next line as the caller. Never write the receptionist's lines, never write both sides, never continue the conversation past your own turn.
- Stick to your persona's preferences. If the receptionist offers options that don't match them, say so and ask for what you want."""


class CallerSimulator:
    def __init__(self, persona: str, opening: str, llm: LLM | None = None, now: datetime | None = None):
        self.persona = persona
        self.opening = opening
        # Without the date, the caller invents its own "today" and argues with a correct agent.
        self.today = (now or datetime(2026, 10, 5, 9, 30)).strftime("%A, %d %B %Y, %I:%M %p")
        self.llm = llm or LLM("simulator", temperature=0.3)

    def next(self, transcript: list[dict[str, Any]]) -> tuple[str, bool]:
        """transcript: [{"speaker": "agent"|"caller", "text": str}]. Returns (utterance, call_ended)."""
        if not any(t["speaker"] == "caller" for t in transcript):
            return self.opening, False
        # The call is rendered as one prompt rather than role-flipped chat turns: small models
        # role-play much more faithfully when asked for "the caller's next line" explicitly.
        convo = "\n".join(f"{'RECEPTIONIST' if t['speaker'] == 'agent' else 'YOU (caller)'}: {t['text']}" for t in transcript)
        prompt = f"The call so far:\n{convo}\n\nWrite only YOUR next line as the caller (or {END} if the call is over). No speaker label."
        reply = self.llm.chat(SIM_SYSTEM.format(persona=self.persona.strip(), end=END, today=self.today), [{"role": "user", "content": prompt}], max_tokens=200)
        before_end, has_end, _ = reply.text.strip().partition(END)
        kept = strip_role_bleed(before_end)
        # Honour END only if nothing was cut: an END that followed bled-in turns belongs to a call that never happened.
        return kept, bool(has_end) and kept == before_end.strip()


_GLUED = re.compile(r"(?<=[.!?])(?=[A-Z])")  # "please.Sure, just a sec" — turns glued without a space
_SPEAKER = re.compile(r"\n?\s*(receptionist|agent|maya|assistant|you \(caller\))\s*:", re.I)


def strip_role_bleed(text: str) -> str:
    """Small simulator models sometimes write the receptionist's reply (and their own next turn)
    into one message. Keep only the caller's first turn; otherwise the agent gets graded on a
    conversation it never had."""
    text = _SPEAKER.split(text)[0]
    text = _GLUED.split(text)[0]
    return re.sub(r"^\s*(caller|patient|me|you)\s*:\s*", "", text, flags=re.I).strip()
