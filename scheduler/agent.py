"""The scheduling agent: an LLM tool-use loop over a Session."""

from __future__ import annotations

from typing import Any

from .llm import LLM
from .playbook import Playbook
from .prompts import build_system_prompt
from .tools import TOOL_SPECS, Session

GREETING = "Thank you for calling Sunrise Family Clinic, this is Maya. How can I help you today?"
MAX_TOOL_ROUNDS = 6


class Agent:
    def __init__(self, session: Session, playbook: Playbook, llm: LLM | None = None):
        self.session = session
        self.playbook = playbook
        self.llm = llm or LLM("agent", temperature=0.0)
        self.messages: list[dict[str, Any]] = [{"role": "assistant", "content": GREETING}]

    def greeting(self) -> str:
        return GREETING

    def respond(self, caller_text: str) -> str:
        """Handle one caller utterance; returns everything the agent says aloud this turn."""
        self.session.turn += 1
        self.session.caller_said.append(caller_text)
        self.messages.append({"role": "user", "content": caller_text})
        spoken: list[str] = []
        for _ in range(MAX_TOOL_ROUNDS):
            system = build_system_prompt(self.session, self.playbook)  # re-rendered: state may have changed
            reply = self.llm.chat(system, self.messages, tools=TOOL_SPECS)
            if reply.text:
                spoken.append(reply.text)
            if not reply.tool_calls:
                self.messages.append({"role": "assistant", "content": reply.text})
                break
            self.messages.append(
                {"role": "assistant", "content": reply.text or None, "tool_calls": [tc.__dict__ for tc in reply.tool_calls]}
            )
            for tc in reply.tool_calls:
                result = self.session.call(tc.name, tc.args)
                self.messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": result})
        else:
            # The model kept calling tools without ever answering: fail safe to a human.
            if self.session.escalation is None:
                self.session.call("escalate_to_human", {"urgency": "routine", "reason": "agent tool loop limit", "summary": caller_text})
            fallback = "I'm sorry, I'm having trouble with our system. Let me transfer you to a colleague at the front desk."
            spoken.append(fallback)
            self.messages.append({"role": "assistant", "content": fallback})
        text = " ".join(spoken).strip()
        return text or "Sorry, could you say that again?"
