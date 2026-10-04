"""Thin, provider-agnostic chat + tool-calling client for any OpenAI-compatible API
(OpenAI, Groq, OpenRouter, Ollama, or a custom LLM_BASE_URL).

Internal (neutral) message format, converted per provider at call time:
  {"role": "user", "content": str}
  {"role": "assistant", "content": str | None, "tool_calls": [{"id", "name", "args"}]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()

DEFAULTS = {
    #            agent / simulator                judge / reflector
    "openai": ("gpt-4.1-mini", "gpt-4.1"),
    "groq": ("openai/gpt-oss-120b", "openai/gpt-oss-120b"),
    "openrouter": ("openai/gpt-4.1-mini", "openai/gpt-4.1"),
    "ollama": ("qwen2.5:14b", "qwen2.5:14b"),
}
BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}
KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": None,
}


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


USAGE: dict[str, Usage] = {}


def provider_name() -> str:
    return os.getenv("LLM_PROVIDER", "openai").strip().lower()


def model_for(role: str) -> str:
    """role in {agent, simulator, judge, reflector}; overridable via AGENT_MODEL etc."""
    env = os.getenv(f"{role.upper()}_MODEL")
    if env:
        return env
    small, big = DEFAULTS.get(provider_name(), DEFAULTS["openai"])
    return small if role in ("agent", "simulator") else big


class LLM:
    def __init__(self, role: str, temperature: float = 0.0, model: str | None = None):
        self.role = role
        self.provider = provider_name()
        self.model = model or model_for(role)
        self.temperature = temperature
        self._client: Any = None

    # ------------------------------------------------------------------ client
    def client(self) -> Any:
        if self._client is not None:
            return self._client
        key_env = KEY_ENVS.get(self.provider, "LLM_API_KEY")
        api_key = os.getenv("LLM_API_KEY") or (os.getenv(key_env) if key_env else None)
        from openai import OpenAI

        base_url = os.getenv("LLM_BASE_URL") or BASE_URLS.get(self.provider)
        if not api_key and self.provider != "ollama":
            raise SystemExit(f"Set {key_env or 'LLM_API_KEY'} (or LLM_API_KEY) in .env — see .env.example")
        self._client = OpenAI(api_key=api_key or "ollama", base_url=base_url, max_retries=0)
        return self._client

    # ------------------------------------------------------------------ public
    def chat(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int = 1024) -> Reply:
        return self._with_retries(lambda: self._chat_once(system, messages, tools, max_tokens))

    def json(self, system: str, prompt: str, max_tokens: int = 2048) -> dict[str, Any]:
        """Single-turn call that must return a JSON object. Retries once on malformed output."""
        msgs: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        for attempt in range(2):
            reply = self.chat(system + "\n\nRespond with a single JSON object and nothing else.", msgs, max_tokens=max_tokens)
            parsed = extract_json(reply.text)
            if parsed is not None:
                return parsed
            msgs = msgs + [
                {"role": "assistant", "content": reply.text},
                {"role": "user", "content": "That was not valid JSON. Return only the JSON object."},
            ]
        raise ValueError(f"{self.role}: model did not return JSON: {reply.text[:300]}")

    # ------------------------------------------------------------------ internals
    def _with_retries(self, fn: Any, attempts: int = 10) -> Reply:
        delay = 2.0
        for i in range(attempts):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001 — provider SDKs raise many types
                status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
                if "per day" in str(e).lower():  # daily quota: waiting minutes won't help, fail loudly
                    raise RuntimeError(f"{self.role} ({self.model}): daily token/request quota exhausted — {str(e)[:300]}") from e
                transient = status in (408, 409, 429, 500, 502, 503, 504, 529) or "timeout" in type(e).__name__.lower() or "connection" in type(e).__name__.lower()
                if not transient or i == attempts - 1:
                    raise
                headers = getattr(getattr(e, "response", None), "headers", None) or {}
                try:
                    wait = float(headers.get("retry-after", delay))
                except (TypeError, ValueError):
                    wait = delay
                time.sleep(min(max(wait, delay), 65))
                delay = min(delay * 2, 30)
        raise RuntimeError("unreachable")

    def _track(self, inp: int, out: int) -> None:
        u = USAGE.setdefault(self.role, Usage())
        u.calls += 1
        u.input_tokens += inp or 0
        u.output_tokens += out or 0

    def _chat_once(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int) -> Reply:
        return self._openai(system, messages, tools, max_tokens)

    def _openai(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int) -> Reply:
        msgs: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "assistant":
                out: dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    out["tool_calls"] = [
                        {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
                        for tc in m["tool_calls"]
                    ]
                msgs.append(out)
            elif m["role"] == "tool":
                msgs.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                msgs.append({"role": "user", "content": m["content"]})
        kwargs: dict[str, Any] = {"model": self.model, "messages": msgs, "max_tokens": max_tokens, "temperature": self.temperature}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            kwargs["parallel_tool_calls"] = False  # one action at a time keeps the trace auditable
        if "gpt-oss" in self.model:  # reasoning models: keep reasoning short so it can't eat the answer budget
            kwargs["reasoning_effort"] = "low"
            kwargs["max_tokens"] = max(max_tokens, 1024)
        if self.provider == "openai":
            kwargs["seed"] = 7
            if re.match(r"^(o\d|gpt-5)", self.model):  # reasoning models: no temperature, different token param
                kwargs.pop("temperature")
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens") * 4
        resp = self.client().chat.completions.create(**kwargs)
        if resp.usage:
            self._track(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable_arguments": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))
        text = re.sub(r"<think>.*?(</think>|$)", "", msg.content or "", flags=re.S)  # qwen-style inline reasoning
        return Reply(text=text.strip(), tool_calls=calls)


def extract_json(text: str) -> dict[str, Any] | None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        val = json.loads(text)
        return val if isinstance(val, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            val = json.loads(text[start : end + 1])
            return val if isinstance(val, dict) else None
        except json.JSONDecodeError:
            return None
    return None
