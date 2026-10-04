# Design note

**Principle: the prompt makes the agent behave well, and code makes sure a badly behaved agent can't do damage.** The tool layer blocks everything that must never happen: acting on an unverified record, booking a slot that was never searched, a wrong-age doctor, overlaps, closed days, a late cancel without fee consent, and a non-atomic reschedule. Everything else is behaviour, which the eval measures and the loop improves.

**Key choices**
- **Scoped tools.** Patient tools only accept a `patient_id` verified *in this call*. Booking only accepts slot ids that `search_slots` returned in this call. Errors come back as data, with a `retryable` flag. Verification never says which field mismatched, and it locks after 3 tries.
- **State lives in code.** Every turn, the prompt is re-rendered with an authoritative call-state block: who is verified, which slots were offered, and what was done. The model doesn't have to reconstruct this from the transcript.
- **Three prompt layers.** A human-owned core, a learned *playbook* of structured rules (`when/do/avoid/why` plus provenance), and the live state. The loop may only edit the playbook, so every learned change is small, diffable and revertible.
- **Layered rubric.** DB-state, tool-trace and deterministic transcript checks run first. The LLM judge only covers what code can't, sees only the spoken words, and must quote evidence verbatim. A transcript judge is blind to "said booked but didn't", "booked the wrong slot", "lied in a tool argument" and "committed before the yes". Each of those has a deterministic check. Live runs found another blind spot: the agent invented a caller identity inside a tool call. A check for that was added.

**Improvement loop** (`harness/loop.py`): run every scenario, cluster failing checks from the *train* split, and rank them by severity × frequency. The reflector returns **one** structured rule; a validator rejects rules containing names, dates, times or ids, so it can't memorise the test. The loop re-runs **every** scenario, including held-out ones the reflector never sees. The gate accepts only if the targeted check improved, nothing regressed, and the train score didn't drop. Suspected regressions are re-run on both playbooks before they're believed. Rejected attempts go back to the reflector with the reason.

**Before → after** (from [`reports/latest_loop_report.md`](reports/latest_loop_report.md)): _pending_

| Playbook | Train pass | Train score | Held-out pass | Held-out score | Critical failures |
|---|---|---|---|---|---|
| v0 (core prompt only) | – | – | – | – | – |
| final | – | – | – | – | – |

**One thing I'd change for a real clinic:** use replayed, de-identified real calls (with ASR n-best lists and timing) as the regression suite instead of an LLM caller. Ship each learned rule in shadow mode, with a human approving the playbook diff. The simulator shares the agent's blind spots: polite callers, no barge-in, perfect audio.

**Limits:** small suite, 1 trial per scenario, so one-run differences are mostly noise. The simulator can drift from its script. The read-back check is a regex heuristic. Audio, latency and interruptions aren't measured. Groq's free tier (200k tokens/day per model) limited how many scenarios each loop run could cover.

**AI help vs. my judgment.** An AI coding assistant did most of the scaffolding, schemas, provider adapters, scenario wording and tests. I overrode the obvious defaults in four places:
1. The loop can't rewrite the system prompt. It may only add small, validated rules.
2. Deterministic checks come first, ahead of an LLM judge.
3. Every scenario is re-run, including a held-out split, not just the failing ones.
4. Slot search originally couldn't express "after 2 PM", so I added time windows to the tool instead of loosening the scenario.
