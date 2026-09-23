# Pre-registration: single agent vs. team

Committed before the dry run (Phase 5). Sections 1 to 3 are filled by the author, and nothing in them is edited after the first benchmark token is spent. Section 4 is filled once, after the dry run and before the full matrix, and is then locked.

## 1. The wall

> Which wall is my single agent hitting? `tool overload` · `context bloat` · `serial bottleneck` · `team boundaries` · or "none, my single agent was fine"

**Wall sentence:**

_(author)_

## 2. Predicted outcome per task type

Success is the rate over 5 runs, e.g. `4/5`.

| Task type | Predicted winner (single / team / tie) | Predicted single success | Predicted team success | Why |
|---|---|---|---|---|
| single | | | | |
| cross_domain | | | | |
| misroute_bait | | | | |
| no_tool | | | | |
| handoff_stress | | | | |
| unanswerable | | | | |
| tool_fails | | | | |

## 3. Predicted coordination failures (team only, per 100 runs)

| Failure | Predicted count per 100 runs |
|---|---|
| loop_detected | |
| cap_breached (turns, tokens, wall-clock) | |
| over-dispatch on no_tool | |
| lost constraint on handoff_stress | |

## 4. Locked safety-net values (filled after the dry run, then never changed)

| Net | Value | Set from |
|---|---|---|
| Max agent turns (team) | | |
| Token budget per task (team) | | |
| Wall-clock timeout (team) | | |
| Loop rule | same worker pair twice in a row (`team.detect_loop`) | code |
| Max iterations (single) | | |
| Wall-clock timeout (single) | | |

---

## Drafting notes: evidence only, not predictions

These are facts from traces and code, for reference while filling sections 1 to 3.

**Current system (Haiku 4.5, real retriever), smoke runs, n=1 each:**

- **The orchestrator wrote the final answer itself.** In `traces/smoke/phase1_team.jsonl`, task `smoke_va`: seq 1 is a handoff to the researcher, and seq 5 is the researcher's worker_turn. At seq 6 the orchestrator emits `direct_answer`: it wrote the reply itself (used_last_worker_output false, 316 output tokens). The writer never ran, and the route is `['researcher']`.
- **Both configs refused without searching.** Task `p2_premium` is an unanswerable premium question. The single agent refused with 0 tool calls (`traces/smoke/phase2_single.jsonl`, summary seq 3). The team's researcher refused with 0 tool calls (`traces/smoke/phase2_team.jsonl`, worker_turn seq 4). The spec asks for a refusal after a bounded search.
- **The writer never ran.** In all 3 of 3 team runs that dispatched a worker, the orchestrator synthesized the reply itself through `direct_answer`, and the writer was never invoked. The runs are `smoke_va` (`phase1_team.jsonl` seq 6), `p2_bail` (`phase2_team.jsonl` seq 7) and `p2_premium` (`phase2_team.jsonl` seq 5). The route was `['researcher']` in each.
- **The orchestrator wrote its own instructions as constraints.** In `p2_bail` (`phase2_team.jsonl` seq 1), the handoff's `payload.constraints` holds `['Provide exact policy language where available', 'Identify coverage limits if applicable']`. The user asked for neither; the field meant for user constraints carried the orchestrator's own instructions.
- **The same question cost the team more.** For `p2_bail`, the team used 10,496 tokens (9,355 in, 1,141 out) and took 15.3 s. The single agent used 6,453 tokens (6,161 in, 292 out) and took 6.0 s. Sources are the summary lines in `phase2_team.jsonl` and `phase2_single.jsonl`.
- **The team's refusal asserted invented facts.** Its `p2_premium` final answer gives a phone number, a website and credit score as a rating factor, with 0 tool calls. Under the original rubric the judge scored this `no_claims`; the fix is in commit `a0b6229`, and the case is now judge sanity case s7.
- **The wall-clock nets differ.** The single agent's timeout is 30 s (`agent.py:35`, `LangGraphAgentRunner(timeout_seconds=30.0)`). The team's is 45 s (`team.py:106`, `MultiAgentTeam(timeout_seconds=45.0)`).

**Legacy system (gpt-4o-mini, stub search_docs, keyword scoring, hardcoded team tokens):** `team_execution_traces.jsonl`, 390 team runs from Sept 16. These are behaviour hints only; none of their metrics are valid.

- **Terminal states:** answered 280, `loop_detected` 108, cap_breached 2.
- **What the loop net actually caught:** all 108 `loop_detected` runs were the same worker dispatched twice in a row, never a ping-pong. The worker-only route ended researcher, researcher in 80 runs and analyst, analyst in 28.
- **The real ping-pong it missed:** t26 run 1 went researcher, analyst, researcher, analyst, four times over. It ended `cap_breached: max_turns_exceeded` because the old loop check compared a route with the orchestrator interleaved and never fired.
- **no_tool tasks** (t07, t08, t09, t33): 0 of 48 runs dispatched a worker.
