# Agent Playground & Evaluation Dashboard

Streamlit front-end for the Assignment 4/5 agents. It drives the existing modules
(`agent.py`, `team.py`, `contracts.py`, `eval_runner.py`) rather than reimplementing them.

## Setup

```bash
cd lesson4
python3 -m pip install -r requirements-dashboard.txt   # adds streamlit + plotly
streamlit run dashboard/app.py
```

`OPENAI_API_KEY` is read from `lesson4/.env` (loaded by `agent.py` / `team.py`).
Without a key the Playground and Benchmark tabs are disabled; **Analytics still works**,
since it only reads Excel files.

Always launch from inside `lesson4/`. The dashboard resolves its own paths absolutely,
but the underlying modules still default to CWD-relative files.

## Layout

| File | Role |
|---|---|
| `app.py` | Entry point: sidebar, three tabs, session state |
| `runner_bridge.py` | Config → runners, trace translation, benchmark loop, xlsx writer |
| `ui_components.py` | Sidebar, trace renderers, metrics, task picker, charts |
| `analytics.py` | Excel discovery, A4/A5 schema normalization (imports no agent code) |

## Tabs

**Playground** prefills from `task_set.json` or takes a free query, runs the single agent,
the team, or both side by side, and paints the trace as it happens: tool calls with
arguments, tool results, routing decisions with the full handoff payload
(summary / constraints / facts / open question), worker turns, and the terminal state.

**Benchmark** selects tasks by category, by id, or all 35, runs them over the chosen
configurations with a progress bar and a live row table, then shows the same
`groupby(config, type)` metrics `eval_runner` produces. Results are written to
`benchmark_results/benchmark_<timestamp>.xlsx` after every task, so an interrupted run
keeps its rows.

**Analytics** loads `assignment_04.xlsx`, `assignment_05.xlsx`, any benchmark run, or an
uploaded workbook. The two historical schemas are normalized to one frame, so latency is
always in ms and rates are always 0–1. Three charts slice success rate, median latency,
and average tokens by task type, coloured per configuration.

## Things worth knowing

- **Nothing existing is overwritten.** Dashboard runs write only under `benchmark_results/`.
  `assignment_04.xlsx`, `assignment_05.xlsx`, `agent_execution_logs.jsonl` and
  `team_execution_traces.jsonl` are left alone.
- **The AGENTS.md editor is in-memory.** Edits apply to the next team run and never touch
  `AGENTS.md` on disk.
- **Team token counts are estimates.** `team.py` books a flat 150 per orchestrator turn and
  750 + 200 per tool call per worker turn. Latency and tool counts are real.
- **The single-agent timeout is checked between graph steps**, so one slow model call can
  overshoot it.
- **Loop detection fires often on multi-step team tasks.** The orchestrator only sees the
  payload facts, so it can re-dispatch the same worker and trip
  `loop_detected: repeating_agent_pair`. That is the pre-existing behaviour documented in
  `SUBMISSION_05.md`, not a dashboard fault.
- Clicking anything during a benchmark reruns the script and stops the loop at the next row
  boundary. Completed rows survive and can still be downloaded.

## Hooks added to the existing modules

Backward-compatible keyword arguments only; every default preserves the previous behaviour.

- `agent.py`: `temperature`, `system_prompt` on the constructor; `on_step` callback on `run_task`.
- `team.py`: `temperature`, `procedural_memory_text`, `max_turns`, `max_tokens`,
  `timeout_seconds`, `log_file`, `on_event`; the handoff payload is now included in the
  trace event, and `.dict()` was updated to `.model_dump()` for pydantic v2.
