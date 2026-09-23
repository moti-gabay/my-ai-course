"""
runner_bridge.py - Adapter between the Streamlit UI and the existing agent modules.

Responsibilities:
  * carry sidebar configuration into LangGraphAgentRunner / MultiAgentTeam
  * translate their two different trace formats into one StepEvent shape
  * drive a benchmark subset over eval_runner's evaluators with a progress callback

Nothing here modifies assignment_04.xlsx / assignment_05.xlsx or the original
JSONL logs; dashboard runs write under lesson5/benchmark_results/.
"""

from __future__ import annotations

import sys
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

LESSON4_DIR = Path(__file__).resolve().parents[1]
if str(LESSON4_DIR) not in sys.path:
    sys.path.insert(0, str(LESSON4_DIR))

from agent import LangGraphAgentRunner, AGENT_SYSTEM_PROMPT  # noqa: E402
from team import MultiAgentTeam  # noqa: E402
from eval_runner import (  # noqa: E402
    load_task_set,
    evaluate_single_agent_run,
    evaluate_team_agent_run,
)

RESULTS_DIR = LESSON4_DIR / "benchmark_results"
TASK_SET_PATH = LESSON4_DIR / "task_set.json"
AGENTS_MD_PATH = LESSON4_DIR / "AGENTS.md"

StepEvent = Dict[str, Any]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    mode: str = "single"                 # "single" | "team" | "both"
    model_name: str = "claude-haiku-4-5"
    temperature: float = 0.0
    # single agent
    max_iterations: int = 10
    timeout_seconds: float = 30.0
    system_prompt: str = AGENT_SYSTEM_PROMPT
    # team
    max_turns: int = 8
    max_tokens: int = 12000
    team_timeout_seconds: float = 45.0
    procedural_memory_text: str = ""

    def modes(self) -> List[str]:
        return ["single", "team"] if self.mode == "both" else [self.mode]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_agents_md() -> str:
    return AGENTS_MD_PATH.read_text(encoding="utf-8") if AGENTS_MD_PATH.exists() else ""


def load_tasks() -> List[Dict[str, Any]]:
    return load_task_set(str(TASK_SET_PATH))


def _ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    return RESULTS_DIR


def build_single_runner(cfg: RunConfig) -> LangGraphAgentRunner:
    return LangGraphAgentRunner(
        model_name=cfg.model_name,
        max_iterations=cfg.max_iterations,
        timeout_seconds=cfg.timeout_seconds,
        log_file=str(_ensure_results_dir() / "dashboard_agent_logs.jsonl"),
        temperature=cfg.temperature,
        system_prompt=cfg.system_prompt,
    )


def build_team_runner(cfg: RunConfig, on_event: Optional[Callable[[Dict], None]] = None) -> MultiAgentTeam:
    return MultiAgentTeam(
        model_name=cfg.model_name,
        temperature=cfg.temperature,
        procedural_memory_text=cfg.procedural_memory_text,
        max_turns=cfg.max_turns,
        max_tokens=cfg.max_tokens,
        timeout_seconds=cfg.team_timeout_seconds,
        log_file=str(_ensure_results_dir() / "dashboard_team_traces.jsonl"),
        on_event=on_event,
    )


# ---------------------------------------------------------------------------
# Trace translation: two source formats -> one StepEvent
#   kind: tool_call | tool_result | handoff | worker_done | summary | final | error
# ---------------------------------------------------------------------------

def _event(kind: str, agent: str, title: str, body: Any, data: Any, seq: int, t0: float) -> StepEvent:
    return {
        "kind": kind,
        "agent": agent,
        "title": title,
        "body": None if body is None else str(body),
        "data": data,
        "seq": seq,
        "t_offset_s": round(time.time() - t0, 2),
    }


def single_step_to_event(raw: Dict[str, Any], seq: int, t0: float) -> Optional[StepEvent]:
    step = raw.get("step")
    if step == "tool_call":
        return _event("tool_call", "single_agent", f"call {raw.get('tool')}", None, raw.get("args"), seq, t0)
    if step == "tool_result":
        return _event("tool_result", "single_agent", f"{raw.get('tool')} returned", raw.get("content"), None, seq, t0)
    if step == "final":
        ok = raw.get("status") == "success"
        return _event(
            "final" if ok else "error",
            "single_agent",
            "Final answer" if ok else f"Stopped: {raw.get('status')}",
            raw.get("content") or raw.get("error_message"),
            {"status": raw.get("status")},
            seq, t0,
        )
    return None


def team_event_to_event(raw: Dict[str, Any], seq: int, t0: float) -> Optional[StepEvent]:
    if raw.get("event_type") == "summary":
        return _event(
            "summary", "team", "Run summary", None,
            {
                "route_history": raw.get("route_history", []),
                "terminal_state": raw.get("terminal_state"),
                "total_tokens": raw.get("total_tokens"),
                "total_turns": raw.get("total_turns"),
                "duration_sec": raw.get("duration_sec"),
            },
            seq, t0,
        )

    event = raw.get("event")
    if event == "handoff_decision":
        return _event(
            "handoff", "orchestrator",
            f"orchestrator -> {raw.get('destination')}",
            raw.get("reason"),
            {
                "destination": raw.get("destination"),
                "payload": raw.get("handoff_payload") or {},
                "duration_ms": raw.get("duration_ms"),
            },
            seq, t0,
        )
    if event == "worker_turn_complete":
        return _event(
            "worker_done", raw.get("agent", "worker"),
            f"{raw.get('agent')} finished turn",
            raw.get("result_preview"),
            {"tool_calls": raw.get("tool_calls", 0), "duration_ms": raw.get("duration_ms")},
            seq, t0,
        )
    return None


# ---------------------------------------------------------------------------
# Live single runs (playground)
# ---------------------------------------------------------------------------

def _playground_task(query: str, task_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if task_meta:
        return {**task_meta, "task": query}
    return {"task_id": "playground", "task": query, "type": "playground", "answerable": True}


class _EventCollector:
    """Collects StepEvents and forwards them to the UI, never raising into the runner."""

    def __init__(self, mapper, sink: Optional[Callable[[StepEvent], None]], t0: float):
        self._mapper = mapper
        self._sink = sink
        self._t0 = t0
        self._seq = 0
        self.events: List[StepEvent] = []

    def __call__(self, raw: Dict[str, Any]) -> None:
        try:
            self._seq += 1
            ev = self._mapper(raw, self._seq, self._t0)
            if ev is None:
                return
            self.events.append(ev)
            if self._sink:
                self._sink(ev)
        except Exception:
            # A raising callback inside agent.run_task would be misreported as an
            # execution_error, so UI failures must never escape.
            pass

    def add(self, ev: StepEvent) -> None:
        self._seq += 1
        ev["seq"] = self._seq
        self.events.append(ev)
        if self._sink:
            try:
                self._sink(ev)
            except Exception:
                pass


def run_single_live(
    cfg: RunConfig,
    query: str,
    on_event: Optional[Callable[[StepEvent], None]] = None,
    task_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    collector = _EventCollector(single_step_to_event, on_event, t0)
    runner = build_single_runner(cfg)
    try:
        result = runner.run_task(_playground_task(query, task_meta), run_number=1, on_step=collector)
    except Exception as exc:
        result = {
            "status": "execution_error",
            "error_message": str(exc),
            "final_answer": f"ERROR: {exc}",
            "latency_seconds": round(time.time() - t0, 3),
            "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "tool_calls_count": 0, "used_tools": [], "is_refused": True,
            "trace_steps": [],
        }
        collector.add(_event("error", "single_agent", "Execution failed", str(exc), None, 0, t0))
    return {"result": result, "events": collector.events}


def run_team_live(
    cfg: RunConfig,
    query: str,
    on_event: Optional[Callable[[StepEvent], None]] = None,
    task_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    collector = _EventCollector(team_event_to_event, on_event, t0)
    task_id = (task_meta or {}).get("task_id", "playground")
    team = build_team_runner(cfg, on_event=collector)
    try:
        result = team.run_task(task_id=task_id, user_query=query, run_num=1)
        # The team loop emits no terminal event of its own.
        collector.add(_event(
            "final", (result.get("route_history") or ["team"])[-1],
            "Final answer", result.get("final_answer"),
            {"terminal_state": result.get("terminal_state")}, 0, t0,
        ))
    except Exception as exc:
        result = {
            "task_id": task_id,
            "final_answer": f"ERROR: {exc}",
            "terminal_state": "error",
            "route_history": [],
            "worker_turns": 0,
            "total_tokens": 0,
            "latency_seconds": round(time.time() - t0, 3),
            "is_refused": True,
        }
        collector.add(_event("error", "team", "Execution failed", str(exc), None, 0, t0))
    return {"result": result, "events": collector.events}


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def make_error_row(task: Dict[str, Any], run_num: int, config: str, exc: Exception, start_t: float) -> Dict[str, Any]:
    """Mirrors the error row shape in eval_runner.evaluate_single_agent_run."""
    agent_key = "single_agent" if config == "single" else "team"
    return {
        "task_id": task["task_id"],
        "task": task["task"],
        "type": task.get("type", "single"),
        "answerable": task.get("answerable", True),
        "success_criteria": task.get("success_criteria", ""),
        "capable_agents": json.dumps(task.get("capable_agents", [])),
        "config": config,
        "run": run_num,
        "answer": f"ERROR: {exc}",
        "success": False,
        "refused": True,
        "terminal_state": "error",
        "route": "[]",
        "agent_turns": 0,
        "tool_calls": 0,
        "routing_correct": False,
        "handoff_correct": False,
        "per_agent_success": json.dumps({agent_key: False}),
        "faithfulness": "Low",
        "latency_ms": round((time.time() - start_t) * 1000, 2),
        "input_tokens": 0,
        "output_tokens": 0,
        "breach_reason": str(exc),
    }


def safe_eval_single(agent_instance, task: Dict[str, Any], run_num: int) -> Dict[str, Any]:
    start_t = time.time()
    try:
        return evaluate_single_agent_run(agent_instance, task, run_num)
    except Exception as exc:
        return make_error_row(task, run_num, "single", exc, start_t)


def safe_eval_team(team_instance, task: Dict[str, Any], run_num: int) -> Dict[str, Any]:
    """evaluate_team_agent_run has no try/except of its own; one bad run must not
    abort the whole matrix."""
    start_t = time.time()
    try:
        return evaluate_team_agent_run(team_instance, task, run_num)
    except Exception as exc:
        return make_error_row(task, run_num, "team", exc, start_t)


def summarize_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """The exact aggregation eval_runner writes to the Sliced_Summary sheet."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return (
        df.groupby(["config", "type"])
        .agg(
            total_runs=("run", "count"),
            success_rate=("success", "mean"),
            refusal_rate=("refused", "mean"),
            latency_p50=("latency_ms", "median"),
            avg_tokens=("input_tokens", "mean"),
            avg_turns=("agent_turns", "mean"),
        )
        .reset_index()
    )


def write_benchmark_xlsx(rows: List[Dict[str, Any]], path: Path) -> Path:
    """Same sheet names as eval_runner, so the file re-opens in the Analytics tab."""
    df = pd.DataFrame(rows)
    path.parent.mkdir(exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Raw_Execution_Logs", index=False)
        summarize_rows(rows).to_excel(writer, sheet_name="Sliced_Summary", index=False)
    return path


def new_benchmark_path() -> Path:
    return _ensure_results_dir() / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"


def run_benchmark_live(
    cfg: RunConfig,
    tasks: List[Dict[str, Any]],
    runs_per_task: int,
    configs: List[str],
    on_row: Callable[[Dict[str, Any], int, int], None],
    output_path: Path,
) -> Path:
    """Task -> config -> run, same nesting as eval_runner.run_benchmark.
    Flushes the workbook after every task so an interrupted run keeps its rows."""
    single_agent = build_single_runner(cfg) if "single" in configs else None
    team_agent = build_team_runner(cfg) if "team" in configs else None

    rows: List[Dict[str, Any]] = []
    total = len(tasks) * len(configs) * runs_per_task
    done = 0

    for task in tasks:
        for config in configs:
            for run_num in range(1, runs_per_task + 1):
                if config == "single":
                    row = safe_eval_single(single_agent, task, run_num)
                else:
                    row = safe_eval_team(team_agent, task, run_num)
                rows.append(row)
                done += 1
                on_row(row, done, total)
        write_benchmark_xlsx(rows, output_path)

    return output_path
