"""
tracing.py - unit-of-work JSONL tracing shared by the single agent and the team.

Every event line carries task_id, run, seq, agent, event and owner. Token counts come
from each AIMessage's usage_metadata and durations from wall-clock, never from constants.

Event types:
  llm_call     one model response: input/output tokens, duration, tool calls requested
  tool_call    one tool execution: full input and full output
  handoff      orchestrator dispatch: from, to, reason, payload_keys, payload
  direct_answer  orchestrator answered without dispatching
  worker_turn  one worker turn: payload received, full output, tokens, duration
  net_breach   a safety net fired: reason
  summary      one per run (also marked event_type="summary")
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class UnitOfWorkTracer:
    def __init__(self, log_file: str, on_event: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.log_file = log_file
        self.on_event = on_event
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    def write(self, entry: Dict[str, Any]) -> None:
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        if self.on_event:
            self.on_event(entry)

    def start_run(self, task_id: str, run: int) -> "RunTrace":
        return RunTrace(self, task_id, run)


class RunTrace:
    """Sequence numbering and per-agent accounting for one run."""

    def __init__(self, tracer: UnitOfWorkTracer, task_id: str, run: int):
        self.tracer, self.task_id, self.run = tracer, task_id, run
        self.seq = 0
        self.per_agent: Dict[str, Dict[str, float]] = {}

    def emit(self, agent: str, event: str, owner: str, **fields: Any) -> Dict[str, Any]:
        self.seq += 1
        entry = {"task_id": self.task_id, "run": self.run, "seq": self.seq,
                 "agent": agent, "event": event, "owner": owner, **fields}
        self.tracer.write(entry)
        return entry

    def account(self, agent: str, input_tokens: int = 0, output_tokens: int = 0,
                duration_ms: float = 0.0, turns: int = 0, tool_calls: int = 0) -> None:
        a = self.per_agent.setdefault(agent, {"turns": 0, "input_tokens": 0, "output_tokens": 0,
                                              "duration_ms": 0.0, "tool_calls": 0})
        a["turns"] += turns
        a["input_tokens"] += input_tokens
        a["output_tokens"] += output_tokens
        a["duration_ms"] = round(a["duration_ms"] + duration_ms, 1)
        a["tool_calls"] += tool_calls

    def totals(self) -> Dict[str, int]:
        return {k: int(sum(a[k] for a in self.per_agent.values()))
                for k in ("input_tokens", "output_tokens", "tool_calls")}

    def summary(self, **fields: Any) -> Dict[str, Any]:
        entry = {"event_type": "summary", "event": "summary", "task_id": self.task_id,
                 "run": self.run, "seq": self.seq + 1, "per_agent": self.per_agent, **fields}
        self.tracer.write(entry)
        return entry


@dataclass
class ReactOutcome:
    text: str = ""
    tool_outputs: List[Dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    breach: Optional[str] = None       # set when `check` stopped the run
    error: Optional[str] = None        # exception text; the run never raises


def run_traced_react(graph, query: str, rt: RunTrace, agent: str,
                     check: Optional[Callable[[], Optional[str]]] = None,
                     config: Optional[Dict[str, Any]] = None) -> ReactOutcome:
    """Stream a create_react_agent graph, tracing every model call and tool call.

    `check` runs after every graph step and returns a breach reason to stop early;
    it sees tokens through rt.account(), which is updated before it is called.
    Tool durations are per tools-node step: parallel calls in one step share it.
    """
    out = ReactOutcome()
    requested: Dict[str, Dict[str, Any]] = {}
    last = time.time()
    try:
        for update in graph.stream({"messages": [HumanMessage(content=query)]},
                                   config=config or {}, stream_mode="updates"):
            now = time.time()
            step_ms = round((now - last) * 1000, 1)
            last = now
            tool_msgs = [m for delta in update.values() if delta
                         for m in delta.get("messages", []) if isinstance(m, ToolMessage)]
            for delta in update.values():
                for m in (delta or {}).get("messages", []):
                    if isinstance(m, AIMessage):
                        usage = m.usage_metadata or {}
                        in_t, out_t = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                        out.input_tokens += in_t
                        out.output_tokens += out_t
                        rt.account(agent, input_tokens=in_t, output_tokens=out_t)
                        rt.emit(agent, "llm_call", owner=agent, duration_ms=step_ms,
                                input_tokens=in_t, output_tokens=out_t,
                                requested_tools=[{"name": tc["name"], "args": tc["args"]} for tc in m.tool_calls],
                                text=m.text)
                        for tc in m.tool_calls:
                            requested[tc["id"]] = tc
                        if not m.tool_calls:
                            out.text = m.text
                    elif isinstance(m, ToolMessage):
                        tc = requested.pop(m.tool_call_id, {})
                        output = m.text if isinstance(m.content, list) else str(m.content)
                        out.tool_outputs.append({"tool": m.name, "input": tc.get("args"), "output": output})
                        rt.account(agent, tool_calls=1)
                        rt.emit(agent, "tool_call", owner=agent, tool=m.name, input=tc.get("args"),
                                output=output, duration_ms=step_ms, parallel_calls=len(tool_msgs))
            if check:
                out.breach = check()
                if out.breach:
                    return out
    except Exception as e:  # recursion limit, API error, tool bug: recorded, never raised
        out.error = f"{type(e).__name__}: {e}"
    return out
