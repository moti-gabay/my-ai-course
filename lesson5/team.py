"""
team.py - Multi-Agent System Core Architecture (Assignment 5)
Contains: Orchestrator, Workers, 4 Safety Nets (with Loop Detector), and JSONL Tracer.
"""

import os
import time
import json
import inspect
from typing import Dict, Any, List, Optional, Callable, Tuple
from pathlib import Path
from dotenv import load_dotenv

# load environment variables
load_dotenv(Path(__file__).with_name(".env"))

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# Import Contracts & Tools
from contracts import (
    AgentName, HandoffPayload, Handoff, TeamState, AGENT_SCOPE_CONTRACTS
)
from tools import ALL_TOOLS, search_docs, calculator, read_policy_page
import retriever
from tracing import UnitOfWorkTracer, RunTrace, ReactOutcome, run_traced_react

DEFAULT_TEAM_LOG = str(Path(__file__).with_name("traces") / "team_traces.jsonl")
WORKERS = (AgentName.RESEARCHER.value, AgentName.ANALYST.value, AgentName.WRITER.value)


# ---------------------------------------------------------------------------
# 1. Safety Nets
# ---------------------------------------------------------------------------

def detect_loop(workers: List[str]) -> bool:
    """The same worker pair twice in a row: R,A,R,A (ping-pong) or R,R,R,R.

    Compares the worker-only route. The orchestrator sits between every worker turn,
    so a route that includes it never repeats a pair and the check never fires.
    """
    return len(workers) >= 4 and workers[-2:] == workers[-4:-2]


class SafetyNetChecker:
    def __init__(
        self,
        max_turns: int = 8,
        max_tokens: int = 12000,
        timeout_seconds: float = 45.0
    ):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def budget_breach(self, tokens_used: int, start_time: float) -> Optional[str]:
        """Token budget and wall-clock nets. Checked before every orchestrator call
        and after every step inside a worker."""
        if tokens_used >= self.max_tokens:
            return "token_budget"
        if (time.time() - start_time) >= self.timeout_seconds:
            return "wall_clock_timeout"
        return None

    def dispatch_breach(self, workers: List[str], next_worker: str) -> Optional[Tuple[str, str]]:
        """Turn cap and loop nets, checked before a dispatch runs.
        Returns (terminal_state, breach_reason)."""
        if len(workers) >= self.max_turns:
            return "cap_breached", "max_turns"
        if detect_loop(workers + [next_worker]):
            return "loop_detected", "repeating_agent_pair"
        return None


# ---------------------------------------------------------------------------
# 2. Helpers for Dynamic LangGraph Version Compatibility
# ---------------------------------------------------------------------------

def create_agent_compat(llm, tools, prompt_text):
    sig = inspect.signature(create_react_agent)
    kwargs = {}
    if "prompt" in sig.parameters:
        kwargs["prompt"] = prompt_text
    elif "state_modifier" in sig.parameters:
        kwargs["state_modifier"] = prompt_text
    elif "messages_modifier" in sig.parameters:
        kwargs["messages_modifier"] = prompt_text

    return create_react_agent(model=llm, tools=tools, **kwargs)


# ---------------------------------------------------------------------------
# 3. Multi-Agent Team Core Implementation
# ---------------------------------------------------------------------------

class MultiAgentTeam:
    def __init__(
        self,
        model_name: str = "claude-haiku-4-5",
        procedural_memory_path: str = "AGENTS.md",
        temperature: float = 0.0,
        procedural_memory_text: Optional[str] = None,
        max_turns: int = 8,
        max_tokens: int = 12000,
        timeout_seconds: float = 45.0,
        log_file: str = DEFAULT_TEAM_LOG,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.llm = ChatAnthropic(model=model_name, temperature=temperature, timeout=30.0, max_tokens=2048)
        # Load embeddings, index and reranker now so model loading never counts as run latency.
        retriever.warm_up()
        self.tracer = UnitOfWorkTracer(log_file=log_file, on_event=on_event)
        self.safety_nets = SafetyNetChecker(
            max_turns=max_turns,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds
        )

        # Load procedural memory: explicit text wins over the file on disk
        self.procedural_memory = ""
        if procedural_memory_text is not None:
            self.procedural_memory = procedural_memory_text
        elif os.path.exists(procedural_memory_path):
            with open(procedural_memory_path, "r", encoding="utf-8") as f:
                self.procedural_memory = f.read()

        # Build Workers
        res_scope = AGENT_SCOPE_CONTRACTS[AgentName.RESEARCHER]
        res_prompt = f"{res_scope['scope']}\n\nHouse Rules:\n{self.procedural_memory}"
        self.researcher_agent = create_agent_compat(
            self.llm, [search_docs, read_policy_page], res_prompt
        )

        ana_scope = AGENT_SCOPE_CONTRACTS[AgentName.ANALYST]
        ana_prompt = f"{ana_scope['scope']}\n\nHouse Rules:\n{self.procedural_memory}"
        self.analyst_agent = create_agent_compat(
            self.llm, [calculator], ana_prompt
        )

        wri_scope = AGENT_SCOPE_CONTRACTS[AgentName.WRITER]
        self.writer_prompt = f"{wri_scope['scope']}\n\nHouse Rules:\n{self.procedural_memory}"

    def orchestrator_step(self, state: TeamState, run_num: int = 1) -> Tuple[Optional[Handoff], Dict[str, int], Optional[str], Any]:
        """
        Orchestrator node: Classifies, routes, or directly answers no_tool queries.
        Returns (handoff or None, token usage, error message or None, raw tool-call args for the trace).
        """
        user_query = state["user_query"]
        last_handoff = state.get("handoff_data") or {}
        facts_data = last_handoff.get("facts", {}) if isinstance(last_handoff, dict) else {}

        orch_system_prompt = f"""You are the Master Orchestrator for an insurance assistant team.
Analyze the request and decide the single best next action.

Available Workers:
1. 'researcher': Retrieves policy text, clauses, deductibles, and limits.
2. 'analyst': Performs exact mathematical calculations, VAT additions, and financial evaluation.
3. 'writer': Formats final output when facts are ready.
4. 'orchestrator': Select this ONLY if you can answer directly with NO worker turns.

Current Payload Facts: {json.dumps(facts_data, ensure_ascii=False)}

CRITICAL ROUTING RULES:
- If 'researcher' has already retrieved the facts AND 'analyst' has calculated the math, route to 'writer' or finish IMMEDIATELY.
- NEVER route back to 'researcher' if policy facts are already present in Payload Facts!
- Do NOT loop infinitely between workers."""

        router_llm = self.llm.with_structured_output(Handoff, method="function_calling", include_raw=True)
        messages = [
            SystemMessage(content=orch_system_prompt),
            HumanMessage(content=f"User Query: {user_query}")
        ]

        try:
            res = router_llm.invoke(messages)
        except Exception as e:
            return None, {}, f"{type(e).__name__}: {e}", None
        raw = res.get("raw")
        usage = (raw.usage_metadata or {}) if raw is not None else {}
        raw_args = [tc["args"] for tc in raw.tool_calls] if raw is not None else None
        if res.get("parsing_error") or res.get("parsed") is None:
            return None, usage, f"orchestrator output did not parse: {res.get('parsing_error')}", raw_args
        return res["parsed"], usage, None, raw_args

    def _run_writer(self, query: str, rt: RunTrace) -> ReactOutcome:
        out = ReactOutcome()
        t0 = time.time()
        try:
            msg = self.llm.invoke([SystemMessage(content=self.writer_prompt), HumanMessage(content=query)])
        except Exception as e:
            out.error = f"{type(e).__name__}: {e}"
            return out
        usage = msg.usage_metadata or {}
        out.text = msg.text
        out.input_tokens, out.output_tokens = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        rt.account(AgentName.WRITER.value, input_tokens=out.input_tokens, output_tokens=out.output_tokens)
        rt.emit(AgentName.WRITER.value, "llm_call", owner=AgentName.WRITER.value,
                duration_ms=round((time.time() - t0) * 1000, 1),
                input_tokens=out.input_tokens, output_tokens=out.output_tokens,
                requested_tools=[], text=out.text)
        return out

    def run_task(self, task_id: str, user_query: str, run_num: int = 1) -> Dict[str, Any]:
        """
        Main Execution Loop for Multi-Agent Task Orchestration
        """
        start_time = time.time()
        rt = self.tracer.start_run(task_id, run_num)
        orch = AgentName.ORCHESTRATOR.value

        def tokens_used() -> int:
            t = rt.totals()
            return t["input_tokens"] + t["output_tokens"]

        # Initialize State
        state: TeamState = {
            "messages": [{"role": "user", "content": user_query}],
            "task_id": task_id,
            "user_query": user_query,
            "last_active": orch,
            "handoff_data": None,
            "agent_turns_count": 0,
            "route_history": [orch],
            "is_terminal": False,
            "terminal_reason": None
        }

        workers: List[str] = []           # ordered worker route, the loop net reads this
        final_answer, last_worker_output = "", ""
        terminal_state, breach_reason = None, None

        while terminal_state is None:
            breach = self.safety_nets.budget_breach(tokens_used(), start_time)
            if breach:
                terminal_state, breach_reason = "cap_breached", breach
                rt.emit(orch, "net_breach", owner=orch, reason=breach)
                break

            # 1. Orchestrator Dispatch Step
            t0 = time.time()
            handoff, usage, err, raw_args = self.orchestrator_step(state, run_num)
            duration = round((time.time() - t0) * 1000, 1)
            in_t, out_t = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
            rt.account(orch, input_tokens=in_t, output_tokens=out_t, duration_ms=duration, turns=1)
            if err:
                terminal_state, breach_reason = "error", err
                rt.emit(orch, "error", owner=orch, error=err, raw_output=raw_args, duration_ms=duration,
                        input_tokens=in_t, output_tokens=out_t)
                break

            if handoff.destination == orch:
                direct = (handoff.direct_answer or "").strip()
                final_answer = direct or last_worker_output
                rt.emit(orch, "direct_answer", owner=orch, reason=handoff.reason,
                        direct_answer=handoff.direct_answer, used_last_worker_output=not direct,
                        duration_ms=duration, input_tokens=in_t, output_tokens=out_t)
                if final_answer:
                    terminal_state = "answered"
                else:
                    terminal_state, breach_reason = "error", "orchestrator finished without an answer"
                break

            dest = handoff.destination
            net = self.safety_nets.dispatch_breach(workers, dest)
            rt.emit(orch, "handoff", owner=orch if net else dest,
                    **{"from": workers[-1] if workers else orch, "to": dest},
                    reason=handoff.reason, payload_keys=[k for k, v in handoff.payload.model_dump().items() if v],
                    payload=handoff.payload.model_dump(), duration_ms=duration,
                    input_tokens=in_t, output_tokens=out_t)
            if net:
                terminal_state, breach_reason = net
                rt.emit(orch, "net_breach", owner=orch, reason=breach_reason,
                        route=workers + [dest])
                break

            state["last_active"] = dest
            state["handoff_data"] = handoff.payload.model_dump()
            state["route_history"].append(dest)
            state["agent_turns_count"] += 1
            workers.append(dest)

            # 2. Worker Execution Steps
            payload = HandoffPayload(**(state["handoff_data"] or {}))
            worker_query = payload.open_question or user_query

            if payload.constraints:
                worker_query += f"\n[Constraints: {', '.join(payload.constraints)}]"
            if payload.facts:
                worker_query += f"\n[Established Facts: {json.dumps(payload.facts)}]"

            t0 = time.time()

            def check() -> Optional[str]:
                return self.safety_nets.budget_breach(tokens_used(), start_time)

            if dest == AgentName.RESEARCHER.value:
                outcome = run_traced_react(self.researcher_agent, worker_query, rt, dest, check=check)
            elif dest == AgentName.ANALYST.value:
                outcome = run_traced_react(self.analyst_agent, worker_query, rt, dest, check=check)
            else:
                outcome = self._run_writer(worker_query, rt)

            duration = round((time.time() - t0) * 1000, 1)
            rt.account(dest, duration_ms=duration, turns=1)
            rt.emit(dest, "worker_turn", owner=dest, received_payload=payload.model_dump(),
                    query=worker_query, output=outcome.text, tool_calls=len(outcome.tool_outputs),
                    input_tokens=outcome.input_tokens, output_tokens=outcome.output_tokens,
                    duration_ms=duration, breach=outcome.breach, error=outcome.error)

            if outcome.error:
                terminal_state, breach_reason = "error", outcome.error
                break
            if outcome.breach:
                terminal_state, breach_reason = "cap_breached", outcome.breach
                rt.emit(dest, "net_breach", owner=dest, reason=outcome.breach)
                break

            last_worker_output = outcome.text
            updated_facts = payload.facts
            updated_facts[dest] = outcome.text

            state["handoff_data"] = HandoffPayload(
                summary=f"{dest} completed step.",
                constraints=payload.constraints,
                facts=updated_facts,
                open_question=f"Synthesize or evaluate step after {dest}"
            ).model_dump()

            if dest == AgentName.WRITER.value:
                final_answer = outcome.text
                terminal_state = "answered"
            else:
                state["last_active"] = orch
                state["route_history"].append(orch)

        if terminal_state in ("cap_breached", "loop_detected"):
            final_answer = f"Could not complete the request: {breach_reason} limit reached."
        elif terminal_state == "error" and not final_answer:
            final_answer = f"ERROR: {breach_reason}"

        state["is_terminal"], state["terminal_reason"] = True, terminal_state
        total_duration = time.time() - start_time
        totals = rt.totals()

        rt.summary(
            terminal_state=terminal_state,
            breach_reason=breach_reason,
            route=workers,
            route_history=state["route_history"],
            agent_turns=len(workers),
            input_tokens=totals["input_tokens"],
            output_tokens=totals["output_tokens"],
            total_tokens=totals["input_tokens"] + totals["output_tokens"],
            tool_calls=totals["tool_calls"],
            duration_ms=round(total_duration * 1000, 1),
            final_answer=final_answer,
        )

        refusal_keywords = ["cannot answer", "missing", "unanswerable", "refuse", "not mentioned", "cap_breached"]
        is_refused = any(kw in final_answer.lower() for kw in refusal_keywords)

        return {
            "task_id": task_id,
            "final_answer": final_answer,
            "terminal_state": terminal_state,
            "breach_reason": breach_reason,
            "route": workers,
            "route_history": state["route_history"],
            "worker_turns": len(workers),
            "input_tokens": totals["input_tokens"],
            "output_tokens": totals["output_tokens"],
            "total_tokens": totals["input_tokens"] + totals["output_tokens"],
            "tool_calls": totals["tool_calls"],
            "per_agent": rt.per_agent,
            "latency_seconds": round(total_duration, 3),
            "is_refused": is_refused
        }


if __name__ == "__main__":
    print("🧪 Dry-running MultiAgentTeam...")
    team = MultiAgentTeam()
    res = team.run_task(task_id="smoke", user_query="What are the minimum liability limits required by Virginia law under the auto policy?")
    print("✅ Result:", json.dumps(res, indent=2, ensure_ascii=False))
