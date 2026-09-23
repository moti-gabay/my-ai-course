"""
team.py - Multi-Agent System Core Architecture (Assignment 5)
Contains: Orchestrator, Workers, 4 Safety Nets (with Loop Detector), and JSONL Tracer.
"""

import os
import time
import json
import inspect
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from dotenv import load_dotenv

# load environment variables
load_dotenv(Path(__file__).with_name(".env"))

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END

# Import Contracts & Tools
from contracts import (
    AgentName, HandoffPayload, Handoff, TeamState, AGENT_SCOPE_CONTRACTS
)
from tools import ALL_TOOLS, search_docs, calculator, read_policy_page

# ---------------------------------------------------------------------------
# 1. JSONL Unit-of-Work Tracer
# ---------------------------------------------------------------------------

class UnitOfWorkTracer:
    def __init__(
        self,
        log_file: str = "team_execution_traces.jsonl",
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.log_file = log_file
        self.on_event = on_event

    def log_event(self, task_id: str, run_num: int, seq: int, agent: str, 
                  event: str, owner: str, duration_ms: float, 
                  input_tokens: int = 0, output_tokens: int = 0, **extra):
        log_entry = {
            "task_id": task_id,
            "run": run_num,
            "seq": seq,
            "agent": agent,
            "event": event,
            "owner": owner,
            "duration_ms": round(duration_ms, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            **extra
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        if self.on_event:
            self.on_event(log_entry)

    def log_summary(self, task_id: str, run_num: int, terminal_state: str, 
                    total_turns: int, total_duration_sec: float, 
                    total_tokens: int, route_history: List[str]):
        summary_entry = {
            "event_type": "summary",
            "task_id": task_id,
            "run": run_num,
            "terminal_state": terminal_state,
            "total_turns": total_turns,
            "duration_sec": round(total_duration_sec, 3),
            "total_tokens": total_tokens,
            "route_history": route_history
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary_entry, ensure_ascii=False) + "\n")
        if self.on_event:
            self.on_event(summary_entry)


# ---------------------------------------------------------------------------
# 2. Safety Nets Checker
# ---------------------------------------------------------------------------

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

    def check_breaches(
        self, 
        state: TeamState, 
        start_time: float, 
        accumulated_tokens: int
    ) -> Optional[str]:
        # 1. Max Agent Turns Net
        if state.get("agent_turns_count", 0) >= self.max_turns:
            return "cap_breached: max_turns_exceeded"

        # 2. Token Budget Net
        if accumulated_tokens >= self.max_tokens:
            return "cap_breached: token_budget_exceeded"

        # 3. Wall-Clock Timeout Net
        if (time.time() - start_time) >= self.timeout_seconds:
            return "cap_breached: wall_clock_timeout"

        # 4. Loop Detection Net (🆕)
        history = state.get("route_history", [])
        if len(history) >= 4:
            # Detect immediate repeating pairs: e.g. ['researcher', 'analyst', 'researcher', 'analyst']
            if history[-4:-2] == history[-2:]:
                return "loop_detected: repeating_agent_pair"

        return None


# ---------------------------------------------------------------------------
# 3. Helpers for Dynamic LangGraph Version Compatibility
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
# 4. Multi-Agent Team Core Implementation
# ---------------------------------------------------------------------------

class MultiAgentTeam:
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini",
        procedural_memory_path: str = "AGENTS.md",
        temperature: float = 0.0,
        procedural_memory_text: Optional[str] = None,
        max_turns: int = 8,
        max_tokens: int = 12000,
        timeout_seconds: float = 45.0,
        log_file: str = "team_execution_traces.jsonl",
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, request_timeout=30.0)
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

    def orchestrator_step(self, state: TeamState, run_num: int = 1) -> Handoff:
        """
        Orchestrator node: Classifies, routes, or directly answers no_tool queries.
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

        router_llm = self.llm.with_structured_output(Handoff, method="function_calling")
        messages = [
            SystemMessage(content=orch_system_prompt),
            HumanMessage(content=f"User Query: {user_query}")
        ]
        
        handoff_decision = router_llm.invoke(messages)
        return handoff_decision

    def run_task(self, task_id: str, user_query: str, run_num: int = 1) -> Dict[str, Any]:
        """
        Main Execution Loop for Multi-Agent Task Orchestration
        """
        start_time = time.time()
        seq = 0
        total_tokens = 0

        # Initialize State
        state: TeamState = {
            "messages": [{"role": "user", "content": user_query}],
            "task_id": task_id,
            "user_query": user_query,
            "last_active": AgentName.ORCHESTRATOR.value,
            "handoff_data": None,
            "agent_turns_count": 0,
            "route_history": [AgentName.ORCHESTRATOR.value],
            "is_terminal": False,
            "terminal_reason": None
        }

        final_answer = ""

        while not state["is_terminal"]:
            seq += 1
            t_turn_start = time.time()

            # Check Safety Nets before turn execution
            breach = self.safety_nets.check_breaches(state, start_time, total_tokens)
            if breach:
                state["is_terminal"] = True
                state["terminal_reason"] = breach
                final_answer = f"[SYSTEM REFUSAL: {breach}]"
                break

            current_owner = state["last_active"]

            # 1. Orchestrator Dispatch Step
            if current_owner == AgentName.ORCHESTRATOR.value:
                handoff = self.orchestrator_step(state, run_num)
                duration = (time.time() - t_turn_start) * 1000
                total_tokens += 150

                self.tracer.log_event(
                    task_id=task_id, run_num=run_num, seq=seq,
                    agent=AgentName.ORCHESTRATOR.value, event="handoff_decision",
                    owner=current_owner, duration_ms=duration,
                    input_tokens=150, output_tokens=50,
                    destination=handoff.destination, reason=handoff.reason,
                    handoff_payload=handoff.payload.model_dump()
                )

                if handoff.destination == AgentName.ORCHESTRATOR.value:
                    state["is_terminal"] = True
                    state["terminal_reason"] = "answered"
                    final_answer = handoff.payload.open_question or "I can assist you with insurance policy lookups and calculations."
                    break
                else:
                    state["last_active"] = handoff.destination
                    state["handoff_data"] = handoff.payload.model_dump()
                    state["route_history"].append(handoff.destination)
                    state["agent_turns_count"] += 1

            # 2. Worker Execution Steps
            else:
                payload = HandoffPayload(**(state["handoff_data"] or {}))
                worker_query = payload.open_question or user_query

                if payload.constraints:
                    worker_query += f"\n[Constraints: {', '.join(payload.constraints)}]"
                if payload.facts:
                    worker_query += f"\n[Established Facts: {json.dumps(payload.facts)}]"

                worker_res_text = ""
                tool_calls_count = 0

                if current_owner == AgentName.RESEARCHER.value:
                    res = self.researcher_agent.invoke({"messages": [HumanMessage(content=worker_query)]})
                    msgs = res.get("messages", [])
                    worker_res_text = msgs[-1].content if msgs else ""
                    tool_calls_count = sum(len(m.tool_calls) for m in msgs if isinstance(m, AIMessage) and m.tool_calls)

                elif current_owner == AgentName.ANALYST.value:
                    res = self.analyst_agent.invoke({"messages": [HumanMessage(content=worker_query)]})
                    msgs = res.get("messages", [])
                    worker_res_text = msgs[-1].content if msgs else ""
                    tool_calls_count = sum(len(m.tool_calls) for m in msgs if isinstance(m, AIMessage) and m.tool_calls)

                elif current_owner == AgentName.WRITER.value:
                    res_msg = self.llm.invoke([
                        SystemMessage(content=self.writer_prompt),
                        HumanMessage(content=worker_query)
                    ])
                    worker_res_text = res_msg.content

                duration = (time.time() - t_turn_start) * 1000
                tokens_used = 750 + (tool_calls_count * 200)
                total_tokens += tokens_used

                self.tracer.log_event(
                    task_id=task_id, run_num=run_num, seq=seq,
                    agent=current_owner, event="worker_turn_complete",
                    owner=current_owner, duration_ms=duration,
                    input_tokens=600, output_tokens=150,
                    tool_calls=tool_calls_count, result_preview=worker_res_text[:100]
                )

                updated_facts = payload.facts
                updated_facts[current_owner] = worker_res_text

                state["handoff_data"] = HandoffPayload(
                    summary=f"{current_owner} completed step.",
                    constraints=payload.constraints,
                    facts=updated_facts,
                    open_question=f"Synthesize or evaluate step after {current_owner}"
                ).model_dump()

                final_answer = worker_res_text

                if current_owner == AgentName.WRITER.value or "final answer:" in worker_res_text.lower():
                    state["is_terminal"] = True
                    state["terminal_reason"] = "answered"
                    break
                else:
                    state["last_active"] = AgentName.ORCHESTRATOR.value
                    state["route_history"].append(AgentName.ORCHESTRATOR.value)

        total_duration = time.time() - start_time
        worker_turns = max(0, state["agent_turns_count"])

        self.tracer.log_summary(
            task_id=task_id,
            run_num=run_num,
            terminal_state=state["terminal_reason"] or "completed",
            total_turns=worker_turns,
            total_duration_sec=total_duration,
            total_tokens=total_tokens,
            route_history=state["route_history"]
        )

        refusal_keywords = ["cannot answer", "missing", "unanswerable", "refuse", "not mentioned", "cap_breached"]
        is_refused = any(kw in final_answer.lower() for kw in refusal_keywords)

        return {
            "task_id": task_id,
            "final_answer": final_answer,
            "terminal_state": state["terminal_reason"],
            "route_history": state["route_history"],
            "worker_turns": worker_turns,
            "total_tokens": total_tokens,
            "latency_seconds": round(total_duration, 3),
            "is_refused": is_refused
        }


if __name__ == "__main__":
    print("🧪 Dry-running MultiAgentTeam...")
    team = MultiAgentTeam()
    res = team.run_task(task_id="t26", user_query="Find the water damage deductible in the policy, then calculate payout for a $15,000 claim with 18% VAT.")
    print("✅ Result:", json.dumps(res, indent=2, ensure_ascii=False))