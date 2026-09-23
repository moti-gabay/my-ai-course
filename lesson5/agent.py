import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from tools import ALL_TOOLS
import retriever
from tracing import UnitOfWorkTracer, run_traced_react
from contracts import WorkerResult
from langgraph.prebuilt import create_react_agent

# טעינת מפתחות ה-API מקובץ ה-.env שליד הסקריפט
load_dotenv(Path(__file__).with_name(".env"))
# ---------------------------------------------------------------------------
# System Prompt & Guardrails Configuration
# ---------------------------------------------------------------------------
AGENT_SYSTEM_PROMPT = """You are a precise, reliable insurance assistant operating as a ReAct agent.

Instructions:
1. Use the search_docs tool to retrieve specific clauses, coverage limits, and terms from the insurance policy corpus.
2. ALWAYS use the calculator tool for arithmetic calculations. Do NOT calculate numbers in your head.
3. If a question cannot be answered using the provided tools or corpus, explicitly decline to answer and state clearly that the required information is missing or unanswerable.
4. Do NOT invoke tools for simple conversational greetings or general knowledge questions that do not depend on policy specifics.
5. Never hallucinate facts, coverage limits, or math results.
"""

# ---------------------------------------------------------------------------
# LangGraph Agent Wrapper with Safety Guardrails & Logging
# ---------------------------------------------------------------------------
class LangGraphAgentRunner:
    def __init__(
        self,
        model_name: str = "claude-haiku-4-5",
        max_iterations: int = 10,
        timeout_seconds: float = 30.0,
        log_file: str = str(Path(__file__).with_name("traces") / "single_traces.jsonl"),
        temperature: float = 0.0,
        system_prompt: Optional[str] = None
    ):
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.log_file = log_file
        self.temperature = temperature
        self.system_prompt = system_prompt or AGENT_SYSTEM_PROMPT

        # אתחול המודל עם תיוג Token usage
        self.llm = ChatAnthropic(model=model_name, temperature=temperature, timeout=self.timeout_seconds, max_tokens=2048)
        
        # Load embeddings, index and reranker now so model loading never counts as run latency.
        retriever.warm_up()

        # יצירת ה-Agent של LangGraph
        self.graph = create_react_agent(
            model=self.llm,
            tools=ALL_TOOLS,
            prompt=self.system_prompt,
            response_format=WorkerResult,  # same structured answer/refused contract as the team workers
        )

    def run_task(
        self,
        task_data: Dict[str, Any],
        run_number: int = 1,
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        מריץ משימה יחידה דרך ה-LangGraph Agent עם מדידת זמנים, ניטור טוקנים, וחישוב מטריקות.
        Traces go through the same UnitOfWorkTracer events as the team (agent="single").
        """
        task_id = task_data.get("task_id", "unknown")
        user_query = task_data.get("task", "")
        start_time = time.time()
        rt = UnitOfWorkTracer(self.log_file, on_event=on_step).start_run(task_id, run_number)

        def check() -> Optional[str]:
            if time.time() - start_time > self.timeout_seconds:
                return "wall_clock_timeout"
            return None

        # הגדרת Recursion limit לבטיחות
        outcome = run_traced_react(self.graph, user_query, rt, "single", check=check,
                                   config={"recursion_limit": self.max_iterations * 2 + 1})
        latency = time.time() - start_time
        rt.account("single", duration_ms=round(latency * 1000, 1), turns=1)

        if outcome.breach:
            status, terminal_state, breach_reason = "timeout_exceeded", "cap_breached", outcome.breach
            error_message = f"Execution exceeded wall-clock limit of {self.timeout_seconds} seconds."
        elif outcome.error and "Recursion limit" in outcome.error:
            status, terminal_state, breach_reason = "max_iterations_exceeded", "cap_breached", "max_iterations"
            error_message = f"Agent exceeded maximum allowed iterations ({self.max_iterations})."
        elif outcome.error:
            status, terminal_state, breach_reason = "execution_error", "error", outcome.error
            error_message = f"Runtime error: {outcome.error}"
        elif outcome.structured is None:
            status, terminal_state, breach_reason = "execution_error", "error", "no structured response"
            error_message = "Runtime error: the agent finished without a structured response."
        else:
            status, breach_reason, error_message = "success", None, None
            terminal_state = "refused" if outcome.structured.refused else "answered"
        if outcome.breach or outcome.error:
            rt.emit("single", "net_breach" if terminal_state == "cap_breached" else "error",
                    owner="single", reason=breach_reason)
        if status == "success":
            final_text = outcome.structured.answer
        else:
            final_text = f"ERROR: Task execution stopped due to: {error_message}"
        # Refusal is the model's own structured flag; a breach or error is not a refusal.
        is_refused = status == "success" and outcome.structured.refused
        refusal_reason = outcome.structured.refusal_reason if is_refused else ""

        totals = rt.totals()
        rt.summary(
            terminal_state=terminal_state,
            breach_reason=breach_reason,
            route=["single"],
            route_history=["single"],
            agent_turns=1,
            input_tokens=totals["input_tokens"],
            output_tokens=totals["output_tokens"],
            total_tokens=totals["input_tokens"] + totals["output_tokens"],
            tool_calls=totals["tool_calls"],
            duration_ms=round(latency * 1000, 1),
            refused=is_refused,
            final_answer=final_text,
        )

        return {
            "task_id": task_id,
            "task_type": task_data.get("type", "single"),
            "run_number": run_number,
            "status": status,
            "terminal_state": terminal_state,
            "breach_reason": breach_reason,
            "error_message": error_message,
            "latency_seconds": round(latency, 4),
            "total_tokens": totals["input_tokens"] + totals["output_tokens"],
            "prompt_tokens": totals["input_tokens"],
            "completion_tokens": totals["output_tokens"],
            "tool_calls_count": totals["tool_calls"],
            "used_tools": sorted({t["tool"] for t in outcome.tool_outputs}),
            "tool_outputs": outcome.tool_outputs,
            "per_agent": rt.per_agent,
            "is_refused": is_refused,
            "refusal_reason": refusal_reason,
            "user_query": user_query,
            "final_answer": final_text,
        }


# ---------------------------------------------------------------------------
# Execution Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    runner = LangGraphAgentRunner(max_iterations=10, timeout_seconds=30.0)
    res = runner.run_task({"task_id": "smoke", "task": "How much will the auto insurer pay for bail bonds after a covered accident?"})
    print(f"Status: {res['status']} | Latency: {res['latency_seconds']}s | Tokens: {res['total_tokens']} | Tool Calls: {res['tool_calls_count']}")
    print(f"Final Answer:\n{res['final_answer']}")
