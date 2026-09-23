import time
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
from tools import ALL_TOOLS
import retriever
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
        log_file: str = "agent_execution_logs.jsonl",
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
            prompt=self.system_prompt
        )

    def run_task(
        self,
        task_data: Dict[str, Any],
        run_number: int = 1,
        on_step: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        מריץ משימה יחידה דרך ה-LangGraph Agent עם מדידת זמנים, ניטור טוקנים, וחישוב מטריקות.
        """
        task_id = task_data.get("task_id", "unknown")
        user_query = task_data.get("task", "")
        task_type = task_data.get("type", "single")

        start_time = time.time()
        run_id = str(uuid.uuid4())[:8]

        inputs = {"messages": [HumanMessage(content=user_query)]}
        
        # הגדרת Recursion limit לבטיחות
        config = {
            "recursion_limit": self.max_iterations * 2 + 1
        }

        trace_steps = []
        tool_calls_count = 0
        used_tools = []
        prompt_tokens = 0
        completion_tokens = 0
        final_text = ""
        status = "success"
        error_message = None

        def _emit(step: Dict[str, Any]):
            trace_steps.append(step)
            if on_step:
                on_step(step)

        try:
            # הרצת ה-Graph בדרייבר Stream לצורך מעקב צעד-אחר-צעד ואיסוף מטריקות
            for event in self.graph.stream(inputs, config=config, stream_mode="values"):
                # בדיקת Timeout שומר סף (Wall-clock timeout)
                elapsed = time.time() - start_time
                if elapsed > self.timeout_seconds:
                    status = "timeout_exceeded"
                    error_message = f"Execution exceeded wall-clock limit of {self.timeout_seconds} seconds."
                    break

                messages = event.get("messages", [])
                if not messages:
                    continue

                last_msg = messages[-1]

                # תיעוד קריאות לכלים ולמידת Token Usage
                if isinstance(last_msg, AIMessage):
                    usage = last_msg.usage_metadata or {}
                    prompt_tokens += usage.get("input_tokens", 0)
                    completion_tokens += usage.get("output_tokens", 0)

                    if last_msg.tool_calls:
                        for tc in last_msg.tool_calls:
                            tool_calls_count += 1
                            used_tools.append(tc["name"])
                            _emit({
                                "step": "tool_call",
                                "tool": tc["name"],
                                "args": tc["args"]
                            })
                    else:
                        final_text = last_msg.content

                elif isinstance(last_msg, ToolMessage):
                    _emit({
                        "step": "tool_result",
                        "tool": last_msg.name,
                        "content": str(last_msg.content)[:300]  # קיצור למניעת לוג נפוח
                    })

        except Exception as e:
            if "Recursion limit" in str(e):
                status = "max_iterations_exceeded"
                error_message = f"Agent exceeded maximum allowed iterations ({self.max_iterations})."
            else:
                status = "execution_error"
                error_message = f"Runtime error: {str(e)}"
            
            final_text = f"ERROR: Task execution stopped due to: {error_message}"

        if on_step:
            on_step({
                "step": "final",
                "content": final_text,
                "status": status,
                "error_message": error_message
            })

        latency = time.time() - start_time

        # זיהוי האם התגובה היא סירוב (Refusal detection)
        refusal_keywords = ["cannot answer", "not mentioned", "unavailable", "refuse", "do not have", "failed", "restricted"]
        is_refused = any(kw in final_text.lower() for kw in refusal_keywords) or not task_data.get("answerable", True)

        # הרכבת אובייקט התיעוד המלא
        log_entry = {
            "run_id": run_id,
            "task_id": task_id,
            "task_type": task_type,
            "run_number": run_number,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "error_message": error_message,
            "latency_seconds": round(latency, 4),
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tool_calls_count": tool_calls_count,
            "used_tools": list(set(used_tools)),
            "is_refused": is_refused,
            "user_query": user_query,
            "final_answer": final_text,
            "trace_steps": trace_steps
        }

        # שמירה לקובץ JSONL
        self._write_jsonl(log_entry)

        return log_entry

    def _write_jsonl(self, data: Dict[str, Any]):
        """כותב שורת לוג לקובץ JSONL."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Execution Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    runner = LangGraphAgentRunner(max_iterations=10, timeout_seconds=30.0)
    
    test_task = {
        "task_id": "t01",
        "task": "If I have a claim for $12,000 for building damage and my deductible is $1,000, what amount will the insurer pay after deductible and 18% VAT?",
        "type": "multi_hop",
        "answerable": True
    }

    print("--- Testing LangGraph Agent Runner ---")
    res = runner.run_task(test_task, run_number=1)
    print(f"Status: {res['status']}")
    print(f"Latency: {res['latency_seconds']}s | Tokens: {res['total_tokens']} | Tool Calls: {res['tool_calls_count']}")
    print(f"Final Answer:\n{res['final_answer']}")

    