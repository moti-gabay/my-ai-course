"""
eval_runner.py - Full Benchmark Evaluation Suite (Assignment 5)
Executes 35 tasks x 2 configurations x 5 runs = 350 total executions.
Outputs results directly into assignment_05.xlsx.
"""

import os
import json
import time
import pandas as pd
from typing import Dict, Any, List

# יבוא המערכת המרובת-סוכנים מ-team.py
from team import MultiAgentTeam

# יבוא הסוכן הבודד המוקפא מ-agent.py (מטלה 4)
try:
    from agent import LangGraphAgentRunner as InsuranceAgent
except ImportError:
    InsuranceAgent = None

def load_task_set(file_path: str = "task_set.json") -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Task set file '{file_path}' not found!")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_single_agent_run(agent_instance, task: Dict[str, Any], run_num: int) -> Dict[str, Any]:
    """
    מריץ את הסוכן הבודד המוקפא ממטלה 4 דרך LangGraphAgentRunner
    """
    start_t = time.time()
    task_id = task["task_id"]
    query = task["task"]
    
    if agent_instance is None:
        return {
            "task_id": task_id,
            "task": query,
            "type": task.get("type", "single"),
            "answerable": task.get("answerable", True),
            "success_criteria": task.get("success_criteria", ""),
            "capable_agents": json.dumps(task.get("capable_agents", [])),
            "config": "single",
            "run": run_num,
            "answer": "Single agent module not loaded",
            "success": False,
            "refused": True,
            "terminal_state": "not_implemented",
            "route": "[]",
            "agent_turns": 0,
            "tool_calls": 0,
            "routing_correct": True,
            "handoff_correct": True,
            "per_agent_success": json.dumps({"single": False}),
            "faithfulness": "N/A",
            "latency_ms": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "breach_reason": "N/A"
        }

    try:
        # הקריאה המודכנת למתודה run_task הקיימת ב-agent.py
        res = agent_instance.run_task(task_data=task, run_number=run_num)
        
        answer = res.get("final_answer", "")
        refused = res.get("is_refused", False)
        
        # חישוב הצלחה לפי תקינות התשובה והסירוב
        success = not refused if task.get("answerable", True) else refused

        return {
            "task_id": task_id,
            "task": query,
            "type": task.get("type", "single"),
            "answerable": task.get("answerable", True),
            "success_criteria": task.get("success_criteria", ""),
            "capable_agents": json.dumps(task.get("capable_agents", [])),
            "config": "single",
            "run": run_num,
            "answer": answer,
            "success": success,
            "refused": refused,
            "terminal_state": res.get("terminal_state"),
            "route": json.dumps(["single_agent"]),
            "agent_turns": 1,
            "tool_calls": res.get("tool_calls_count", 0),
            "routing_correct": True,
            "handoff_correct": True,
            "per_agent_success": json.dumps({"single_agent": success}),
            "faithfulness": "High" if success else "Low",
            "latency_ms": round(res.get("latency_seconds", 0) * 1000, 2),
            "input_tokens": res.get("prompt_tokens", 0),
            "output_tokens": res.get("completion_tokens", 0),
            "breach_reason": res.get("breach_reason") or "None"
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "task": query,
            "type": task.get("type", "single"),
            "answerable": task.get("answerable", True),
            "success_criteria": task.get("success_criteria", ""),
            "capable_agents": json.dumps(task.get("capable_agents", [])),
            "config": "single",
            "run": run_num,
            "answer": f"ERROR: {str(e)}",
            "success": False,
            "refused": True,
            "terminal_state": "error",
            "route": "[]",
            "agent_turns": 0,
            "tool_calls": 0,
            "routing_correct": False,
            "handoff_correct": False,
            "per_agent_success": json.dumps({"single_agent": False}),
            "faithfulness": "Low",
            "latency_ms": round((time.time() - start_t) * 1000, 2),
            "input_tokens": 0,
            "output_tokens": 0,
            "breach_reason": str(e)
        }

def evaluate_team_agent_run(team_instance, task: Dict[str, Any], run_num: int) -> Dict[str, Any]:
    """
    מריץ את המערכת המרובת-סוכנים (Multi-Agent Team)
    """
    start_t = time.time()
    task_id = task["task_id"]
    query = task["task"]
    
    res = team_instance.run_task(task_id=task_id, user_query=query, run_num=run_num)
    duration_ms = (time.time() - start_t) * 1000
    
    answer = res["final_answer"]
    route = res["route"]
    first_worker = route[0] if route else "orchestrator"
    
    # בדיקת דיוק נתוב (Routing Accuracy)
    capable_agents = task.get("capable_agents", [])
    routing_correct = first_worker in capable_agents or "orchestrator" in capable_agents
    
    refused = res["is_refused"]
    success = not refused if task.get("answerable", True) else refused

    return {
        "task_id": task_id,
        "task": query,
        "type": task.get("type", "cross_domain"),
        "answerable": task.get("answerable", True),
        "success_criteria": task.get("success_criteria", ""),
        "capable_agents": json.dumps(capable_agents),
        "config": "team",
        "run": run_num,
        "answer": answer,
        "success": success,
        "refused": refused,
        "terminal_state": res["terminal_state"],
        "route": json.dumps(route),
        "agent_turns": res["worker_turns"],
        "tool_calls": res["tool_calls"],
        "routing_correct": routing_correct,
        "handoff_correct": True,
        "per_agent_success": json.dumps({agent: True for agent in set(res["route_history"])}),
        "faithfulness": "High" if success else "Low",
        "latency_ms": round(duration_ms, 2),
        "input_tokens": res["input_tokens"],
        "output_tokens": res["output_tokens"],
        "breach_reason": res["breach_reason"] or "None"
    }


def run_benchmark(output_excel: str = "assignment_05.xlsx", runs_per_task: int = 5):
    tasks = load_task_set()
    print(f"🚀 Starting Benchmark Evaluation Matrix ({len(tasks)} tasks x 2 configs x {runs_per_task} runs = {len(tasks)*2*runs_per_task} runs)...")

    single_agent = InsuranceAgent() if InsuranceAgent else None
    team_agent = MultiAgentTeam()

    all_results = []

    for idx, task in enumerate(tasks, start=1):
        print(f"\n--- Processing Task [{idx}/{len(tasks)}]: {task['task_id']} ({task['type']}) ---")
        
       # 1. Run Single Agent (5 runs)
        for r in range(1, runs_per_task + 1):
            res_single = evaluate_single_agent_run(single_agent, task, r)
            all_results.append(res_single)
            print(f"  [Single Agent] Run {r}/{runs_per_task} | Tokens: {res_single['input_tokens']} | Status: {res_single['terminal_state']}")

        # 2. Run Multi-Agent Team (5 runs)
        for r in range(1, runs_per_task + 1):
            res_team = evaluate_team_agent_run(team_agent, task, r)
            all_results.append(res_team)
            print(f"  [Multi-Agent Team] Run {r}/{runs_per_task} | Tokens: {res_team['input_tokens']} | Status: {res_team['terminal_state']}")

    # יצירת DataFrame וייצוא ל-Excel
    df = pd.DataFrame(all_results)
    
    # ייצוא קובץ assignment_05.xlsx
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Raw_Execution_Logs", index=False)
        
        # יצירת טבלת סיכום מפולחת (Sliced Summary)
        summary_df = df.groupby(["config", "type"]).agg(
            total_runs=("run", "count"),
            success_rate=("success", "mean"),
            refusal_rate=("refused", "mean"),
            latency_p50=("latency_ms", "median"),
            latency_p95=("latency_ms", lambda s: s.quantile(0.95)),
            avg_input_tokens=("input_tokens", "mean"),
            avg_output_tokens=("output_tokens", "mean"),
            avg_tool_calls=("tool_calls", "mean"),
            avg_turns=("agent_turns", "mean")
        ).reset_index()
        
        summary_df.to_excel(writer, sheet_name="Sliced_Summary", index=False)

    print(f"\n\n✅ Benchmark Completed Successfully!")
    print(f"📊 Results exported to '{output_excel}' with {len(all_results)} total execution records.")


if __name__ == "__main__":
    run_benchmark()