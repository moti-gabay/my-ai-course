"""
eval_runner.py - Full Benchmark Evaluation Suite (Assignment 5)
Executes 35 tasks x 2 configurations x 5 runs = 350 total executions.
Outputs results directly into assignment_05.xlsx.
"""

import os
import json
import time
import pandas as pd
from typing import Dict, Any, List, Optional

# יבוא המערכת המרובת-סוכנים מ-team.py
from team import MultiAgentTeam
from contracts import AgentName, AGENT_SCOPE_CONTRACTS
from scoring import JUDGE, RunView, evaluate, judge_rubrics
import judge

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


def _predicate(task: Dict[str, Any]) -> Dict[str, Any]:
    crit = task.get("success_criteria")
    # Pre-Phase-3 task sets store prose criteria; those can only be judged.
    return crit if isinstance(crit, dict) else {"judge": str(crit or "")}


def _judge_cost(*results: Optional[Dict[str, Any]]) -> Dict[str, int]:
    live = [r for r in results if r and not r.get("cached")]
    return {"judge_input_tokens": sum(r["input_tokens"] for r in live),
            "judge_output_tokens": sum(r["output_tokens"] for r in live)}


def score_run(task: Dict[str, Any], config: str, answer: str, refused: bool, terminal_state: str,
              tool_outputs: List[Dict[str, Any]], worker_turns: Optional[int], tool_calls: int,
              use_judge: bool = True) -> Dict[str, Any]:
    """Success, refusal and faithfulness for one run. Reads the run's outputs and the task's
    predicate/reference; never the task type or the answerable label, and never the config
    inside a judge prompt (config only picks the no_dispatch meaning)."""
    pred = _predicate(task)
    outcome = evaluate(pred, RunView(answer=answer, refused=refused, config=config,
                                     worker_turns=worker_turns, tool_calls=tool_calls))
    success_judge = None
    if outcome == JUDGE:
        if use_judge:
            success_judge = judge.judge_task_success(task["task"], task.get("reference_answer", ""), answer,
                                                     "\n".join(judge_rubrics(pred)))
            success, method = success_judge["verdict"] == "pass", "judge"
        else:
            success, method = None, "judge_skipped"
    else:
        success, method = outcome, "code"

    faith = None
    if use_judge and terminal_state in ("answered", "refused"):
        faith = judge.judge_faithfulness(answer, tool_outputs)
    return {
        "success": success,
        "success_method": method,
        "success_explanation": success_judge["explanation"] if success_judge else "",
        "faithfulness": faith["verdict"] if faith else "n/a",
        "faithfulness_explanation": faith["explanation"] if faith else "",
        **_judge_cost(success_judge, faith),
    }


def _row(task: Dict[str, Any], config: str, run_num: int) -> Dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "task": task["task"],
        "type": task.get("type", "single"),
        "answerable": task.get("answerable", True),
        "success_criteria": json.dumps(task.get("success_criteria", ""), ensure_ascii=False),
        "capable_agents": json.dumps(task.get("capable_agents", [])),
        "config": config,
        "run": run_num,
    }


def evaluate_single_agent_run(agent_instance, task: Dict[str, Any], run_num: int,
                              use_judge: bool = True) -> Dict[str, Any]:
    """
    מריץ את הסוכן הבודד המוקפא ממטלה 4 דרך LangGraphAgentRunner
    """
    start_t = time.time()
    row = _row(task, "single", run_num)
    try:
        # הקריאה המודכנת למתודה run_task הקיימת ב-agent.py
        res = agent_instance.run_task(task_data=task, run_number=run_num)
    except Exception as e:
        return {**row, "answer": f"ERROR: {e}", "success": False, "success_method": "code",
                "refused": False, "terminal_state": "error", "route": "[]", "agent_turns": 0,
                "tool_calls": 0, "routing_correct": "n/a", "handoff_correct": "n/a",
                "per_agent_success": json.dumps({"single": False}), "faithfulness": "n/a",
                "latency_ms": round((time.time() - start_t) * 1000, 2), "input_tokens": 0,
                "output_tokens": 0, "breach_reason": str(e)}

    scores = score_run(task, "single", res["final_answer"], res["is_refused"], res["terminal_state"],
                       res.get("tool_outputs", []), None, res.get("tool_calls_count", 0), use_judge)
    return {
        **row,
        "answer": res["final_answer"],
        "refused": res["is_refused"],
        "refusal_reason": res.get("refusal_reason", ""),
        "terminal_state": res["terminal_state"],
        "route": json.dumps(["single"]),
        "agent_turns": 1,
        "tool_calls": res.get("tool_calls_count", 0),
        "routing_correct": "n/a",
        "handoff_correct": "n/a",
        "per_agent_success": json.dumps({"single": scores["success"]}),
        "latency_ms": round(res.get("latency_seconds", 0) * 1000, 2),
        "input_tokens": res.get("prompt_tokens", 0),
        "output_tokens": res.get("completion_tokens", 0),
        "breach_reason": res.get("breach_reason") or "None",
        **scores,
    }


def _handoff_correct(task: Dict[str, Any], turns: List[Dict[str, Any]]) -> Any:
    """Code check: every required constraint appears in the constraints the LAST worker received."""
    required = task.get("required_constraints") or []
    if not required or not turns:
        return "n/a"
    received = " | ".join(turns[-1]["received_payload"].get("constraints") or []).lower()
    return all(rc.lower() in received for rc in required)


def evaluate_team_agent_run(team_instance, task: Dict[str, Any], run_num: int,
                            use_judge: bool = True) -> Dict[str, Any]:
    """
    מריץ את המערכת המרובת-סוכנים (Multi-Agent Team)
    """
    start_t = time.time()
    res = team_instance.run_task(task_id=task["task_id"], user_query=task["task"], run_num=run_num)
    duration_ms = (time.time() - start_t) * 1000

    # Routing accuracy: the first agent that handled the request must be in capable_agents.
    route = res["route"]
    first_agent = route[0] if route else AgentName.ORCHESTRATOR.value
    capable_agents = task.get("capable_agents", [])
    routing_correct = first_agent in capable_agents

    scores = score_run(task, "team", res["final_answer"], res["is_refused"], res["terminal_state"],
                       res["tool_outputs"], res["worker_turns"], res["tool_calls"], use_judge)

    # Per-agent success: each worker judged on what it received; the orchestrator on routing (code).
    per_agent: Dict[str, Any] = {AgentName.ORCHESTRATOR.value: routing_correct}
    per_agent_detail, turn_judgements = [], []
    for turn in res["turns"]:
        if not use_judge:
            continue
        contract = AGENT_SCOPE_CONTRACTS[AgentName(turn["agent"])]
        v = judge.judge_agent_turn(f"{contract['name']}: {contract['scope']}", turn["received_payload"],
                                   turn["tool_outputs"], turn["output"])
        turn_judgements.append(v)
        ok = v["verdict"] == "pass"
        per_agent[turn["agent"]] = per_agent.get(turn["agent"], True) and ok
        per_agent_detail.append({"agent": turn["agent"], "verdict": v["verdict"], "explanation": v["explanation"]})
    cost = _judge_cost(*turn_judgements)

    return {
        **_row(task, "team", run_num),
        "answer": res["final_answer"],
        "refused": res["is_refused"],
        "refusal_reason": res.get("refusal_reason", ""),
        "terminal_state": res["terminal_state"],
        "route": json.dumps(route),
        "agent_turns": res["worker_turns"],
        "tool_calls": res["tool_calls"],
        "routing_correct": routing_correct,
        "handoff_correct": _handoff_correct(task, res["turns"]),
        "per_agent_success": json.dumps(per_agent),
        "per_agent_detail": json.dumps(per_agent_detail, ensure_ascii=False),
        "latency_ms": round(duration_ms, 2),
        "input_tokens": res["input_tokens"],
        "output_tokens": res["output_tokens"],
        "breach_reason": res["breach_reason"] or "None",
        **scores,
        "judge_input_tokens": scores["judge_input_tokens"] + cost["judge_input_tokens"],
        "judge_output_tokens": scores["judge_output_tokens"] + cost["judge_output_tokens"],
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