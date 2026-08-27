import os
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ייבוא הרכיבים מהקבצים הקודמים
from agent import LangGraphAgentRunner
from tools import search_docs

# ---------------------------------------------------------------------------
# 1. Static RAG Baseline Implementation (Assignment 3 Baseline)
# ---------------------------------------------------------------------------
def run_static_rag(user_query: str, model_name: str = "gpt-4o-mini") -> Dict[str, Any]:
    """
    מריץ הזרמת מידע ישירה (Static RAG) במעבר יחיד - ללא ReAct וללא כלים דינמיים.
    """
    start_time = time.time()
    llm = ChatOpenAI(model=model_name, temperature=0)

    # 1. שליפת מסמכים סטטית יחידה
    retrieved_context = search_docs.invoke(user_query)

    # 2. הרכבת System Prompt
    system_prompt = f"""You are an insurance policy assistant. 
Answer the user query based ONLY on the provided context below.
If the information is not contained in the context, explicitly state that you cannot answer.

Context:
{retrieved_context}
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]

    try:
        response = llm.invoke(messages)
        latency = time.time() - start_time

        usage = getattr(response, "response_metadata", {}).get("token_usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        final_text = response.content
        refusal_keywords = ["cannot answer", "not mentioned", "unavailable", "refuse", "do not have"]
        is_refused = any(kw in final_text.lower() for kw in refusal_keywords)

        return {
            "status": "success",
            "latency_seconds": round(latency, 4),
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tool_calls_count": 0,
            "is_refused": is_refused,
            "final_answer": final_text,
            "error_message": None
        }

    except Exception as e:
        return {
            "status": "error",
            "latency_seconds": round(time.time() - start_time, 4),
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "tool_calls_count": 0,
            "is_refused": True,
            "final_answer": f"ERROR: Static RAG failed: {str(e)}",
            "error_message": str(e)
        }


# ---------------------------------------------------------------------------
# 2. Automated Judge / Evaluator Function
# ---------------------------------------------------------------------------
def evaluate_success(task: Dict[str, Any], final_answer: str, is_refused: bool, tool_calls_count: int) -> bool:
    """
    בודק האם התשובה עמדה ב-success_criteria של המשימה.
    """
    task_type = task.get("type")
    
    #1. משימות unanswerable או tool_fails - מצפות לסירוב
    if task_type in ["unanswerable", "tool_fails"] or not task.get("answerable", True):
        return is_refused or "error" in final_answer.lower() or "cannot" in final_answer.lower()

    # 2. משימות no_tool - מצפות לאפס קריאות לכלים
    if task_type == "no_tool":
        return tool_calls_count == 0 and len(final_answer.strip()) > 10

    # 3. בדיקת טקסט חופשי לפי קריטריון הצלחה מדויק
    criteria = task.get("success_criteria", "").lower()
    
    # חילוץ מילות מפתח לבדיקה מתוך ה-criteria
    if "12,980" in criteria or "12980" in criteria:
        return "12,980" in final_answer or "12980" in final_answer
    if "1890" in criteria:
        return "1890" in final_answer
    if "375" in criteria:
        return "375" in final_answer

    # ברירת מחדל: בדיקת אורך ותקינות
    return len(final_answer.strip()) > 15 and not is_refused


# ---------------------------------------------------------------------------
# 3. Experiment Matrix Runner
# ---------------------------------------------------------------------------
def run_experiment_matrix(
    task_file: str = "task_set.json",
    runs_per_task: int = 5,
    output_excel: str = "assignment_04.xlsx"
):
    """
    מריץ מטריצה מלאה: 25 משימות × 5 הרצות × 2 קונפיגורציות = 250 הרצות.
    """
    print("🚀 Loading Task Set...")
    with open(task_file, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    agent_runner = LangGraphAgentRunner(max_iterations=10, timeout_seconds=30.0)
    
    all_raw_logs = []

    print(f"📊 Starting Experiment Matrix: {len(tasks)} tasks | {runs_per_task} runs per task | 2 Configurations\n")

    for idx, task in enumerate(tasks, 1):
        task_id = task["task_id"]
        task_type = task["type"]
        print(f"[{idx}/{len(tasks)}] Running Task {task_id} ({task_type}): '{task['task'][:60]}...'")

        for run_num in range(1, runs_per_task + 1):
            # A. הרצת Static RAG Baseline
            rag_res = run_static_rag(task["task"])
            rag_success = evaluate_success(task, rag_res["final_answer"], rag_res["is_refused"], rag_res["tool_calls_count"])
            
            all_raw_logs.append({
                "Config": "Static RAG",
                "Task_ID": task_id,
                "Task_Type": task_type,
                "Run_Number": run_num,
                "Success": 1 if rag_success else 0,
                "Latency_Sec": rag_res["latency_seconds"],
                "Total_Tokens": rag_res["total_tokens"],
                "Tool_Calls": rag_res["tool_calls_count"],
                "Refused": rag_res["is_refused"],
                "Status": rag_res["status"],
                "Final_Answer": rag_res["final_answer"]
            })

            # B. הרצת LangGraph Agent
            agent_res = agent_runner.run_task(task, run_number=run_num)
            agent_success = evaluate_success(task, agent_res["final_answer"], agent_res["is_refused"], agent_res["tool_calls_count"])

            all_raw_logs.append({
                "Config": "LangGraph Agent",
                "Task_ID": task_id,
                "Task_Type": task_type,
                "Run_Number": run_num,
                "Success": 1 if agent_success else 0,
                "Latency_Sec": agent_res["latency_seconds"],
                "Total_Tokens": agent_res["total_tokens"],
                "Tool_Calls": agent_res["tool_calls_count"],
                "Refused": agent_res["is_refused"],
                "Status": agent_res["status"],
                "Final_Answer": agent_res["final_answer"]
            })

    # ---------------------------------------------------------------------------
    # 4. Aggregating Results & Generating assignment_04.xlsx
    # ---------------------------------------------------------------------------
    df_raw = pd.DataFrame(all_raw_logs)

    # חישוב מטריקות מקובצות (Summary Metrics)
    summary_rows = []
    for (config, t_type), group in df_raw.groupby(["Config", "Task_Type"]):
        summary_rows.append({
            "Configuration": config,
            "Task_Type": t_type,
            "Total_Runs": len(group),
            "Success_Rate_%": round(group["Success"].mean() * 100, 2),
            "Latency_p50_Sec": round(group["Latency_Sec"].median(), 3),
            "Latency_p95_Sec": round(group["Latency_Sec"].quantile(0.95), 3),
            "Avg_Tokens": round(group["Total_Tokens"].mean(), 1),
            "Avg_Tool_Calls": round(group["Tool_Calls"].mean(), 2),
            "Refusal_Rate_%": round(group["Refused"].mean() * 100, 2)
        })

    df_summary = pd.DataFrame(summary_rows)

    # שמירה ל-Excel עם 2 גיליונות
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary Metrics", index=False)
        df_raw.to_excel(writer, sheet_name="Raw Executions Log", index=False)

    print(f"\n✅ Experiment completed successfully!")
    print(f"📁 Results saved to {output_excel}")


if __name__ == "__main__":
    run_experiment_matrix(runs_per_task=5)