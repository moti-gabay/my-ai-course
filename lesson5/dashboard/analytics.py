"""
analytics.py - Excel result loading & schema normalization.

Deliberately imports nothing from agent.py / team.py so the Analytics tab works
offline (no API key, no LangChain import cost).

Two historical schemas are supported and normalized to one frame:
  a4  assignment_04.xlsx : "Summary Metrics" + "Raw Executions Log"  (seconds, %)
  a5  assignment_05.xlsx : "Raw_Execution_Logs" + "Sliced_Summary"   (ms, 0-1)
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import pandas as pd
import streamlit as st

LESSON4_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = LESSON4_DIR / "benchmark_results"

Schema = Literal["a4", "a5", "unknown"]

SUMMARY_COLS = [
    "config", "task_type", "total_runs", "success_rate", "refusal_rate",
    "latency_p50_ms", "avg_tokens", "avg_turns_or_tool_calls",
]
RAW_COLS = [
    "config", "task_id", "task_type", "run", "success", "refused",
    "terminal_state", "latency_ms", "total_tokens", "tool_calls", "answer",
]

# Fixed slice order so charts and tables always read in the same sequence.
TASK_TYPE_ORDER = [
    "single", "multi_hop", "cross_domain", "handoff_stress",
    "misroute_bait", "no_tool", "tool_fails", "unanswerable",
]

SCHEMA_NOTES = {
    "a4": (
        "Assignment 4 schema. `avg_tokens` is total tokens per run and the turns "
        "column holds tool calls. Latency was stored in seconds and is shown here in ms."
    ),
    "a5": (
        "Assignment 5 schema. The file's own summary sheet averages **input** tokens only "
        "and counts agent turns. Recompute from raw to average total tokens instead."
    ),
    "unknown": "Unrecognized sheet names. Showing the sheets as-is.",
}


# ---------------------------------------------------------------------------
# Discovery & loading
# ---------------------------------------------------------------------------

def list_result_files() -> List[Path]:
    """Assignment workbooks first, then dashboard benchmark runs (newest first)."""
    assignments = sorted(LESSON4_DIR.glob("*.xlsx"))
    runs = sorted(
        RESULTS_DIR.glob("*.xlsx"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return assignments + runs


@st.cache_data(show_spinner=False)
def load_workbook(path_str: str, mtime: float) -> Dict[str, pd.DataFrame]:
    """mtime is part of the cache key only, so edits on disk invalidate the cache."""
    return pd.read_excel(path_str, sheet_name=None)


@st.cache_data(show_spinner=False)
def load_workbook_bytes(data: bytes) -> Dict[str, pd.DataFrame]:
    return pd.read_excel(io.BytesIO(data), sheet_name=None)


def detect_schema(sheets: Dict[str, pd.DataFrame]) -> Schema:
    names = set(sheets)
    if {"Summary Metrics", "Raw Executions Log"} & names:
        return "a4"
    if "Raw_Execution_Logs" in names or "Sliced_Summary" in names:
        return "a5"
    return "unknown"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _safe_json(value: Any, fallback: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def _as_bool(series: pd.Series) -> pd.Series:
    """A4 stores Success as 0/1 int; A5 stores real bools."""
    if series.dtype == bool:
        return series
    return series.fillna(0).astype(float).astype(bool)


def normalize_summary_a4(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "config": df["Configuration"],
        "task_type": df["Task_Type"],
        "total_runs": df["Total_Runs"],
        "success_rate": df["Success_Rate_%"] / 100.0,
        "refusal_rate": df["Refusal_Rate_%"] / 100.0,
        "latency_p50_ms": df["Latency_p50_Sec"] * 1000.0,
        "avg_tokens": df["Avg_Tokens"],
        "avg_turns_or_tool_calls": df["Avg_Tool_Calls"],
    })
    out["latency_p95_ms"] = df["Latency_p95_Sec"] * 1000.0
    return out


def normalize_summary_a5(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "config": df["config"],
        "task_type": df["type"],
        "total_runs": df["total_runs"],
        "success_rate": df["success_rate"],
        "refusal_rate": df["refusal_rate"],
        "latency_p50_ms": df["latency_p50"],
        "avg_tokens": df["avg_tokens"],
        "avg_turns_or_tool_calls": df["avg_turns"],
    })


def normalize_raw_a4(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "config": df["Config"],
        "task_id": df["Task_ID"],
        "task_type": df["Task_Type"],
        "run": df["Run_Number"],
        "success": _as_bool(df["Success"]),
        "refused": _as_bool(df["Refused"]),
        "terminal_state": df["Status"],
        "latency_ms": df["Latency_Sec"] * 1000.0,
        "total_tokens": df["Total_Tokens"],
        "tool_calls": df["Tool_Calls"],
        "answer": df["Final_Answer"],
    })


def normalize_raw_a5(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "config": df["config"],
        "task_id": df["task_id"],
        "task_type": df["type"],
        "run": df["run"],
        "success": _as_bool(df["success"]),
        "refused": _as_bool(df["refused"]),
        "terminal_state": df["terminal_state"],
        "latency_ms": df["latency_ms"],
        "total_tokens": df["input_tokens"] + df["output_tokens"],
        "tool_calls": df["tool_calls"],
        "answer": df["answer"],
    })
    # JSON-encoded columns are parsed for display only; the stored strings are untouched.
    out["route_str"] = df["route"].map(lambda v: " -> ".join(_safe_json(v, [])))
    out["capable_agents_str"] = df["capable_agents"].map(lambda v: ", ".join(_safe_json(v, [])))
    out["agent_turns"] = df["agent_turns"]
    out["routing_correct"] = _as_bool(df["routing_correct"])
    out["input_tokens"] = df["input_tokens"]
    out["output_tokens"] = df["output_tokens"]
    out["breach_reason"] = df["breach_reason"].fillna("None").astype(str)
    return out


def summarize_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Same aggregation as eval_runner.run_benchmark, but averaging total tokens."""
    if raw.empty:
        return pd.DataFrame(columns=SUMMARY_COLS)
    return (
        raw.groupby(["config", "task_type"])
        .agg(
            total_runs=("run", "count"),
            success_rate=("success", "mean"),
            refusal_rate=("refused", "mean"),
            latency_p50_ms=("latency_ms", "median"),
            avg_tokens=("total_tokens", "mean"),
            avg_turns_or_tool_calls=("tool_calls", "mean"),
        )
        .reset_index()
    )


def load_normalized(sheets: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, Schema]:
    """-> (summary, raw, schema). Summary is recomputed when the file has no summary sheet."""
    schema = detect_schema(sheets)

    if schema == "a4":
        raw = normalize_raw_a4(sheets["Raw Executions Log"]) if "Raw Executions Log" in sheets else pd.DataFrame(columns=RAW_COLS)
        summary = normalize_summary_a4(sheets["Summary Metrics"]) if "Summary Metrics" in sheets else summarize_raw(raw)
    elif schema == "a5":
        raw = normalize_raw_a5(sheets["Raw_Execution_Logs"]) if "Raw_Execution_Logs" in sheets else pd.DataFrame(columns=RAW_COLS)
        summary = normalize_summary_a5(sheets["Sliced_Summary"]) if "Sliced_Summary" in sheets else summarize_raw(raw)
    else:
        return pd.DataFrame(columns=SUMMARY_COLS), pd.DataFrame(columns=RAW_COLS), schema

    return sort_by_task_type(summary), raw, schema


def sort_by_task_type(df: pd.DataFrame) -> pd.DataFrame:
    """Stable slice ordering; unknown types keep their relative position at the end."""
    if df.empty or "task_type" not in df:
        return df
    rank = {t: i for i, t in enumerate(TASK_TYPE_ORDER)}
    out = df.copy()
    out["_rank"] = out["task_type"].map(lambda t: rank.get(t, len(rank)))
    return out.sort_values(["config", "_rank", "task_type"]).drop(columns="_rank").reset_index(drop=True)


def filter_frame(df: pd.DataFrame, configs: List[str], task_types: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df
    if configs:
        out = out[out["config"].isin(configs)]
    if task_types:
        out = out[out["task_type"].isin(task_types)]
    return out.reset_index(drop=True)
