"""
ui_components.py - Reusable Streamlit widgets, trace renderers and charts.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .runner_bridge import RunConfig, AGENT_SYSTEM_PROMPT
from .analytics import TASK_TYPE_ORDER

MODEL_CHOICES = ["claude-haiku-4-5", "claude-sonnet-5"]

# Validated categorical palette, assigned to configs in fixed order (never cycled,
# never re-assigned when a filter drops a series).
SERIES_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]

AGENT_ICONS = {
    "single_agent": "🤖",
    "orchestrator": "🧭",
    "researcher": "📚",
    "analyst": "🧮",
    "writer": "✍️",
    "team": "👥",
}


def api_key_present() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(default_agents_md: str) -> RunConfig:
    """Reads the widget state and returns a fresh RunConfig on every rerun."""
    sb = st.sidebar
    sb.title("⚙️ Configuration")

    if api_key_present():
        sb.caption("✅ ANTHROPIC_API_KEY loaded from lesson5/.env")
    else:
        sb.error("No ANTHROPIC_API_KEY. Playground and Benchmark are disabled; Analytics still works.")

    mode = sb.radio(
        "Mode",
        options=["single", "team", "both"],
        format_func=lambda m: {"single": "Single ReAct Agent", "team": "Multi-Agent Team", "both": "Both (compare)"}[m],
        key="cfg_mode",
    )

    model = sb.selectbox("Model", MODEL_CHOICES, key="cfg_model")
    override = sb.text_input("Model override", key="cfg_model_override", placeholder="leave blank to use the dropdown")
    model_name = override.strip() or model

    temperature = sb.slider("Temperature", 0.0, 2.0, step=0.1, key="cfg_temperature")

    with sb.expander("Single agent limits", expanded=False):
        max_iterations = st.number_input("Max iterations", 1, 50, key="cfg_max_iterations")
        timeout_seconds = st.number_input(
            "Timeout (s)", 5.0, 300.0, step=5.0, key="cfg_timeout",
            help="Checked between graph steps, so a single slow model call can overshoot it.",
        )
        system_prompt = st.text_area("System prompt", height=260, key="cfg_system_prompt")

    with sb.expander("Team safety nets & procedural memory", expanded=False):
        max_turns = st.number_input("Max worker turns", 1, 30, key="cfg_max_turns")
        max_tokens = st.number_input("Token budget", 1000, 100000, step=1000, key="cfg_max_tokens")
        team_timeout = st.number_input("Wall-clock timeout (s)", 5.0, 300.0, step=5.0, key="cfg_team_timeout")
        agents_md = st.text_area(
            "AGENTS.md (procedural memory)", height=320, key="cfg_agents_md",
            help="Injected into every worker system prompt. Edits apply to the next run and never touch AGENTS.md on disk.",
        )

    return RunConfig(
        mode=mode,
        model_name=model_name,
        temperature=temperature,
        max_iterations=int(max_iterations),
        timeout_seconds=float(timeout_seconds),
        system_prompt=system_prompt or AGENT_SYSTEM_PROMPT,
        max_turns=int(max_turns),
        max_tokens=int(max_tokens),
        team_timeout_seconds=float(team_timeout),
        procedural_memory_text=agents_md if agents_md is not None else default_agents_md,
    )


def sidebar_defaults(default_agents_md: str) -> Dict[str, Any]:
    """Initial values for every cfg_* widget key."""
    return {
        "cfg_mode": "single",
        "cfg_model": MODEL_CHOICES[0],
        "cfg_model_override": "",
        "cfg_temperature": 0.0,
        "cfg_max_iterations": 10,
        "cfg_timeout": 30.0,
        "cfg_system_prompt": AGENT_SYSTEM_PROMPT,
        "cfg_max_turns": 8,
        "cfg_max_tokens": 12000,
        "cfg_team_timeout": 45.0,
        "cfg_agents_md": default_agents_md,
    }


# ---------------------------------------------------------------------------
# Trace rendering (one renderer for both architectures)
# ---------------------------------------------------------------------------

def render_step_event(container, ev: Dict[str, Any]) -> None:
    kind = ev.get("kind")
    icon = AGENT_ICONS.get(ev.get("agent", ""), "•")
    stamp = f"{ev.get('t_offset_s', 0):.2f}s"
    title = f"{icon} {ev.get('title')}  ·  {stamp}"

    if kind == "tool_call":
        with container.expander(title, expanded=False):
            st.json(ev.get("data") or {})

    elif kind == "tool_result":
        with container.expander(title, expanded=False):
            st.code(ev.get("body") or "", language="text")

    elif kind == "handoff":
        with container.expander(title, expanded=True):
            if ev.get("body"):
                st.caption(f"Reason: {ev['body']}")
            payload = (ev.get("data") or {}).get("payload") or {}
            st.markdown("**Handoff payload**")
            st.markdown(f"- **Summary:** {payload.get('summary') or '_empty_'}")
            constraints = payload.get("constraints") or []
            st.markdown(f"- **Constraints:** {', '.join(constraints) if constraints else '_none_'}")
            st.markdown(f"- **Open question:** {payload.get('open_question') or '_empty_'}")
            st.markdown("- **Facts:**")
            st.json(payload.get("facts") or {})

    elif kind == "worker_done":
        data = ev.get("data") or {}
        label = f"{title}  ·  {data.get('tool_calls', 0)} tool calls"
        with container.expander(label, expanded=False):
            st.markdown(ev.get("body") or "_no output_")

    elif kind == "summary":
        data = ev.get("data") or {}
        route = " → ".join(data.get("route_history") or [])
        container.caption(
            f"Route: {route}  ·  terminal state: `{data.get('terminal_state')}`  ·  "
            f"{data.get('total_turns')} worker turns  ·  ~{data.get('total_tokens')} tokens"
        )

    elif kind == "final":
        container.markdown("**Final answer**")
        container.markdown(ev.get("body") or "_empty_")

    elif kind == "error":
        container.error(f"{ev.get('title')}\n\n{ev.get('body') or ''}")


def render_trace(container, events: List[Dict[str, Any]]) -> None:
    for ev in events:
        render_step_event(container, ev)


class LiveTrace:
    """Callback target that paints each StepEvent as it arrives."""

    def __init__(self, container, status=None):
        self.container = container
        self.status = status

    def __call__(self, ev: Dict[str, Any]) -> None:
        render_step_event(self.container, ev)
        if self.status is not None and ev.get("kind") not in {"final", "error"}:
            self.status.update(label=f"{AGENT_ICONS.get(ev.get('agent',''), '•')} {ev.get('title')}")


def render_run_metrics(result: Dict[str, Any], mode: str) -> None:
    cols = st.columns(5)
    cols[0].metric("Latency", f"{result.get('latency_seconds', 0):.2f}s")
    cols[1].metric("Tokens", f"{result.get('total_tokens', 0):,}")
    if mode == "single":
        cols[2].metric("Tool calls", result.get("tool_calls_count", 0))
        cols[3].metric("Status", result.get("status", "?"))
    else:
        cols[2].metric("Worker turns", result.get("worker_turns", 0))
        cols[3].metric("Terminal state", str(result.get("terminal_state") or "?"))
    cols[4].metric("Refused", "yes" if result.get("is_refused") else "no")

    if mode == "team":
        st.caption("Team token counts are estimates: team.py books a flat 150 per orchestrator turn and 750 + 200/tool-call per worker turn.")


# ---------------------------------------------------------------------------
# Benchmark widgets
# ---------------------------------------------------------------------------

def render_task_picker(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {t["task_id"]: t for t in tasks}
    categories = sorted({t["type"] for t in tasks})

    select_all = st.checkbox(f"Select all {len(tasks)} tasks", key="bm_all")

    col_a, col_b = st.columns(2)
    picked_types = col_a.multiselect("By category", categories, key="bm_types", disabled=select_all)
    labels = {f"{t['task_id']} [{t['type']}] {t['task'][:60]}": t["task_id"] for t in tasks}
    picked_labels = col_b.multiselect("By task id", list(labels), key="bm_ids", disabled=select_all)

    if select_all:
        chosen = list(by_id)
    else:
        chosen = [t["task_id"] for t in tasks if t["type"] in picked_types]
        chosen += [labels[lbl] for lbl in picked_labels]

    ordered = [by_id[tid] for tid in dict.fromkeys(chosen)]
    return ordered


def render_summary_table(df: pd.DataFrame, rate_cols=("success_rate", "refusal_rate")) -> None:
    if df.empty:
        st.info("No rows yet.")
        return
    column_config = {}
    for col in rate_cols:
        if col in df.columns:
            column_config[col] = st.column_config.ProgressColumn(
                col.replace("_", " ").title(), min_value=0.0, max_value=1.0, format="%.2f"
            )
    for col in ("latency_p50_ms", "latency_p50", "avg_tokens", "avg_turns", "avg_turns_or_tool_calls"):
        if col in df.columns:
            column_config[col] = st.column_config.NumberColumn(col.replace("_", " ").title(), format="%.1f")
    st.dataframe(df, width="stretch", hide_index=True, column_config=column_config)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def build_color_map(configs: List[str]) -> Dict[str, str]:
    """Colour follows the config, not its rank, so filtering never repaints survivors.
    Build this from the UNFILTERED frame once and reuse it for every chart."""
    return {cfg: SERIES_COLORS[i % len(SERIES_COLORS)] for i, cfg in enumerate(configs)}


def chart_grouped_bar(
    df: pd.DataFrame,
    y: str,
    title: str,
    y_label: str,
    color_map: Dict[str, str],
    percent: bool = False,
) -> go.Figure:
    """One measure per chart, grouped by task type, coloured by config."""
    data = df.copy()
    if percent:
        data[y] = data[y] * 100.0

    order = [t for t in TASK_TYPE_ORDER if t in set(data["task_type"])]
    order += [t for t in data["task_type"].unique() if t not in order]

    fig = px.bar(
        data,
        x="task_type",
        y=y,
        color="config",
        barmode="group",
        title=title,
        category_orders={"task_type": order, "config": list(color_map)},
        color_discrete_map=color_map,
        labels={"task_type": "", y: y_label, "config": ""},
    )
    fig.update_traces(
        marker_line_width=0,
        hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>" + y_label + ": %{y:.1f}<extra></extra>",
    )
    fig.update_layout(
        bargap=0.28,
        bargroupgap=0.08,
        margin=dict(l=8, r=8, t=56, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        height=380,
        title=dict(font=dict(size=15)),
    )
    fig.update_yaxes(
        rangemode="tozero",
        title_text=y_label,
        gridcolor="rgba(128,128,128,0.22)",
        zerolinecolor="rgba(128,128,128,0.4)",
    )
    fig.update_xaxes(showgrid=False, tickangle=-35)
    if percent:
        fig.update_yaxes(range=[0, 100], ticksuffix="%")
    return fig
