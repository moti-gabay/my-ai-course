"""
app.py - LLM Playground & Evaluation Dashboard for the Assignment 4/5 agents.

Run:  cd lesson4 && streamlit run dashboard/app.py
Needs OPENAI_API_KEY in lesson4/.env for the Playground and Benchmark tabs;
the Analytics tab reads Excel results and works without a key.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

LESSON4_DIR = Path(__file__).resolve().parents[1]
if str(LESSON4_DIR) not in sys.path:
    sys.path.insert(0, str(LESSON4_DIR))

from dashboard import analytics as an  # noqa: E402
from dashboard import runner_bridge as rb  # noqa: E402
from dashboard import ui_components as ui  # noqa: E402

st.set_page_config(page_title="Agent Playground & Eval", page_icon="🧪", layout="wide")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_tasks() -> list:
    return rb.load_tasks()


@st.cache_data(show_spinner=False)
def cached_agents_md() -> str:
    return rb.default_agents_md()


def init_state() -> None:
    defaults = ui.sidebar_defaults(cached_agents_md())
    defaults.update({
        "pg_task_pick": "(custom query)",
        "pg_query": "",
        "pg_result": None,
        "bm_all": False,
        "bm_types": [],
        "bm_ids": [],
        "bm_runs": 1,
        "bm_configs": ["single"],
        "bm_rows": [],
        "bm_output_path": None,
        "bm_status": "idle",
        "an_last_source": None,
        "an_recompute": False,
        "an_search": "",
    })
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    # A rerun during a benchmark aborts the loop; flag the partial result.
    if st.session_state["bm_status"] == "running":
        st.session_state["bm_status"] = "interrupted"


init_state()
TASKS = cached_tasks()
cfg = ui.render_sidebar(cached_agents_md())

st.title("🧪 LLM Playground & Evaluation Dashboard")
tab_playground, tab_benchmark, tab_analytics = st.tabs(["Playground", "Benchmark", "Analytics"])


# ---------------------------------------------------------------------------
# Tab 1 - Playground & live trace
# ---------------------------------------------------------------------------

def _fill_query_from_task() -> None:
    pick = st.session_state["pg_task_pick"]
    if pick == "(custom query)":
        return
    task_id = pick.split(" ", 1)[0]
    match = next((t for t in TASKS if t["task_id"] == task_id), None)
    if match:
        st.session_state["pg_query"] = match["task"]


def _run_one(mode: str, query: str, task_meta):
    """Runs one architecture, painting its trace live into a fresh status block."""
    status = st.status(f"Running {mode}…", expanded=True)
    with status:
        container = st.container()
        trace = ui.LiveTrace(container, status)
        runner = rb.run_single_live if mode == "single" else rb.run_team_live
        payload = runner(cfg, query, on_event=trace, task_meta=task_meta)

    result = payload["result"]
    failed = result.get("status", "success") != "success" or result.get("terminal_state") == "error"
    status.update(label=f"{mode} finished in {result.get('latency_seconds', 0):.2f}s",
                  state="error" if failed else "complete", expanded=False)
    ui.render_run_metrics(result, mode)
    return payload


with tab_playground:
    st.subheader("Interactive playground")

    labels = ["(custom query)"] + [f"{t['task_id']} [{t['type']}] {t['task'][:70]}" for t in TASKS]
    st.selectbox("Prefill from task_set.json", labels, key="pg_task_pick", on_change=_fill_query_from_task)

    with st.form("pg_form"):
        st.text_area("Query", key="pg_query", height=120, placeholder="Ask the insurance agent something…")
        submitted = st.form_submit_button(
            "▶ Run", type="primary", disabled=not ui.api_key_present()
        )

    query = st.session_state["pg_query"].strip()

    if submitted and not query:
        st.warning("Enter a query first.")
    elif submitted:
        pick = st.session_state["pg_task_pick"]
        task_meta = None
        if pick != "(custom query)":
            task_id = pick.split(" ", 1)[0]
            task_meta = next((t for t in TASKS if t["task_id"] == task_id), None)

        st.session_state["pg_result"] = {"query": query, "config": cfg.to_dict(), "runs": {}}
        modes = cfg.modes()
        columns = st.columns(len(modes))
        for column, mode in zip(columns, modes):
            with column:
                st.markdown(f"#### {'🤖 Single ReAct agent' if mode == 'single' else '👥 Multi-agent team'}")
                st.session_state["pg_result"]["runs"][mode] = _run_one(mode, query, task_meta)

    elif st.session_state["pg_result"]:
        saved = st.session_state["pg_result"]
        st.caption(f"Last run · {saved['config']['model_name']} · temp {saved['config']['temperature']} · “{saved['query'][:90]}”")
        runs = saved["runs"]
        columns = st.columns(len(runs))
        for column, (mode, payload) in zip(columns, runs.items()):
            with column:
                st.markdown(f"#### {'🤖 Single ReAct agent' if mode == 'single' else '👥 Multi-agent team'}")
                ui.render_run_metrics(payload["result"], mode)
                with st.expander("Execution trace", expanded=True):
                    ui.render_trace(st.container(), payload["events"])
    else:
        st.info("Pick a task or type a query, then press Run. The trace appears here step by step.")


# ---------------------------------------------------------------------------
# Tab 2 - Benchmark runner
# ---------------------------------------------------------------------------

LIVE_COLS = ["task_id", "type", "config", "run", "success", "refused", "terminal_state",
             "agent_turns", "tool_calls", "latency_ms", "input_tokens", "output_tokens"]

with tab_benchmark:
    st.subheader("Benchmark runner")

    selected = ui.render_task_picker(TASKS)

    col_a, col_b, col_c = st.columns([1, 2, 2])
    runs_per_task = col_a.number_input("Runs per task", 1, 10, key="bm_runs")
    configs = col_b.multiselect("Configurations", ["single", "team"], key="bm_configs")
    total_runs = len(selected) * len(configs) * int(runs_per_task)
    col_c.metric("Planned executions", total_runs)

    start = st.button("🚀 Run benchmark", type="primary",
                      disabled=not ui.api_key_present() or total_runs == 0)
    st.caption("Results are written to lesson4/benchmark_results/ after every task. "
               "assignment_04.xlsx and assignment_05.xlsx are never modified.")

    if start:
        st.session_state["bm_rows"] = []
        st.session_state["bm_status"] = "running"
        output_path = rb.new_benchmark_path()
        st.session_state["bm_output_path"] = str(output_path)

        progress = st.progress(0.0, text="Starting…")
        live_table = st.empty()

        def on_row(row, done, total):
            st.session_state["bm_rows"].append(row)
            progress.progress(
                done / total,
                text=f"{done}/{total} · {row['task_id']} · {row['config']} run {row['run']} → {row['terminal_state']}",
            )
            live_table.dataframe(
                pd.DataFrame(st.session_state["bm_rows"])[LIVE_COLS].tail(15),
                width="stretch", hide_index=True,
            )

        rb.run_benchmark_live(cfg, selected, int(runs_per_task), configs, on_row, output_path)
        st.session_state["bm_status"] = "done"
        progress.progress(1.0, text=f"Finished {total_runs} executions")

    if st.session_state["bm_status"] == "interrupted":
        st.warning("The previous benchmark was interrupted by a rerun. Rows completed before the interruption are kept below.")

    rows = st.session_state["bm_rows"]
    if rows:
        st.divider()
        st.markdown(f"### Metrics by category · {len(rows)} executions")
        ui.render_summary_table(rb.summarize_rows(rows), rate_cols=("success_rate", "refusal_rate"))

        with st.expander("Raw execution rows", expanded=False):
            st.dataframe(pd.DataFrame(rows)[LIVE_COLS], width="stretch", hide_index=True)

        path = Path(st.session_state["bm_output_path"])
        if st.session_state["bm_status"] == "interrupted" and not path.exists():
            rb.write_benchmark_xlsx(rows, path)
        if path.exists():
            st.download_button(
                "⬇ Download results (.xlsx)",
                data=path.read_bytes(),
                file_name=path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.caption(f"Saved to `{path}` — reload it in the Analytics tab.")


# ---------------------------------------------------------------------------
# Tab 3 - Analytics & file viewer
# ---------------------------------------------------------------------------

with tab_analytics:
    st.subheader("Analytics & file viewer")

    files = an.list_result_files()
    col_file, col_upload = st.columns([2, 2])
    choice = col_file.selectbox(
        "Result file",
        [str(p) for p in files],
        format_func=lambda p: str(Path(p).relative_to(LESSON4_DIR)),
        key="an_source",
    ) if files else None
    uploaded = col_upload.file_uploader("…or upload a workbook", type=["xlsx"])

    sheets = None
    source_id = None
    if uploaded is not None:
        sheets = an.load_workbook_bytes(uploaded.getvalue())
        source_id = f"upload:{uploaded.name}"
        st.caption(f"Showing uploaded file **{uploaded.name}**")
    elif choice:
        sheets = an.load_workbook(choice, Path(choice).stat().st_mtime)
        source_id = choice

    # A new workbook has its own configs and task types; stale filters would
    # silently hide slices that only exist in the new file.
    if source_id != st.session_state.get("an_last_source"):
        st.session_state["an_last_source"] = source_id
        for key in ("an_configs", "an_types"):
            st.session_state.pop(key, None)

    if not sheets:
        st.info("No .xlsx results found. Run a benchmark or upload a workbook.")
    else:
        summary, raw, schema = an.load_normalized(sheets)
        st.caption(f"Schema `{schema}` · sheets: {', '.join(sheets)} — {an.SCHEMA_NOTES[schema]}")

        if schema == "unknown":
            for name, frame in sheets.items():
                st.markdown(f"**{name}**")
                st.dataframe(frame, width="stretch", hide_index=True)
        else:
            all_configs = sorted(set(summary["config"]) | set(raw["config"]))
            color_map = ui.build_color_map(all_configs)
            all_types = [t for t in an.TASK_TYPE_ORDER if t in set(summary["task_type"])]
            all_types += [t for t in summary["task_type"].unique() if t not in all_types]

            f1, f2, f3 = st.columns([2, 3, 2])
            sel_configs = f1.multiselect("Configurations", all_configs, default=all_configs, key="an_configs")
            sel_types = f2.multiselect("Task types", all_types, default=all_types, key="an_types")
            recompute = f3.toggle("Recompute from raw", key="an_recompute",
                                  help="Recompute the summary from the raw sheet, averaging total tokens instead of the file's own convention.")

            source = an.summarize_raw(raw) if recompute else summary
            view = an.filter_frame(an.sort_by_task_type(source), sel_configs, sel_types)

            st.markdown("### Sliced summary")
            ui.render_summary_table(view)

            if view.empty:
                st.info("No rows match the current filters.")
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.plotly_chart(ui.chart_grouped_bar(view, "success_rate", "Success rate by task type", "Success rate", color_map, percent=True), width="stretch")
                with c2:
                    st.plotly_chart(ui.chart_grouped_bar(view, "latency_p50_ms", "Median latency by task type", "Latency p50 (ms)", color_map), width="stretch")
                with c3:
                    st.plotly_chart(ui.chart_grouped_bar(view, "avg_tokens", "Average tokens by task type", "Tokens per run", color_map), width="stretch")

            st.markdown("### Raw executions")
            raw_view = an.filter_frame(raw, sel_configs, sel_types)
            search = st.text_input("Filter rows (task id or answer text)", key="an_search")
            if search:
                needle = search.lower()
                mask = raw_view["task_id"].astype(str).str.lower().str.contains(needle) | \
                    raw_view["answer"].astype(str).str.lower().str.contains(needle)
                raw_view = raw_view[mask]
            st.caption(f"{len(raw_view)} rows")
            st.dataframe(raw_view, width="stretch", hide_index=True)
