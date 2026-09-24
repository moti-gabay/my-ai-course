"""
project_cost.py - project the cost of Phases 6-7 from a dry-run workbook.

Per config it takes the measured tokens per run (agent + judge), then multiplies by the
number of runs each remaining phase needs. The dry run samples only a few tasks, so the
projection is a bracket: cheapest task, mean, most expensive task.

Usage: .venv/bin/python project_cost.py benchmark_results/dryrun.xlsx
Writes: results/cost_projection.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

# $ per million tokens (Anthropic list prices)
PRICES = {"claude-haiku-4-5": (1.00, 5.00), "claude-sonnet-5": (2.00, 10.00)}
AGENT_MODEL, JUDGE_MODEL = "claude-haiku-4-5", "claude-sonnet-5"

N_TASKS, N_RUNS = 41, 5
PHASES = [  # (label, config, runs)
    ("Phase 6: full matrix, single", "single", N_TASKS * N_RUNS),
    ("Phase 6: full matrix, team", "team", N_TASKS * N_RUNS),
    ("Phase 6: team without AGENTS.md", "team", N_TASKS * N_RUNS),
    ("Phase 7: fix 1 re-run, team", "team", N_TASKS * N_RUNS),
    ("Phase 7: fix 2 re-run, team", "team", N_TASKS * N_RUNS),
]


def dollars(model: str, tokens_in: float, tokens_out: float) -> float:
    p_in, p_out = PRICES[model]
    return tokens_in / 1e6 * p_in + tokens_out / 1e6 * p_out


def main(path: str) -> None:
    df = pd.read_excel(path, sheet_name="Raw_Execution_Logs")
    df["agent_usd"] = [dollars(AGENT_MODEL, i, o) for i, o in zip(df.input_tokens, df.output_tokens)]
    df["judge_usd"] = [dollars(JUDGE_MODEL, i, o) for i, o in zip(df.judge_input_tokens, df.judge_output_tokens)]
    df["usd"] = df.agent_usd + df.judge_usd

    print(f"Dry run: {path}  ({len(df)} runs)\n")
    cols = ["task_id", "type", "config", "terminal_state", "success", "input_tokens", "output_tokens",
            "judge_input_tokens", "judge_output_tokens", "latency_ms", "agent_usd", "judge_usd", "usd"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    per_run = {}
    print("\nPer run, by config (low = cheapest task, high = most expensive task):")
    for config, g in df.groupby("config"):
        per_run[config] = {"low": g.usd.min(), "mean": g.usd.mean(), "high": g.usd.max(),
                           "mean_agent_tokens": [g.input_tokens.mean(), g.output_tokens.mean()],
                           "mean_judge_tokens": [g.judge_input_tokens.mean(), g.judge_output_tokens.mean()]}
        r = per_run[config]
        print(f"  {config:6}  agent tokens {r['mean_agent_tokens'][0]:,.0f} in / {r['mean_agent_tokens'][1]:,.0f} out"
              f"  judge tokens {r['mean_judge_tokens'][0]:,.0f} in / {r['mean_judge_tokens'][1]:,.0f} out"
              f"  $/run low {r['low']:.4f}  mean {r['mean']:.4f}  high {r['high']:.4f}")

    print("\nProjection:")
    print(f"  {'phase':<36} {'runs':>5} {'low $':>9} {'mean $':>9} {'high $':>9}")
    rows, totals = [], {"low": 0.0, "mean": 0.0, "high": 0.0}
    for label, config, runs in PHASES:
        r = per_run[config]
        row = {"phase": label, "config": config, "runs": runs, **{k: r[k] * runs for k in totals}}
        rows.append(row)
        for k in totals:
            totals[k] += row[k]
        print(f"  {label:<36} {runs:>5} {row['low']:>9.2f} {row['mean']:>9.2f} {row['high']:>9.2f}")
    print(f"  {'total':<36} {sum(r['runs'] for r in rows):>5} {totals['low']:>9.2f} {totals['mean']:>9.2f} {totals['high']:>9.2f}")
    print("\nJudge cost is an upper bound: the cache makes identical (answer, inputs) pairs free on rerun.")

    out = Path(__file__).with_name("results") / "cost_projection.json"
    out.write_text(json.dumps({"dry_run": path, "prices_per_mtok": PRICES, "per_run_usd": per_run,
                               "phases": rows, "total_usd": totals}, indent=2, default=float) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/dryrun.xlsx")
