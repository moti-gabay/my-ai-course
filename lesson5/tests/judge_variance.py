"""Measure judge stability: repeat one sanity check N times with the cache bypassed.
Run: .venv/bin/python tests/judge_variance.py s9_premise_from_question s10c_real_t32_single --n 5
Writes results/judge_variance.json (appends one entry per case per run)."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import judge  # noqa: E402

KIND = {"faithfulness": (judge.FAITHFULNESS_RUBRIC, judge.FaithfulnessVerdict),
        "task_success": (judge.SUCCESS_RUBRIC, judge.SuccessVerdict),
        "agent_turn": (judge.AGENT_TURN_RUBRIC, judge.AgentTurnVerdict)}


def user_prompt(kind: str, inputs: dict) -> str:
    """Rebuild the exact user prompt the judge function sends (same helpers, no cache)."""
    if kind == "faithfulness":
        return (f"QUESTION:\n{inputs.get('question') or '(not provided)'}\n\n"
                f"TOOL OUTPUTS:\n{judge._format_tool_outputs(inputs['tool_outputs'])}\n\nANSWER TO JUDGE:\n{inputs['answer']}")
    raise NotImplementedError(kind)


ap = argparse.ArgumentParser()
ap.add_argument("case_ids", nargs="+")
ap.add_argument("--n", type=int, default=5)
a = ap.parse_args()
cases = {c["id"]: c for c in json.loads((ROOT / "tests" / "judge_sanity.json").read_text(encoding="utf-8"))}
out_path = ROOT / "results" / "judge_variance.json"
log = json.loads(out_path.read_text()) if out_path.exists() else []
for cid in a.case_ids:
    chk = cases[cid]["checks"][0]
    system, schema = KIND[chk["judge"]]
    verdicts = []
    for i in range(a.n):
        resp = judge._client_().messages.parse(model=judge.JUDGE_MODEL, max_tokens=16000, system=system,
                                               messages=[{"role": "user", "content": user_prompt(chk["judge"], chk["inputs"])}],
                                               output_format=schema)
        v = resp.parsed_output.verdict if resp.parsed_output else "judge_error"
        verdicts.append(v)
        print(f"{cid} #{i + 1}: {v}  ({resp.usage.input_tokens}/{resp.usage.output_tokens})")
    expected = chk["expected"] if isinstance(chk["expected"], list) else [chk["expected"]]
    agree = sum(v in expected for v in verdicts)
    print(f"{cid}: {dict(Counter(verdicts))}  agree with expected {agree}/{a.n}\n")
    log.append({"case": cid, "model": judge.JUDGE_MODEL, "n": a.n, "verdicts": verdicts,
                "expected": expected, "agree": f"{agree}/{a.n}"})
out_path.write_text(json.dumps(log, indent=2) + "\n")
