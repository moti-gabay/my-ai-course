"""Run the Sonnet judge on tests/judge_sanity.json and report agreement with the expected verdicts.

A case agrees only if every one of its verdicts matches. Results (with explanations and
token usage) go to results/judge_sanity_results.json. Cached verdicts cost nothing on rerun.
Run: .venv/bin/python tests/run_judge_sanity.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import judge  # noqa: E402

FNS = {"task_success": judge.judge_task_success, "faithfulness": judge.judge_faithfulness,
       "agent_turn": judge.judge_agent_turn}

cases = json.loads((ROOT / "tests" / "judge_sanity.json").read_text(encoding="utf-8"))
results, agree_cases, agree_checks, n_checks, tok_in, tok_out = [], 0, 0, 0, 0, 0
print(f"judge model: {judge.JUDGE_MODEL}\n")
for case in cases:
    case_ok, rows = True, []
    for chk in case["checks"]:
        r = FNS[chk["judge"]](**chk["inputs"])
        allowed = chk["expected"] if isinstance(chk["expected"], list) else [chk["expected"]]
        ok = r["verdict"] in allowed
        shown = "|".join(allowed)
        case_ok &= ok
        agree_checks += ok
        n_checks += 1
        tok_in += r["input_tokens"] if not r["cached"] else 0
        tok_out += r["output_tokens"] if not r["cached"] else 0
        rows.append({"judge": chk["judge"], "expected": chk["expected"], **r, "agree": ok})
        print(f"{'OK  ' if ok else 'DIFF'} {case['id']:<22} {chk['judge']:<13} expected={shown:<20} got={r['verdict']:<20}"
              f" tokens={r['input_tokens']}/{r['output_tokens']}{' (cached)' if r['cached'] else ''}")
        print(f"      explanation: {r['explanation']}")
    agree_cases += case_ok
    results.append({"id": case["id"], "about": case["about"], "agree": case_ok, "checks": rows})

print(f"\nagreement: {agree_cases}/{len(cases)} cases, {agree_checks}/{n_checks} verdicts")
print(f"tokens this run (uncached calls): input {tok_in}, output {tok_out}")
out = ROOT / "results" / "judge_sanity_results.json"
out.write_text(json.dumps({"model": judge.JUDGE_MODEL, "agreement_cases": f"{agree_cases}/{len(cases)}",
                           "agreement_verdicts": f"{agree_checks}/{n_checks}", "cases": results},
                          indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {out}")
