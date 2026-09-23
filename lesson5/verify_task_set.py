"""
verify_task_set.py - check task_set.json against the real corpus. No LLM calls.

  1. schema and composition (spec: 30-45 tasks, 8+ cross_domain, 4 misroute_bait, 3+ no_tool,
     2 handoff_stress, 2+ unanswerable)
  2. every evidence quote appears verbatim (whitespace-normalised) on the cited page/section
  3. every amount a predicate checks is grounded: it appears in a cited quote, in the task
     text, or is the task's computed calculation result
  4. every reference answer passes its own predicate (or defers to the judge)

Run: .venv/bin/python verify_task_set.py      exit code 1 on any failure
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import retriever
from scoring import JUDGE, RunView, evaluate, numbers_in

HERE = Path(__file__).resolve().parent
AGENTS = {"orchestrator", "researcher", "analyst", "writer"}
TOOLS = {"search_docs", "calculator", "read_policy_page"}
TYPES = {"single", "cross_domain", "misroute_bait", "no_tool", "handoff_stress", "unanswerable", "tool_fails"}
REQUIRED = ["task_id", "type", "task", "answerable", "success_criteria", "reference_answer", "evidence",
            "capable_agents", "expected_agents", "source"]
COMPOSITION = {"cross_domain": (8, None), "misroute_bait": (4, 4), "no_tool": (3, None),
               "handoff_stress": (2, 2), "unanswerable": (2, None)}


WORD_NUMBERS = {"half": 0.5, "two": 2, "three": 3, "twelve": 12, "fourteen": 14}


def grounded_numbers(text: str) -> set:
    """Numbers stated in text: digits, number words, and percentages as factors (18% -> 0.18, 1.18)."""
    nums = set(numbers_in(text))
    nums |= {v for w, v in WORD_NUMBERS.items() if re.search(rf"\b{w}\b", text, re.I)}
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%", text):
        p = float(m.group(1)) / 100
        nums |= {p, round(1 + p, 6)}
    return nums


def calculations(t: dict) -> list:
    c = t.get("calculation")
    return c if isinstance(c, list) else ([c] if c else [])


def norm(text: str) -> str:
    return " ".join(text.replace("’", "'").split())


def amounts(pred) -> list:
    if isinstance(pred, dict):
        own = [float(pred["contains_amount"])] if "contains_amount" in pred else []
        subs = pred.get("all") or pred.get("any") or ([pred["not"]] if "not" in pred else [])
        return own + [a for p in subs for a in amounts(p)]
    return []


def main() -> int:
    tasks = json.loads((HERE / "task_set.json").read_text(encoding="utf-8"))
    failures = []

    def fail(task_id, msg):
        failures.append(f"{task_id}: {msg}")
        print(f"  FAIL {task_id}: {msg}")

    print("== 1. schema and composition")
    ids = [t.get("task_id") for t in tasks]
    if len(ids) != len(set(ids)):
        fail("-", "duplicate task ids")
    for t in tasks:
        tid = t.get("task_id", "?")
        for field in REQUIRED:
            if field not in t:
                fail(tid, f"missing field {field}")
        if t["type"] not in TYPES:
            fail(tid, f"unknown type {t['type']}")
        if not set(t["capable_agents"]) <= AGENTS or not t["capable_agents"]:
            fail(tid, f"bad capable_agents {t['capable_agents']}")
        if not set(t["expected_agents"]) <= AGENTS:
            fail(tid, f"bad expected_agents {t['expected_agents']}")
        if not set(t.get("inject_faults", {})) <= TOOLS:
            fail(tid, f"bad inject_faults {t.get('inject_faults')}")
        if (t["type"] == "tool_fails") != bool(t.get("inject_faults")):
            fail(tid, "inject_faults must be set exactly on tool_fails tasks")
        if (t["type"] == "handoff_stress") != bool(t.get("required_constraints")):
            fail(tid, "required_constraints must be set exactly on handoff_stress tasks")
        try:
            evaluate(t["success_criteria"], RunView(answer="", refused=False, config="team", worker_turns=1, tool_calls=1))
        except (ValueError, KeyError, TypeError) as e:
            fail(tid, f"invalid predicate: {e}")
    counts = Counter(t["type"] for t in tasks)
    print(f"  total {len(tasks)}  " + "  ".join(f"{k}={counts[k]}" for k in sorted(TYPES)))
    if not 30 <= len(tasks) <= 45:
        fail("-", f"total {len(tasks)} outside 30-45")
    for typ, (lo, hi) in COMPOSITION.items():
        if counts[typ] < lo or (hi is not None and counts[typ] > hi):
            fail("-", f"{typ}={counts[typ]}, spec wants {lo}{'+' if hi is None else ''}")

    print("\n== 2. evidence quotes on the cited page")
    for t in tasks:
        for e in t["evidence"]:
            try:
                page_text = norm(retriever.read_page(e["policy"], e["page"]))
            except ValueError as err:
                fail(t["task_id"], f"cannot read {e['policy']} {e['page']}: {err}")
                continue
            ok = norm(e["quote"]) in page_text
            print(f"  {'PASS' if ok else 'FAIL'} {t['task_id']:4} {e['policy']:<10} {e['page']:<7} \"{norm(e['quote'])[:90]}\"")
            if not ok:
                fail(t["task_id"], f"quote not found on {e['policy']} {e['page']}")
        if t["answerable"] and not t["evidence"] and t["type"] not in ("no_tool",) and not calculations(t):
            fail(t["task_id"], "answerable task without evidence")

    print("\n== 3. predicate amounts are grounded")
    for t in tasks:
        grounded = grounded_numbers(" ".join(e["quote"] for e in t["evidence"])) | grounded_numbers(t["task"])
        inputs = set(grounded)
        grounded |= {float(c["result"]) for c in calculations(t)}
        for a in amounts(t["success_criteria"]):
            ok = any(abs(a - g) <= 0.01 for g in grounded)
            print(f"  {'PASS' if ok else 'FAIL'} {t['task_id']:4} amount {a:,.2f}")
            if not ok:
                fail(t["task_id"], f"amount {a} not in any quote, the task text, or the calculation result")
        for c in calculations(t):
            for n in numbers_in(c["expression"]):
                ok = any(abs(n - g) <= 1e-6 for g in inputs)
                print(f"  {'PASS' if ok else 'FAIL'} {t['task_id']:4} calculation input {n:g} ({c['expression']})")
                if not ok:
                    fail(t["task_id"], f"calculation input {n} is not in any quote or the task text")

    print("\n== 4. reference answer passes its own predicate")
    for t in tasks:
        no_tool = t["type"] == "no_tool"
        refused = not t["answerable"] or t["type"] == "tool_fails"
        run = RunView(answer=t["reference_answer"], refused=refused, config="team",
                      worker_turns=0 if no_tool else 1, tool_calls=0 if no_tool else 1)
        r = evaluate(t["success_criteria"], run)
        ok = r in (True, JUDGE)
        print(f"  {'PASS' if ok else 'FAIL'} {t['task_id']:4} {t['type']:<15} -> {r}")
        if not ok:
            fail(t["task_id"], "reference answer fails its own predicate")

    print(f"\n{len(failures)} failure(s)")
    for f in failures:
        print("  " + f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
